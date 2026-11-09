"""Continuous planar-base door task, isolated from the frozen fixed-base task.

The three base DOFs are an ideal holonomic drive, not a wheel model. Navigation
and arm preparation integrate mj_step. Only episode initialization resets data.
Geometry planning uses a separate MjData and never advances the physical door.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from door_stress_scenes import _style_door, _style_handles
from grasp_plan import GraspPlan
from perception import estimate_frame, observe
from prm import segment_box_distance_squared
from simulate import Experiment, ROOT, smooth

BASE_JOINTS = ('mobile_x', 'mobile_y', 'mobile_yaw')
BASE_ACTUATORS = ('mobile_x_servo', 'mobile_y_servo', 'mobile_yaw_servo')
STOW = np.array([0., -math.pi/2, -2*math.pi/3, 0.])
BASE_POSES = ((.6, -1., math.pi/2), (.3, -.9, math.pi/2),
              (.6, 1., -math.pi/2), (.3, .9, -math.pi/2))
GRASP_ANGLES = (-15., 22.)


@dataclass(frozen=True)
class MobileSettings:
    speed_m_s: float = .3
    yaw_speed_rad_s: float = .6
    grid_m: float = .12
    edge_step_m: float = .025
    joint_step_rad: float = .05
    arm_speed_rad_s: float = .7
    arm_samples: int = 180
    clearance_m: float = .008
    seed: int = 42

    def __post_init__(self):
        if self.seed != 42:
            raise ValueError('New experiments use seed 42.')
        if any(not math.isfinite(v) or v <= 0 for v in (
                self.speed_m_s, self.yaw_speed_rad_s, self.grid_m,
                self.edge_step_m, self.joint_step_rad, self.arm_speed_rad_s, self.clearance_m)):
            raise ValueError('Motion rates and collision resolutions must be positive and finite.')
        if self.arm_samples < 0:
            raise ValueError('Arm sample budget must be nonnegative.')


def scene_xml(scene):
    root = ET.parse(ROOT/'scene.xml').getroot()
    world = root.find('worldbody')
    door = world.find("body[@name='door']")
    door.find("joint[@name='door_hinge']").set('range', f'{-math.pi/2} {math.pi/2}')
    _style_door(door, scene.door_style)
    _style_handles(door, scene.handle_style, scene.handle_faces, scene.handle_x, scene.handle_z)
    for obstacle in scene.obstacles:
        obstacle.append(world)
    base = world.find("body[@name='base']")
    base.set('pos', '0 0 0')
    base.set('euler', '0 0 0')
    # A constrained planar base keeps its height; there is no simulated wheel
    # suspension or ground traction. All horizontal/yaw motion is force driven.
    for i, (name, kind, axis, extent) in enumerate((
            ('mobile_x', 'slide', '1 0 0', '-4 4'),
            ('mobile_y', 'slide', '0 1 0', '-4 4'),
            ('mobile_yaw', 'hinge', '0 0 1', '-6.28 6.28'))):
        base.insert(i, ET.Element('joint', name=name, type=kind, axis=axis,
                                  range=extent, damping='15', armature='1'))
    base.find("geom[@name='base_box']").set('pos', '0 0 .16')
    camera = world.find("camera[@name='perception']")
    world.remove(camera)
    camera.set('pos', '.06 0 1.1')
    camera.set('xyaxes', '0 -1 0 .242535625 0 .9701425')
    camera.set('fovy', '65')
    base.append(camera)
    actuator = root.find('actuator')
    for name, joint, kp, kv, force in zip(
            BASE_ACTUATORS, BASE_JOINTS, (30000, 30000, 5000),
            (1800, 1800, 350), (1800, 1800, 800)):
        ET.SubElement(actuator, 'position', name=name, joint=joint,
                      kp=str(kp), kv=str(kv), forcerange=f'{-force} {force}')
    return ET.tostring(root, encoding='unicode')


class MobileDoor(Experiment):
    def __init__(self, scene, start_pose=(.6, -1.6, math.pi/2), settings=None):
        self.scene = scene
        self.start_pose = np.asarray(start_pose, dtype=float)
        if self.start_pose.shape != (3,) or not np.isfinite(self.start_pose).all():
            raise ValueError('Start pose must be finite [x, y, yaw].')
        self.settings = settings or MobileSettings()
        self.reset_count = 0
        self.initialized = False
        active = scene.handle_faces[0]
        super().__init__(base=(0., 0.), base_yaw=0., friction=scene.friction,
                         hinge_resistance=scene.hinge_resistance,
                         model_xml=scene_xml(scene), handle_body='handle' if active == 'front' else 'back_handle',
                         handle_geom='handle_bar' if active == 'front' else 'back_handle_bar')
        self.initialized = True
        self.trace = []
        self.phase_reports = []
        self.integration_steps = 0
        self._last_trace_time = -1.
        for face in scene.handle_faces:
            self.m.geom('handle_bar' if face == 'front' else 'back_handle_bar').friction[0] = scene.handle_friction
        self.record('initial')

    def reset(self, target=None, yaw=None):
        if self.initialized:
            raise RuntimeError('Mobile episodes cannot reset between phases; create a new episode instead.')
        self.reset_count += 1
        self.base_q = np.array([self.m.joint(n).qposadr[0] for n in BASE_JOINTS])
        self.base_dof = np.array([self.m.joint(n).dofadr[0] for n in BASE_JOINTS])
        self.base_ctrl = np.array([self.m.actuator(n).id for n in BASE_ACTUATORS])
        self.arm_qadr = np.r_[self.lift_q, self.arm_q]
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.base_q] = self.start_pose
        self.d.qpos[self.arm_qadr] = STOW
        self.d.qpos[self.finger_q] = .04
        self.base_target = self.start_pose.copy()
        self.d.ctrl[:6] = [*STOW, .04, .04]
        self.d.ctrl[self.base_ctrl] = self.base_target
        mujoco.mj_forward(self.m, self.d)
        self.sync_base()

    def sync_base(self):
        self.base = self.d.qpos[self.base_q[:2]].copy()
        self.base_yaw = float(self.d.qpos[self.base_q[2]])

    def command(self, point, yaw, aperture):
        self.sync_base()
        q = self.ik(point, yaw)
        if q is None:
            return False
        self.d.ctrl[:6] = [point[2]-.85, *q, aperture, aperture]
        self.d.ctrl[self.base_ctrl] = self.base_target
        self.d.qfrc_applied[:] = 0.
        self.d.qfrc_applied[self.robot_dof] = self.d.qfrc_bias[self.robot_dof]
        return True

    def set_side(self, side):
        if side not in self.scene.handle_faces:
            raise ValueError('Requested handle face is absent.')
        self.operating_side = side
        self.opening_direction = 1 if side == 'front' else -1
        self.ik_branch = self._default_ik_branch = -1 if side == 'front' else 1
        self.grasp_yaw_offset = 0. if side == 'front' else math.pi
        self.grasp_yaw_sign = 1 if side == 'front' else -1
        self.hbody = self.m.body('handle' if side == 'front' else 'back_handle').id
        self.bar = self.m.geom('handle_bar' if side == 'front' else 'back_handle_bar').id

    def record(self, phase, force=False):
        if not force and self.d.time-self._last_trace_time < .05-1e-8:
            return
        self._last_trace_time = float(self.d.time)
        self.trace.append(dict(time_s=float(self.d.time), phase=phase,
                               base_pose=self.d.qpos[self.base_q].tolist(),
                               base_velocity=self.d.qvel[self.base_dof].tolist(),
                               door_angle_deg=float(np.degrees(self.d.qpos[self.door_q]))))

    def _step(self, arm, base):
        self.base_target = np.asarray(base, dtype=float).copy()
        self.d.ctrl[:6] = [*arm, .04, .04]
        self.d.ctrl[self.base_ctrl] = self.base_target
        self.d.qfrc_applied[:] = 0.
        self.d.qfrc_applied[self.robot_dof] = self.d.qfrc_bias[self.robot_dof]
        mujoco.mj_step(self.m, self.d)
        self.integration_steps += 1
        self.sync_base()

    def track(self, arm_path=None, base_path=None, phase='navigation'):
        arm_path = np.asarray(arm_path if arm_path is not None else [self.d.qpos[self.arm_qadr]]*2)
        base_path = np.asarray(base_path if base_path is not None else [self.d.qpos[self.base_q]]*2)
        if len(arm_path) != len(base_path):
            if len(arm_path) == 2:
                arm_path = np.tile(arm_path[0], (len(base_path), 1))
            elif len(base_path) == 2:
                base_path = np.tile(base_path[0], (len(arm_path), 1))
            else:
                raise ValueError('Arm and base paths must align.')
        started = time.perf_counter()
        begin = float(self.d.time)
        start_pose = self.d.qpos[self.base_q].copy()
        stop, contact = 'arrived', None
        next_check = self.d.time
        dt = self.m.opt.timestep
        rates_arm = np.array([.08, *([self.settings.arm_speed_rad_s]*3)])
        rates_base = np.array([self.settings.speed_m_s, self.settings.speed_m_s, self.settings.yaw_speed_rad_s])
        for a, b, u, v in zip(arm_path, arm_path[1:], base_path, base_path[1:]):
            duration = max(.15, 1.6*float(np.max(np.abs(b-a)/rates_arm)),
                           1.6*float(np.linalg.norm(v[:2]-u[:2]))/self.settings.speed_m_s,
                           1.6*abs(float(v[2]-u[2]))/self.settings.yaw_speed_rad_s)
            for i in range(1, math.ceil(duration/dt)+1):
                blend = smooth(min(1., i*dt/duration))
                self._step(a+blend*(b-a), u+blend*(v-u))
                if self.d.time >= next_check:
                    mujoco.mj_forward(self.m, self.d)
                    next_check += .01
                    self.record(phase)
                    unexpected = self.contact_metrics()[-1]
                    if unexpected:
                        stop, contact = phase+'_collision', unexpected[0]
                        break
                    if abs(self.d.qpos[self.door_q]) > math.radians(.5):
                        stop = 'door_moved_during_'+phase
                        break
            if stop != 'arrived':
                break
        if stop == 'arrived':
            for _ in range(math.ceil(.8/dt)):
                self._step(arm_path[-1], base_path[-1])
                if self.d.time >= next_check:
                    mujoco.mj_forward(self.m, self.d)
                    next_check += .01
                    self.record(phase)
                    unexpected = self.contact_metrics()[-1]
                    if unexpected:
                        stop, contact = phase+'_collision', unexpected[0]
                        break
            arm_error = float(np.max(np.abs(self.d.qpos[self.arm_qadr]-arm_path[-1])))
            xy_error = float(np.linalg.norm(self.d.qpos[self.base_q[:2]]-base_path[-1,:2]))
            yaw_error = abs(float(self.d.qpos[self.base_q[2]]-base_path[-1,2]))
            if stop == 'arrived' and (arm_error > .035 or xy_error > .008 or yaw_error > .02):
                stop = 'tracking_error'
        else:
            arm_error = xy_error = yaw_error = None
        self.record(phase, True)
        report = dict(success=stop == 'arrived', stop_reason=stop, collision=contact is not None,
                      first_contact=contact, start_time_s=begin, end_time_s=float(self.d.time),
                      wall_seconds=time.perf_counter()-started, start_base_pose=start_pose.tolist(),
                      end_base_pose=self.d.qpos[self.base_q].tolist(),
                      arm_error_max=arm_error, xy_error_m=xy_error, yaw_error_rad=yaw_error)
        self.phase_reports.append(report)
        return report

    def trial_continuous(self, plan, observer=None):
        """Reuse the original contact skill without resetting physical state.

        The legacy skill reads d.time as its local clock. Rebase only that
        scalar during its call, then restore the episode clock. qpos/qvel,
        contacts, actuator dynamics and passive door state are preserved.
        """
        offset = float(self.d.time)
        before_q, before_v = self.d.qpos.copy(), self.d.qvel.copy()
        self.d.time = 0.
        entered = []
        def sample(exp, terminal):
            if not entered:
                entered.append(bool(np.array_equal(before_q, exp.d.qpos) and np.array_equal(before_v, exp.d.qvel)))
            local = float(exp.d.time)
            exp.d.time = offset+local
            exp.record('opening', True)
            if observer is not None:
                observer(exp, terminal)
            exp.d.time = local
        try:
            result, rows, states = super().trial(plan=plan, reset_before_trial=False,
                                                observer=sample, observation_interval=.05)
        finally:
            self.integration_steps += round(self.d.time/self.m.opt.timestep)
            self.d.time += offset
            self.sync_base()
        for row in rows:
            row['episode_time_s'] = offset+row['time_s']
        result.update(continuous_state_at_skill_entry=bool(entered and entered[0]),
                      episode_start_time_s=offset, episode_end_time_s=float(self.d.time),
                      base_reset_count=self.reset_count)
        return result, rows, states


class CollisionChecker:
    def __init__(self, exp):
        self.exp = exp
        # A separate geometry model supplies positive-distance contacts for a
        # clearance margin. It never changes live collision/contact physics.
        self.m = mujoco.MjModel.from_xml_string(scene_xml(exp.scene))
        self.m.geom_margin[:] = exp.settings.clearance_m
        self.data = mujoco.MjData(self.m)
        self.data.qpos[:] = exp.d.qpos
        self.checks = 0
        self.panel = self.m.geom('door_panel').id
        self.arm_capsules = [self.m.geom(n).id for n in ('upper_link', 'forearm_link', 'wrist_link')]
        self.base_box=self.m.geom('base_box').id
        self.base_radius=float(np.linalg.norm(self.m.geom_size[self.base_box,:2]))+.015
        base=self.m.body('base').id
        self.robot_geoms=set()
        for gid,body in enumerate(self.m.geom_bodyid):
            ancestor=int(body)
            while ancestor and ancestor!=base:
                ancestor=int(self.m.body_parentid[ancestor])
            if ancestor==base:self.robot_geoms.add(gid)

    def invalid(self, base, arm=None):
        self.checks += 1
        self.data.qpos[self.exp.base_q] = base
        self.data.qpos[self.exp.arm_qadr] = STOW if arm is None else arm
        self.data.qpos[self.exp.finger_q] = .04
        mujoco.mj_forward(self.m, self.data)
        # MuJoCo box-box contact margins alone do not give a continuous
        # clearance certificate near rotated crate corners. Use a conservative
        # circumscribed base footprint, then check the rest of the robot below.
        center=self.data.geom_xpos[self.base_box]
        for gid in range(self.m.ngeom):
            if gid in self.robot_geoms or not self.m.geom_contype[gid]:
                continue
            kind=self.m.geom_type[gid]
            relative=self.data.geom_xmat[gid].reshape(3,3).T@(center-self.data.geom_xpos[gid])
            size=self.m.geom_size[gid]
            if kind==mujoco.mjtGeom.mjGEOM_BOX:
                if abs(relative[2])>size[2]+self.m.geom_size[self.base_box,2]:
                    continue
                excess=np.maximum(np.abs(relative[:2])-size[:2],0.)
                if excess@excess<self.base_radius**2:
                    return 'base_footprint:'+self.m.geom(gid).name
            elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
                if abs(relative[2])<=size[1]+self.m.geom_size[self.base_box,2] and np.linalg.norm(relative[:2])<size[0]+self.base_radius:
                    return 'base_footprint:'+self.m.geom(gid).name
        for contact in self.data.contact:
            robot_count=sum(int(g) in self.robot_geoms for g in (contact.geom1,contact.geom2))
            # Door/floor and obstacle/floor support are static geometry pairs;
            # they cannot invalidate every free-motion query. Positive safety
            # clearance applies only between robot and environment geometry.
            if not robot_count:
                continue
            threshold=self.exp.settings.clearance_m if robot_count==1 else -1e-5
            if contact.dist < threshold:
                return ':'.join(self.m.geom(int(g)).name for g in (contact.geom1, contact.geom2))
        rotation = self.data.geom_xmat[self.panel].reshape(3, 3)
        center = self.data.geom_xpos[self.panel]
        for gid in self.arm_capsules:
            axis = self.data.geom_xmat[gid].reshape(3, 3)[:,2]
            pos, size = self.data.geom_xpos[gid], self.m.geom_size[gid]
            a = rotation.T@(pos-size[1]*axis-center)
            b = rotation.T@(pos+size[1]*axis-center)
            if segment_box_distance_squared(a, b, self.m.geom_size[self.panel]) < (size[0]+self.exp.settings.clearance_m)**2:
                return self.m.geom(gid).name+':door_panel'
        return None

    def base_edge(self, a, b):
        count = max(1, math.ceil(max(np.linalg.norm(np.asarray(b)[:2]-np.asarray(a)[:2])/self.exp.settings.edge_step_m,
                                     abs(b[2]-a[2])/.04)))
        return all(self.invalid(q) is None for q in np.linspace(a,b,count+1))

    def arm_edge(self, a, b):
        count = max(1, math.ceil(float(np.max(np.abs((np.asarray(b)-a)/np.array([.008, *([self.exp.settings.joint_step_rad]*3)]))))))
        return all(self.invalid(self.exp.d.qpos[self.exp.base_q], q) is None for q in np.linspace(a,b,count+1))


def plan_navigation(exp, goal):
    """Deterministic bounded A*; collision-check the complete stowed robot."""
    checker = CollisionChecker(exp)
    start, goal = exp.d.qpos[exp.base_q].copy(), np.asarray(goal, float)
    turned = np.r_[start[:2], goal[2]]
    if not checker.base_edge(start, turned):
        return dict(path_found=False, reason='start_rotation_blocked', checks=checker.checks)
    if checker.invalid(goal):
        return dict(path_found=False, reason='goal_in_collision', checks=checker.checks)
    if checker.base_edge(turned, goal):
        path = [start, turned, goal]
    else:
        step = exp.settings.grid_m
        minimum, maximum = np.array([-1.8,-1.9]), np.array([2.4,1.9])
        def point(node): return np.r_[start[:2]+step*np.array(node), goal[2]]
        origin = (0,0)
        queue, distance, parent = [(float(np.linalg.norm(goal[:2]-start[:2])),0.,origin)], {origin:0.}, {}
        valid = {origin:True}
        found = None
        while queue:
            _, cost, node = heapq.heappop(queue)
            if cost > distance[node]+1e-9:
                continue
            p = point(node)
            if np.linalg.norm(p[:2]-goal[:2]) < step*1.5 and checker.base_edge(p,goal):
                found = node
                break
            for delta in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                nxt = (node[0]+delta[0], node[1]+delta[1])
                q = point(nxt)
                if np.any(q[:2]<minimum) or np.any(q[:2]>maximum):
                    continue
                weight = float(np.linalg.norm(q[:2]-p[:2]))
                new = cost+weight
                if new >= distance.get(nxt, math.inf)-1e-9:
                    continue
                if nxt not in valid:
                    valid[nxt] = checker.invalid(q) is None
                if not valid[nxt] or not checker.base_edge(p,q):
                    continue
                distance[nxt], parent[nxt] = new, node
                heapq.heappush(queue,(new+float(np.linalg.norm(q[:2]-goal[:2])),new,nxt))
        if found is None:
            return dict(path_found=False, reason='no_path_in_bounded_grid', checks=checker.checks)
        nodes = [found]
        while nodes[-1] != origin:
            nodes.append(parent[nodes[-1]])
        raw = [point(node) for node in nodes[::-1]]+[goal]
        # Deterministic farthest-visible shortcut, still with fine edge checks.
        reduced, i = [raw[0]], 0
        while i < len(raw)-1:
            j = len(raw)-1
            while j>i+1 and not checker.base_edge(raw[i],raw[j]):
                j -= 1
            reduced.append(raw[j]); i=j
        path = [start, turned, *reduced[1:]]
    return dict(path_found=True, reason='collision_checked_path', path=np.asarray(path).tolist(),
                geometric_length_m=float(sum(np.linalg.norm(b[:2]-a[:2]) for a,b in zip(path,path[1:]))),
                checks=checker.checks)


def plan_arm(exp, plan):
    checker = CollisionChecker(exp)
    yaw = math.pi/2+plan.yaw_offset_rad(exp)
    pre = np.asarray(plan.point_world_m)-plan.approach_distance_m*np.array([math.cos(yaw),math.sin(yaw),0.])
    ik = exp.ik(pre,yaw,branch=plan.ik_branch)
    start = exp.d.qpos[exp.arm_qadr].copy()
    if ik is None:
        return dict(path_found=False, reason='pregrasp_ik_unreachable', checks=checker.checks)
    goal = np.r_[pre[2]-.85,ik]
    if checker.invalid(exp.d.qpos[exp.base_q],start) or checker.invalid(exp.d.qpos[exp.base_q],goal):
        return dict(path_found=False, reason='pregrasp_endpoint_collision', checks=checker.checks)
    if checker.arm_edge(start,goal):
        return dict(path_found=True, reason='direct', path=np.array([start,goal]).tolist(), checks=checker.checks)
    # RRT in bounded arm joints, all edges checked. Finite failure is not proof
    # that the base or scene has no possible opening action.
    rng = np.random.default_rng(exp.settings.seed)
    limits = exp.m.jnt_range[[exp.m.joint('lift_z').id,*exp.arm_j]]
    nodes, parents = [start], [-1]
    scale = np.array([.24,math.pi,math.pi,math.pi])
    for i in range(exp.settings.arm_samples):
        sample = goal if i%4==0 else rng.uniform(limits[:,0],limits[:,1])
        nearest = int(np.argmin([np.linalg.norm((q-sample)/scale) for q in nodes]))
        delta = sample-nodes[nearest]
        q = nodes[nearest]+delta*min(1.,.25/max(1e-9,float(np.linalg.norm(delta/scale))))
        if not checker.arm_edge(nodes[nearest],q):
            continue
        nodes.append(q); parents.append(nearest)
        if checker.arm_edge(q,goal):
            path, j = [goal], len(nodes)-1
            while j>=0:
                path.append(nodes[j]); j=parents[j]
            return dict(path_found=True,reason='rrt',path=np.array(path[::-1]).tolist(),checks=checker.checks)
    return dict(path_found=False,reason='no_arm_path_within_budget',checks=checker.checks)


def original_proprio(exp, local_time=0.):
    """Keep the original 19-D checkpoint contract; report mobile state separately."""
    names = ('door_hinge','lift_z','shoulder','elbow','wrist_yaw','left_slide','right_slide')
    qa = [exp.m.joint(n).qposadr[0] for n in names]
    va = [exp.m.joint(n).dofadr[0] for n in names]
    exp.sync_base()
    return np.array([*exp.d.qpos[qa],*exp.d.qvel[va],*exp.base,
                     math.sin(exp.base_yaw),math.cos(exp.base_yaw),local_time/15.5],np.float32)


def reached_view(exp):
    obs = observe(exp, width=448, height=448)
    from PIL import Image
    rgb = np.asarray(Image.fromarray(obs['rgb']).resize((224,224), Image.Resampling.BILINEAR))
    return obs, rgb, original_proprio(exp)


def execute_candidate(scene, base_id, grasp_id, start_pose=(.6,-1.6,math.pi/2), settings=None,
                      capture=True, keep_trace=False, initial_observation=None,
                      base_selector=None, grasp_selector=None):
    """One independent episode, continuous from start through the opening hold."""
    exp = MobileDoor(scene,start_pose,settings)
    begin = time.perf_counter()
    if initial_observation is not None:
        initial_rgb, initial_state = initial_observation
        np.testing.assert_array_equal(initial_state,original_proprio(exp))
    else:
        _, initial_rgb, initial_state = reached_view(exp) if capture else (None,None,None)
    if base_selector is not None:
        base_id=int(base_selector(initial_rgb,initial_state))
        if base_id==len(BASE_POSES):
            record=dict(scene_id=scene.scene_id,base_id=base_id,grasp_id=None,refused=True,
                        success=False,collision=False,nav_success=False,opening_tested=False,
                        base_reset_count=1,base_displacement_m=0.,measured_path_length_m=0.,
                        max_trace_step_m=0.,elapsed_wall_seconds=time.perf_counter()-begin,
                        result=dict(outcome='E_policy_abstained',stop_reason='policy_refused'))
            return record,dict(initial_rgb=initial_rgb,initial_proprio=initial_state),exp.trace,exp
    side = 'front' if BASE_POSES[base_id][1]<0 else 'back'
    nav_start = time.perf_counter()
    query = plan_navigation(exp,BASE_POSES[base_id])
    nav = exp.track(base_path=query['path']) if query['path_found'] else dict(
        success=False,collision=False,stop_reason=query['reason'])
    nav_seconds = time.perf_counter()-nav_start
    nav_xy=np.array([r['base_pose'][:2] for r in exp.trace])
    nav_length=float(np.linalg.norm(np.diff(nav_xy,axis=0),axis=1).sum()) if len(nav_xy)>1 else 0.
    arrays = dict(initial_rgb=initial_rgb,initial_proprio=initial_state)
    record = dict(scene_id=scene.scene_id,base_id=base_id,grasp_id=grasp_id,side=side,
                  base_pose=list(BASE_POSES[base_id]),grasp_angle_deg=GRASP_ANGLES[grasp_id],
                  navigation=dict(query=query,execution=nav),nav_success=nav['success'],
                  nav_collision=nav['collision'],navigation_wall_seconds=nav_seconds,
                  navigation_path_length_m=nav_length,
                  path_length_m=query.get('geometric_length_m',0.),
                  preparation=None,opening_tested=False,grasp_observed=False,
                  success=False,collision=nav['collision'],failure_stage='navigation',
                  result=dict(outcome='A_navigation_failed',stop_reason=nav['stop_reason']),
                  safe_opening_fraction=0.,shortfall=1.,milestones=[False,False,False],
                  physical_scope='candidate controller and finite planner; not universal infeasibility')
    if nav['success'] and side in scene.handle_faces:
        exp.set_side(side)
        obs, arrival_rgb, arrival_state = reached_view(exp) if capture else (observe(exp,448,448),None,None)
        arrays.update(arrival_rgb=arrival_rgb,arrival_proprio=arrival_state)
        try:
            estimate = estimate_frame(obs)
        except ValueError:
            estimate = None
        if estimate is None:
            record.update(failure_stage='perception', result=dict(outcome='B_perception_failed',stop_reason='handle_not_detected'))
        else:
            record['grasp_observed']=True
            record['estimated_handle_center_world_m']=list(estimate['center_world_m'])
            if grasp_selector is not None:
                commands=[]
                for angle in GRASP_ANGLES:
                    yaw=exp.grasp_yaw_offset+exp.grasp_yaw_sign*math.radians(angle)
                    commands.append([*estimate['center_world_m'],math.sin(yaw),math.cos(yaw)])
                grasp_id=int(grasp_selector(arrival_rgb,arrival_state,np.array(commands,np.float32)))
                if not 0<=grasp_id<len(GRASP_ANGLES):
                    raise ValueError('Invalid learned grasp choice.')
                record.update(grasp_id=grasp_id,grasp_angle_deg=GRASP_ANGLES[grasp_id])
            plan = GraspPlan(tuple(estimate['center_world_m']),approach_angle_deg=GRASP_ANGLES[grasp_id],
                             ik_branch=exp.ik_branch,grip_force_limit_N=6.)
            yaw_offset=plan.yaw_offset_rad(exp)
            arrays['grasp_command']=np.array([*plan.point_world_m,math.sin(yaw_offset),math.cos(yaw_offset)],np.float32)
            arm_query = plan_arm(exp,plan)
            arm = exp.track(arm_path=arm_query['path'],phase='pregrasp') if arm_query['path_found'] else dict(
                success=False,collision=False,stop_reason=arm_query['reason'])
            record['preparation']=dict(query=arm_query,execution=arm)
            if arm['success']:
                result,rows,_=exp.trial_continuous(plan)
                from hjepa_scene import task_reward
                metrics=task_reward(result,rows)
                record.update(opening_tested=True,result=result,**metrics,
                              failure_stage=None if metrics['success'] else 'opening')
            else:
                record.update(failure_stage='pregrasp',collision=arm['collision'],
                              result=dict(outcome='B_preparation_failed',stop_reason=arm['stop_reason']))
    elif nav['success']:
        record.update(failure_stage='perception',result=dict(outcome='B_handle_face_absent',stop_reason='handle_face_absent'))
    if 'reward' not in record:
        record['reward']=-3.-5.*record['collision']
    trace=exp.trace
    xy=np.array([r['base_pose'][:2] for r in trace])
    record.update(base_reset_count=exp.reset_count,integration_steps=exp.integration_steps,
                  base_displacement_m=float(np.linalg.norm(exp.d.qpos[exp.base_q[:2]]-exp.start_pose[:2])),
                  measured_path_length_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()) if len(xy)>1 else 0.,
                  max_trace_step_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).max()) if len(xy)>1 else 0.,
                  elapsed_wall_seconds=time.perf_counter()-begin,episode_duration_s=float(exp.d.time))
    return record,arrays,trace if keep_trace else [],exp
