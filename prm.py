"""Joint-space PRM for free motion to pregrasp, with MuJoCo collision checks.

Nodes are [lift_m, shoulder_rad, elbow_rad, wrist_rad]. The door stays closed
and fingers open. Edges are lazily checked by bounded-joint interpolation;
angles are NOT wrapped through their physical joint limits. Finding a path
certifies only sampled geometric checks, never full opening/contact success.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import math
import time

import mujoco
import numpy as np


def segment_box_distance_squared(a, b, half_size):
    """Exact segment/AABB squared distance, including deep crossings.

    In box-local coordinates the distance is piecewise quadratic. Evaluate
    its minimizer on intervals separated by crossings of the six box planes.
    """
    a, b, half_size = np.asarray(a), np.asarray(b), np.asarray(half_size)
    delta = b-a
    cuts = [0., 1.]
    for k in range(3):
        if abs(delta[k]) > 1e-12:
            cuts.extend(float(t) for t in ((-half_size[k]-a[k])/delta[k],
                                          (half_size[k]-a[k])/delta[k]) if 0 < t < 1)
    cuts = sorted(set(cuts))
    best = math.inf
    for lo, hi in zip(cuts, cuts[1:]):
        mid = a+(lo+hi)/2*delta
        active = np.abs(mid) > half_size
        bound = np.sign(mid)*half_size
        denom = float(np.sum(delta[active]**2))
        t = lo if denom < 1e-24 else float(np.clip(
            -np.dot(delta[active], (a-bound)[active])/denom, lo, hi))
        for value in (lo, t, hi):
            diff = np.maximum(np.abs(a+value*delta)-half_size, 0.)
            best = min(best, float(diff@diff))
    return best


@dataclass(frozen=True)
class PRMSettings:
    samples: int = 240
    neighbors: int = 12
    seed: int = 42
    joint_step_rad: float = .06
    lift_step_m: float = .008
    penetration_tolerance_m: float = 1e-5
    initial_q: tuple = (0., -math.pi/2, -2*math.pi/3, 0.)

    def __post_init__(self):
        if self.samples < 0 or self.neighbors < 1 or self.seed < 0:
            raise ValueError('Invalid PRM sample count, neighbor count or seed')
        if not all(np.isfinite(v) and v > 0 for v in (self.joint_step_rad, self.lift_step_m)):
            raise ValueError('PRM collision-check resolution must be positive')
        if not np.isfinite(self.penetration_tolerance_m) or self.penetration_tolerance_m < 0:
            raise ValueError('Invalid penetration tolerance')
        if len(self.initial_q) != 4 or not np.all(np.isfinite(self.initial_q)):
            raise ValueError('initial_q must contain four finite joint coordinates')
        object.__setattr__(self, 'initial_q', tuple(self.initial_q))

    def to_dict(self):
        return asdict(self)


class Roadmap:
    def __init__(self, experiment, settings=None):
        self.exp = experiment
        self.settings = settings or PRMSettings()
        self.m = experiment.m
        self.probe = mujoco.MjData(self.m)
        self.qadr = np.r_[experiment.lift_q, experiment.arm_q]
        self.limits = self.m.jnt_range[[self.m.joint('lift_z').id, *experiment.arm_j]].copy()
        self.scale = np.array([.24, math.pi, math.pi, math.pi])
        self.steps = np.array([self.settings.lift_step_m, *([self.settings.joint_step_rad]*3)])
        self.nodes = np.empty((0, 4))
        self.edges = {}
        self.edge_validity = {}
        self.state_checks = self.link_checks = 0
        self.rejected_samples = self.attempted_samples = 0
        self.built = False
        self.door_box = self.m.geom('door_panel').id
        self.arm_capsules = [self.m.geom(n).id for n in ('upper_link', 'forearm_link', 'wrist_link')]

    def invalid_reason(self, q):
        self.state_checks += 1
        q = np.asarray(q, dtype=float)
        if q.shape != (4,) or not np.all(np.isfinite(q)):
            return 'invalid_joint_vector'
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            return 'joint_limits'
        mujoco.mj_resetData(self.m, self.probe)
        self.probe.qpos[self.qadr] = q
        self.probe.qpos[self.exp.finger_q] = .04
        mujoco.mj_forward(self.m, self.probe)
        # Check long capsules against the door independently of generated
        # contact points; deeply crossing initial configurations need rejection.
        rotation = self.probe.geom_xmat[self.door_box].reshape(3, 3)
        center = self.probe.geom_xpos[self.door_box]
        for gid in self.arm_capsules:
            axis = self.probe.geom_xmat[gid].reshape(3, 3)[:, 2]
            pos, size = self.probe.geom_xpos[gid], self.m.geom_size[gid]
            a = rotation.T@(pos-size[1]*axis-center)
            b = rotation.T@(pos+size[1]*axis-center)
            radius = max(0., size[0]-self.settings.penetration_tolerance_m)
            if segment_box_distance_squared(a, b, self.m.geom_size[self.door_box]) < radius*radius:
                return 'collision:'+self.m.geom(gid).name+':door_panel'
        for c in self.probe.contact:
            if c.dist < -self.settings.penetration_tolerance_m:
                return 'collision:' + ':'.join(self.m.geom(int(g)).name for g in (c.geom1, c.geom2))
        return None

    def clear(self, q):
        return self.invalid_reason(q) is None

    def link(self, a, b):
        self.link_checks += 1
        count = max(1, int(np.ceil(np.max(np.abs(np.asarray(b)-a)/self.steps))))
        return all(self.clear(q) for q in np.linspace(a, b, count+1))

    def build(self):
        if self.built:
            return
        rng = np.random.default_rng(self.settings.seed)
        nodes = []
        # A finite attempt budget also terminates in heavily obstructed scenes.
        for _ in range(max(1, self.settings.samples*8)):
            if len(nodes) >= self.settings.samples:
                break
            q = rng.uniform(self.limits[:, 0], self.limits[:, 1])
            self.attempted_samples += 1
            if self.clear(q):
                nodes.append(q)
            else:
                self.rejected_samples += 1
        self.nodes = np.asarray(nodes).reshape(-1, 4)
        self.edges = {i: {} for i in range(len(nodes))}
        for i, q in enumerate(self.nodes):
            distances = np.linalg.norm((self.nodes-q)/self.scale, axis=1)
            for j in np.argsort(distances)[1:self.settings.neighbors+1]:
                self.edges[i][int(j)] = self.edges[int(j)][i] = float(distances[j])
        self.built = True

    @staticmethod
    def shortest_path(graph, start, goal):
        queue, distance, parent = [(0., start)], {start: 0.}, {}
        while queue:
            cost, u = heapq.heappop(queue)
            if cost > distance[u]:
                continue
            if u == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                return path[::-1]
            for v, weight in graph[u].items():
                next_cost = cost+weight
                if next_cost < distance.get(v, math.inf):
                    distance[v], parent[v] = next_cost, u
                    heapq.heappush(queue, (next_cost, v))
        return None

    def query(self, goal, start=None, include_graph=False):
        before, begin = (self.state_checks, self.link_checks), time.perf_counter()
        start = np.array(self.settings.initial_q if start is None else start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        result = dict(path_found=False, status='not_searched', path_q=[], initial_q=start.tolist(),
                      goal_q=goal.tolist(), settings=self.settings.to_dict())
        reason_start, reason_goal = self.invalid_reason(start), self.invalid_reason(goal)
        graph, points, raw = {}, np.empty((0, 4)), []
        direct = False
        if reason_start:
            result.update(status='invalid_start', reason=reason_start)
        elif reason_goal:
            result.update(status='invalid_goal', reason=reason_goal)
        else:
            direct = self.link(start, goal)
            if direct:
                result.update(path_found=True, status='direct_path', path_q=[start.tolist(), goal.tolist()])
            else:
                self.build()
                points = np.vstack([self.nodes, start, goal])
                n, target = len(self.nodes), len(self.nodes)+1
                graph = {i: dict(v) for i, v in self.edges.items()}
                graph[n], graph[target] = {}, {}
                # Successful and failed roadmap edges can be reused across queries.
                for (i, j), valid in self.edge_validity.items():
                    if not valid:
                        graph[i].pop(j, None); graph[j].pop(i, None)
                for u in (n, target):
                    distances = np.linalg.norm((self.nodes-points[u])/self.scale, axis=1)
                    for j in np.argsort(distances)[:self.settings.neighbors]:
                        graph[u][int(j)] = graph[int(j)][u] = float(distances[j])
                while True:
                    raw = self.shortest_path(graph, n, target)
                    if raw is None:
                        result.update(status='no_path_within_budget', reason='finite_roadmap_disconnected')
                        break
                    failed = False
                    for i, j in zip(raw, raw[1:]):
                        key = tuple(sorted((i, j)))
                        valid = self.edge_validity.get(key) if max(i, j) < n else None
                        if valid is None:
                            valid = self.link(points[i], points[j])
                            if max(i, j) < n:
                                self.edge_validity[key] = valid
                        if not valid:
                            graph[i].pop(j, None); graph[j].pop(i, None)
                            failed = True
                            break
                    if not failed:
                        result.update(path_found=True, status='roadmap_path',
                                      path_q=points[raw].tolist(), raw_path_indices=list(map(int, raw)))
                        break
        result.update(direct_path_clear=direct, accepted_nodes=len(self.nodes),
            attempted_samples=self.attempted_samples, rejected_samples=self.rejected_samples,
            state_checks=self.state_checks-before[0], link_checks=self.link_checks-before[1],
            elapsed_seconds=time.perf_counter()-begin,
            interpretation='Geometric pregrasp query only. Finite failure does not prove infeasibility.')
        if include_graph:
            result['graph'] = dict(nodes=points.tolist(), candidate_edges=[
                [i, j] for i in graph for j in graph[i] if i < j],
                checked_edges=[[i, j, bool(v)] for (i, j), v in self.edge_validity.items()],
                start_index=len(self.nodes), goal_index=len(self.nodes)+1)
        return result

    def plan(self, plan, include_graph=False):
        point = np.asarray(plan.point_world_m)
        yaw = math.pi/2 + plan.yaw_offset_rad(self.exp)
        pre = point-plan.approach_distance_m*np.array([math.cos(yaw), math.sin(yaw), 0.])
        arm = self.exp.ik(pre, yaw, branch=plan.ik_branch)
        if arm is None:
            return dict(path_found=False, status='goal_ik_unreachable', path_q=[],
                initial_q=list(self.settings.initial_q), pregrasp_world_m=pre.tolist(),
                reason='pregrasp has no IK on selected branch within joint limits',
                interpretation='This target/branch is excluded; not a proof about other grasps or branches.')
        result = self.query(np.r_[point[2]-.85, arm], include_graph=include_graph)
        result['pregrasp_world_m'] = pre.tolist()
        return result


def execute_path(exp, path, viewer=None, record=False):
    """Track the roadmap in joint space using mj_step, with collision stops.

    Only the initial episode reset assigns qpos. All motion follows joint
    servos with robot-only bias compensation. The door remains passive.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 4 or len(path) < 2:
        raise ValueError('Expected at least two four-coordinate waypoints')
    qadr = np.r_[exp.lift_q, exp.arm_q]
    mujoco.mj_resetData(exp.m, exp.d)
    exp.d.qpos[qadr] = path[0]
    exp.d.qpos[exp.finger_q] = .04
    mujoco.mj_forward(exp.m, exp.d)
    rows, states = [], []
    rates = np.array([.08, .7, .7, .7])
    dt = exp.m.opt.timestep
    stop, steps, last_sample = 'path_executed', 0, -.02
    collision = None
    from simulate import smooth

    def command_and_check(q):
        nonlocal steps, last_sample, collision, stop
        exp.d.ctrl[:] = [*q, .04, .04]
        exp.d.qfrc_applied[:] = 0.
        exp.d.qfrc_applied[exp.robot_dof] = exp.d.qfrc_bias[exp.robot_dof]
        mujoco.mj_step(exp.m, exp.d)
        steps += 1
        if exp.d.time-last_sample < .02-1e-9:
            return True
        last_sample = exp.d.time
        mujoco.mj_forward(exp.m, exp.d)
        unexpected = exp.contact_metrics()[-1]
        if record:
            rows.append(dict(time_s=float(exp.d.time), lift_m=float(exp.d.qpos[qadr[0]]),
                shoulder_rad=float(exp.d.qpos[qadr[1]]), elbow_rad=float(exp.d.qpos[qadr[2]]),
                wrist_rad=float(exp.d.qpos[qadr[3]]), unexpected_contact=';'.join(unexpected)))
            states.append(exp.d.qpos.copy())
        if unexpected:
            collision, stop = unexpected[0], 'pregrasp_execution_collision'
            return False
        if abs(exp.d.qpos[exp.door_q]) > math.radians(.5):
            stop = 'door_moved_during_free_motion'
            return False
        if viewer is not None:
            viewer.sync()
            time.sleep(.02)
        return True

    ok = True
    for a, b in zip(path, path[1:]):
        duration = max(.25, 1.5*float(np.max(np.abs(b-a)/rates)))
        n = int(math.ceil(duration/dt))
        for i in range(1, n+1):
            if not command_and_check(a+smooth(i/n)*(b-a)):
                ok = False
                break
        if not ok:
            break
    if ok:
        for _ in range(int(math.ceil(.8/dt))):
            if not command_and_check(path[-1]):
                ok = False
                break
    error = np.abs(exp.d.qpos[qadr]-path[-1])
    if ok and (error[0] > .005 or np.any(error[1:] > .03) or
               np.max(np.abs(exp.d.qvel[exp.robot_dof])) > .08):
        ok, stop = False, 'pregrasp_tracking_not_settled'
    report = dict(success=bool(ok), stop_reason=stop, collision=collision,
        duration_s=float(exp.d.time), integration_steps=steps, joint_error=error.tolist(),
        actual_goal_q=exp.d.qpos[qadr].tolist(), door_angle_deg=float(np.degrees(exp.d.qpos[exp.door_q])))
    # The opening skill's schedule uses its own clock, preserving qpos/qvel.
    if ok:
        exp.d.time = 0.
    return report, rows, states
