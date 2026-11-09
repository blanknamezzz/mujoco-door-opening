"""Crossed scene groups and coordinated base/arm opening, isolated from v1.

Each geometry group contains front/rear/both/no obstruction variants and stays
in one split. A coordinated controller rotates the commanded base pose with
the same hinge arc as the hand; all physical states still integrate mj_step.
"""
from dataclasses import asdict, replace
import hashlib
import json
import math
import time
import numpy as np

from door_stress_scenes import StressDoorScene, Obstacle, scenario_suite, DOOR_STYLES
from grasp_plan import GraspPlan
from mobile_door import (MobileDoor, MobileSettings, BASE_POSES, GRASP_ANGLES,
                         original_proprio, reached_view, plan_navigation, plan_arm)
from perception import estimate_frame, observe
from simulate import smooth, rot

LAYOUTS=('front_blocked','rear_blocked','both_blocked','free')
PLANNERS=('fixed','orbit')


def geometry_identity(spec):
    fields={k:spec[k] for k in StressDoorScene.__dataclass_fields__ if k in spec and k not in
            ('scene_id','description','wall_layout','expected_open_side')}
    fields['start_pose']=spec['start_pose']
    return hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def mirror(obstacles):
    return tuple(replace(o,name='obstacle_mirror_'+o.name,pos=(o.pos[0],-o.pos[1],o.pos[2]),
                         euler_deg=(o.euler_deg[0],o.euler_deg[1],-o.euler_deg[2])) for o in obstacles)


def make_manifest(group_counts,stream='mobile_crossed_v2_20261108'):
    from run_mobile_reward_experiment import generator
    templates=[s for s in scenario_suite() if s.handle_style in ('lever','t-bar','d-pull','offset-bar')]
    specs=[]
    for split,count in group_counts.items():
        rng=generator(split,stream)
        for index in range(count):
            template=templates[index%len(templates)]
            front=template.obstacles if template.expected_open_side=='back' else mirror(template.obstacles)
            front=tuple(replace(o,pos=(o.pos[0]+float(rng.uniform(-.025,.025)),
                                      o.pos[1]+float(rng.uniform(-.025,.025)),o.pos[2])) for o in front)
            rear=mirror(front)
            group_id=f'{stream}_{split}_g{index:03d}'
            base=replace(template,door_style=list(DOOR_STYLES)[int(rng.integers(len(DOOR_STYLES)))],
                         handle_x=template.handle_x+float(rng.uniform(-.022,.022)),
                         handle_z=template.handle_z+float(rng.uniform(-.012,.012)),
                         handle_friction=float(rng.uniform(.085,.30)),hinge_resistance=float(rng.uniform(.85,1.45)))
            start=[.6+float(rng.uniform(-.04,.04)),-1.65,math.pi/2]
            for layout,obstacles in zip(LAYOUTS,(front,rear,front+rear,())):
                scene=replace(base,scene_id=group_id+'_'+layout,obstacles=obstacles,
                              wall_layout=layout,expected_open_side='back' if layout=='front_blocked' else 'front')
                spec=dict(**asdict(scene),split=split,group_id=group_id,layout=layout,
                          template_id=template.scene_id,start_pose=start)
                spec['geometry_sha256']=geometry_identity(spec)
                specs.append(spec)
    identities=[s['geometry_sha256'] for s in specs]
    if len(identities)!=len(set(identities)):raise AssertionError('Repeated physical geometry.')
    group_splits={}
    for spec in specs:group_splits.setdefault(spec['group_id'],set()).add(spec['split'])
    if any(len(v)!=1 for v in group_splits.values()):raise AssertionError('Group crosses split.')
    return dict(schema_version=2,seed=42,stream=stream,group_counts=group_counts,
                counts={k:4*v for k,v in group_counts.items()},scenes=specs,
                scope='crossed visual/geometry groups; four layouts share appearance and physics parameters; known handle families',
                test_outcomes_available_at_creation=False)


def restore_scene(spec):
    fields={k:spec[k] for k in StressDoorScene.__dataclass_fields__ if k in spec}
    fields['obstacles']=tuple(Obstacle(**o) for o in fields['obstacles'])
    fields['handle_faces']=tuple(fields['handle_faces'])
    return StressDoorScene(**fields)


class MotionDoor(MobileDoor):
    def __init__(self,scene,start_pose=(.6,-1.6,math.pi/2),settings=None,planner='fixed'):
        if planner not in PLANNERS:raise ValueError('Unknown opening planner.')
        self.opening_planner=planner
        self._opening_plan=None
        self._orbit_origin=None
        super().__init__(scene,start_pose,settings)

    def command(self,point,yaw,aperture):
        if self.opening_planner=='orbit' and self._opening_plan is not None:
            plan=self._opening_plan
            fraction=smooth((self.d.time-plan.timing.open_start)/plan.timing.open_s)
            theta=self.opening_direction*math.radians(plan.opening_angle_deg)*fraction
            xy=(rot(theta)@np.r_[self._orbit_origin[:2],0.])[:2]
            self.base_target=np.r_[xy,self._orbit_origin[2]-theta]
        return super().command(point,yaw,aperture)

    def trial_continuous(self,plan,observer=None):
        origin=self.d.qpos[self.base_q].copy()
        max_speed=1.5*math.radians(plan.opening_angle_deg)*float(np.linalg.norm(origin[:2]))/plan.timing.open_s
        if self.opening_planner=='orbit' and max_speed>self.settings.speed_m_s+1e-9:
            raise ValueError('Orbital plan exceeds the declared planar speed limit.')
        self._orbit_origin=origin
        self._opening_plan=plan
        try:
            result,rows,states=super().trial_continuous(plan,observer)
        finally:
            self._opening_plan=None
        result.update(opening_planner=self.opening_planner,
                      opening_base_start_pose=origin.tolist(),
                      opening_base_end_pose=self.d.qpos[self.base_q].tolist(),
                      planned_peak_base_speed_m_s=max_speed if self.opening_planner=='orbit' else 0.)
        return result,rows,states


def factorized_targets(records):
    from staged_mobile_reward import credit_targets
    credit=credit_targets(records)
    table={(e['base_id'],e['grasp_id']):e for e in records}
    return dict(base=credit['feasible_bases'].astype(np.float32),
                side=credit['feasible_sides'].astype(np.float32),
                grasp=np.array([[table[b,g]['success'] for g in range(2)] for b in range(4)],np.float32),
                grasp_mask=credit['grasp_training_mask'],
                grasp_weight=np.array([[1.+float(table[b,g]['collision']) for g in range(2)] for b in range(4)],np.float32))


def factorized_loss(position_scores,grasp_scores,targets):
    import torch
    from torch.nn import functional as F
    device=position_scores.device
    refusal=position_scores[4]
    feasible=torch.as_tensor(targets['base'],device=device,dtype=torch.bool)
    pools=[]
    for start,end in ((0,2),(2,4)):
        # Positive side credit must reach a measured feasible instance, rather
        # than reinforcing the currently highest-scored bad base on that side.
        # These instance labels are training targets, never inference features.
        eligible=feasible[start:end]
        candidates=position_scores[start:end]
        pools.append(candidates[eligible].amax() if eligible.any() else candidates.amax())
    side=torch.stack(pools)-refusal
    base=position_scores[:4]-refusal
    side_loss=F.binary_cross_entropy_with_logits(side,torch.as_tensor(targets['side'],device=device))
    base_loss=F.binary_cross_entropy_with_logits(base,torch.as_tensor(targets['base'],device=device))
    mask=torch.as_tensor(targets['grasp_mask'],device=device)
    if mask.any():
        target=torch.as_tensor(targets['grasp'],device=device)
        weight=torch.as_tensor(targets['grasp_weight'],device=device)
        grasp_loss=F.binary_cross_entropy_with_logits(grasp_scores[mask],target[mask],weight=weight[mask])
    else:grasp_loss=position_scores.sum()*0.
    return side_loss+.25*base_loss+grasp_loss


# Execution fork of mobile_door.execute_candidate; v1 source and runs stay intact.
def execute_candidate_v2(scene, base_id, grasp_id, start_pose=(.6,-1.6,math.pi/2), settings=None,
                      capture=True, keep_trace=False, initial_observation=None,
                      base_selector=None, grasp_selector=None, planner="fixed"):
    """One independent episode, continuous from start through the opening hold."""
    exp = MotionDoor(scene,start_pose,settings,planner=planner)
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
    record.update(planner=planner,base_reset_count=exp.reset_count,integration_steps=exp.integration_steps,
                  base_displacement_m=float(np.linalg.norm(exp.d.qpos[exp.base_q[:2]]-exp.start_pose[:2])),
                  measured_path_length_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()) if len(xy)>1 else 0.,
                  max_trace_step_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).max()) if len(xy)>1 else 0.,
                  elapsed_wall_seconds=time.perf_counter()-begin,episode_duration_s=float(exp.d.time))
    return record,arrays,trace if keep_trace else [],exp
