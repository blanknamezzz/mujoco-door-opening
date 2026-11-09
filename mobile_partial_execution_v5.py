"""v5 observation interface, preserving v3 navigation/contact execution.

Only reached RGB-D exposure changes. Queries and rendering do not advance or
reset physics. Initial callbacks receive no post frame. Extra post depth is
read at the state reached by the original inspection motion.
"""
import math
import time
import numpy as np
from mobile_active_v3 import (MotionDoor,reached_view,original_proprio,inspect,
    _finish,REFUSE,INSPECT,BASE_POSES,GRASP_ANGLES,plan_inspection_navigation,
    plan_arm,GraspPlan,estimate_frame,pose_from_state)


def read_reached_frame(exp):
    q,v,when=exp.d.qpos.copy(),exp.d.qvel.copy(),float(exp.d.time)
    result=reached_view(exp)
    np.testing.assert_array_equal(q,exp.d.qpos)
    np.testing.assert_array_equal(v,exp.d.qvel)
    if when!=float(exp.d.time):raise AssertionError('Camera acquisition advanced physics.')
    return result


def geometric_station(mask,state,order='nearest'):
    mask=np.asarray(mask,bool);state=np.asarray(state)
    if mask.shape!=(4,):raise ValueError('Four station constraints required.')
    pose=pose_from_state(state) if state.shape==(19,) else state
    if pose.shape!=(3,) or not np.isfinite(pose).all():raise ValueError('Current finite base pose required.')
    allowed=np.flatnonzero(mask)
    if not len(allowed):return REFUSE
    if order=='nearest':priorities=np.linalg.norm(np.asarray(BASE_POSES)[:,:2]-pose[:2],axis=1)
    elif order=='rear_first':priorities=np.array([2.,3.,0.,1.])
    else:raise ValueError('Unknown station ordering.')
    return int(min(allowed,key=lambda b:(priorities[b],b)))


def geometry_depth_action(initial_map,state,inspect_unknown=False):
    station=geometric_station(initial_map['mask'],state)
    if station==REFUSE:return REFUSE
    if inspect_unknown and initial_map['queries'][station]['sampled_free_fraction']<.4:return INSPECT
    return station


def execute_partial_candidate(scene,base_id,grasp_id,start_pose=(.6,-1.65,math.pi/2),
                             inspection=False,initial_observation=None,keep_trace=False,
                             pre_selector=None,post_selector=None,grasp_selector=None):
    exp=MotionDoor(scene,start_pose,planner='fixed');begin=time.perf_counter()
    if initial_observation is None:initial_frame,initial_rgb,initial_state=read_reached_frame(exp)
    else:
        initial_frame,initial_rgb,initial_state=initial_observation
        np.testing.assert_array_equal(initial_state,original_proprio(exp))
    arrays=dict(initial_rgb=initial_rgb,initial_proprio=initial_state,initial_frame=initial_frame,
                post_rgb=initial_rgb.copy(),post_proprio=initial_state.copy(),post_view_available=False,post_frame=initial_frame)
    scan=dict(attempted=False,success=False,collision=False,unsafe_abort=False,
              path_length_m=0.,duration_s=0.,wall_seconds=0.,actual_base_pose=list(start_pose))
    record=dict(scene_id=scene.scene_id,base_id=base_id,grasp_id=grasp_id,refused=False,
        inspection=scan,success=False,collision=False,continuation_collision=False,
        nav_success=False,nav_collision=False,opening_tested=False,grasp_observed=False,
        failure_stage=None,result=dict(outcome='not_started'),safe_opening_fraction=0.,shortfall=1.,
        milestones=[False,False,False],reward=-3.,navigation=None,preparation=None)
    if pre_selector is not None:
        action=int(pre_selector(dict(initial_rgb=initial_rgb,initial_proprio=initial_state,initial_frame=initial_frame)))
        if not 0<=action<=INSPECT:raise ValueError('Invalid initial decision.')
        inspection=action==INSPECT
        if action==REFUSE:
            record.update(base_id=REFUSE,grasp_id=None,refused=True,
                          result=dict(outcome='E_policy_abstained',stop_reason='policy_refused'))
            return _finish(exp,record,arrays,begin,keep_trace)
        if not inspection:base_id=action
    if inspection:
        scan,post_rgb,post_state=inspect(exp);record['inspection']=scan
        post_frame,post_rgb,post_state=read_reached_frame(exp)
        arrays.update(post_rgb=post_rgb,post_proprio=post_state,post_view_available=scan['success'],post_frame=post_frame)
        if scan['unsafe_abort']:
            record.update(failure_stage='inspection',
                          result=dict(outcome='A_inspection_failed',stop_reason=scan['execution']['stop_reason']))
            return _finish(exp,record,arrays,begin,keep_trace)
        if post_selector is not None:
            base_id=int(post_selector(dict(arrays)))
            if not 0<=base_id<=REFUSE:raise ValueError('Inspection cannot repeat or select an invalid base.')
            if base_id==REFUSE:
                record.update(base_id=REFUSE,grasp_id=None,refused=True,
                              result=dict(outcome='E_policy_abstained',stop_reason='policy_refused_after_observation'))
                return _finish(exp,record,arrays,begin,keep_trace)
    if base_id is None or not 0<=base_id<4:raise ValueError('A physical station must be selected.')
    side='front' if BASE_POSES[base_id][1]<0 else 'back'
    record.update(base_id=base_id,side=side,base_pose=list(BASE_POSES[base_id]),
                  grasp_angle_deg=GRASP_ANGLES[grasp_id])
    nav_start=time.perf_counter();trace_start=len(exp.trace)
    query=plan_inspection_navigation(exp,BASE_POSES[base_id])
    nav=exp.track(base_path=query['path']) if query['path_found'] else dict(
        success=False,collision=False,stop_reason=query['reason'])
    positions=[np.array(r['base_pose'][:2]) for r in exp.trace[trace_start:]]
    nav_length=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()) if len(positions)>1 else 0.
    record.update(navigation=dict(query=query,execution=nav),nav_success=bool(nav['success']),
        nav_collision=bool(nav['collision']),continuation_collision=bool(nav['collision']),
        navigation_wall_seconds=time.perf_counter()-nav_start,navigation_path_length_m=nav_length,
        failure_stage='navigation',result=dict(outcome='A_navigation_failed',stop_reason=nav['stop_reason']))
    if nav['success'] and side in scene.handle_faces:
        exp.set_side(side);obs,arrival_rgb,arrival_state=reached_view(exp)
        arrays.update(arrival_rgb=arrival_rgb,arrival_proprio=arrival_state)
        try:estimate=estimate_frame(obs)
        except ValueError:estimate=None
        if estimate is None:
            record.update(failure_stage='perception',result=dict(outcome='B_perception_failed',stop_reason='handle_not_detected'))
        else:
            record.update(grasp_observed=True,estimated_handle_center_world_m=list(estimate['center_world_m']))
            if grasp_selector is not None:
                commands=[]
                for angle in GRASP_ANGLES:
                    yaw=exp.grasp_yaw_offset+exp.grasp_yaw_sign*math.radians(angle)
                    commands.append([*estimate['center_world_m'],math.sin(yaw),math.cos(yaw)])
                grasp_id=int(grasp_selector(arrival_rgb,arrival_state,np.array(commands,np.float32)))
                if not 0<=grasp_id<2:raise ValueError('Invalid grasp choice.')
                record.update(grasp_id=grasp_id,grasp_angle_deg=GRASP_ANGLES[grasp_id])
            plan=GraspPlan(tuple(estimate['center_world_m']),approach_angle_deg=GRASP_ANGLES[grasp_id],
                           ik_branch=exp.ik_branch,grip_force_limit_N=6.)
            yaw=plan.yaw_offset_rad(exp)
            arrays['grasp_command']=np.array([*plan.point_world_m,math.sin(yaw),math.cos(yaw)],np.float32)
            arm_query=plan_arm(exp,plan)
            preparation=exp.track(arm_path=arm_query['path'],phase='pregrasp') if arm_query['path_found'] else dict(
                success=False,collision=False,stop_reason=arm_query['reason'])
            record['preparation']=dict(query=arm_query,execution=preparation)
            if preparation['success']:
                result,rows,_=exp.trial_continuous(plan)
                from hjepa_scene import task_reward
                metrics=task_reward(result,rows)
                record.update(opening_tested=True,result=result,**metrics)
                record.update(continuation_collision=metrics['collision'],failure_stage=None if metrics['success'] else 'opening')
            else:
                record.update(failure_stage='pregrasp',continuation_collision=preparation['collision'],
                              result=dict(outcome='B_preparation_failed',stop_reason=preparation['stop_reason']))
    elif nav['success']:
        record.update(failure_stage='perception',result=dict(outcome='B_handle_face_absent',stop_reason='handle_face_absent'))
    return _finish(exp,record,arrays,begin,keep_trace)
