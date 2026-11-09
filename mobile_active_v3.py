"""Real inspection option and continuation utilities for the mobile task.

No camera teleport or midpoint reset is used. Every counterfactual independently
integrates its complete inspection/navigation/preparation/opening episode.
"""
from __future__ import annotations
import math
import time
import heapq
import numpy as np
import mujoco

from mobile_crossed_v2 import MotionDoor,restore_scene,make_manifest,geometry_identity
from mobile_door import BASE_POSES,GRASP_ANGLES,CollisionChecker,plan_navigation,plan_arm,reached_view,original_proprio
from grasp_plan import GraspPlan
from perception import estimate_frame

INSPECT_POSE=(.6,1.65,-math.pi/2)
REFUSE=4
INSPECT=5
OBS_PATH_WEIGHT=.2
OBS_TIME_WEIGHT=.1


def relative_command(goal,current):
    goal,current=np.asarray(goal,float),np.asarray(current,float)
    if goal.shape!=(3,) or current.shape!=(3,) or not np.isfinite([goal,current]).all():
        raise ValueError('Finite three-coordinate poses required.')
    return np.array([goal[0]-current[0],goal[1]-current[1],
                     math.sin(goal[2]-current[2]),math.cos(goal[2]-current[2])],np.float32)


def pose_from_state(state):
    state=np.asarray(state)
    if state.shape!=(19,):raise ValueError('Preserve the 19-D checkpoint state contract.')
    return np.array([state[14],state[15],math.atan2(state[16],state[17])])


def base_commands(state):
    current=pose_from_state(state)
    return np.array([relative_command(p,current) for p in BASE_POSES]+[np.zeros(4)],np.float32)


def observation_cost(path_m,duration_s):
    if not np.isfinite([path_m,duration_s]).all() or path_m<0 or duration_s<0:
        raise ValueError('Observation costs require finite nonnegative measured distance/time.')
    return OBS_PATH_WEIGHT*path_m/6.+OBS_TIME_WEIGHT*duration_s/60.


class InspectionChecker(CollisionChecker):
    """Whole-robot signed-distance clearance for the longer observation route.

    Native contact generation can omit a narrow rotated-box corner interval.
    Bound-sphere broadphase plus mj_geomDistance checks actual surfaces, rather
    than assuming absence of a generated contact proves positive clearance.
    """
    def __init__(self,exp):
        super().__init__(exp)
        self.robot=np.array(sorted(self.robot_geoms),int)
        self.environment=np.array([g for g in range(self.m.ngeom) if g not in self.robot_geoms
            and self.m.geom_contype[g] and self.m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_PLANE],int)
        self.distance_checks=0

    def invalid(self,base,arm=None):
        reason=super().invalid(base,arm)
        if reason is not None:return reason
        centers=self.data.geom_xpos
        delta=centers[self.robot,None,:]-centers[self.environment][None,:,:]
        reach=self.m.geom_rbound[self.robot,None]+self.m.geom_rbound[self.environment][None,:]+.02
        candidates=np.argwhere((delta*delta).sum(-1)<reach*reach)
        for ri,ei in candidates:
            robot,environment=int(self.robot[ri]),int(self.environment[ei])
            self.distance_checks+=1
            if mujoco.mj_geomDistance(self.m,self.data,robot,environment,.02,None)<.02-1e-9:
                return 'clearance:'+self.m.geom(environment).name+':'+self.m.geom(robot).name
        return None

    def base_edge(self,a,b):
        count=max(1,math.ceil(max(np.linalg.norm(np.asarray(b)[:2]-np.asarray(a)[:2])/.02,
                                  abs(b[2]-a[2])/.03)))
        return all(self.invalid(q) is None for q in np.linspace(a,b,count+1))


def inspect(exp):
    begin=time.perf_counter();time0=float(exp.d.time);start=exp.d.qpos[exp.base_q[:2]].copy()
    query=plan_inspection_navigation(exp,INSPECT_POSE)
    execution=exp.track(base_path=query['path'],phase='inspection') if query['path_found'] else dict(
        success=False,collision=False,stop_reason=query['reason'])
    xy=[start]+[np.array(r['base_pose'][:2]) for r in exp.trace if r['phase']=='inspection']
    distance=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()) if len(xy)>1 else 0.
    unsafe=bool(execution['collision'] or abs(exp.d.qpos[exp.door_q])>math.radians(.5))
    _,rgb,state=reached_view(exp)
    return dict(attempted=True,success=bool(execution['success']),collision=bool(execution['collision']),
                unsafe_abort=unsafe,query=query,execution=execution,
                path_length_m=distance,duration_s=float(exp.d.time-time0),wall_seconds=time.perf_counter()-begin,
                actual_base_pose=exp.d.qpos[exp.base_q].tolist()),rgb,state


def _finish(exp,record,arrays,begin,keep_trace):
    trace=exp.trace;xy=np.array([r['base_pose'][:2] for r in trace])
    record.update(base_reset_count=exp.reset_count,integration_steps=exp.integration_steps,
        base_displacement_m=float(np.linalg.norm(exp.d.qpos[exp.base_q[:2]]-exp.start_pose[:2])),
        measured_path_length_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()) if len(xy)>1 else 0.,
        max_trace_step_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).max()) if len(xy)>1 else 0.,
        episode_duration_s=float(exp.d.time),elapsed_wall_seconds=time.perf_counter()-begin)
    scan=record['inspection']
    record.update(inspection_attempted=scan['attempted'],inspection_success=scan['success'],
                  inspection_collision=scan['collision'],inspection_unsafe_abort=scan['unsafe_abort'],
                  observation_cost=observation_cost(scan['path_length_m'],scan['duration_s']))
    record['collision']=bool(record.get('continuation_collision',False) or scan['collision'])
    if not record['opening_tested']:record['reward']=-3.-5.*record['collision']
    return record,arrays,trace if keep_trace else [],exp


def execute_active_candidate(scene,base_id,grasp_id,start_pose=(.6,-1.65,math.pi/2),
                             inspection=False,initial_observation=None,keep_trace=False,
                             pre_selector=None,post_selector=None,grasp_selector=None):
    exp=MotionDoor(scene,start_pose,planner='fixed');begin=time.perf_counter()
    if initial_observation is None:_,initial_rgb,initial_state=reached_view(exp)
    else:
        initial_rgb,initial_state=initial_observation
        np.testing.assert_array_equal(initial_state,original_proprio(exp))
    arrays=dict(initial_rgb=initial_rgb,initial_proprio=initial_state,
                post_rgb=initial_rgb.copy(),post_proprio=initial_state.copy(),post_view_available=False)
    scan=dict(attempted=False,success=False,collision=False,unsafe_abort=False,
              path_length_m=0.,duration_s=0.,wall_seconds=0.,actual_base_pose=list(start_pose))
    record=dict(scene_id=scene.scene_id,base_id=base_id,grasp_id=grasp_id,refused=False,
        inspection=scan,success=False,collision=False,continuation_collision=False,
        nav_success=False,nav_collision=False,opening_tested=False,grasp_observed=False,
        failure_stage=None,result=dict(outcome='not_started'),safe_opening_fraction=0.,shortfall=1.,
        milestones=[False,False,False],reward=-3.,navigation=None,preparation=None)
    if pre_selector is not None:
        action=int(pre_selector(dict(initial_rgb=initial_rgb,initial_proprio=initial_state)))
        if not 0<=action<=INSPECT:raise ValueError('Invalid initial decision.')
        inspection=action==INSPECT
        if action==REFUSE:
            record.update(base_id=REFUSE,grasp_id=None,refused=True,
                          result=dict(outcome='E_policy_abstained',stop_reason='policy_refused'))
            return _finish(exp,record,arrays,begin,keep_trace)
        if not inspection:base_id=action
    if inspection:
        scan,post_rgb,post_state=inspect(exp);record['inspection']=scan
        arrays.update(post_rgb=post_rgb,post_proprio=post_state,post_view_available=scan['success'])
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


def continuation_returns(records):
    """Task-flow returns, not invented opening-component outcomes."""
    table={(r['base_id'],r['grasp_id']):r for r in records}
    if len(table)!=8 or set(table)!={(b,g) for b in range(4) for g in range(2)}:
        raise ValueError('One complete physical action table required.')
    if len({r['scene_id'] for r in records})!=1:raise ValueError('Mixed scenes.')
    if any(r['success'] and not (r['opening_tested'] and r['nav_success']) for r in records):
        raise ValueError('Success requires actual navigation and opening.')
    if any(r.get('inspection_unsafe_abort') and (r['grasp_observed'] or r['opening_tested']) for r in records):
        raise ValueError('Unsafe inspection cannot invent a continuation observation or outcome.')
    utility=np.array([[2.*table[b,g]['success']-1.-table[b,g]['continuation_collision']
                       for g in range(2)] for b in range(4)],np.float32)
    mask=np.array([[table[b,g]['grasp_observed'] for g in range(2)] for b in range(4)],bool)
    return utility,mask


def continuation_targets(scene,g0,g1,q1,cost_scale):
    """Detach/freeze downstream policy decisions; look up train outcomes after choice.

    Inspection is valued by the current post-observation P/G, not an oracle-best
    counterfactual continuation. Inputs here are teacher predictions on train only.
    """
    u0=scene['branches'][0]['utility'];u1=scene['branches'][1]['utility']
    choice0=np.asarray(g0).argmax(-1);choice1=np.asarray(g1).argmax(-1)
    p0=np.array([u0[b,choice0[b]] for b in range(4)],np.float32)
    p1=np.array([u1[b,choice1[b]] for b in range(4)],np.float32)
    scan=scene['branches'][1]['records'][0]['inspection']
    safe=not scan['unsafe_abort']
    if not safe:value=-1.-float(scan['collision'])
    else:
        selected=int(np.asarray(q1).argmax())
        value=0. if selected==REFUSE else float(u1[selected,choice1[selected]])
    value-=cost_scale*observation_cost(scan['path_length_m'],scan['duration_s'])
    return dict(p0=p0,p1=p1,inspect=np.float32(value),post_mask=safe)


# Inspection-specific A* fork; frozen v1/v2 navigation stays untouched.
def plan_inspection_navigation(exp, goal):
    """Deterministic bounded A*; collision-check the complete stowed robot."""
    checker = InspectionChecker(exp)
    start, goal = exp.d.qpos[exp.base_q].copy(), np.asarray(goal, float)
    turned = np.r_[start[:2], goal[2]]
    if not checker.base_edge(start, turned):
        return dict(path_found=False, reason='start_rotation_blocked', checks=checker.checks,distance_checks=checker.distance_checks)
    if checker.invalid(goal):
        return dict(path_found=False, reason='goal_in_collision', checks=checker.checks,distance_checks=checker.distance_checks)
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
            return dict(path_found=False, reason='no_path_in_bounded_grid', checks=checker.checks,distance_checks=checker.distance_checks)
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
                checks=checker.checks,distance_checks=checker.distance_checks)

