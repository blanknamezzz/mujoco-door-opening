"""Causal observation API, real continuous motion and detached policy targets."""
import copy
import math
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import mujoco
import hjepa_runtime
import torch

from mobile_active_v3 import (relative_command,base_commands,pose_from_state,observation_cost,
    continuation_returns,continuation_targets,execute_active_candidate,InspectionChecker,INSPECT,REFUSE)
from mobile_crossed_v2 import MotionDoor
from run_mobile_active_experiment import _position_scores,choose
from door_stress_scenes import scenario_by_id
from test_mobile_door import action_table


def scene_fixture():
    branches={}
    for b in (0,1):
        records=action_table()
        for r in records:
            r.update(continuation_collision=False,inspection_unsafe_abort=False,episode_duration_s=15.5,
                inspection=dict(attempted=bool(b),success=bool(b),collision=False,unsafe_abort=False,
                                path_length_m=3.*b,duration_s=30.*b))
        utility,mask=continuation_returns(records)
        branches[b]=dict(records=records,utility=utility,grasp_mask=mask,
                         feasible_bases=np.array([False,False,True,False]))
    return dict(spec=dict(scene_id='fixture',group_id='g',layout='front_blocked'),branches=branches,
                union_bases=np.array([False,False,True,False]),union_sides=np.array([False,True]))


class ActiveContracts(unittest.TestCase):
    def test_relative_commands_keep_real_current_pose_and_unique_refusal(self):
        state=np.zeros(19,np.float32);state[14:18]=[.6,1.65,-1.,0.]
        command=base_commands(state)
        self.assertTrue((command[:4,1]<0).all())
        np.testing.assert_array_equal(command[4],np.zeros(4))
        self.assertTrue((np.abs(command[:4]).sum(-1)>0).all())
        np.testing.assert_allclose(pose_from_state(state),[.6,1.65,-math.pi/2],atol=1e-6)
        with self.assertRaises(ValueError):relative_command([1,2],[0,0,0])

    def test_side_view_routing_is_not_based_on_negative_relative_y(self):
        state=np.zeros(19,np.float32);state[14:18]=[.6,1.65,-1.,0.]
        linear=torch.nn.Linear(6,1)
        with torch.no_grad():linear.weight.zero_();linear.weight[0,0]=1.;linear.bias.fill_(10.)
        model=SimpleNamespace(position_policy=linear)
        scores=_position_scores(model,torch.tensor([[1.,0.]]),torch.tensor([[3.,0.]]),state)
        torch.testing.assert_close(scores,torch.tensor([11.,11.,13.,13.,0.]))

    def test_observation_cost_is_paid_and_has_no_repeat_bonus(self):
        self.assertAlmostEqual(observation_cost(6.,60.),.3)
        self.assertEqual(observation_cost(0.,0.),0.)
        for values in ((-1,1),(1,float('nan'))):
            with self.assertRaises(ValueError):observation_cost(*values)

    def test_inspection_target_uses_chosen_continuation_not_oracle_success(self):
        scene=scene_fixture();g=np.zeros((4,2));q=np.array([-1.,-1.,.5,.9,0.])
        target=continuation_targets(scene,g,g,q,1.)
        self.assertLess(target['inspect'],-1.)  # teacher chose bad base3, though base2 can open
        q=np.array([-1.,-1.,.5,-1.,0.]);g[2]=[-1.,1.]
        target=continuation_targets(scene,g,g,q,0.)
        self.assertEqual(target['inspect'],-1.)  # teacher G chose failed grasp1
        q[:4]=-1.;target=continuation_targets(scene,g,g,q,1.)
        self.assertAlmostEqual(target['inspect'],-observation_cost(3,30),places=6)

    def test_unsafe_observation_has_no_post_or_grasp_supervision(self):
        scene=scene_fixture()
        for record in scene['branches'][1]['records']:
            record.update(success=False,nav_success=False,opening_tested=False,grasp_observed=False,inspection_unsafe_abort=True)
            record['inspection'].update(unsafe_abort=True,collision=True)
        utility,mask=continuation_returns(scene['branches'][1]['records'])
        self.assertFalse(mask.any())
        scene['branches'][1].update(utility=utility,grasp_mask=mask)
        target=continuation_targets(scene,np.zeros((4,2)),np.zeros((4,2)),np.zeros(5),1.)
        self.assertFalse(target['post_mask']);self.assertLess(target['inspect'],-2.)

    def test_policy_can_refuse_after_inspection_and_pays_real_cost(self):
        scene=scene_fixture()
        prediction=(np.array([-1.,-1.,.2,-1.,0.]),np.array(.7),np.array([-1.,-1.,-1.,-1.,0.]),
                    np.zeros((4,2)),np.zeros((4,2)))
        d=choose(scene,prediction,'learned','test')
        self.assertTrue(d['inspection_attempted']);self.assertTrue(d['refused'])
        self.assertTrue(d['feasible']);self.assertLess(d['net_utility'],0.)
        d=choose(scene,prediction,'never','test')
        self.assertFalse(d['inspection_attempted']);self.assertTrue(d['success'])
        with self.assertRaises(ValueError):choose(scene,prediction,'invalid','test')

    def test_real_inspection_then_opening_preserves_state_and_staging(self):
        seen=[]
        def pre(arrays):
            self.assertEqual(set(arrays),{'initial_rgb','initial_proprio'});seen.append('pre');return INSPECT
        def post(arrays):
            self.assertTrue(arrays['post_view_available'])
            self.assertGreater(pose_from_state(arrays['post_proprio'])[1],1.64)
            self.assertNotIn('arrival_rgb',arrays);seen.append('post');return 2
        def grasp(rgb,state,commands):seen.append('grasp');return 0
        with patch('mujoco.mj_resetData',wraps=mujoco.mj_resetData) as resets:
            record,arrays,trace,exp=execute_active_candidate(scenario_by_id('oak-lever-wall'),None,0,
                pre_selector=pre,post_selector=post,grasp_selector=grasp,keep_trace=True)
        self.assertEqual(resets.call_count,1);self.assertEqual(seen,['pre','post','grasp'])
        self.assertTrue(record['inspection_success']);self.assertTrue(record['success'])
        self.assertTrue(record['result']['continuous_state_at_skill_entry'])
        self.assertGreater(record['observation_cost'],0.)
        self.assertFalse(np.array_equal(arrays['initial_rgb'],arrays['post_rgb']))
        self.assertTrue(all(b['time_s']>=a['time_s']-1e-8 for a,b in zip(trace,trace[1:])))
        self.assertEqual(exp.m.neq,0)
        self.assertNotIn(exp.m.joint('door_hinge').id,exp.m.actuator_trnid[:,0])

    def test_inspection_signed_distance_rejects_the_measured_pad_wall_corner(self):
        exp=MotionDoor(scenario_by_id('oak-lever-wall'))
        before=exp.d.qpos.copy()
        checker=InspectionChecker(exp)
        reason=checker.invalid([.1526015857,-1.3226096597,-math.pi/2])
        self.assertIsNotNone(reason)
        self.assertIn('wall_front',reason)
        np.testing.assert_array_equal(exp.d.qpos,before)


if __name__=='__main__':unittest.main()
