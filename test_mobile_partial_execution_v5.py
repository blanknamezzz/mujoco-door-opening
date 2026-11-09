"""Real camera availability and unchanged continuous contact execution."""
from dataclasses import replace
import math
import unittest
import numpy as np
from mobile_partial_execution_v5 import (geometric_station,geometry_depth_action,
    execute_partial_candidate)


class PartialExecutionContracts(unittest.TestCase):
    def test_geometry_rule_uses_current_pose_constraints_and_visibility(self):
        state=np.zeros(19);state[14:18]=[.6,-1.65,1,0]
        self.assertEqual(geometric_station([True,False,True,False],state),0)
        self.assertEqual(geometric_station([True,False,True,False],state,'rear_first'),2)
        self.assertEqual(geometric_station([False]*4,state),4)
        evidence=dict(mask=[False,False,True,False],queries=[dict(sampled_free_fraction=0.)]*4)
        self.assertEqual(geometry_depth_action(evidence,state,True),5)
        evidence['queries']=[dict(sampled_free_fraction=.6)]*4
        self.assertEqual(geometry_depth_action(evidence,state,True),2)

    def test_refusal_callback_has_only_an_initial_actual_rgbd_frame(self):
        from door_stress_scenes import scenario_by_id
        captured=[]
        def pre(observation):
            self.assertEqual(set(observation),{'initial_rgb','initial_proprio','initial_frame'})
            self.assertEqual(observation['initial_frame']['depth_m'].shape,(448,448))
            captured.append(observation);return 4
        record,arrays,trace,exp=execute_partial_candidate(scenario_by_id('oak-lever-wall'),None,0,
            pre_selector=pre,keep_trace=True)
        self.assertEqual(len(captured),1);self.assertTrue(record['refused'])
        self.assertEqual(exp.reset_count,1);self.assertEqual(exp.d.time,0.)
        self.assertFalse(record['inspection_attempted']);self.assertEqual(record['measured_path_length_m'],0.)

    def test_added_depth_exposure_preserves_full_inspection_and_opening_physics(self):
        from door_stress_scenes import scenario_by_id
        from mobile_active_v3 import execute_active_candidate
        scene=scenario_by_id('oak-lever-wall');start=(.6,-1.65,math.pi/2)
        old,old_arrays,old_trace,old_exp=execute_active_candidate(scene,2,0,start_pose=start,inspection=True,keep_trace=True)
        new,arrays,trace,exp=execute_partial_candidate(scene,2,0,start_pose=start,inspection=True,keep_trace=True)
        for key in ('success','collision','nav_success','opening_tested','inspection_success','integration_steps','episode_duration_s'):
            self.assertEqual(old[key],new[key],key)
        self.assertTrue(new['inspection_success']);self.assertTrue(new['opening_tested'])
        self.assertTrue(new['result']['continuous_state_at_skill_entry'])
        np.testing.assert_array_equal(old_exp.d.qpos,exp.d.qpos)
        np.testing.assert_array_equal(old_exp.d.qvel,exp.d.qvel)
        self.assertEqual(old_trace,trace);self.assertEqual(exp.reset_count,1)
        np.testing.assert_array_equal(old_arrays['post_proprio'],arrays['post_proprio'])
        self.assertEqual(arrays['post_frame']['depth_m'].shape,(448,448))


if __name__=='__main__':unittest.main()
