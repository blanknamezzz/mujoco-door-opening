"""Continuous physical execution, planner isolation and stage-credit contracts."""
from unittest.mock import patch
import unittest
import math
import mujoco
import numpy as np
from door_stress_scenes import scenario_by_id
from mobile_door import MobileDoor, CollisionChecker, execute_candidate, original_proprio
from staged_mobile_reward import credit_targets, shared_targets


def action_table(successes=((2,0),)):
    return [dict(scene_id='fixture',base_id=b,grasp_id=g,side='front' if b<2 else 'back',
                 nav_success=True,nav_collision=False,measured_path_length_m=1.,
                 opening_tested=True,grasp_observed=True,success=(b,g) in successes,
                 collision=False,reward=999.,safe_opening_fraction=.99)
            for b in range(4) for g in range(2)]


class MobileDoorContracts(unittest.TestCase):
    def test_base_is_actuated_and_keeps_original_checkpoint_state_contract(self):
        exp=MobileDoor(scenario_by_id('oak-lever-wall'))
        self.assertEqual(len(exp.base_q),3)
        self.assertEqual(len(exp.base_ctrl),3)
        self.assertEqual(original_proprio(exp).shape,(19,))
        self.assertFalse(exp.m.neq)
        self.assertNotIn(exp.m.joint('door_hinge').id,exp.m.actuator_trnid[:,0])
        with self.assertRaisesRegex(RuntimeError,'cannot reset'):
            exp.reset()

    def test_collision_probe_does_not_change_live_state(self):
        exp=MobileDoor(scenario_by_id('oak-lever-wall'))
        q,v,t=exp.d.qpos.copy(),exp.d.qvel.copy(),exp.d.time
        margins=exp.m.geom_margin.copy()
        checker=CollisionChecker(exp)
        self.assertIsNone(checker.invalid(exp.start_pose))
        checker.invalid([.6,1.,-math.pi/2])
        np.testing.assert_array_equal(exp.d.qpos,q)
        np.testing.assert_array_equal(exp.d.qvel,v)
        self.assertEqual(exp.d.time,t)
        np.testing.assert_array_equal(exp.m.geom_margin,margins)

    def test_continuous_navigation_preparation_and_opening_never_reset_mid_episode(self):
        with patch('mujoco.mj_resetData',wraps=mujoco.mj_resetData) as resets:
            record,_,trace,_=execute_candidate(scenario_by_id('oak-lever-wall'),2,0,
                                               capture=False,keep_trace=True)
        self.assertEqual(resets.call_count,1)
        self.assertTrue(record['nav_success'])
        self.assertTrue(record['success'])
        self.assertTrue(record['result']['continuous_state_at_skill_entry'])
        self.assertGreater(record['base_displacement_m'],2.5)
        self.assertLess(record['max_trace_step_m'],.025)
        times=np.array([x['time_s'] for x in trace])
        self.assertTrue((np.diff(times)>=-1e-8).all())

    def test_partial_opening_cannot_compensate_for_wrong_side(self):
        table=action_table();targets=credit_targets(table)
        self.assertLess(targets['position'][0],0.)
        self.assertGreater(targets['position'][2],targets['position'][3])
        before=targets['position'].copy()
        for episode in table:episode.update(reward=-1e9,safe_opening_fraction=1.)
        np.testing.assert_array_equal(before,credit_targets(table)['position'])

    def test_grasp_failure_does_not_relabel_correct_side_and_nav_failure_is_gated(self):
        table=action_table();targets=credit_targets(table)
        self.assertGreater(targets['position'][3],0.)
        self.assertEqual(targets['grasp'][2,1],-1.)
        table[6].update(nav_success=False,opening_tested=False,grasp_observed=False)
        table[7].update(nav_success=False,opening_tested=False,grasp_observed=False)
        targets=credit_targets(table)
        self.assertLess(targets['position'][3],0.)
        self.assertTrue(np.isnan(targets['opening'][3]).all())
        self.assertFalse(targets['grasp_training_mask'][3].any())

    def test_refusal_best_when_all_tested_candidates_fail(self):
        targets=credit_targets(action_table(()))
        self.assertEqual(int(targets['position'].argmax()),4)
        with self.assertRaises(ValueError):credit_targets(action_table()[:-1])
        with self.assertRaises(ValueError):shared_targets(action_table(),np.ones((4,2)))

if __name__=='__main__':unittest.main()
