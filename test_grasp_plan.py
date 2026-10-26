"""Physics and replay contracts for parameterized grasp-and-opening plans."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

from grasp_plan import GraspPlan, SkillTiming
from hjepa_scene import action_at
from simulate import Experiment, ROOT, skill_target


CENTER = (.8, -.11, .85)


class GraspPlanContracts(unittest.TestCase):
    def test_serialized_plan_keeps_all_execution_parameters(self):
        plan = GraspPlan(CENTER, height_offset_m=.03, approach_angle_deg=8., ik_branch=1,
                         grip_force_limit_N=20., timing=SkillTiming(open_s=14.))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'plan.json'
            plan.save(path)
            self.assertEqual(GraspPlan.load(path), plan)
        self.assertEqual(plan.point_world_m, (.8, -.11, .88))
        back = Experiment(operating_side='back')
        self.assertAlmostEqual(plan.yaw_offset_rad(back), math.pi-math.radians(8.))

    def test_invalid_or_unsafe_parameters_are_rejected(self):
        for values in [dict(ik_branch=0), dict(grip_force_limit_N=36), dict(grip_force_limit_N=0),
                       dict(approach_angle_deg=float('nan')), dict(approach_distance_m=-1),
                       dict(opening_angle_deg=91)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                GraspPlan(CENTER, **values)
        for values in [dict(open_s=0), dict(close_s=.7), dict(hold_s=.5)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                SkillTiming(**values)

    def test_custom_timing_and_packed_actions_use_same_schedule(self):
        timing = SkillTiming(settle_s=1., approach_s=3., close_s=2., open_s=14., hold_s=3.)
        point = np.array(CENTER)
        self.assertEqual(skill_target(5., point, timing=timing)[3], 'close')
        target, yaw, aperture, phase = skill_target(13., point, timing=timing)
        # Half the opening duration should command 45 degrees of door rotation.
        expected = np.array([(.8-.11)/math.sqrt(2), (-.8-.11)/math.sqrt(2), .85])
        np.testing.assert_allclose(target, expected)
        self.assertAlmostEqual(yaw, math.pi/4)
        self.assertEqual((aperture, phase), (0., 'open'))
        self.assertEqual(skill_target(20., point, timing=timing)[3], 'hold')
        commands = [skill_target(t, point, timing=timing) for t in np.linspace(13., 13.5, 10, endpoint=False)]
        packed = np.mean([[*p, math.sin(y), math.cos(y), a/.04] for p, y, a, _ in commands], axis=0)
        np.testing.assert_allclose(action_at(13., point, 0., timing=timing), packed, atol=1e-7)

    def test_default_plan_matches_legacy_physical_trajectory(self):
        old, old_rows, old_states = Experiment().trial(np.array(CENTER))
        new, new_rows, new_states = Experiment().trial(plan=GraspPlan(CENTER))
        self.assertEqual(new['outcome'], 'D_full_open')
        self.assertEqual(new['final_angle_deg'], old['final_angle_deg'])
        self.assertEqual(len(new_rows), len(old_rows))
        np.testing.assert_array_equal(new_states, old_states)

    def test_slower_plan_runs_through_new_hold_time(self):
        plan = GraspPlan(CENTER, height_offset_m=.03, timing=SkillTiming(open_s=14.))
        result, rows, _ = Experiment().trial(plan=plan)
        self.assertEqual(result['outcome'], 'D_full_open')
        self.assertTrue(result['held_open_one_second'])
        self.assertAlmostEqual(rows[-1]['time_s'], plan.timing.end, delta=.011)
        self.assertEqual(result['point_world_m'], [.8, -.11, .88])
        self.assertEqual(result['execution_plan']['timing']['open_s'], 14.)

    def test_force_cap_changes_physics_and_is_enforced(self):
        exp = Experiment()
        measured = []
        plan = GraspPlan(CENTER, grip_force_limit_N=.05)
        result, _, _ = exp.trial(plan=plan, observer=lambda e, _: measured.append(e.d.actuator_force[-2:].copy()))
        np.testing.assert_allclose(exp.m.actuator('left_grip').forcerange, [-.05, .05])
        self.assertLessEqual(np.max(np.abs(measured)), .05+1e-9)
        self.assertNotEqual(result['outcome'], 'D_full_open')
        self.assertFalse(result['grasp_acquired'])

    def test_positive_branch_reaches_actuator_targets_and_metadata(self):
        exp = Experiment()
        plan = GraspPlan(CENTER, ik_branch=1)
        result, rows, states = exp.trial(plan=plan, mode='reach_only')
        self.assertTrue(rows)
        self.assertEqual(exp.ik_branch, 1)
        self.assertGreater(states[0][exp.arm_q[1]], 0.)
        self.assertEqual(result['execution_plan']['ik_branch'], 1)

    def test_default_trial_does_not_inherit_previous_plan_force(self):
        exp = Experiment()
        exp.trial(plan=GraspPlan(CENTER, grip_force_limit_N=.05))
        reused, _, reused_states = exp.trial(CENTER)
        fresh, _, fresh_states = Experiment().trial(CENTER)
        self.assertEqual(reused['outcome'], 'D_full_open')
        self.assertEqual(reused['outcome'], fresh['outcome'])
        np.testing.assert_array_equal(reused_states, fresh_states)
        np.testing.assert_array_equal(exp.m.actuator('left_grip').forcerange, [-35., 35.])

    def test_default_trial_restores_constructor_branch_after_plan(self):
        exp = Experiment(ik_branch=1)
        exp.trial(plan=GraspPlan(CENTER, ik_branch=-1), mode='reach_only')
        _, _, reused = exp.trial(CENTER, mode='reach_only')
        _, _, fresh = Experiment(ik_branch=1).trial(CENTER, mode='reach_only')
        self.assertEqual(exp.ik_branch, 1)
        np.testing.assert_array_equal(reused, fresh)

    def test_live_perception_interfaces_remain_available(self):
        from perception import observe, estimate_frame
        self.assertTrue(callable(observe))
        self.assertTrue(callable(estimate_frame))

    def test_nonzero_angle_and_height_change_initial_joint_configuration(self):
        exp = Experiment()
        plan = GraspPlan(CENTER, height_offset_m=.03, approach_angle_deg=8.)
        initial = []
        exp.trial(plan=plan, mode='reach_only', observer=lambda e, _: initial.append(e.d.qpos.copy()))
        self.assertTrue(initial)
        self.assertAlmostEqual(initial[0][exp.lift_q], .03)
        world_yaw = exp.base_yaw + sum(initial[0][exp.arm_q])
        self.assertAlmostEqual(world_yaw, math.pi/2+math.radians(8.))

    def test_saved_plan_cli_reproduces_result_and_rejects_overrides(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            plan = GraspPlan(CENTER, timing=SkillTiming(open_s=12.))
            path = output/'plan.json'
            plan.save(path)
            command = [sys.executable, str(ROOT/'simulate.py'), '--plan', str(path), '--output', str(output)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            result = json.loads((output/'nominal.json').read_text())
            self.assertEqual(result['execution_plan'], plan.to_dict())
            self.assertEqual(result['outcome'], 'D_full_open')
            rejected = subprocess.run(command+['--grip-force', '10'], capture_output=True, text=True)
            self.assertEqual(rejected.returncode, 2)
            self.assertIn('cannot be combined', rejected.stderr)


if __name__ == '__main__':
    unittest.main()
