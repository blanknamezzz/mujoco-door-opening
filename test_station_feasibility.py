"""Guard against confusing a search miss with a station-level impossibility."""
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from grasp_plan import GraspPlan
from simulate import Experiment, rot
from station_feasibility import (SearchSettings, candidate_plans, diagnose_station,
                                 workspace_certificate)


class StationFeasibilityTests(unittest.TestCase):
    def test_failed_grasp_is_rescued_at_the_same_base(self):
        r = diagnose_station((.8, -1.), settings=SearchSettings(max_rollouts=4))
        self.assertEqual(r['status'], 'feasible_with_alternative_grasp')
        self.assertNotEqual(r['baseline_result']['result']['outcome'], 'D_full_open')
        self.assertEqual(r['attempts'][-1]['result']['outcome'], 'D_full_open')
        self.assertEqual(r['changed_parameters'], ['approach_angle_deg'])
        self.assertEqual(r['witness_plan']['approach_angle_deg'], 8.)
        for a in r['attempts']:
            self.assertEqual(a['base_xy_m'], [.8, -1.])
            self.assertTrue(a['preparation']['execution']['success'])

    def test_close_initial_handle_can_leave_the_entire_contact_envelope(self):
        exp = Experiment(base=(1.4, -.4))
        c = workspace_certificate(exp)
        self.assertTrue(c['certified_exclusion'])
        self.assertGreater(c['excess_m'], .05)
        # Initial handle distance is within the envelope, but opening is not.
        self.assertLess(np.linalg.norm(np.array([.8, -.11])-exp.base), c['handle_axis_radius_bound_m'])
        r = diagnose_station((1.4, -.4))
        self.assertEqual(r['status'], 'excluded_by_workspace_bound')
        self.assertEqual(r['physical_rollouts'], 0)
        self.assertIsNone(r['witness_plan'])

    def test_success_tolerance_does_not_require_reaching_exactly_90_degrees(self):
        exp = Experiment(base=(1.3, -.4))
        relaxed = workspace_certificate(exp)
        exact = workspace_certificate(exp, success_tolerance_deg=0.)
        self.assertFalse(relaxed['certified_exclusion'])
        self.assertTrue(exact['certified_exclusion'])
        self.assertEqual(relaxed['minimum_success_angle_deg'], 88.)

    def test_budget_failure_preserves_unknown_after_real_partial_opening(self):
        r = diagnose_station((.3, -.9), settings=SearchSettings(max_rollouts=1))
        self.assertEqual(r['status'], 'unconfirmed_after_search')
        self.assertFalse(r['certificate']['certified_exclusion'])
        self.assertEqual(r['baseline_result']['result']['stop_reason'], 'arm_workspace_limit')
        self.assertAlmostEqual(r['baseline_result']['result']['final_angle_deg'], 42.5, delta=.2)
        self.assertEqual(r['stop_reason'], 'rollout_budget_exhausted')
        self.assertIsNone(r['witness_plan'])

    def test_even_exhausting_a_finite_pool_does_not_prove_absence(self):
        p = GraspPlan((.8, -.11, .85))
        cfg = SearchSettings(angles_deg=(0.,), heights_m=(0.,), branches=(-1,),
                             forces_N=(35.,), open_times_s=(10.,), max_rollouts=4)
        r = diagnose_station((.3, -.9), baseline=p, settings=cfg)
        self.assertEqual(r['status'], 'unconfirmed_after_search')
        self.assertEqual(r['stop_reason'], 'finite_candidate_set_exhausted')
        self.assertEqual(r['physical_rollouts'], 1)

    def test_unsupported_model_fails_open_and_zero_budget_cannot_claim_feasible(self):
        exp = Experiment()
        exp.m.jnt_axis[exp.arm_j[1]] = [0., 1., 0.]
        c = workspace_certificate(exp)
        self.assertFalse(c['certified_exclusion'])
        self.assertEqual(c['status'], 'unsupported_geometry')
        shifted = Experiment()
        shifted.m.jnt_pos[shifted.arm_j[0]] = [.05, 0., 0.]
        self.assertEqual(workspace_certificate(shifted)['status'], 'unsupported_geometry')
        r = diagnose_station((.8, -1.), settings=SearchSettings(max_rollouts=0))
        self.assertEqual(r['status'], 'unconfirmed_after_search')
        self.assertEqual(r['physical_rollouts'], 0)

    def test_contact_envelope_contains_sampled_fk_pad_corners(self):
        exp = Experiment()
        c = workspace_certificate(exp)
        probe = mujoco.MjData(exp.m)
        rng = np.random.default_rng(42)
        for _ in range(100):
            probe.qpos[exp.arm_q] = rng.uniform(-math.pi, math.pi, 3)
            for name in ('left_slide', 'right_slide'):
                j = exp.m.joint(name).id
                probe.qpos[exp.m.jnt_qposadr[j]] = rng.uniform(*exp.m.jnt_range[j])
            mujoco.mj_forward(exp.m, probe)
            tcp = probe.site_xpos[exp.site, :2]
            self.assertLessEqual(np.linalg.norm(tcp-exp.base), c['tcp_max_radius_m']+1e-9)
            for gid in exp.pads:
                for sign in ((-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),
                             (1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1)):
                    corner=probe.geom_xpos[gid]+probe.geom_xmat[gid].reshape(3,3)@(exp.m.geom_size[gid]*sign)
                    self.assertLessEqual(np.linalg.norm(corner[:2]-exp.base)+c['bar_radius_m'],
                                         c['handle_axis_radius_bound_m']+1e-9)

    def test_collection_preflight_adds_a_separate_station_bound_without_opening(self):
        from collect_grasp_data import preflight_job
        from prm import PRMSettings
        p = GraspPlan((.8, -.11, .85))
        m = dict(angles_deg=[0., 8.], physics=dict(plan=p.to_dict(),
            prm=PRMSettings().to_dict(), friction=.8, hinge_resistance=1.2))
        scene = dict(scene_id='train_000', split='train', base_xy_m=[1.4, -.4], base_yaw_deg=90.)
        r = preflight_job((m, scene))
        self.assertTrue(r['station_workspace_check']['certified_exclusion'])
        self.assertEqual(len(r['queries']), 2)
        self.assertNotIn('reward', r)

    def test_plan_pool_and_invalid_ranges(self):
        p = GraspPlan((.8, -.11, .85))
        plans = list(candidate_plans(p, SearchSettings()))
        self.assertTrue(any(a.ik_branch==1 for a in plans))
        self.assertTrue(any(a.height_offset_m==.07 for a in plans))
        self.assertTrue(any(a.approach_angle_deg==-60 for a in plans))
        self.assertTrue(any(a.grip_force_limit_N==20 and a.timing.open_s==14 for a in plans))
        for cfg in [dict(max_rollouts=-1),dict(angles_deg=(float('nan'),)),dict(forces_N=(36.,))]:
            with self.assertRaises(ValueError):SearchSettings(**cfg)


if __name__ == '__main__':
    unittest.main()
