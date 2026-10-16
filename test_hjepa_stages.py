"""Stage interventions must preserve state and isolate policy decisions."""
from copy import deepcopy
import math
import unittest
import numpy as np
from evaluate_hjepa_stages import select, stage_metrics, summarize
from hjepa_scene import action_at
from simulate import Experiment, skill_target


class StageEvaluationContracts(unittest.TestCase):
    def test_camera_orbit_preserves_calibration_and_nominal_view(self):
        from evaluate_hjepa_views import CameraScene
        from hjepa_scene import DoorScene
        normal = DoorScene('normal').experiment()
        viewed = CameraScene('normal').experiment()
        cid = normal.m.camera('perception').id
        np.testing.assert_allclose(normal.d.cam_xpos[cid], viewed.d.cam_xpos[cid], atol=1e-12)
        np.testing.assert_allclose(normal.d.cam_xmat[cid], viewed.d.cam_xmat[cid], atol=1e-12)
        for yaw, elevation in [(-30., -10.), (30., 10.)]:
            exp = CameraScene('test', view_yaw_deg=yaw, view_elevation_deg=elevation).experiment()
            direction = np.array([.8, -.11, .85])-exp.d.cam_xpos[cid]
            direction /= np.linalg.norm(direction)
            np.testing.assert_allclose(-exp.d.cam_xmat[cid].reshape(3, 3)[:, 2], direction, atol=1e-12)

    def test_policy_crosses_change_only_the_requested_decision(self):
        def trial(label):
            return dict(result={'label': label}, metrics={})
        bases = [dict(feasible=[True, True], direct_scores={'pre': [2., 0.], 'post': [0., 2.]},
                      trials=[trial(f'{b}/0'), trial(f'{b}/1')]) for b in range(2)]
        table = dict(case={'scene': {'scene_id': 'test'}, 'family': 'test'},
                     position_scores={'pre': [2., 0.], 'post': [0., 2.]}, bases=bases)
        for p, b in [('pre', 0), ('post', 1)]:
            for g, gid in [('pre', 0), ('post', 1)]:
                self.assertEqual(select(table, p, g)['result']['label'], f'{b}/{gid}')
        # Arbitrary future outcome changes must not affect either decision.
        changed = deepcopy(table)
        for b in changed['bases']:
            for t in b['trials']:
                t['result'] = {'success': True, 'reward': 10000.}
        self.assertEqual((select(changed)['base_id'], select(changed)['grasp_id']), (1, 1))

    def test_stage_denominators_do_not_hide_upstream_failures(self):
        samples = [dict(result={'outcome': 'A_initial_unreachable'}, rows=[]),
                   dict(result={'outcome': 'B_reachable_grasp_failed', 'stop_reason': 'grasp_not_acquired'},
                        rows=[dict(time_s=3., door_deg=0., unexpected_contact='',
                                   slip_detected=False, contact_lost=False)]),
                   dict(result={'outcome': 'C_grasped_but_failed', 'grasp_acquired': True,
                                'stop_reason': 'arm_workspace_limit'},
                        rows=[dict(time_s=7., door_deg=40., unexpected_contact='',
                                   slip_detected=False, contact_lost=False)])]
        trials = [dict(result=s['result'], metrics=stage_metrics(**s)) for s in samples]
        metrics = summarize(trials)
        self.assertEqual(metrics['stage_counts'], dict(position=2, align=2, grasp=1, open_hold=0))
        self.assertEqual(metrics['conditional_rates']['grasp/align'], .5)
        self.assertEqual(metrics['failure_stages'], {'position_unreachable': 1, 'grasp': 1, 'opening': 1})

    def test_target_angle_changes_cartesian_commands_and_real_hold(self):
        point = np.array([.8, -.11, .85])
        p, yaw, aperture, phase = skill_target(14., point, opening_angle_deg=30.)
        self.assertAlmostEqual(yaw, math.pi/3)
        self.assertEqual(phase, 'hold')
        self.assertAlmostEqual(np.linalg.norm(p[:2]), np.linalg.norm(point[:2]))
        self.assertFalse(np.allclose(action_at(14., point, 0., opening_angle_deg=30.),
                                     action_at(14., point, 0.)))
        result, rows, _ = Experiment().trial(point, opening_angle_deg=30.)
        self.assertEqual(result['outcome'], 'D_target_open')
        self.assertTrue(result['held_open_one_second'])
        self.assertAlmostEqual(result['final_angle_deg'], 30., delta=2.)
        self.assertTrue(stage_metrics(result, rows, 30.)['success'])
        with self.assertRaises(ValueError):
            Experiment().trial(point, opening_angle_deg=120.)


if __name__ == '__main__':
    unittest.main()
