"""Physical PRM contracts: free endpoints, blocked edge, detour and execution."""
import unittest
import csv
from pathlib import Path
import tempfile
import numpy as np

from grasp_plan import GraspPlan
from prm import PRMSettings, Roadmap, execute_path, segment_box_distance_squared
from prm_examples import cases
from simulate import Experiment
from train_grasp_rl import rollout


class PRMTests(unittest.TestCase):
    def test_direct_query_checks_states_and_edges(self):
        roadmap = Roadmap(Experiment())
        q = roadmap.plan(GraspPlan((.8, -.11, .85), approach_angle_deg=8))
        self.assertEqual(q['status'], 'direct_path')
        self.assertEqual(len(q['path_q']), 2)
        self.assertTrue(roadmap.link(*q['path_q']))
        self.assertGreater(q['state_checks'], 2)

    def test_blocked_direct_edge_has_a_valid_repeatable_detour(self):
        case = cases()[1]
        exp = Experiment(model_xml=case['model_xml'])
        settings = PRMSettings(samples=500, neighbors=18)
        roadmap = Roadmap(exp, settings)
        q = roadmap.plan(GraspPlan((.8, -.11, .85), approach_angle_deg=8))
        self.assertEqual(q['status'], 'roadmap_path')
        self.assertFalse(q['direct_path_clear'])
        self.assertGreater(len(q['path_q']), 2)
        for a, b in zip(q['path_q'], q['path_q'][1:]):
            self.assertTrue(roadmap.link(a, b))
        again = Roadmap(Experiment(model_xml=case['model_xml']), settings).plan(
            GraspPlan((.8, -.11, .85), approach_angle_deg=8))
        np.testing.assert_array_equal(q['path_q'], again['path_q'])
        report, _, _ = execute_path(exp, q['path_q'])
        self.assertTrue(report['success'])
        self.assertGreater(report['integration_steps'], 0)
        self.assertAlmostEqual(exp.d.time, 0.)
        self.assertLess(abs(report['door_angle_deg']), .5)

    def test_no_samples_is_unconfirmed_not_infeasible(self):
        case = cases()[1]
        q = Roadmap(Experiment(model_xml=case['model_xml']), PRMSettings(samples=0)).plan(
            GraspPlan((.8, -.11, .85), approach_angle_deg=8))
        self.assertFalse(q['path_found'])
        self.assertEqual(q['status'], 'no_path_within_budget')

    def test_joint_limits_and_invalid_start_are_explicit(self):
        r = Roadmap(Experiment())
        valid = r.plan(GraspPlan((.8, -.11, .85)))['goal_q']
        q = r.query(valid, start=[1., 0., 0., 0.])
        self.assertEqual(q['status'], 'invalid_start')
        self.assertEqual(q['reason'], 'joint_limits')
        obstacle = Roadmap(Experiment(model_xml=cases()[1]['model_xml']))
        target = obstacle.plan(GraspPlan((.8, -.11, .85), approach_angle_deg=8))['goal_q']
        start = np.asarray(obstacle.settings.initial_q)
        q = obstacle.query(target, start=start+.55*(np.asarray(target)-start))
        self.assertEqual(q['status'], 'invalid_start')
        self.assertTrue(q['reason'].startswith('collision:'))

    def test_unreachable_target_skips_opening_and_keeps_reason(self):
        config = dict(base_yaw_rad=np.pi/2, friction=.8, hinge_resistance=1.2,
                      plan=GraspPlan((.8, -.11, .85)).to_dict(), prm=PRMSettings().to_dict())
        record, rows, states, _ = rollout(config, (2., -1.2), 8.)
        self.assertFalse(record['opening_tested'])
        self.assertEqual(record['preparation']['query']['status'], 'goal_ik_unreachable')
        self.assertEqual(record['result']['stop_reason'], 'prm_goal_ik_unreachable')
        self.assertFalse(rows)
        self.assertFalse(states)

    def test_real_preparation_then_full_open_without_pregrasp_reset(self):
        config = dict(base_yaw_rad=np.pi/2, friction=.8, hinge_resistance=1.2,
                      plan=GraspPlan((.8, -.11, .85)).to_dict(), prm=PRMSettings().to_dict())
        record, _, _, _ = rollout(config, (.8, -1.), 8.)
        self.assertTrue(record['preparation']['execution']['success'])
        self.assertTrue(record['opening_tested'])
        self.assertEqual(record['result']['outcome'], 'D_full_open')

    def test_configuration_rejects_nonfinite_and_invalid_settings(self):
        for kwargs in [dict(joint_step_rad=0), dict(samples=-1), dict(neighbors=0),
                       dict(initial_q=[0, 0, float('nan'), 0])]:
            with self.assertRaises(ValueError):
                PRMSettings(**kwargs)

    def test_deep_door_crossing_is_not_mistaken_for_free_space(self):
        self.assertEqual(segment_box_distance_squared([-2,0,0], [2,0,0], [1,1,1]), 0.)
        self.assertEqual(segment_box_distance_squared([-2,2,0], [2,2,0], [1,1,1]), 1.)
        r = Roadmap(Experiment())
        self.assertIn('door_panel', r.invalid_reason([0., 0., 0., 0.]))

    def test_execution_collision_is_recorded_and_exportable(self):
        from collect_grasp_data import export_csv
        from compare_grasp_learning import summarize
        case = cases()[1]
        # Intentionally coarse LINK misses the small obstacle. The physical
        # controller must catch the collision, skip opening and save the row.
        config = dict(base_yaw_rad=np.pi/2, friction=.8, hinge_resistance=1.2,
            plan=GraspPlan((.8, -.11, .85)).to_dict(), model_xml=case['model_xml'],
            prm=PRMSettings(samples=0, joint_step_rad=10., lift_step_m=1.).to_dict())
        record, _, _, _ = rollout(config, (.6, -1.), 8.)
        self.assertEqual(record['preparation']['query']['status'], 'direct_path')
        self.assertFalse(record['opening_tested'])
        self.assertEqual(record['result']['stop_reason'], 'pregrasp_execution_collision')
        self.assertEqual(record['result']['unexpected_contact_samples'], 1)
        row = dict(scene_id='failure_test', split='test', base_yaw_deg=90., action_index=0, **record)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'records.csv'
            export_csv(path, [row])
            with path.open() as file:
                saved = list(csv.DictReader(file))[0]
        self.assertEqual(saved['opening_tested'], 'False')
        self.assertEqual(saved['collision'], '1')
        self.assertEqual(float(saved['safe_max_angle_deg']), 0.)
        self.assertEqual(summarize([row], {('failure_test', 0): row})['collisions'], 1)


if __name__ == '__main__':
    unittest.main()
