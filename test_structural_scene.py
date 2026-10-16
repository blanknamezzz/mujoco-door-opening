"""Structural failures must arise from contact/limits, never side labels."""
import math
import unittest
import numpy as np
from hjepa_scene import action_at, grasps
from perception import observe, estimate_frame
from planning import choose_grasp
from structural_scene import STRUCTURAL_BASES, structural_scenes


def execute(name, base_id):
    exp = structural_scenes()[name].experiment(STRUCTURAL_BASES[base_id])
    point, yaw = grasps(estimate_frame(observe(exp)), exp.grasp_yaw_offset, exp.grasp_yaw_sign)[0]
    result, rows, _ = exp.trial(point, yaw, reset_before_trial=False)
    return exp, result, rows


class StructuralContracts(unittest.TestCase):
    def test_closed_one_way_limits_are_not_observable_in_initial_rgbd(self):
        scenes = structural_scenes()
        positive = scenes['one-way-positive'].experiment()
        negative = scenes['one-way-negative'].experiment()
        frames = [observe(e) for e in [positive, negative]]
        np.testing.assert_array_equal(frames[0]['rgb'], frames[1]['rgb'])
        np.testing.assert_array_equal(frames[0]['depth_m'], frames[1]['depth_m'])
        np.testing.assert_array_equal(positive.d.qpos, negative.d.qpos)

    def test_free_swing_succeeds_with_opposite_signed_angles(self):
        for bid, direction in [(0, 1), (2, -1)]:
            exp, result, _ = execute('free-two-way', bid)
            self.assertEqual(result['outcome'], 'D_full_open')
            self.assertEqual(result['opening_direction'], direction)
            self.assertAlmostEqual(result['final_signed_angle_deg'], direction*90., delta=2.)
            self.assertEqual(exp.m.neq, 0)
            self.assertNotIn(exp.m.joint('door_hinge').id, exp.m.actuator_trnid[:, 0])

    def test_pillar_blocks_door_for_two_same_side_bases_but_not_reverse(self):
        for bid in [0, 1]:
            _, result, _ = execute('pillar-front', bid)
            self.assertTrue(result['grasp_acquired'])
            self.assertEqual(result['stop_reason'], 'door_obstructed')
            self.assertIn('door_panel', result['first_unexpected_contact'])
            self.assertIn('pillar_0', result['first_unexpected_contact'])
            self.assertTrue(35 < result['final_angle_deg'] < 45)
        _, reverse, _ = execute('pillar-front', 2)
        self.assertEqual(reverse['outcome'], 'D_full_open')

    def test_forbidden_hinge_direction_holds_closed_despite_acquired_grasp(self):
        exp, result, _ = execute('one-way-positive', 2)
        self.assertTrue(result['grasp_acquired'])
        self.assertTrue(result['slip_detected'] or result['contact_lost'])
        self.assertLess(abs(result['final_signed_angle_deg']), .1)
        self.assertAlmostEqual(exp.m.jnt_range[exp.m.joint('door_hinge').id, 0], 0.)

    def test_one_sided_handle_is_physically_inaccessible_from_back(self):
        exp = structural_scenes()['front-access-only'].experiment(STRUCTURAL_BASES[2])
        with self.assertRaisesRegex(ValueError, 'Handle detection failed'):
            estimate_frame(observe(exp))
        # Explicit geometry-only diagnostic, not a perception fallback: try the
        # real front handle through the panel from the wrong face.
        result, _, _ = exp.trial(np.array([.8, -.11, .85]), math.pi, reset_before_trial=False)
        self.assertFalse(result['grasp_acquired'])
        self.assertEqual(result['stop_reason'], 'unexpected_collision')
        self.assertIn('door_panel', result['first_unexpected_contact'])

    def test_geometry_screens_signed_hinge_and_obstacle_path(self):
        for name, bid, expected, reason in [('free-two-way', 2, 90., 'full_geometric_path'),
                ('pillar-front', 0, 38., 'collision:pillar_0:door_panel'),
                ('one-way-positive', 2, 0., 'hinge_direction_limit')]:
            exp = structural_scenes()[name].experiment(STRUCTURAL_BASES[bid])
            selected = choose_grasp(exp, estimate_frame(observe(exp)), candidate_offsets=[(0., 0.)])['selected']
            self.assertEqual(selected['geometric_open_deg'], expected)
            self.assertEqual(selected['reason'], reason)

    def test_packed_action_schedule_mirrors_actual_swing(self):
        front = action_at(10., np.array([.8, -.11, .85]), 0.)
        back = action_at(10., np.array([.8, .11, .85]), math.pi, opening_direction=-1)
        np.testing.assert_allclose(back, front*np.array([1, -1, 1, -1, 1, 1]), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
