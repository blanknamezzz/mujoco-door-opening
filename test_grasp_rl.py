"""Numerical gradient, reward safety, and checkpoint tests for grasp RL."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from grasp_plan import GraspPlan
from train_grasp_rl import Adam, Policy, load_checkpoint, reward, save_checkpoint


class GraspRLTests(unittest.TestCase):
    def test_policy_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(7)
        policy = Policy(3, rng)
        policy.params["w2"][:] = rng.normal(0, .2, (16, 3))
        x = rng.normal(size=(4, 4))
        actions = np.array([0, 2, 1, 0])
        advantages = np.array([1., -.4, .2, -.7])
        entropy_coef = .03
        grads = policy.gradient(x, actions, advantages, entropy_coef)

        def objective():
            _, p = policy.forward(x)
            entropy = -(p*np.log(p)).sum(axis=1)
            return np.mean(advantages*np.log(p[np.arange(4), actions]) + entropy_coef*entropy)

        for key, values in policy.params.items():
            for index in np.ndindex(values.shape):
                original = values[index]
                values[index] = original + 1e-6
                plus = objective()
                values[index] = original - 1e-6
                minus = objective()
                values[index] = original
                self.assertAlmostEqual(grads[key][index], (plus-minus)/2e-6, places=7)

    def test_reward_prefers_safe_hold_over_collision_near_full_open(self):
        plan = GraspPlan((.8, -.11, .85))
        row = dict(door_deg=89.7, unexpected_contact="", slip_detected=False,
                   contact_lost=False, time_s=15.5)
        success = dict(outcome="D_full_open", stop_reason="full_open_and_hold",
                       grasp_acquired=True, final_angle_deg=89.7)
        crash = dict(success, outcome="C_grasped_but_failed",
                     stop_reason="unexpected_collision", unexpected_contact_samples=1)
        good, _ = reward(success, [row], plan)
        bad, terms = reward(crash, [dict(row, unexpected_contact="arm:finger")], plan)
        self.assertGreater(good, bad + 10)
        self.assertEqual(terms["safe_progress"], 0)
        unreachable, _ = reward(dict(outcome="A_initial_unreachable",
            stop_reason="initial_grasp_or_pregrasp_unreachable"), [], plan)
        self.assertTrue(np.isfinite(unreachable))
        self.assertLess(unreachable, 0)

    def test_checkpoint_roundtrip_and_normalized_policy(self):
        policy = Policy(3, np.random.default_rng(1))
        policy.params["b2"][:] = [-1., .1, 1.]
        config = dict(schema_version=1, angles_deg=[-2., 0., 2.])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"policy.npz"
            save_checkpoint(path, policy, config)
            restored, restored_config = load_checkpoint(path)
        self.assertEqual(config, restored_config)
        x = np.array([.2, 1., 1., 0.])
        np.testing.assert_array_equal(policy.forward(x)[1], restored.forward(x)[1])
        self.assertAlmostEqual(restored.forward(x)[1].sum(), 1.)

    def test_on_policy_learning_increases_reward(self):
        # Independent known bandit verifies update direction, without physics cost.
        rng = np.random.default_rng(3)
        policy = Policy(3, rng)
        adam = Adam(policy.params, .02)
        x = np.array([.2, .9, 1., 0.])
        returns = np.array([-1., 0., 1.])
        baseline = 0.
        for _ in range(60):
            _, p = policy.forward(x)
            actions = rng.choice(3, size=16, p=p)
            values = returns[actions]
            adam.step(policy.params, policy.gradient(np.tile(x, (16, 1)), actions,
                       values-baseline, .02))
            baseline = .8*baseline + .2*values.mean()
        self.assertGreater(policy.forward(x)[1][2], .9)


if __name__ == "__main__":
    unittest.main()
