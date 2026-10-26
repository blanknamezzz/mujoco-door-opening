"""Numerical and data-isolation checks for the SL/RL comparison."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from collect_grasp_data import make_scenes, validate_records
from compare_grasp_learning import (RewardModel, encode, fit_supervised, load_model,
                                    normalizer, save_model, training_arrays)


class LearningComparisonTests(unittest.TestCase):
    def manifest(self):
        return dict(scenes=make_scenes([4, 2, 2], 5), angles_deg=[-2., 0., 2.],
                    physics=dict(plan=dict(handle_center_world_m=[.8, -.11, .85])))

    def test_splits_unique_reproducible(self):
        a = make_scenes([36, 12, 24], 20261026)
        self.assertEqual(a, make_scenes([36, 12, 24], 20261026))
        self.assertEqual(len({s['scene_id'] for s in a}), 72)
        self.assertEqual(len({tuple(s['base_xy_m']+[s['base_yaw_deg']]) for s in a}), 72)

    def test_normalizer_ignores_heldout_contexts(self):
        m = self.manifest()
        before = normalizer(m)
        for s in m['scenes']:
            if s['split'] != 'train':
                s['base_xy_m'] = [1000., -1000.]
        self.assertEqual(before, normalizer(m))

    def test_training_rejects_heldout_records(self):
        m = self.manifest()
        bad = [dict(scene_id='test_000', split='test', action_index=0, reward=10.)]
        with self.assertRaises(ValueError):
            training_arrays(m, bad, normalizer(m))

    def test_missing_and_duplicate_data_rejected(self):
        m = self.manifest()
        with self.assertRaises(ValueError):
            validate_records(m, [], 'train')
        with self.assertRaises(ValueError):
            validate_records(m, [dict(scene_id='train_000', action_index=0)]*12, 'train')

    def test_reward_gradient_finite_difference(self):
        rng = np.random.default_rng(2)
        model = RewardModel(3, rng)
        model.params['w2'][:] = rng.normal(0, .2, (16, 3))
        x, actions, targets = rng.normal(size=(5, 4)), np.array([0, 1, 2, 1, 0]), rng.normal(size=5)
        gradients, _ = model.gradient(x, actions, targets)
        def objective():
            predictions = model.forward(x)[1][np.arange(5), actions]
            return -np.mean((predictions-targets)**2)
        for key, values in model.params.items():
            for index in np.ndindex(values.shape):
                old = values[index]
                values[index] = old+1e-6
                plus = objective()
                values[index] = old-1e-6
                minus = objective()
                values[index] = old
                self.assertAlmostEqual(gradients[key][index], (plus-minus)/2e-6, places=7)

    def test_supervised_can_fit_known_rewards_and_reload(self):
        m = self.manifest()
        records = [dict(scene_id=s['scene_id'], split='train', action_index=a, reward=float(a*5))
                   for s in m['scenes'] if s['split'] == 'train' for a in range(3)]
        norm = normalizer(m)
        model, history = fit_supervised(m, records, norm, 3, 300, .01)
        self.assertLess(history[-1]['train_reward_mse'], .01)
        metadata = dict(method='supervised', angles_deg=m['angles_deg'], normalizer=norm)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.npz'
            save_model(path, model, metadata)
            restored, meta = load_model(path)
        x = encode(m, m['scenes'][0], norm)
        np.testing.assert_array_equal(model.forward(x)[1], restored.forward(x)[1])
        self.assertEqual(meta, metadata)


if __name__ == '__main__':
    unittest.main()
