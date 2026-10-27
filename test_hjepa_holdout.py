"""Test-only data stays out of fitting; replay compares meaningful actions."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from evaluate_structural_training import assert_replay_matches
from prepare_hjepa_splits import assert_scene_disjoint
from structural_scene import diverse_structural_scenes, independent_test_scenes
from train_hjepa import read_data


class HoldoutContracts(unittest.TestCase):
    def test_training_loader_never_opens_test_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            for split in ('train', 'val'):
                np.savez(path/(split+'.npz'), value=np.ones(1))
            manifest = dict(scenes=[dict(scene_id=s, split=s) for s in ('train', 'val', 'test')],
                episodes=[dict(scene_id=s, split=s, file=s+'.npz') for s in ('train', 'val', 'test')])
            (path/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            _, records = read_data(path, required_splits=('train', 'val'), include_splits=('train', 'val'))
            self.assertEqual([ep['split'] for ep, _ in records], ['train', 'val'])
            self.assertFalse((path/'test.npz').exists())
            manifest['evaluation_only'] = True
            (path/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Evaluation-only'):
                read_data(path, required_splits=('train', 'val'), include_splits=('train', 'val'))

    def test_independent_test_is_new_geometry_and_all_test(self):
        training = diverse_structural_scenes()
        heldout = independent_test_scenes()
        self.assertEqual(len(heldout), 48)
        self.assertEqual(heldout, independent_test_scenes())
        self.assertEqual({s['split'] for s in heldout}, {'test'})
        assert_scene_disjoint(training, heldout)
        renamed = dict(training[0], scene_id='different_name', split='test')
        with self.assertRaisesRegex(ValueError, 'geometry overlap'):
            assert_scene_disjoint(training, [renamed])

    def test_perception_failure_has_no_grasp_but_still_requires_matching_base_and_outcome(self):
        cached = dict(abstained=False, base_id=2, grasp_id=0, success=False,
                      result=dict(outcome='B_perception_failed'))
        plan = dict(base_id=2, grasp_id=None, result=dict(outcome='B_perception_failed'), metrics=dict(success=False))
        assert_replay_matches(plan, cached)
        wrong_base = dict(plan, base_id=1)
        with self.assertRaisesRegex(AssertionError, 'base differs'):
            assert_replay_matches(wrong_base, cached)
        wrong_result = deepcopy(cached)
        wrong_result['result']['outcome'] = 'A_initial_unreachable'
        with self.assertRaisesRegex(AssertionError, 'perception failure differs'):
            assert_replay_matches(plan, wrong_result)

    def test_actual_grasp_mismatch_is_not_suppressed(self):
        cached = dict(abstained=False, base_id=2, grasp_id=1, success=True, result=dict(outcome='D_full_open'))
        plan = dict(base_id=2, grasp_id=0, result=dict(outcome='D_full_open'), metrics=dict(success=True))
        with self.assertRaisesRegex(AssertionError, 'grasp differs'):
            assert_replay_matches(plan, cached)


if __name__ == '__main__':
    unittest.main()
