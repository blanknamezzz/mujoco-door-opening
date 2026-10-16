"""Training split and refusal contracts; no hidden direction supervision."""
import json
import unittest
from unittest.mock import patch
import numpy as np
import hjepa_runtime
import torch
from structural_scene import structural_training_scenes, structural_scene_from_spec, STRUCTURAL_BASES
from hjepa_scene import task_reward


class StructuralTrainingContracts(unittest.TestCase):
    def test_folding_feature_statistics_preserves_head_predictions(self):
        from refit_structural_position import fold_standardization
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.LayerNorm(8),
                                               torch.nn.GELU(), torch.nn.Linear(8, 1))
            def forward(self, x):
                return self.net(x)
        head = Head().eval()
        x = torch.randn(7, 6)
        mean, std = torch.tensor([1.2, -.3]), torch.tensor([.2, 1.7])
        normalized = x.clone()
        normalized[:, :2] = (normalized[:, :2]-mean)/std
        folded = fold_standardization(head, mean, std)
        torch.testing.assert_close(folded(x), head(normalized), atol=1e-6, rtol=1e-6)

    def test_position_backbone_probe_does_not_replace_world_latents(self):
        from hjepa_model import DoorHJEPA
        torch.set_num_threads(2)
        model = DoorHJEPA().eval()
        pixels, state = torch.rand(1, 3, 224, 224), torch.zeros(1, 19)
        with torch.no_grad():
            world = model.encode(pixels, state)
            raw = model.model.get_level(1).encoder.encoder(model.normalize_pixels(pixels)).last_hidden_state[:, 0]
            model.config['position_encoder'] = 'backbone_cls'
            torch.testing.assert_close(model.encode_position(pixels, state), raw)
            torch.testing.assert_close(model.encode(pixels, state), world)

    def test_scene_specs_roundtrip_and_balanced_disjoint_splits(self):
        specs = structural_training_scenes()
        self.assertEqual(specs, structural_training_scenes())
        self.assertEqual(len(specs), 24)
        self.assertEqual({s['split']: sum(t['split'] == s['split'] for t in specs) for s in specs},
                         {'train': 12, 'val': 6, 'test': 6})
        self.assertEqual(len({s['scene_id'] for s in specs}), 24)
        for spec in json.loads(json.dumps(specs)):
            scene = structural_scene_from_spec(spec)
            self.assertEqual(scene.scene_id, spec['scene_id'])
            self.assertNotIn(spec['family'], ['one-way-positive', 'one-way-negative'])
            if len(scene.handle_faces) == 2:
                self.assertEqual((scene.hinge_min_deg, scene.hinge_max_deg), (-90., 90.))

    def test_refusal_returns_without_moving_or_observing_an_arrival(self):
        from hjepa_model import DoorHJEPA, compact_config
        from run_hjepa import select_and_execute
        class Reject(torch.nn.Module):
            def forward(self, inputs):
                return torch.arange(len(inputs), device=inputs.device, dtype=torch.float32)[:, None]
        torch.set_num_threads(2)
        model = DoorHJEPA(compact_config()).eval()
        model.position_policy = Reject()
        ckpt = dict(state_mean=torch.zeros(19), state_std=torch.ones(19), upstream_commit='test',
                    positioning_abstention=True, base_budget=5, grasp_budget=4)
        scene = structural_scene_from_spec(structural_training_scenes()[0])
        with patch('run_hjepa.observe', return_value={'rgb': np.zeros((64, 64, 3), np.uint8)}) as camera, \
             patch('simulate.Experiment.trial') as execute:
            plan, rows, states = select_and_execute(model, ckpt, scene)
        self.assertEqual(camera.call_count, 1)
        execute.assert_not_called()
        self.assertIsNone(plan['base_id'])
        self.assertEqual(plan['result']['outcome'], 'E_policy_abstained')
        self.assertEqual(rows, [])
        self.assertEqual(states, [])
        self.assertEqual(task_reward(plan['result'], [])['milestones'], [False]*3)
        self.assertEqual(plan['metrics']['reward'], 0.)

    def test_alternating_updates_cache_frozen_latents_and_skip_refusal_arrival(self):
        from train_hjepa import alternate_policy_updates
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(19, 2)
                self.position_policy = torch.nn.Linear(6, 1)
                self.grasp_policy = torch.nn.Linear(7, 1)
                self.calls = 0
            def encode(self, pixels, states):
                assert torch.isfinite(pixels).all(), 'Encoded fictitious refusal arrival'
                self.calls += 1
                return self.encoder(states)
        def records(split):
            result = []
            for gid, reward in enumerate([1., -1.]):
                x = dict(initial_rgb=np.zeros((4, 4, 3), np.float32), initial_proprio=np.zeros(19, np.float32),
                         arrival_rgb=np.zeros((4, 4, 3), np.float32), arrival_proprio=np.zeros(19, np.float32),
                         base=np.ones(4, np.float32), grasp=np.full(5, gid, np.float32))
                result.append((dict(scene_id=split, base_id=0, split=split, reward=reward), x))
            x = dict(x, base=np.zeros(4, np.float32), arrival_rgb=np.full((4, 4, 3), np.nan, np.float32))
            result.append((dict(scene_id=split, base_id=4, split=split, reward=0., positioning_only=True), x))
            return result
        model = Tiny()
        original = {k: v.clone() for k, v in model.encoder.state_dict().items()}
        selected = alternate_policy_updates(model, records('train'), torch.zeros(19), torch.ones(19),
                                            'cpu', 4, validation=records('val'), select_best=True)
        self.assertEqual(model.calls, 4)  # initial + one physical arrival per split
        self.assertEqual(selected['updates_executed'], 4)
        for key, value in model.encoder.state_dict().items():
            torch.testing.assert_close(value, original[key])

    def test_heldout_selector_does_not_rank_using_future_rewards(self):
        from evaluate_structural_training import choose
        class Feature(torch.nn.Module):
            def forward(self, x):
                return x[:, 2:3]
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))
                self.position_policy = Feature()
                self.grasp_policy = Feature()
            def encode(self, pixels, states):
                return torch.zeros(len(pixels), 2)
        x = dict(initial_rgb=np.zeros((4, 4, 3), np.uint8), initial_proprio=np.zeros(19, np.float32),
                 arrival_rgb=np.zeros((4, 4, 3), np.uint8), arrival_proprio=np.zeros(19, np.float32),
                 base=np.ones(4, np.float32), grasp=np.ones(5, np.float32))
        table = {0: [(dict(base_id=0, grasp_id=0, reward=-10000., current_reachable=True), x),
                     (dict(base_id=0, grasp_id=1, reward=10000., current_reachable=True),
                      dict(x, grasp=np.zeros(5, np.float32)))]}
        ckpt = dict(state_mean=torch.zeros(19), state_std=torch.ones(19))
        selected, _ = choose(Model(), ckpt, table)
        self.assertEqual(selected['grasp_id'], 0)
        table[0][0][0]['reward'], table[0][1][0]['reward'] = 1e9, -1e9
        selected, _ = choose(Model(), ckpt, table)
        self.assertEqual(selected['grasp_id'], 0)


if __name__ == '__main__':
    unittest.main()
