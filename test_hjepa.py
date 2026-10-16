"""Contract tests: physical execution, no label leakage and official hierarchy."""
import math
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from hjepa_scene import action_at, lookahead_steps, proprio, scene_set, task_reward
from simulate import Experiment


class SceneContracts(unittest.TestCase):
    def test_base_yaw_matches_forward_kinematics(self):
        target = np.array([.8, -.11, .85])
        for yaw in [0., .7, math.pi/2, 2.1]:
            exp = Experiment(base_yaw=yaw)
            exp.reset(target, math.pi/2)
            np.testing.assert_allclose(exp.d.site_xpos[exp.site], target, atol=1e-8)

    def test_scene_split_is_deterministic_and_disjoint(self):
        scenes = scene_set(12, 42)
        self.assertEqual(scenes, scene_set(12, 42))
        self.assertEqual({s['split'] for s in scenes}, {'train', 'val', 'test'})
        self.assertEqual(len({s['scene_id'] for s in scenes}), 12)

    def test_reward_uses_final_shortfall_and_only_safe_prefix(self):
        def row(t, angle, collision=''):
            return dict(time_s=t, door_deg=angle, unexpected_contact=collision,
                        slip_detected=False, contact_lost=False)
        result = dict(outcome='C_grasped_but_failed', grasp_acquired=True, final_angle_deg=20.)
        rows = [row(2.5, 0), row(5, 45), row(8, 80, 'arm:door'), row(9, 89)]
        reward = task_reward(result, rows)
        self.assertAlmostEqual(reward['safe_opening_fraction'], .5)
        self.assertAlmostEqual(reward['shortfall'], 70/90)
        self.assertTrue(reward['collision'])
        self.assertEqual(reward['milestones'], [True, True, True])
        changed = task_reward(result, rows, incomplete_weight=0.)
        self.assertAlmostEqual(changed['reward']-reward['reward'], 3*70/90)

    def test_unreachable_is_a_negative_policy_example(self):
        exp = Experiment(base=(2., -1.2))
        calls = []
        result, rows, _ = exp.trial(np.array([.8, -.11, .85]), observer=lambda *x: calls.append(x))
        self.assertEqual(result['outcome'], 'A_initial_unreachable')
        self.assertLess(task_reward(result, rows)['reward'], 0)
        self.assertEqual(calls, [])

    def test_action_contract(self):
        point = np.array([.8, -.11, .85])
        self.assertEqual(action_at(5., point, 0.).shape, (6,))
        self.assertTrue(np.isfinite(action_at(5., point, .1)).all())
        self.assertEqual(len(proprio(Experiment())), 19)
        self.assertEqual([lookahead_steps(x) for x in [0., .25, .5, 1.]], [0, 14, 17, 31])


class OfficialModelContracts(unittest.TestCase):
    def test_full_profile_matches_upstream_factory_except_door_action_dimension(self):
        from copy import deepcopy
        from hjepa_model import DoorHJEPA, default_config
        from hjepa_utils import create_world_model
        from omegaconf import OmegaConf
        torch.set_num_threads(2)
        config = default_config()
        model = DoorHJEPA(config)
        native = deepcopy(config)
        native['level1']['wm']['action_dim'] = 2
        reference, _ = create_world_model(OmegaConf.create(native))
        actual_shapes = {k: tuple(v.shape) for k, v in model.model.state_dict().items()}
        native_shapes = {k: tuple(v.shape) for k, v in reference.state_dict().items()}
        self.assertEqual(set(actual_shapes), set(native_shapes))
        differences = [k for k in actual_shapes if actual_shapes[k] != native_shapes[k]]
        self.assertEqual(differences, ['jepas.0.action_embed.patch_embed.weight'])
        self.assertEqual(actual_shapes[differences[0]], (10, 30, 1))
        self.assertEqual(native_shapes[differences[0]], (10, 10, 1))
        self.assertEqual(sum(p.numel() for p in model.model.parameters()), 23222914)
        self.assertEqual(model.model.get_level(1).encoder.encoder.config.num_hidden_layers, 12)
        self.assertEqual(model.sigreg_level1.num_proj, 1024)
        self.assertEqual(config['loader']['batch_size'], 128)
        self.assertEqual((config['frame_skip'], config['clip_frames'], config['raw_clip_frames']), (5, 7, 35))
        with self.assertRaises(ValueError):
            model.normalize_pixels(torch.zeros(1, 3, 64, 64))

    def test_recomputation_preserves_official_vit_outputs_and_gradients(self):
        from hjepa_model import DoorHJEPA
        torch.set_num_threads(2)
        torch.manual_seed(7)
        model = DoorHJEPA().train()
        vit = model.model.get_level(1).encoder.encoder
        pixels = torch.randn(2, 3, 224, 224)
        weight = vit.embeddings.patch_embeddings.projection.weight
        original = vit(pixels).last_hidden_state[:, :1]
        gradient = torch.autograd.grad(original.square().mean(), weight)[0]
        model.enable_activation_checkpointing(chunk_size=1)
        recomputed = vit(pixels).last_hidden_state
        recomputed_gradient = torch.autograd.grad(recomputed.square().mean(), weight)[0]
        torch.testing.assert_close(recomputed, original, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(recomputed_gradient, gradient, rtol=1e-4, atol=1e-6)

    def test_arrival_perception_failure_is_reported_without_a_grasp(self):
        from unittest.mock import patch
        from hjepa_model import DoorHJEPA, compact_config
        from hjepa_scene import DoorScene
        from run_hjepa import select_and_execute
        torch.set_num_threads(2)
        model = DoorHJEPA(compact_config()).eval()
        checkpoint = dict(state_mean=torch.zeros(19), state_std=torch.ones(19),
                          upstream_commit='test')
        with patch('run_hjepa.observe', return_value={'rgb': np.zeros((64, 64, 3), np.uint8)}), \
             patch('run_hjepa.estimate_frame', side_effect=ValueError('occluded')):
            plan, rows, states = select_and_execute(model, checkpoint, DoorScene('test'), base_mode='fixed')
        self.assertEqual(plan['result']['outcome'], 'B_perception_failed')
        self.assertIsNone(plan['grasp_id'])
        self.assertLess(plan['metrics']['reward'], 0.)
        self.assertEqual(rows, [])
        self.assertEqual(states, [])

    def test_joint_gradients_hierarchy_and_checkpoint(self):
        from hjepa_model import DoorHJEPA, compact_config, load_checkpoint, save_checkpoint
        torch.set_num_threads(2)
        torch.manual_seed(2)
        model = DoorHJEPA(compact_config())
        self.assertEqual(model.model.__class__.__module__, 'models.hjepa')
        batch = dict(pixels_level1=torch.rand(2, 9, 3, 64, 64),
                     proprio_level1=torch.randn(2, 9, 19), action_level1=torch.randn(2, 9, 6))
        out = model.predictive_loss(batch)
        self.assertEqual(out['embed_1'].shape, (2, 3, 96))
        self.assertEqual(out['embed_2'].shape, (2, 3, 64))
        self.assertTrue(torch.isfinite(out['loss']))
        # Upper-level objective must train the lower encoder, as in H-JEPA.
        out['loss_level2'].backward()
        pixel = model.model.get_level(1).encoder.pixel_encoder.encoder
        self.assertGreater(sum(p.grad.abs().sum().item() for p in pixel.parameters() if p.grad is not None), 0.)
        with torch.no_grad():
            z = torch.randn(2, 96)
            a = torch.randn(2, 8, 6)
            self.assertEqual(model.imagine(z, a, 1).shape, (2, 96))
            self.assertEqual(model.imagine(z, a, 2).shape, (2, 64))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model.pt'
            save_checkpoint(path, model, torch.zeros(19), torch.ones(19), step=1)
            restored, metadata = load_checkpoint(path)
            self.assertEqual(metadata['step'], 1)
            self.assertEqual(len(restored.state_dict()), len(model.state_dict()))


if __name__ == '__main__':
    unittest.main()
