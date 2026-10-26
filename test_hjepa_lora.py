"""Adapter zero-init, immutable base tensors and portable loading contracts."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
import hjepa_runtime
import torch
from hjepa_lora import LoRALinear, install_lora, resume_lora, freeze_core_statistics, frozen_core_digest, adapter_state
from hjepa_model import DoorHJEPA, compact_config, save_checkpoint, load_checkpoint
from structural_scene import diverse_structural_scenes, structural_scene_from_spec


class LoRAContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_zero_initialization_and_optimizer_leave_original_weights_exact(self):
        layer = torch.nn.Linear(8, 6)
        original = deepcopy(layer.state_dict())
        adapter = LoRALinear(layer, rank=2)
        x = torch.randn(10, 8)
        torch.testing.assert_close(adapter(x), torch.nn.functional.linear(x, original['weight'], original['bias']),
                                   atol=0, rtol=0)
        optimizer = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=.01)
        adapter(x).square().mean().backward()
        self.assertGreater(adapter.lora_B.grad.abs().sum().item(), 0)
        optimizer.step()
        torch.testing.assert_close(adapter.weight, original['weight'], atol=0, rtol=0)
        torch.testing.assert_close(adapter.bias, original['bias'], atol=0, rtol=0)

    def test_full_checkpoint_and_adapter_only_loading_match_and_check_base_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            base, full, delta = [Path(folder)/n for n in ('base.pt', 'full.pt', 'adapter.pt')]
            model = DoorHJEPA(compact_config()).eval()
            save_checkpoint(base, model, torch.zeros(19), torch.ones(19))
            install_lora(model, rank=2)
            for name, p in model.named_parameters():
                if 'lora_B' in name:
                    p.data.normal_(std=.001)
            save_checkpoint(full, model, torch.zeros(19), torch.ones(19))
            torch.save(dict(adapter=adapter_state(model), config=model.config,
                state_mean=torch.zeros(19), state_std=torch.ones(19),
                base_checkpoint=dict(path=str(base), sha256=hashlib.sha256(base.read_bytes()).hexdigest()),
                metadata=dict(positioning_abstention=True)), delta)
            full_model, _ = load_checkpoint(full)
            delta_model, ckpt = load_checkpoint(delta)
            self.assertTrue(ckpt['positioning_abstention'])
            for key, tensor in full_model.state_dict().items():
                torch.testing.assert_close(tensor, delta_model.state_dict()[key], atol=0, rtol=0)
            base.write_bytes(base.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_checkpoint(delta)

    def test_frozen_core_buffers_do_not_drift_in_training_mode(self):
        model = DoorHJEPA(compact_config())
        # Include a BN buffer to detect otherwise silent base changes.
        model.model.test_bn = torch.nn.BatchNorm1d(4)
        install_lora(model, rank=2)
        before = frozen_core_digest(model)
        model.train()
        freeze_core_statistics(model)
        model.model.test_bn(torch.randn(8, 4))
        self.assertEqual(before, frozen_core_digest(model))
        self.assertTrue(all('lora_' in n for n, p in model.model.named_parameters() if p.requires_grad))

    def test_continuation_reuses_adapter_without_stacking_or_resetting(self):
        model = DoorHJEPA(compact_config())
        install_lora(model, rank=2)
        for name, p in model.named_parameters():
            if 'lora_B' in name:
                p.data.normal_(std=.01)
        before = {n: p.clone() for n, p in model.named_parameters()}
        parameter_ids = {n: id(p) for n, p in model.named_parameters()}
        model.requires_grad_(False)
        result = resume_lora(model, rank=2)
        self.assertTrue(result['continued_existing_adapter'])
        self.assertEqual(parameter_ids, {n: id(p) for n, p in model.named_parameters()})
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, before[name], atol=0, rtol=0)
            if name.startswith('model.'):
                self.assertEqual(p.requires_grad, 'lora_' in name)
        with self.assertRaisesRegex(ValueError, 'saved LoRA'):
            resume_lora(model, rank=3)

    def test_diverse_suite_is_reproducible_balanced_and_physically_constructible(self):
        scenes = diverse_structural_scenes()
        self.assertEqual(scenes, diverse_structural_scenes())
        self.assertEqual(len(scenes), 240)
        self.assertEqual({s: sum(x['split'] == s for x in scenes) for s in ('train', 'val', 'test')},
                         dict(train=192, val=24, test=24))
        self.assertEqual(len({s['scene_id'] for s in scenes}), 240)
        for family in {s['family'] for s in scenes}:
            spec = next(s for s in scenes if s['family'] == family)
            scene = structural_scene_from_spec(spec)
            for base in (scene.base_candidates[0], scene.base_candidates[2]):
                exp = scene.experiment(base)
                self.assertTrue(torch.isfinite(torch.tensor(exp.d.cam_xpos)).all())


if __name__ == '__main__':
    unittest.main()
