"""Outcome credit assignment, abstention, native alignment and serialization."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
import hjepa_runtime
import torch
from decomposed_reward import (DecomposedPolicy, configure_decomposition, outcome_target,
                               preference_loss, terminal_utility)
from hjepa_lora import adapter_state, install_lora, frozen_core_digest
from hjepa_model import DoorHJEPA, MLP, compact_config, load_checkpoint, save_checkpoint


class DecomposedContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(14)

    def test_terminal_outcome_dominates_partial_milestones(self):
        self.assertGreater(terminal_utility(True,False),terminal_utility(False,False,True))
        self.assertGreater(terminal_utility(False,False,True),terminal_utility(False,False))
        self.assertGreater(terminal_utility(False,False),terminal_utility(False,True))
        episode=dict(success=False,collision=False,current_reachable=False,grasp_observed=False,
                     milestones=[True,False,False],safe_opening_fraction=0.,shortfall=1.)
        self.assertEqual(outcome_target(episode)[2],0.)
        with self.assertRaisesRegex(ValueError,'invented'):
            outcome_target(dict(episode,positioning_only=True))

    def test_zero_blend_preserves_original_policy_and_refusal_is_known_zero(self):
        scalar=MLP(10,128,1).eval()
        x=torch.randn(5,10)
        head=DecomposedPolicy(scalar,4,True,blend=0).eval()
        torch.testing.assert_close(head(x),scalar(x),atol=0,rtol=0)
        head.blend=1.
        physical=x.clone()
        physical[:,-4:]=1.
        before=head(physical)
        head.score_offset=.4
        torch.testing.assert_close(head(physical),before+.4)
        x[:,-4:]=0.
        torch.testing.assert_close(head(x),torch.zeros(5,1),atol=0,rtol=0)
        head.score_offset=0.
        x[:,-4:]=1.
        head.net[-1].bias.data[2:]=20.  # large milestone/progress predictions are not terminal bonuses
        self.assertTrue((head(x)<0).all())

    def test_preferences_push_success_above_refusal_above_failure(self):
        scores=torch.tensor([-.5,0.,.5],requires_grad=True)
        utilities=torch.tensor([1.,0.,-1.])
        loss=preference_loss(scores,utilities)
        loss.backward()
        self.assertLess(scores.grad[0].item(),0)
        self.assertGreater(scores.grad[2].item(),0)
        tied=preference_loss(scores,torch.ones(3))
        self.assertEqual(tied.item(),0.)

    def test_new_heads_do_not_change_core_and_adapter_roundtrips_all_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            model=DoorHJEPA(compact_config()).eval()
            base=root/'base.pt'
            save_checkpoint(base,model,torch.zeros(19),torch.ones(19))
            original=frozen_core_digest(model)
            install_lora(model,rank=2)
            configure_decomposition(model,dict(version='outcome_components_v2',blend=1.,position_score_offset=.37))
            self.assertEqual(original,frozen_core_digest(model))
            for n,p in model.named_parameters():
                if 'lora_B' in n: p.data.normal_(std=.001)
            full=root/'full.pt'
            save_checkpoint(full,model,torch.zeros(19),torch.ones(19))
            delta=root/'adapter.pt'
            torch.save(dict(adapter=adapter_state(model),config=model.config,
                state_mean=torch.zeros(19),state_std=torch.ones(19),
                base_checkpoint=dict(path=str(base),sha256=hashlib.sha256(base.read_bytes()).hexdigest())),delta)
            loaded,_=load_checkpoint(full)
            compact,_=load_checkpoint(delta)
            self.assertEqual(compact.position_policy.score_offset,.37)
            for n,v in loaded.state_dict().items():
                torch.testing.assert_close(v,compact.state_dict()[n],atol=0,rtol=0)
            features=torch.randn(3,int(model.config['level1']['wm']['embed_dim'])+4)
            torch.testing.assert_close(loaded.position_policy(features),compact.position_policy(features),atol=0,rtol=0)

    def test_progress_targets_follow_native_temporal_crops(self):
        model=DoorHJEPA().eval()
        config=model.config
        frames=config['clip_frames']
        progress=torch.arange(frames,dtype=torch.float32)[None,:,None].expand(2,-1,-1)
        batch=dict(pixels_level1=torch.zeros(2,frames,3,224,224,dtype=torch.uint8),
                   proprio_level1=torch.zeros(2,frames,19),action_level1=torch.zeros(2,frames,30),
                   door_progress_level1=progress)
        with torch.no_grad(): out=model.predictive_loss(batch,'test')
        self.assertEqual(out['embed_1'].shape[:2],out['door_progress_level1'].shape[:2])
        self.assertEqual(out['embed_2'].shape[:2],out['door_progress_level2'].shape[:2])
        torch.testing.assert_close(out['door_progress_level2'],progress[:,::2])
        self.assertTrue((out['door_progress_level1'][:,1:]-out['door_progress_level1'][:,:-1]==1).all())


if __name__=='__main__': unittest.main()
