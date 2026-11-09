"""Capacity, freezing and checkpoint contracts for depth-conditioned heads."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import hjepa_runtime
import torch
from mobile_outcome_v4 import configure_outcomes
from mobile_partial_policy_v5 import (configure_partial_policy,load_partial_checkpoint,
    current_map_features,partial_gate_features)


class PartialPolicyContracts(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)

    def model(self,use_map):
        from hjepa_model import DoorHJEPA,compact_config
        model=DoorHJEPA(compact_config());configure_outcomes(model);configure_partial_policy(model,use_map)
        return model

    def test_new_slots_preserve_the_initial_existing_head(self):
        from hjepa_model import DoorHJEPA,compact_config
        model=DoorHJEPA(compact_config());configure_outcomes(model)
        old_weight=model.position_policy.net[0].weight.detach().clone()
        x=torch.randn(3,model.position_policy.net[0].in_features);before=model.position_policy(x)
        configure_partial_policy(model,True)
        after=model.position_policy(torch.cat([x,torch.randn(3,4)],-1))
        torch.testing.assert_close(old_weight,model.position_policy.net[0].weight[:,:-4],rtol=0,atol=0)
        self.assertEqual(float(model.position_policy.net[0].weight[:,-4:].abs().sum()),0.)
        # Changing GEMM input width can change float32 reduction roundoff.
        torch.testing.assert_close(before,after,rtol=1e-6,atol=1e-7)

    def test_image_control_zeros_map_features_with_matching_capacity(self):
        a=self.model(True);b=self.model(False)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        values=np.ones((4,4),np.float32)
        self.assertEqual(float(current_map_features(b,values,'cpu').abs().sum()),0.)
        self.assertEqual(float(current_map_features(a,values,'cpu').sum()),16.)

    def test_checkpoint_roundtrip_and_initial_map_only_gate(self):
        from hjepa_model import save_checkpoint
        model=self.model(True);state=np.zeros(19,np.float32);state[17]=1.
        width=model.inspection_policy.net[0].in_features-22
        feature=partial_gate_features(model,torch.zeros(1,width),state,np.ones((4,4)),1.)
        self.assertEqual(feature.shape[-1],model.inspection_policy.net[0].in_features)
        torch.testing.assert_close(feature[0,-16:],torch.ones(16))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.pt';save_checkpoint(path,model,torch.zeros(19),torch.ones(19),seed=42)
            loaded,source=load_partial_checkpoint(path)
        self.assertTrue(all(torch.equal(v,loaded.state_dict()[n]) for n,v in model.state_dict().items()))
        self.assertEqual(source['seed'],42)


if __name__=='__main__':unittest.main()
