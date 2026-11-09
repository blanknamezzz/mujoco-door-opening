"""Outcome identification, independent gate freezing and sweep geometry checks."""
from copy import deepcopy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import numpy as np
import hjepa_runtime
import torch

from mobile_outcome_v4 import (SUCCESS,OTHER,COLLISION,REFUSAL,record_outcome,
    outcome_values,outcome_loss,configure_outcomes,load_outcome_checkpoint,
    continuation_class,station_values)
from mobile_sweep_v4 import DoorSweepChecker


class OutcomeContracts(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)

    def test_scalar_mean_cannot_identify_collision_composition(self):
        # Both distributions have expected utility -1. Rich labels distinguish
        # a mostly ordinary failure from a larger collision probability.
        logits=torch.tensor([[.01,.97,.02],[.1,.7,.2]]).log()
        values=outcome_values(logits)
        torch.testing.assert_close(values,torch.tensor([-1.,-1.]))
        labels=torch.tensor([OTHER,OTHER])
        self.assertLess(float(outcome_loss(logits,labels,'scalar')),1e-10)
        a=outcome_loss(logits[:1],labels[:1],'factorized')
        b=outcome_loss(logits[1:],labels[1:],'factorized')
        self.assertGreater(float(b),float(a))

    def test_risk_changes_only_the_declared_collision_value(self):
        logits=torch.tensor([[20.,-20.,-20.],[-20.,20.,-20.],[-20.,-20.,20.]])
        torch.testing.assert_close(outcome_values(logits),torch.tensor([1.,-1.,-2.]))
        torch.testing.assert_close(outcome_values(logits,3.),torch.tensor([1.,-1.,-4.]))
        self.assertEqual(record_outcome(dict(success=False,collision=True)),COLLISION)
        with self.assertRaises(ValueError):record_outcome(dict(success=True,collision=True))

    def test_continuation_label_follows_frozen_choice_not_best_outcome(self):
        records=[dict(success=False,collision=False) for _ in range(8)]
        records[1]['success']=True
        scan=dict(unsafe_abort=False,collision=False)
        records[0]['inspection']=scan
        scene=dict(branches={1:dict(records=records)})
        p=torch.tensor([[10.,-10.,-10.],[-10.,10.,-10.],[-10.,10.,-10.],[-10.,10.,-10.]])
        g=torch.tensor([[[10.,-10.,-10.],[-10.,10.,-10.]]]*4)
        self.assertEqual(continuation_class(scene,p,g),OTHER)
        self.assertEqual(continuation_class(scene,p,g,mask=[False]*4),REFUSAL)
        scan.update(unsafe_abort=True,collision=True)
        self.assertEqual(continuation_class(scene,p,g),COLLISION)

    def test_mask_retains_known_refusal_and_does_not_relabel_geometry(self):
        logits=torch.tensor([[9.,0.,0.]]*4)
        value=station_values(logits,mask=[False]*4)
        self.assertEqual(int(value.argmax()),4)
        self.assertEqual(float(value[-1]),0.)
        self.assertTrue((value[:4]<-1000).all())

    def test_gate_update_keeps_selected_continuation_exact(self):
        from hjepa_model import DoorHJEPA,compact_config
        model=DoorHJEPA(compact_config());configure_outcomes(model)
        model.requires_grad_(False);model.inspection_policy.requires_grad_(True)
        fixed={n:v.clone() for n,v in model.state_dict().items() if not n.startswith('inspection_policy.')}
        optimizer=torch.optim.AdamW(model.inspection_policy.parameters(),lr=.001)
        x=torch.randn(2,model.inspection_policy.net[0].in_features)
        raw=model.inspection_policy(x)
        loss=outcome_loss(raw[:,:4],torch.tensor([REFUSAL,OTHER]),'factorized')
        loss.backward();optimizer.step()
        self.assertTrue(all(torch.equal(v,model.state_dict()[n]) for n,v in fixed.items()))
        self.assertTrue(any(p.grad is not None for p in model.inspection_policy.parameters()))
        self.assertTrue(all(p.grad is None for p in model.position_policy.parameters()))

    def test_outcome_checkpoint_roundtrip_keeps_all_head_shapes(self):
        from hjepa_model import DoorHJEPA,compact_config,save_checkpoint
        model=DoorHJEPA(compact_config());configure_outcomes(model)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt'
            save_checkpoint(path,model,torch.zeros(19),torch.ones(19),seed=42)
            restored,source=load_outcome_checkpoint(path)
        self.assertEqual(restored.position_policy.net[-1].out_features,3)
        self.assertEqual(restored.grasp_policy.net[-1].out_features,3)
        self.assertEqual(restored.inspection_policy.net[-1].out_features,5)
        self.assertTrue(all(torch.equal(v,restored.state_dict()[n]) for n,v in model.state_dict().items()))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA normalization handoff requires CUDA')
    def test_gpu_reload_moves_state_normalization_with_model(self):
        from hjepa_model import DoorHJEPA,compact_config,save_checkpoint
        from train_hjepa import state_tensor
        model=DoorHJEPA(compact_config());configure_outcomes(model)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt'
            save_checkpoint(path,model,torch.zeros(19),torch.ones(19),seed=42)
            restored,source=load_outcome_checkpoint(path,'cuda')
        self.assertEqual(next(restored.parameters()).device.type,'cuda')
        self.assertEqual(source['state_mean'].device.type,'cuda')
        self.assertEqual(source['state_std'].device.type,'cuda')
        state=state_tensor(np.zeros((1,19),np.float32),source['state_mean'],source['state_std'],'cuda')
        torch.testing.assert_close(state,torch.zeros((1,19),device='cuda'))


class SweepContracts(unittest.TestCase):
    def test_guard_predicts_obstruction_and_does_not_mutate_physics(self):
        from door_stress_scenes import scenario_by_id
        from mobile_crossed_v2 import MotionDoor
        scene=scenario_by_id('oak-lever-wall');exp=MotionDoor(scene)
        q,v=exp.d.qpos.copy(),exp.d.qvel.copy();when=exp.d.time
        mask,queries=DoorSweepChecker(scene).mask()
        self.assertFalse(mask[0]);self.assertTrue(mask[2])
        self.assertIn('wall',queries[0]['pair'])
        np.testing.assert_array_equal(q,exp.d.qpos);np.testing.assert_array_equal(v,exp.d.qvel)
        self.assertEqual(when,exp.d.time)

    def test_free_scene_retains_far_station_but_rejects_base_sweep_collision(self):
        from door_stress_scenes import scenario_by_id
        scene=replace(scenario_by_id('oak-lever-wall'),obstacles=())
        mask,queries=DoorSweepChecker(scene).mask()
        self.assertTrue(mask[0]);self.assertTrue(mask[2]);self.assertFalse(mask[1])
        self.assertIn('base_box',queries[1]['pair'])
        with self.assertRaises(ValueError):DoorSweepChecker(scene,step_deg=0.)


if __name__=='__main__':unittest.main()
