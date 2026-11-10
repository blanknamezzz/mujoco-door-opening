"""Contracts for real selected-action supervision and decision regret."""
import unittest
import numpy as np
import hjepa_runtime
import torch
from mobile_inspection_regret_v6 import paired_teacher, inspection_loss


def synthetic_scene():
    scene = dict(spec=dict(scene_id='synthetic', group_id='g', layout='free'),
                 maps=dict(initial=dict(mask=[True]*4), fused=dict(mask=[True]*4)),
                 union_bases=np.ones(4, bool), union_sides=np.ones(2, bool), branches={})
    for branch in (0, 1):
        scan = dict(unsafe_abort=False, collision=False, success=bool(branch),
                    path_length_m=3. if branch else 0., duration_s=30. if branch else 0.)
        records=[]
        for base in range(4):
            for grasp in range(2):
                records.append(dict(base_id=base,grasp_id=grasp,inspection=scan,
                    success=bool(grasp==1),collision=False,continuation_collision=False,
                    nav_success=True,measured_path_length_m=1.,episode_duration_s=4.))
        scene['branches'][branch]=dict(records=records,feasible_bases=np.ones(4,bool))
    p=torch.tensor([[4.,-2.,-3.],[-3.,4.,-3.],[-3.,4.,-3.],[-3.,4.,-3.]])
    g0=torch.tensor([[[4.,-2.,-3.],[2.,-2.,-3.]]]*4)
    g1=g0.flip(1)
    return scene,dict(p0=p,p1=p,g0=g0,g1=g1)


class InspectionRegretContracts(unittest.TestCase):
    def test_teacher_uses_actual_frozen_choice_instead_of_best_candidate(self):
        scene,pred=synthetic_scene();target=paired_teacher(scene,pred)
        self.assertEqual(target['direct_choice'],[0,0])
        self.assertEqual(target['inspect_choice'],[0,1])
        self.assertEqual(target['direct_actual_utility'],-1.)
        self.assertEqual(target['inspect_actual_utility'],1.)
        self.assertAlmostEqual(target['cost'],.15)
        self.assertAlmostEqual(target['advantage'],1.85)
        # A successful unchosen direct grasp must not change the direct label.
        self.assertTrue(scene['branches'][0]['records'][1]['success'])

    def test_collision_risk_changes_actual_counterfactual_label(self):
        scene,pred=synthetic_scene()
        record=scene['branches'][0]['records'][0]
        record.update(collision=True,continuation_collision=True)
        self.assertAlmostEqual(paired_teacher(scene,pred,3.)['advantage']-
                               paired_teacher(scene,pred,1.)['advantage'],2.)

    def decision_gradient(self,advantage,eligible=True):
        raw=torch.zeros(1,5,requires_grad=True)
        costs=torch.nn.functional.softplus(raw[:,4]).detach()
        result=inspection_loss(raw,torch.tensor([0]),costs,torch.tensor([1.]),
            torch.tensor([.5]),torch.tensor([advantage]),torch.tensor([eligible]),'decision_regret')
        value_grad=torch.autograd.grad(result['predicted_inspection_value'].sum(),raw,retain_graph=True)[0]
        loss_grad=torch.autograd.grad(result['loss'],raw)[0]
        return result,float((value_grad*loss_grad).sum())

    def test_regret_pushes_inspection_value_in_correct_direction(self):
        self.assertLess(self.decision_gradient(1.5)[1],0.)
        self.assertGreater(self.decision_gradient(-.2)[1],0.)

    def test_zero_regret_and_forced_refusal_have_no_decision_gradient(self):
        for advantage,eligible in ((0.,True),(1.,False)):
            result,direction=self.decision_gradient(advantage,eligible)
            self.assertEqual(float(result['regret']),0.)
            self.assertEqual(direction,0.)
            self.assertTrue(torch.isfinite(result['loss']))


if __name__=='__main__':unittest.main()
