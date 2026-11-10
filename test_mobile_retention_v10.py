"""Teacher detachment, visibility, objective scale and declared rate changes."""
import unittest
import hjepa_runtime
import torch
from experiment_reproducibility import seed_everything
from mobile_retention_v10 import output_kl, retention_components, pg_rates
from mobile_branch_pcgrad_v9 import branch_losses
seed_everything(42)


class RetentionContracts(unittest.TestCase):
    def fixture(self):
        prediction={f'p{b}':torch.randn(4,3,requires_grad=True) for b in (0,1)}
        prediction.update({f'g{b}':torch.randn(4,2,3,requires_grad=True) for b in (0,1)})
        teacher={key:value.detach().clone().requires_grad_(True) for key,value in prediction.items()}
        target=dict(p0=[0]*4,p1=[1]*4,g0=[[0,1]]*4,g1=[[1,0]]*4,post_mask=True)
        scene=dict(branches=[dict(grasp_mask=[[True,True]]*4),dict(grasp_mask=[[True,True]]*4)])
        return prediction,teacher,target,scene

    def test_teacher_has_no_gradient_and_identical_outputs_have_zero_kl(self):
        student=torch.tensor([[1.,-2.,.5]],requires_grad=True)
        teacher=student.detach().clone().requires_grad_(True)
        loss=output_kl(student,teacher);loss.backward()
        self.assertAlmostEqual(float(loss.detach()),0.,places=6)
        self.assertLess(float(student.grad.abs().max()),1e-6)
        self.assertIsNone(teacher.grad)

    def test_retention_step_moves_toward_frozen_distribution(self):
        student=torch.tensor([[0.,2.,1.]],requires_grad=True)
        teacher=torch.tensor([[2.,0.,-1.]])
        before=output_kl(student,teacher);before.backward()
        after=output_kl(student.detach()-.1*student.grad,teacher)
        self.assertLess(float(after),float(before.detach()))

    def test_zero_weight_matches_original_objective_and_gradient(self):
        prediction,teacher,target,scene=self.fixture()
        ce=sum(branch_losses(prediction,target,scene))
        expected=torch.autograd.grad(ce,list(prediction.values()),retain_graph=True)
        result=retention_components(prediction,teacher,target,scene,0.)
        actual=torch.autograd.grad(result['loss'],list(prediction.values()))
        self.assertTrue(torch.allclose(ce,result['loss']))
        self.assertTrue(all(torch.allclose(a,b) for a,b in zip(actual,expected)))

    def test_retention_ignores_unsafe_post_position_and_unobserved_grasps(self):
        prediction,teacher,target,scene=self.fixture();target['post_mask']=False
        scene['branches'][0]['grasp_mask']=[[True,False]]+[[False,False]]*3
        scene['branches'][1]['grasp_mask']=[[False,False]]*4
        for key in ('p1','g0','g1'):
            teacher[key]=torch.zeros_like(teacher[key],requires_grad=False)
            teacher[key][...,2]=20.
        result=retention_components(prediction,teacher,target,scene,1.)
        result['kl'].backward()
        self.assertEqual(float(prediction['p1'].grad.abs().sum()),0.)
        self.assertIsNone(prediction['g1'].grad)
        self.assertEqual(float(prediction['g0'].grad[0,1].abs().sum()),0.)
        self.assertEqual(float(prediction['g0'].grad[1:].abs().sum()),0.)
        self.assertGreater(float(prediction['g0'].grad[0,0].abs().sum()),0.)

    def test_only_pg_rates_change_by_declared_factor(self):
        normal,small=pg_rates(1.),pg_rates(.1)
        self.assertAlmostEqual(small['head']/normal['head'],.1)
        self.assertAlmostEqual(small['lora']/normal['lora'],.1)
        with self.assertRaises(ValueError):pg_rates(.01)

    def test_teacher_cache_rejects_heldout_before_encoding(self):
        from run_mobile_retention import fixed_teacher
        for split in ('val','test'):
            with self.assertRaises(ValueError):
                fixed_teacher(None,None,None,[dict(spec=dict(split=split))])


if __name__=='__main__':unittest.main()
