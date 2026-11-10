"""Projection math, exact objective scale, frozen parameters and visibility."""
import unittest
import hjepa_runtime
import torch
from mobile_branch_pcgrad_v9 import (two_task_update, assign_branch_gradients,
                                     branch_losses)


class BranchPCGradContracts(unittest.TestCase):
    def test_conflicting_projection_uses_original_other_gradient(self):
        first, second = torch.tensor([1., 0.]), torch.tensor([-1., 1.])
        update, stats = two_task_update(first, second, 'pcgrad')
        self.assertTrue(torch.allclose(update, torch.tensor([.5, 1.5])))
        self.assertTrue(stats['projected'])
        self.assertAlmostEqual(stats['projected_pair_dot'], .5)
        self.assertGreaterEqual(stats['direction_dot_direct'], 0.)
        self.assertGreaterEqual(stats['direction_dot_post'], 0.)

    def test_nonconflicting_and_zero_tasks_keep_original_sum_scale(self):
        for first, second in ((torch.tensor([2., 1.]), torch.tensor([1., 3.])),
                              (torch.zeros(2), torch.tensor([1., -2.]))):
            update, stats = two_task_update(first, second, 'pcgrad')
            self.assertTrue(torch.equal(update, first + second))
            self.assertFalse(stats['projected'])
        update, _ = two_task_update(torch.tensor([1., 0.]), torch.tensor([-1., 1.]), 'sum')
        self.assertTrue(torch.equal(update, torch.tensor([0., 1.])))

    def test_sum_matches_autograd_and_unused_parameter_stays_none(self):
        first = torch.nn.Parameter(torch.tensor([.5, -1.]))
        unused = torch.nn.Parameter(torch.tensor([4.]))
        frozen = torch.tensor([3.], requires_grad=False)
        losses = ((first*frozen).square().sum(), (first+2.).square().sum())
        expected = torch.autograd.grad(sum(losses), first, retain_graph=True)[0]
        assign_branch_gradients([('position_policy.weight', first), ('unused', unused)], losses, 'sum')
        self.assertTrue(torch.allclose(first.grad, expected))
        self.assertIsNone(unused.grad)
        self.assertIsNone(frozen.grad)
        with self.assertRaises(ValueError):
            assign_branch_gradients([('frozen', frozen)], losses, 'sum')

    def test_unobserved_grasps_and_unsafe_post_position_receive_no_labels(self):
        prediction = {f'{kind}{branch}': torch.zeros(4, 3, requires_grad=True)
                      for branch in (0, 1) for kind in ('p', 'g')}
        target = dict(p0=[0]*4, p1=[0]*4, g0=[0]*4, g1=[0]*4, post_mask=False)
        scene = dict(branches=[dict(grasp_mask=[True, False, False, False]),
                               dict(grasp_mask=[False]*4)])
        losses = branch_losses(prediction, target, scene)
        sum(losses).backward()
        self.assertEqual(float(prediction['p1'].grad.abs().sum()), 0.)
        self.assertIsNone(prediction['g1'].grad)
        self.assertEqual(float(prediction['g0'].grad[1:].abs().sum()), 0.)
        self.assertGreater(float(prediction['g0'].grad[0].abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
