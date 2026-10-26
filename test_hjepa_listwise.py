import unittest
import torch
from train_hjepa_listwise import successful_set_loss, correct_action_mask


class ListwiseTests(unittest.TestCase):
    def test_only_impossible_scenes_target_refusal(self):
        self.assertEqual(correct_action_mask([True,False,False],[False,False,True]).tolist(),[True,False,False])
        self.assertEqual(correct_action_mask([False,False,True],[False,False,True]).tolist(),[False,False,True])

    def test_multiple_successes_are_interchangeable(self):
        mask=torch.tensor([[True,True,False]])
        a=successful_set_loss(torch.tensor([[2.,0.,-1.]]),mask)
        b=successful_set_loss(torch.tensor([[0.,2.,-1.]]),mask)
        self.assertTrue(torch.allclose(a,b))

    def test_gradient_reduces_failed_actions_and_refusal(self):
        scores=torch.tensor([[0.,0.,0.]],requires_grad=True)
        successful_set_loss(scores,torch.tensor([[True,False,False]])).backward()
        self.assertLess(scores.grad[0,0],0)
        self.assertGreater(scores.grad[0,1],0)
        self.assertGreater(scores.grad[0,2],0)

    def test_missing_correct_action_is_rejected(self):
        with self.assertRaises(ValueError):
            successful_set_loss(torch.zeros(1,3),torch.zeros(1,3,dtype=torch.bool))


if __name__=='__main__': unittest.main()
