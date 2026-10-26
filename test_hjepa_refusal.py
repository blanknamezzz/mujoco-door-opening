import unittest
import numpy as np
import torch
from door_side_reward import side_feasibility_loss,validation_key
from position_observation import position_decision_logits
from calibrate_hjepa_refusal import fit_offset,summarize_offset


class RefusalTests(unittest.TestCase):
    def test_binary_loss_allows_both_sides_and_trains_neither_to_refuse(self):
        scores=torch.zeros(2,5,requires_grad=True)
        sides=torch.tensor([[0,0,1,1,2]]).expand(2,-1)
        rewards=torch.tensor([[2.,2.,0.],[-1.,-1.,0.]])
        base=torch.zeros(2,5)
        loss,_=side_feasibility_loss(scores,sides,rewards,base);loss.backward()
        self.assertTrue((scores.grad[0,:4]<0).all());self.assertGreater(float(scores.grad[0,4]),0)
        self.assertTrue((scores.grad[1,:4]>0).all());self.assertLess(float(scores.grad[1,4]),0)

    def test_binary_loss_uses_relative_scores_and_wrong_side_labels(self):
        sides=torch.tensor([[0,0,1,1,2]])
        targets=torch.tensor([[-2.,2.,0.]])
        base=torch.zeros(1,5)
        good=torch.tensor([[-2.,-3.,2.,1.,0.]])
        bad=torch.tensor([[2.,1.,-2.,-3.,0.]])
        a=side_feasibility_loss(good,sides,targets,base)[0]
        b=side_feasibility_loss(bad,sides,targets,base)[0]
        self.assertLess(float(a),float(b))
        torch.testing.assert_close(a,side_feasibility_loss(good+10,sides,targets,base)[0])

    def test_offset_changes_only_explicit_refusal_and_never_physical_order(self):
        class Model:
            config={}
            def position_policy(self,x):return x[...,:1]
        model=Model();features=torch.zeros(5,196);features[:,0]=torch.arange(5.)
        features[:4,-4:]=1.
        raw=model.position_policy(features).squeeze(-1)
        self.assertTrue(torch.equal(raw,position_decision_logits(model,features)))
        model.config['position_refusal_offset']=3.
        adjusted=position_decision_logits(model,features)
        torch.testing.assert_close(adjusted[:4],raw[:4]);self.assertEqual(float(adjusted[4]),7.)
        torch.testing.assert_close(position_decision_logits(model,features[:4]),raw[:4])
        features[3,-4:]=0.
        with self.assertRaises(ValueError):position_decision_logits(model,features)

    def test_validation_offset_finds_refusal_without_changing_opening_choice(self):
        rows=[dict(physical_score=1.,refusal_score=0.,success=True,collision=False,feasible=True,chosen_side_can_open=True),
              dict(physical_score=.5,refusal_score=0.,success=False,collision=False,feasible=False,chosen_side_can_open=False)]
        best,candidates=fit_offset(rows)
        self.assertEqual(best['correct_task_decisions'],2)
        self.assertEqual(best['opening_successes'],1);self.assertEqual(best['correct_rejections'],1)
        self.assertEqual(best['threshold'],.75)
        self.assertGreaterEqual(validation_key(best,'door-side'),validation_key(summarize_offset(rows,0),'door-side'))
        self.assertTrue(any(c['threshold']==0 for c in candidates))

    def test_refusal_score_tie_keeps_physical_first_argmax(self):
        rows=[dict(physical_score=1.,refusal_score=.5,success=True,collision=False,feasible=True,chosen_side_can_open=True)]
        self.assertEqual(summarize_offset(rows,.5)['opening_successes'],1)
        self.assertEqual(summarize_offset(rows,.6)['false_rejections'],1)
        with self.assertRaises(ValueError):fit_offset([])


if __name__=='__main__':unittest.main()
