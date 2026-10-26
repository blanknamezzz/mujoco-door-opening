import unittest
from unittest.mock import patch
import numpy as np
import torch
from door_side_reward import (side_reward_targets,door_side_loss,decision_side_metrics,validation_key)


def action_table(successful_bases=()):
    groups=[]
    for b in range(4):
        groups.append([dict(scene_id='scene',base_id=b,grasp_id=g,operating_side='front' if b<2 else 'back',
            base_pose=[.6,-1. if b<2 else 1.,0.],success=b in successful_bases and g==0,
            reward=999.,milestones=[True]*3,safe_opening_fraction=.99) for g in range(4)])
    groups.append([dict(scene_id='scene',base_id=4,grasp_id=0,positioning_only=True,success=False)])
    return groups


def loss_for(scores,targets):
    return door_side_loss(scores,
        torch.tensor(targets['sides'][None]),torch.tensor(targets['side_rewards'][None]),
        torch.tensor(targets['base_rewards'][None]))[0]


class DoorSideRewardTests(unittest.TestCase):
    def test_wrong_side_progress_cannot_outweigh_openable_side(self):
        t=side_reward_targets(action_table((2,3)))
        np.testing.assert_array_equal(t['side_rewards'],[-2,2,0])
        np.testing.assert_array_equal(t['position_rewards'],[-2,-2,2.25,2.25,0])

    def test_side_credit_is_separate_from_base_and_grasp_failure(self):
        groups=action_table((0,))
        t=side_reward_targets(groups)
        np.testing.assert_array_equal(t['position_rewards'],[2.25,1.75,-2,-2,0])
        # A bad G selection at grasp #1 does not make this side impossible.
        selected=groups[0][1]
        metrics=decision_side_metrics(selected,[e for g in groups for e in g])
        self.assertFalse(selected['success'])
        self.assertTrue(metrics['side_decision_correct'])
        self.assertFalse(metrics['within_side_base_error'])
        metrics=decision_side_metrics(groups[1][0],[e for g in groups for e in g])
        self.assertTrue(metrics['within_side_base_error'])

    def test_neither_side_opens_refusal_is_best(self):
        groups=action_table()
        t=side_reward_targets(groups)
        np.testing.assert_array_equal(t['position_rewards'],[-1,-1,-1,-1,0])
        self.assertTrue(decision_side_metrics(groups[-1][0],[e for g in groups for e in g])['side_decision_correct'])

    def test_both_sides_open_have_equal_credit(self):
        t=side_reward_targets(action_table((0,1,2,3)))
        np.testing.assert_array_equal(t['side_rewards'],[2,2,0])
        a=loss_for(torch.tensor([[1.,0.,.2,.1,-1.]]),t)
        b=loss_for(torch.tensor([[.2,.1,1.,0.,-1.]]),t)
        self.assertTrue(torch.allclose(a,b))

    def test_gradient_pushes_openable_side_above_wrong_side(self):
        t=side_reward_targets(action_table((2,3)))
        scores=torch.zeros(1,5,requires_grad=True)
        loss_for(scores,t).backward()
        self.assertTrue((scores.grad[0,:2]>0).all())
        self.assertTrue((scores.grad[0,2:4]<0).all())

    def test_duplicating_candidate_does_not_improve_side_score(self):
        t=side_reward_targets(action_table((0,1,2,3)))
        original=loss_for(torch.tensor([[.3,.1,.2,.1,0.]]),t)
        # Identical command duplicates cannot bias the front/back/refusal pool.
        scores=torch.tensor([[.3,.1,.3,.2,.1,0.]])
        sides=torch.tensor([[0,0,0,1,1,2]])
        expanded=door_side_loss(scores,sides,torch.tensor([[2.,2.,0.]]),torch.tensor([[1.,1.,1.,1.,1.,0.]]))[0]
        self.assertTrue(torch.allclose(original,expanded))

    def test_targets_ignore_legacy_reward_and_scene_family(self):
        groups=action_table((0,))
        before=side_reward_targets(groups)
        for g in groups:
            for e in g:e.update(reward=-1e9,family='incorrect-label',milestones=[False]*3)
        np.testing.assert_array_equal(before['position_rewards'],side_reward_targets(groups)['position_rewards'])

    def test_partial_or_mixed_scene_tables_are_rejected(self):
        groups=action_table((0,)); groups[0].pop()
        with self.assertRaises(ValueError):side_reward_targets(groups)
        groups=action_table((0,)); groups[1][0]['scene_id']='different'
        with self.assertRaises(ValueError):side_reward_targets(groups)
        groups=action_table((0,)); groups[2][0]['operating_side']='front'
        with self.assertRaises(ValueError):side_reward_targets(groups)

    def test_missing_metadata_does_not_create_side_credit(self):
        self.assertFalse(decision_side_metrics(dict(base_id=0),[dict(base_id=0,success=True)])['side_metrics_available'])

    def test_side_validation_does_not_silently_use_overall_metric(self):
        s=dict(scenes=2,side_evaluated_scenes=2,correct_side_decisions=2,correct_task_decisions=1,
            opening_successes=1,mean_terminal_utility_v2=0.)
        self.assertEqual(validation_key(s,'door-side'),(2,1,1,0.))
        s['side_evaluated_scenes']=0
        with self.assertRaises(ValueError):validation_key(s,'door-side')

    def test_evaluator_separates_wrong_side_from_bad_base(self):
        from evaluate_structural_training import evaluate_records
        groups=action_table((0,))
        for g in groups:
            for e in g:e.update(split='val',result=dict(unexpected_contact_samples=0))
        records=[(e,{}) for g in groups for e in g]
        manifest=dict(scenes=[dict(scene_id='scene',family='example')])
        with patch('evaluate_structural_training.choose',return_value=(groups[1][0],{})):
            s,_=evaluate_records(None,{},records,manifest,split='val')
        self.assertEqual((s['correct_side_decisions'],s['wrong_side_errors'],s['within_side_base_errors']),(1,0,1))
        self.assertEqual(s['correct_task_decisions'],0)
        with patch('evaluate_structural_training.choose',return_value=(groups[2][0],{})):
            s,_=evaluate_records(None,{},records,manifest,split='val')
        self.assertEqual((s['correct_side_decisions'],s['wrong_side_errors'],s['within_side_base_errors']),(0,1,0))


if __name__=='__main__':unittest.main()
