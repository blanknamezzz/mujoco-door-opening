"""Training split integrity, collision emphasis and safe checkpoint selection."""
import random
import unittest
import hjepa_runtime
import torch
from mobile_branch_balance_v8 import ScenePairSampler,weighted_outcome_loss,pg_selection_key


class BranchBalanceContracts(unittest.TestCase):
    def scenes(self):
        return [dict(spec=dict(split='train',group_id=str(group),layout=layout,scene_id=f'{group}_{layout}'))
                for group in range(3) for layout in ('front','rear','both','free')]

    def test_paired_batches_have_two_distinct_layouts_and_preserve_budget(self):
        sampler=ScenePairSampler(self.scenes(),'paired');rng=random.Random(42)
        for _ in range(100):
            pair=sampler.sample(rng);self.assertEqual(len(pair),2)
            self.assertEqual(pair[0]['spec']['group_id'],pair[1]['spec']['group_id'])
            self.assertNotEqual(pair[0]['spec']['layout'],pair[1]['spec']['layout'])

    def test_sampler_rejects_heldout_or_broken_groups(self):
        scenes=self.scenes();scenes[0]['spec']['split']='val'
        with self.assertRaises(ValueError):ScenePairSampler(scenes,'random')
        with self.assertRaises(ValueError):ScenePairSampler(self.scenes()[:-1],'paired')

    def test_collision_weight_increases_collision_relative_gradient(self):
        for weight,expected in ((1.,1.),(4.,4.)):
            logits=torch.zeros(2,3,requires_grad=True)
            weighted_outcome_loss(logits,torch.tensor([0,2]),weight).backward()
            self.assertAlmostEqual(abs(float(logits.grad[1,2]/logits.grad[0,0])),expected,places=5)

    def test_extra_direct_accuracy_cannot_hide_new_collisions(self):
        def metrics(correct,collisions=0):
            return dict(correct_decisions=correct,collisions=collisions,openings=15,
                        mean_net_utility=.4,mean_path_length_m=1.)
        initial=pg_selection_key(metrics(24),metrics(35))
        unsafe=pg_selection_key(metrics(25,2),metrics(35))
        safer_improvement=pg_selection_key(metrics(25),metrics(35))
        self.assertLess(unsafe,initial);self.assertGreater(safer_improvement,initial)


if __name__=='__main__':unittest.main()
