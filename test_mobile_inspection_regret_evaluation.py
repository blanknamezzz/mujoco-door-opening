"""Guard against test leakage and unintended continuation differences."""
from copy import deepcopy
from pathlib import Path
import unittest
import hjepa_runtime
import torch
from run_mobile_inspection_regret_evaluation import (assert_same_continuation,validate_test_manifest,make_policies)


class RegretEvaluationContracts(unittest.TestCase):
    def test_policy_routes_keep_declared_v5_checkpoint_and_modes(self):
        policies=make_policies(Path('new-training'),Path('prior-v5'))
        self.assertEqual(len(policies),6)
        self.assertEqual([policies['frozen_v5_'+name]['mode'] for name in ('learned','always','never')],
                         ['learned','always','never'])
        self.assertTrue(all(policies[name]['model']==policies['frozen_v5_learned']['model']
                            for name in ('frozen_v5_always','frozen_v5_never')))
        self.assertNotEqual(policies['joint_regret']['model'],policies['frozen_v5_learned']['model'])

    def test_smoke_cannot_read_frozen_formal_geometry(self):
        formal=dict(scenes=[dict(split='test',geometry_sha256='frozen')])
        smoke=dict(scenes=[dict(split='test',geometry_sha256='other')])
        self.assertEqual(len(validate_test_manifest(smoke,formal,True)),1)
        with self.assertRaises(ValueError):validate_test_manifest(formal,formal,True)
        with self.assertRaises(ValueError):validate_test_manifest(smoke,formal,False)

    def test_only_inspection_updates_can_share_frozen_predictions(self):
        baseline=dict(model={'model.lora_A':torch.tensor([1.]),'position_policy.weight':torch.tensor([2.]),
            'grasp_policy.weight':torch.tensor([3.]),'inspection_policy.weight':torch.tensor([4.])},
            state_mean=torch.zeros(1),state_std=torch.ones(1),config={'mobile_partial_v5':{'use_map_features':True}})
        changed=deepcopy(baseline);changed['model']['inspection_policy.weight']+=1.
        changed['config']['inspection_regret_v6']={'arm':'joint_regret'}
        assert_same_continuation(baseline,changed)
        for name in ('model.lora_A','position_policy.weight','grasp_policy.weight'):
            invalid=deepcopy(changed);invalid['model'][name]+=1.
            with self.assertRaises(AssertionError):assert_same_continuation(baseline,invalid)
        invalid=deepcopy(changed);invalid['state_mean']+=1.
        with self.assertRaises(AssertionError):assert_same_continuation(baseline,invalid)
        invalid=deepcopy(changed);invalid['config']['mobile_partial_v5']['use_map_features']=False
        with self.assertRaises(AssertionError):assert_same_continuation(baseline,invalid)


if __name__=='__main__':unittest.main()
