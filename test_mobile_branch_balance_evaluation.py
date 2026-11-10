"""Equivalence must cover actual inference; changed encoders need new caches."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import hjepa_runtime
import torch
from run_mobile_branch_balance_evaluation import inference_digest,coalesce_policies,evaluate_unique


class BranchEvaluationContracts(unittest.TestCase):
    def checkpoint(self):
        return dict(config={'mobile_partial_v5':{'use_map_features':True}},
            model={'encoder.lora_A':torch.tensor([1.]),'position_policy.weight':torch.tensor([2.]),
                   'inspection_policy.weight':torch.tensor([3.])},state_mean=torch.zeros(1),state_std=torch.ones(1))

    def test_only_training_metadata_is_ignored_in_inference_identity(self):
        original=self.checkpoint();copy=deepcopy(original)
        copy['config'].update(branch_balance_v8={'arm':'paired_ce'},inspection_regret_v6={'arm':'joint_regret'})
        self.assertEqual(inference_digest(original),inference_digest(copy))
        for kind in ('encoder','I','norm','config'):
            changed=deepcopy(copy)
            if kind=='encoder':changed['model']['encoder.lora_A']+=1.
            elif kind=='I':changed['model']['inspection_policy.weight']+=1.
            elif kind=='norm':changed['state_mean']+=1.
            else:changed['config']['mobile_partial_v5']['use_map_features']=False
            self.assertNotEqual(inference_digest(original),inference_digest(changed))

    def test_modes_stay_distinct_and_primary_represents_equivalent_aliases(self):
        policies={'old':dict(model='a',mode='learned',risk=1.),'primary':dict(model='b',mode='learned',risk=1.),
            'changed':dict(model='c',mode='learned',risk=1.),'always':dict(model='a',mode='always',risk=1.),
            'never':dict(model='a',mode='never',risk=1.)}
        reps,aliases=coalesce_policies(policies,{'a':'same','b':'same','c':'other'},'primary')
        self.assertEqual(len(reps),4);self.assertEqual(aliases['old'],'primary')
        self.assertNotEqual(aliases['always'],aliases['never']);self.assertEqual(aliases['changed'],'changed')

    def test_updated_encoders_do_not_receive_another_checkpoints_cache(self):
        class Model:
            def requires_grad_(self,value):return self
        policies={'first':dict(model='first',mode='learned',risk=1.),'second':dict(model='second',mode='learned',risk=1.)}
        def load(path,device):return Model(),{'tag':str(path)}
        def predict(model,source,scenes):return source['tag']
        def evaluate(model,scenes,cached,subset):return ({name:cached for name in subset},{name:[] for name in subset})
        with patch('mobile_partial_policy_v5.load_partial_checkpoint',side_effect=load),\
             patch('run_mobile_partial_training.predict',side_effect=predict) as calls,\
             patch('run_mobile_partial_training.evaluate',side_effect=evaluate),patch('torch.cuda.empty_cache'):
            metrics,_=evaluate_unique([],policies)
        self.assertEqual(calls.call_count,2);self.assertNotEqual(metrics['first'],metrics['second'])


if __name__=='__main__':unittest.main()
