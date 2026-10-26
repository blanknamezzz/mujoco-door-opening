"""Fit one global rejection threshold on validation; freeze all learned tensors.

This calibrates a decision rule, not the probability outputs. A completely new
test suite must follow; already inspected test results are not selection data.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import time
import hjepa_runtime
import numpy as np
import torch
from decomposed_reward import configure_decomposition, terminal_utility
from evaluate_structural_training import choose, evaluate_records
from hjepa_model import load_checkpoint, save_checkpoint
from hjepa_lora import adapter_state, frozen_core_digest
from run_hjepa_lora_experiment import hash_file, write_json
from train_hjepa import read_data


def calibrate(args):
    started=time.monotonic()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'selected.pt').exists(): raise FileExistsError('Preserve previous calibration.')
    manifest,records=read_data(args.data,required_splits=('val',),include_splits=('val',))
    model,source=load_checkpoint(args.initial,'cuda')
    if model.config.get('reward_decomposition',{}).get('blend')!=1.:
        raise ValueError('Calibrate the actually trained decomposed model, not its baseline fallback.')
    model.requires_grad_(False)
    frozen=frozen_core_digest(model)
    original_state={n:v.clone() for n,v in model.state_dict().items()}
    tables=defaultdict(lambda:defaultdict(list))
    for e,x in records: tables[e['scene_id']][e['base_id']].append((e,x))
    rows=[]
    for scene_id,table in tables.items():
        chosen,scores=choose(model,dict(source,positioning_abstention=False),table)
        rows.append(dict(scene_id=scene_id,score=max(scores['position_scores']),
            success=chosen['success'],collision=chosen['collision'],
            feasible=any(e['success'] for g in table.values() for e,_ in g)))
    values=sorted({row['score'] for row in rows})
    thresholds=[values[0]-1.,0.,values[-1]+1.]+[(a+b)/2 for a,b in zip(values,values[1:])]
    candidates=[]
    for threshold in thresholds:
        opens=sum(x['success'] and x['score']>=threshold for x in rows)
        rejects=sum(not x['feasible'] and x['score']<threshold for x in rows)
        false_rejects=sum(x['feasible'] and x['score']<threshold for x in rows)
        utility=np.mean([terminal_utility(x['success'],x['collision'],x['score']<threshold) for x in rows])
        candidates.append(dict(threshold=threshold,opening_successes=opens,correct_rejections=rejects,
            false_rejections=false_rejects,correct_task_decisions=opens+rejects,mean_terminal_utility_v2=float(utility)))
    best=max(candidates,key=lambda x:(x['correct_task_decisions'],x['opening_successes'],x['mean_terminal_utility_v2'],-abs(x['threshold'])))
    old,baseline=load_checkpoint(args.baseline,'cuda')
    baseline_summary,_=evaluate_records(old,baseline,records,manifest,split='val')
    del old
    model.position_policy.score_offset=-best['threshold']
    model.config['reward_decomposition']['position_score_offset']=-best['threshold']
    calibrated_summary,_=evaluate_records(model,source,records,manifest,split='val')
    assert calibrated_summary['correct_task_decisions']==best['correct_task_decisions']
    adopted=(calibrated_summary['correct_task_decisions'],calibrated_summary['opening_successes'],calibrated_summary['mean_terminal_utility_v2']) > (
        baseline_summary['correct_task_decisions'],baseline_summary['opening_successes'],baseline_summary['mean_terminal_utility_v2'])
    if frozen_core_digest(model)!=frozen or any(not torch.equal(v,model.state_dict()[n]) for n,v in original_state.items()):
        raise AssertionError('Threshold fitting changed a learned tensor.')
    base=source['adapter_base_checkpoint']
    budgets={k:source[k] for k in ('base_budget','grasp_budget','positioning_abstention','candidate_base_poses','abstention_reward','adapter_base_checkpoint')}
    selected,selected_source=(model,source) if adopted else load_checkpoint(args.baseline,'cuda')
    if not adopted:
        configure_decomposition(selected,dict(version='outcome_components_v2',blend=0.))
    selected_step=source['step'] if adopted else 0
    for filename in ('best.pt','selected.pt'):
        save_checkpoint(args.output/filename,selected,selected_source['state_mean'].cpu(),selected_source['state_std'].cpu(),
            step=selected_step,**budgets)
    torch.save(dict(adapter=adapter_state(selected),config=selected.config,base_checkpoint=base,
        state_mean=selected_source['state_mean'].cpu(),state_std=selected_source['state_std'].cpu(),
        metadata=dict(step=selected_step,**budgets)),args.output/'adapter.pt')
    prior_report=json.loads((args.initial.parent/'training.json').read_text())
    calibration=dict(parameter_count=1,threshold=best['threshold'],position_score_offset=-best['threshold'],
        fit_split='val',fit_scenes=len(rows),trained_tensor_changes=0,probability_calibration_claim=False,
        candidates=candidates,selected_validation=calibrated_summary,baseline_validation=baseline_summary,
        adopted=adopted,test_arrays_opened=0)
    write_json(args.output/'decision_calibration.json',calibration)
    report=dict(history=deepcopy(prior_report['history']),unadapted_validation=baseline_summary,
        steps=prior_report['steps'],joint_training_steps_this_phase=0,joint_training_source=str(args.initial),
        selected_world_model_step=selected_step,selected_policy_updates=0,selected_world_model_unchanged=True,
        initialization=dict(path=str(args.initial),sha256=hash_file(args.initial),step=source['step'],optimizer_resumed=False),
        adapter_base_checkpoint=base,lora=prior_report['lora'],reward_decomposition=selected.config['reward_decomposition'],
        selected_method_adopted=adopted,scenes=manifest['scenes'],test_arrays_opened=0,
        frozen_core_unchanged=True,all_learned_tensors_unchanged=True,decision_calibration=calibration,
        checkpoint_selection='one global threshold fit on validation correct decisions, then openings, then terminal utility',
        elapsed_seconds=time.monotonic()-started,core_training_elapsed_seconds=prior_report['elapsed_seconds'],
        batch_size=128,precision='bf16-mixed',selected_checkpoint='selected.pt',
        note='Calibration development score is not an independent test score. No test scenes or per-scene lookup rule.')
    report['history'].append(dict(step=source['step'],policy_validation=calibrated_summary,
        val=0.,train=0.,phase='decision_threshold_calibration'))
    write_json(args.output/'training.json',report)
    write_json(args.output/'resolved_config.json',selected.config)
    write_json(args.output/'adapter_contract.json',dict(frozen_core_sha256=frozen,all_learned_tensors_unchanged=True,
        core_training_steps=prior_report['steps'],calibration_parameters=1,test_arrays_opened=0))
    print(json.dumps(dict(adopted=adopted,threshold=best['threshold'],validation=calibrated_summary,
        baseline_validation=baseline_summary,elapsed_seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('runs/hjepa/diverse_lora_trainval'))
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=Path('runs/hjepa/lora_continued_model/selected.pt'))
    p.add_argument('--output',type=Path,required=True)
    calibrate(p.parse_args())
