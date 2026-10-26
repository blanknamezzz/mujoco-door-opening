"""One global refusal-score offset, selected on validation, with frozen tensors."""
import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import json
import time
import hjepa_runtime
import numpy as np
import torch
from experiment_reproducibility import seed_everything
from hjepa_model import load_checkpoint,save_checkpoint
from hjepa_lora import adapter_state,frozen_core_digest
from train_hjepa import read_data
from evaluate_structural_training import choose,evaluate_records
from door_side_reward import decision_side_metrics,validation_key
from decomposed_reward import terminal_utility
from run_hjepa_lora_experiment import hash_file,write_json


def summarize_offset(rows, offset):
    """Emulate float32 inference addition, including physical-first argmax ties."""
    rejected=[bool(np.float32(r['refusal_score'])+np.float32(offset)>np.float32(r['physical_score'])) for r in rows]
    return dict(threshold=float(offset),scenes=len(rows),side_evaluated_scenes=len(rows),
        correct_side_decisions=sum((not r['feasible']) if a else r['chosen_side_can_open'] for r,a in zip(rows,rejected)),
        opening_successes=sum(r['success'] and not a for r,a in zip(rows,rejected)),
        correct_rejections=sum(not r['feasible'] and a for r,a in zip(rows,rejected)),
        false_rejections=sum(r['feasible'] and a for r,a in zip(rows,rejected)),
        correct_task_decisions=sum((not r['feasible']) if a else r['success'] for r,a in zip(rows,rejected)),
        mean_terminal_utility_v2=float(np.mean([terminal_utility(r['success'],r['collision'],a) for r,a in zip(rows,rejected)])))


def fit_offset(rows):
    if not rows:raise ValueError('Nonempty validation cohort required.')
    values=sorted({r['physical_score']-r['refusal_score'] for r in rows})
    if not np.isfinite(values).all():raise ValueError('Finite scores required.')
    offsets=sorted(set([0.,values[0]-1.,values[-1]+1.]+[(a+b)/2 for a,b in zip(values,values[1:])]))
    candidates=[summarize_offset(rows,o) for o in offsets]
    best=max(candidates,key=lambda s:(*validation_key(s,'door-side'),-abs(s['threshold'])))
    return best,candidates


def calibrate(args):
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Use a new calibration output.')
    args.output.mkdir(parents=True);started=time.monotonic();repro=seed_everything(args.seed);torch.set_num_threads(4)
    manifest,records=read_data(args.data,required_splits=('val',),include_splits=('val',),policy_only=True)
    if any(s['split'] not in ('train','val') for s in manifest['scenes']):raise ValueError('No test selection permitted.')
    model,source=load_checkpoint(args.initial,'cuda');model.requires_grad_(False)
    if not model.is_official or not source.get('positioning_abstention'):raise ValueError('Official model with explicit refusal required.')
    if model.config.get('reward_decomposition'):raise ValueError('Use the historical decomposed calibrator for component heads.')
    original={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    frozen=frozen_core_digest(model);initial_hash=hash_file(args.initial)
    # Fit an absolute offset; preserve the starting setting for the receipt.
    old_offset=model.config.get('position_refusal_offset',0.)
    model.config['position_refusal_offset']=0.
    tables=defaultdict(lambda:defaultdict(list))
    for e,x in records:tables[e['scene_id']][e['base_id']].append((e,x))
    rows=[]
    for sid,table in tables.items():
        groups=list(table.values());refusals=[i for i,g in enumerate(groups) if g[0][0].get('positioning_only')]
        if len(refusals)!=1 or refusals[0]!=len(groups)-1:raise ValueError('Physical-first candidates, then one refusal required.')
        chosen,scores=choose(model,source,table,force_position_execution=True)
        assert not chosen.get('positioning_only')
        raw=scores['position_scores'];physical=raw[:-1]
        flat=[e for g in groups for e,_ in g]
        side=decision_side_metrics(chosen,flat)
        if not side['side_metrics_available']:raise ValueError('Complete measured side labels required.')
        rows.append(dict(scene_id=sid,physical_score=max(physical),refusal_score=raw[-1],
            base_id=chosen['base_id'],grasp_id=chosen['grasp_id'],success=chosen['success'],collision=chosen['collision'],
            feasible=any(e['success'] for e in flat if not e.get('positioning_only')),
            chosen_side_can_open=side['chosen_side_can_open']))
    baseline,_=evaluate_records(model,source,records,manifest,split='val')
    best,candidates=fit_offset(rows)
    model.config['position_refusal_offset']=best['threshold']
    selected,_=evaluate_records(model,source,records,manifest,split='val')
    for metric in ('correct_side_decisions','correct_task_decisions','opening_successes','correct_rejections','false_rejections'):
        if selected[metric]!=best[metric]:raise AssertionError('Scanned and actual calibration disagree: '+metric)
    assert validation_key(selected,'door-side')>=validation_key(baseline,'door-side')
    assert frozen_core_digest(model)==frozen and all(torch.equal(v,model.state_dict()[n].cpu()) for n,v in original.items())
    budgets={k:source[k] for k in ('base_budget','grasp_budget','positioning_abstention','candidate_base_poses','abstention_reward','adapter_base_checkpoint')}
    budgets.update(seed=args.seed,reproducibility=repro)
    for name in ('best.pt','selected.pt'):save_checkpoint(args.output/name,model,source['state_mean'].cpu(),source['state_std'].cpu(),step=source['step'],**budgets)
    torch.save(dict(adapter=adapter_state(model),config=model.config,base_checkpoint=budgets['adapter_base_checkpoint'],
        state_mean=source['state_mean'].cpu(),state_std=source['state_std'].cpu(),metadata=dict(step=source['step'],**budgets)),args.output/'adapter.pt')
    previous=json.loads((args.initial.parent/'training.json').read_text())
    calibration=dict(threshold=best['threshold'],parameter_count=1,fit_split='val',fit_scenes=len(rows),
        inherited_offset=old_offset,candidates=candidates,rows=rows,baseline_validation=baseline,selected_validation=selected,
        learned_tensor_changes=0,test_arrays_opened=0,probability_calibration_claim=False,
        selection='same side/task/openings/utility validation tuple, then smallest absolute offset')
    report=deepcopy(previous)
    report.update(decision_calibration=calibration,selected_validation=selected,unadapted_validation=baseline,
        selected_world_model_step=source['step'],selected_method_adopted=best['threshold']!=0.,
        checkpoint_selection='Global refusal offset fitted on validation side/task correctness, openings and utility',
        scenes=manifest['scenes'],test_arrays_opened=0,all_learned_tensors_unchanged=True,
        calibration_source=dict(path=str(args.initial),sha256=initial_hash),seed=args.seed,reproducibility=repro,
        elapsed_seconds=time.monotonic()-started)
    write_json(args.output/'training.json',report);write_json(args.output/'decision_calibration.json',calibration)
    write_json(args.output/'adapter_contract.json',dict(frozen_core_sha256=frozen,learned_tensor_changes=0,seed=args.seed,test_arrays_opened=0))
    assert hash_file(args.initial)==initial_hash
    print(json.dumps(dict(complete=True,threshold=best['threshold'],baseline=baseline,selected=selected,elapsed_seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('runs/hjepa/coverage_corpus_seed42/expanded_policy'))
    p.add_argument('--initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    calibrate(p.parse_args())
