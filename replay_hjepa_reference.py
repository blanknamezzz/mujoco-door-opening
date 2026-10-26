"""Replay an already frozen candidate against its cached held-out choices."""
import argparse
import json
import os
from pathlib import Path
import time
import hjepa_runtime
import torch
from hjepa_model import load_checkpoint
from structural_scene import structural_scene_from_spec
from run_hjepa import select_and_execute
from evaluate_structural_training import assert_replay_matches
from simulate import save_trial
from run_hjepa_lora_experiment import hash_file,write_json
from experiment_reproducibility import seed_everything


def validate_replay_cohort(report, manifest, key):
    rows=report['decisions'][key]
    cached={d['scene_id']:d for d in rows}
    specs=manifest['scenes']
    ids={s['scene_id'] for s in specs}
    if (not cached or len(cached)!=len(rows) or len(ids)!=len(specs)
        or len(cached)!=report['test_scene_count'] or ids!=set(cached)
        or any(s['split']!='test' for s in specs)):
        raise ValueError('A complete, unique, matching frozen test cohort is required.')
    return cached


def replay(args):
    started=time.monotonic();reproducibility=seed_everything(42);torch.set_num_threads(2)
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Fresh replay output required.')
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint_hash=hash_file(args.checkpoint)
    report=json.loads(args.report.read_text())
    manifest_path=Path(report['dataset'])/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest_hash=hash_file(manifest_path)
    cached=validate_replay_cohort(report,manifest,args.key)
    model,source=load_checkpoint(args.checkpoint,'cuda')
    print(json.dumps(dict(pid=os.getpid(),cases=len(cached),sensor_mode=model.config.get('position_views','single'))),flush=True)
    rows=[]
    for spec in manifest['scenes']:
        plan,trajectory,states=select_and_execute(model,source,structural_scene_from_spec(spec),method='direct')
        target=cached[spec['scene_id']]
        assert_replay_matches(plan,target)
        if abs(plan['metrics']['reward']-target['reward'])>1e-7:raise AssertionError('Physical reward mismatch.')
        save_trial('paired_replay',plan,trajectory,states,args.output/spec['scene_id'])
        rows.append(dict(scene_id=spec['scene_id'],selection_matches=True,outcome_matches=True,reward_matches=True,
            result=plan['result'],position_views=plan['position_views']))
    assert hash_file(args.checkpoint)==checkpoint_hash and hash_file(manifest_path)==manifest_hash
    write_json(args.output/'replay.json',rows)
    write_json(args.output/'verification.json',dict(cases=len(rows),all_actions_outcomes_rewards_match=True,
        checkpoint_sha256=checkpoint_hash,manifest_sha256=manifest_hash,sensor_mode=model.config.get('position_views','single'),
        selected_by_test=False,purpose='Physical verification only; no fitting or checkpoint reselection',
        reproducibility=reproducibility,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(complete=True,cases=len(rows),elapsed_seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--key',default='reference_3')
    p.add_argument('--output',type=Path,required=True)
    replay(p.parse_args())
