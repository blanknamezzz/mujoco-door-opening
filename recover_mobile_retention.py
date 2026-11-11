"""Finish the interrupted fourth arm of the frozen v10 matched experiment.

This creates a new ignored run. It preserves the original run and reuses only
completed, hash-checked development artifacts; the unfinished arm is retrained
from step zero with the original seed, batches, teacher and manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from run_mobile_reward_experiment import ROOT, ZONE, file_hash, write_json
from mobile_retention_v10 import ARMS
from run_mobile_retention import read, train_pg, train_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interrupted', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--estimate-seconds', type=int, required=True)
    args = parser.parse_args()
    args.interrupted = args.interrupted.resolve()
    args.output = args.output.resolve()
    if ROOT/'runs' not in args.output.parents or (args.output.exists() and any(args.output.iterdir())):
        raise ValueError('Recovery requires a new empty ignored runs directory.')
    if args.estimate_seconds < 1:
        raise ValueError('Positive runtime estimate required.')
    if (args.interrupted/'report.json').exists():
        raise ValueError('Original experiment already finished.')
    begin, now = time.perf_counter(), datetime.now(ZONE)
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = dict(pid=os.getpid(), seed=42, status='starting', started_at=now.isoformat(),
        runtime_estimate_seconds=args.estimate_seconds, queue_estimate_seconds=0,
        expected_completion=(now+timedelta(seconds=args.estimate_seconds)).isoformat(),
        interrupted_run=str(args.interrupted), interrupted_phase='pg_low_lr_retention_step200',
        recovery='retrain fourth arm from step zero; copy completed three arms only', test_outcome_reads=0)
    write_json(args.output/'receipt.json', receipt)
    from experiment_reproducibility import seed_everything
    receipt['reproducibility'] = seed_everything(42)
    import torch
    torch.set_num_threads(4)
    old = read(args.interrupted/'protocol.json')
    args.initial = Path(old['initial'])
    args.development = Path(old['development'])
    args.maps = Path(old['maps'])
    args.prior = Path(old['prior'])
    args.pg_steps = int(old['pg_steps'])
    args.gate_steps = int(old['gate_steps'])
    if old['arms'] != ARMS or file_hash(args.initial) != old['initial_sha256']:
        raise AssertionError('Frozen arm specification or initial weights changed.')
    if read(args.interrupted/'receipt.json').get('status') == 'complete':
        raise ValueError('Original experiment finished after recovery was requested.')
    for name, digest in read(args.interrupted/'source_hashes.json').items():
        if file_hash(ROOT/name) != digest:
            raise AssertionError('Original v10 source changed: '+name)
    for name in ('manifest.json','protocol.json','preserved_hashes.json',
                 'reused_development_hashes.json','teacher_logits.pt','teacher_cache.json',
                 'source_hashes.json'):
        shutil.copy2(args.interrupted/name, args.output/name)
    shutil.copytree(args.interrupted/'source', args.output/'source')
    completed = ('standard','low_lr','retention')
    inherited = {}
    for arm in completed:
        folder = args.interrupted/arm
        for name in ('pg_training.json','gate_training.json','validation.json',
                     'selected_pg.pt','last_pg.pt','selected.pt','last.pt',
                     'training_batches.json','retention_diagnostics.json'):
            if not (folder/name).is_file():
                raise AssertionError('Completed arm lacks '+arm+'/'+name)
        shutil.copytree(folder, args.output/arm)
        inherited[arm] = {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()}
        if any(file_hash(args.output/arm/name) != digest for name,digest in inherited[arm].items()):
            raise AssertionError('Copied arm differs from interrupted run.')
    original_manifest_sha = file_hash(args.interrupted/'manifest.json')
    if file_hash(args.output/'manifest.json') != original_manifest_sha:
        raise AssertionError('Frozen test manifest changed.')
    if file_hash(args.output/'teacher_logits.pt') != read(args.output/'teacher_cache.json')['cache_sha256']:
        raise AssertionError('Frozen teacher cache changed.')
    write_json(args.output/'recovery.json', dict(seed=42, interrupted_run=str(args.interrupted),
        original_manifest_sha256=original_manifest_sha, inherited_completed_arm_hashes=inherited,
        unfinished_arm_old_checkpoint_preserved=str(args.interrupted/'low_lr_retention'/'selected_pg.pt'),
        unfinished_arm_restarted_at_step=0, test_outcome_reads=0))
    from run_mobile_outcome_experiment import attach_development
    manifest = read(args.output/'manifest.json')
    specs = [s for s in manifest['scenes'] if s['split'] in ('train','val')]
    scenes = attach_development(args,specs)
    if file_hash(args.output/'reused_development_hashes.json') != file_hash(args.interrupted/'reused_development_hashes.json'):
        raise AssertionError('Reused development dataset changed.')
    for scene in scenes:
        scene['maps'] = read(args.maps/'data'/scene['spec']['scene_id']/'maps.json')
    receipt.update(status='training', startup_seconds=time.perf_counter()-begin)
    write_json(args.output/'receipt.json', receipt)
    pg = train_pg(args,'low_lr_retention',scenes)
    gate = train_gate(args,'low_lr_retention',scenes)
    training = {arm:dict(pg=read(args.output/arm/'pg_training.json'),
                         gate=read(args.output/arm/'gate_training.json')) for arm in ARMS}
    validation = {arm:training[arm]['gate']['validation'] for arm in ARMS}
    phases = {f'{phase}_{arm}':training[arm][phase]['seconds']
              for arm in ARMS for phase in ('pg','gate')}
    if len({row['pg']['batches_sha256'] for row in training.values()}) != 1:
        raise AssertionError('Matched scene batches changed during recovery.')
    if len({row['pg']['teacher_cache_sha256'] for row in training.values()}) != 1:
        raise AssertionError('Fixed teacher changed during recovery.')
    baseline = torch.load(args.initial,map_location='cpu',weights_only=True)
    original = [name for name in baseline['model'] if 'lora_' not in name and not
        name.startswith(('position_policy.','grasp_policy.','inspection_policy.'))]
    reference_path = Path(old['reference_path'])
    if file_hash(reference_path) != old['reference_sha256']:
        raise AssertionError('Frozen reference changed.')
    reference = torch.load(reference_path,map_location='cpu',weights_only=True)
    audits,identity = {},{}
    for arm in ARMS:
        selected_pg = torch.load(args.output/arm/'selected_pg.pt',map_location='cpu',weights_only=True)
        audits[arm] = {}
        for label in ('selected_pg','last_pg','selected','last'):
            cp = selected_pg if label == 'selected_pg' else torch.load(
                args.output/arm/(label+'.pt'),map_location='cpu',weights_only=True)
            if any(not torch.equal(baseline['model'][n],cp['model'][n]) for n in original):
                raise AssertionError('Original core changed: '+arm+'/'+label)
            if any(not torch.equal(baseline[k],cp[k]) for k in ('state_mean','state_std')):
                raise AssertionError('Normalization changed: '+arm+'/'+label)
            fixed = (all(torch.equal(value,cp['model'][name]) for name,value in selected_pg['model'].items()
                if not name.startswith('inspection_policy.')) if label in ('selected','last') else None)
            if fixed is False:
                raise AssertionError('Non-I parameters changed during gate training.')
            changed = sum(not torch.equal(value,cp['model'][name]) for name,value in baseline['model'].items()
                          if 'lora_' in name)
            if label == 'last_pg' and not changed:
                raise AssertionError('LoRA did not update.')
            if label == 'selected':
                identity[arm] = all(torch.equal(value,cp['model'][name])
                                    for name,value in reference['model'].items())
            audits[arm][label] = dict(step=cp['step'],original_tensor_count=len(original),
                original_and_norms_exact=True,changed_lora_tensors=changed,selected_pg_exact_outside_I=fixed)
            if cp is not selected_pg:
                del cp
                gc.collect()
        del selected_pg
        gc.collect()
    protected = read(args.output/'preserved_hashes.json')
    if any(file_hash(path) != digest for path,digest in protected.items()):
        raise AssertionError('Prior protected input changed.')
    hashes = read(args.output/'source_hashes.json')
    if any(file_hash(ROOT/name) != digest or file_hash(args.output/'source'/name) != digest
           for name,digest in hashes.items()):
        raise AssertionError('Source snapshot changed.')
    for arm in completed:
        if any(file_hash(args.output/arm/name) != digest for name,digest in inherited[arm].items()):
            raise AssertionError('Inherited completed arm changed.')
    write_json(args.output/'tensor_audit.json',dict(seed=42,checkpoints=audits,
        all_original422_and_norms_exact=True,all_non_I_frozen_during_gate=True,
        selected_tensor_identity_to_reference=identity,test_outcome_reads=0))
    from run_mobile_active_experiment import best_policy
    selected = best_policy(validation)
    frozen = dict(seed=42,selected_policy=selected,selected_before_test=True,
        manifest_sha256=file_hash(args.output/'manifest.json'),
        checkpoint_sha256={arm:file_hash(args.output/arm/'selected.pt') for arm in ARMS},
        reference_path=str(reference_path),reference_sha256=file_hash(reference_path),
        validation_metrics=validation,selected_tensor_identity_to_reference=identity,
        new_test_required=not all(identity.values()),
        test_references=old['test_references'],test_outcomes_collected=False,
        frozen_at=datetime.now(ZONE).isoformat())
    write_json(args.output/'frozen_for_test.json',frozen)
    phases['startup_io_selection_audit_recovery'] = time.perf_counter()-begin-pg['seconds']-gate['seconds']
    report = dict(seed=42,selected_policy=selected,training=training,validation=validation,
        phase_seconds=phases,actual_recovery_seconds=time.perf_counter()-begin,
        interrupted_completed_arm_seconds=sum(training[a]['pg']['seconds']+training[a]['gate']['seconds'] for a in completed),
        test_outcome_reads=0,all_original422_and_norms_exact=True,
        selected_tensor_identity_to_reference=identity,new_test_required=frozen['new_test_required'],
        status='training_complete_fresh_test_pending' if frozen['new_test_required'] else
               'training_complete_same_policy_no_new_test')
    write_json(args.output/'report.json',report)
    receipt.update(status='complete',exit_code=0,actual_seconds=report['actual_recovery_seconds'],
        completed_at=datetime.now(ZONE).isoformat(),phase_seconds=phases)
    write_json(args.output/'receipt.json',receipt)
    print(json.dumps(dict(status=report['status'],selected=selected,
        validation=validation,identity=identity,seconds=report['actual_recovery_seconds'])),flush=True)


if __name__ == '__main__':
    if os.environ.get('PYTHONHASHSEED') != '42':
        raise SystemExit('Set PYTHONHASHSEED=42 before launching recovery.')
    try:
        main()
    except Exception as error:
        import sys
        if '--output' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--output')+1])/'receipt.json'
            if path.exists():
                receipt=read(path)
                if receipt.get('pid') == os.getpid():
                    receipt.update(status='failed',exit_code=1,error=repr(error))
                    write_json(path,receipt)
        raise
