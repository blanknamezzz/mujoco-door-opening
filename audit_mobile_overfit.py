"""Audit frozen mobile policies on development splits without reading test outcomes."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from run_mobile_reward_experiment import ROOT, ZONE, file_hash, write_json
from run_mobile_retention import read


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--estimate-seconds',type=int,required=True)
    args=parser.parse_args()
    args.training=args.training.resolve();args.output=args.output.resolve()
    if ROOT/'runs' not in args.output.parents or (args.output.exists() and any(args.output.iterdir())):
        raise ValueError('A new ignored runs directory is required.')
    args.output.mkdir(parents=True,exist_ok=True)
    begin,now=time.perf_counter(),datetime.now(ZONE)
    receipt=dict(pid=os.getpid(),seed=42,status='starting',started_at=now.isoformat(),
        queue_estimate_seconds=0,runtime_estimate_seconds=args.estimate_seconds,
        expected_completion=(now+timedelta(seconds=args.estimate_seconds)).isoformat(),
        scope='development-only train/validation audit of two frozen policies',test_outcome_reads=0)
    write_json(args.output/'receipt.json',receipt)
    from experiment_reproducibility import seed_everything
    receipt['reproducibility']=seed_everything(42)
    import torch
    torch.set_num_threads(4)
    protocol=read(args.training/'protocol.json')
    selection=read(args.training/'frozen_for_test.json')
    if file_hash(args.training/'manifest.json')!=selection['manifest_sha256']:
        raise AssertionError('Frozen manifest changed.')
    manifest=read(args.training/'manifest.json')
    scenespec=[s for s in manifest['scenes'] if s['split'] in ('train','val')]
    if len(scenespec)!=108 or sum(s['split']=='train' for s in scenespec)!=72:
        raise AssertionError('Unexpected development split.')
    from mobile_crossed_v2 import geometry_identity
    identities={split:{geometry_identity(s) for s in manifest['scenes'] if s['split']==split}
                for split in ('train','val','test')}
    groups={split:{s['group_id'] for s in manifest['scenes'] if s['split']==split}
            for split in ('train','val','test')}
    if any(identities[a]&identities[b] or groups[a]&groups[b]
           for a,b in (('train','val'),('train','test'),('val','test'))):
        raise AssertionError('Split leakage by geometry or group.')
    from run_mobile_active_experiment import load_scenes,summarize
    scenes=load_scenes(scenespec,Path(protocol['development']))
    for scene in scenes:
        scene['maps']=read(Path(protocol['maps'])/'data'/scene['spec']['scene_id']/'maps.json')
    from mobile_partial_policy_v5 import load_partial_checkpoint
    from run_mobile_inspection_regret import prepare,evaluate
    results={};checkpoints={};phase={}
    for arm in ('standard','retention'):
        tick=time.perf_counter()
        path=args.training/arm/'selected.pt'
        if file_hash(path)!=selection['checkpoint_sha256'][arm]:
            raise AssertionError('Frozen checkpoint changed: '+arm)
        model,source=load_partial_checkpoint(path,'cuda')
        model.requires_grad_(False)
        folder=args.output/arm;folder.mkdir()
        data=prepare(SimpleNamespace(output=folder),model,source,scenes)
        rows={};decisions={}
        for split in ('train','val'):
            subset=[s for s in scenes if s['spec']['split']==split]
            summary,choice=evaluate(model,subset,data,arm)
            rows[split]=summary;decisions[split]=choice
        layout={split:{name:summarize([d for d in decisions[split] if d['layout']==name])
                       for name in ('front_blocked','rear_blocked','both_blocked','free')}
                for split in ('train','val')}
        results[arm]=dict(summary=rows,by_layout=layout,
            train_val_correct_rate_gap=rows['train']['correct_decisions']/72-rows['val']['correct_decisions']/36,
            train_val_net_utility_gap=rows['train']['mean_net_utility']-rows['val']['mean_net_utility'])
        write_json(folder/'decisions.json',decisions)
        checkpoints[arm]=file_hash(path)
        phase[arm]=time.perf_counter()-tick
        print(json.dumps(dict(arm=arm,seconds=phase[arm],train=rows['train'],val=rows['val'])),flush=True)
        del model,source,data
        gc.collect();torch.cuda.empty_cache()
    report=dict(seed=42,scope='development-only frozen-policy audit',
        training_run=str(args.training),training_selection_sha256=file_hash(args.training/'frozen_for_test.json'),
        manifest_sha256=file_hash(args.training/'manifest.json'),checkpoint_sha256=checkpoints,
        scenes={split:sum(s['split']==split for s in manifest['scenes']) for split in ('train','val','test')},
        groups={k:len(v) for k,v in groups.items()},geometry_overlap=0,group_overlap=0,
        test_outcome_reads=0,results=results,phase_seconds=phase,
        actual_seconds=time.perf_counter()-begin)
    write_json(args.output/'report.json',report)
    receipt.update(status='complete',exit_code=0,actual_seconds=report['actual_seconds'],
        completed_at=datetime.now(ZONE).isoformat())
    write_json(args.output/'receipt.json',receipt)


if __name__=='__main__':
    if os.environ.get('PYTHONHASHSEED')!='42':
        raise SystemExit(subprocess.run([sys.executable,*sys.argv],cwd=ROOT,
            env=dict(os.environ,PYTHONHASHSEED='42',CUBLAS_WORKSPACE_CONFIG=':4096:8')).returncode)
    try:main()
    except Exception as error:
        if '--output' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--output')+1])/'receipt.json'
            if path.exists():
                receipt=json.loads(path.read_text(encoding='utf-8'))
                if receipt.get('pid')==os.getpid():
                    receipt.update(status='failed',exit_code=1,error=repr(error))
                    write_json(path,receipt)
        raise
