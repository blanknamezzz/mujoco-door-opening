"""Train/validation-only partial-map calibration, never read test outcomes."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from run_mobile_reward_experiment import ROOT,ZONE,write_json,file_hash
from mobile_partial_fusion_v7 import PROFILES,FusedPartialDoorMap


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def counts(scene,maps,full,records):
    result=Counter()
    for key,branch in (('initial',0),('fused',1)):
        mask=maps[key]['mask']
        result[key+'_extra_exclusions_vs_full']+=sum(a and not b for a,b in zip(full,mask))
        result[key+'_full_blocks_allowed']+=sum(not a and b for a,b in zip(full,mask))
        for r in records[branch]:
            result[key+'_successful_candidates_vetoed']+=int(r['success'] and not mask[r['base_id']])
            result[key+'_collision_candidates_allowed']+=int(r['collision'] and mask[r['base_id']])
    return result


def selection_key(result,initial_collision_limit):
    # Initial hidden obstacles are unknown, not oracle-visible. Require no
    # increase over baseline there before optimizing complete fused evidence.
    return (int(result['initial_collision_candidates_allowed']<=initial_collision_limit),
        -result['fused_collision_candidates_allowed'],-result['fused_successful_candidates_vetoed'],
        -result['fused_extra_exclusions_vs_full'],-result['fused_full_blocks_allowed'],
        -result['initial_successful_candidates_vetoed'])


def validation_pg_diagnostic(args,protocol,protected):
    """Inspect both branches of existing PG snapshots on validation only."""
    import gc
    from types import SimpleNamespace
    import torch
    from run_mobile_outcome_experiment import attach_development
    from run_mobile_partial_training import predict,evaluate
    from mobile_partial_policy_v5 import load_partial_checkpoint
    prior=Path(protocol['prior_training']);development=Path(protocol['development']);maps=Path(protocol['maps'])
    specs=[s for s in read(args.training/'manifest.json')['scenes'] if s['split']=='val']
    scenes=attach_development(SimpleNamespace(output=args.output,development=development),specs)
    protected.update(read(args.output/'reused_development_hashes.json'))
    for scene in scenes:scene['maps']=read(maps/'data'/scene['spec']['scene_id']/'maps.json')
    begin=time.perf_counter();result={}
    for label in ('selected_pg','last_pg'):
        path=prior/'depth_features'/(label+'.pt');protected[str(path)]=file_hash(path)
        model,checkpoint=load_partial_checkpoint(path,'cuda');model.requires_grad_(False)
        policies={label+'_'+mode:dict(mode=mode,risk=1.) for mode in ('never','always')}
        summaries,decisions=evaluate(model,scenes,predict(model,checkpoint,scenes),policies)
        result[label]=dict(step=checkpoint['step'],metrics=summaries,checkpoint_sha256=protected[str(path)])
        write_json(args.output/('validation_'+label+'.json'),dict(summary=summaries,decisions=decisions))
        del model,checkpoint;gc.collect();torch.cuda.empty_cache()
    result.update(seconds=time.perf_counter()-begin,split='validation',counterfactual_only=True,
                  new_physical_replays=0,weight_updates=0,test_outcome_reads=0)
    write_json(args.output/'validation_pg_diagnostic.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--training',type=Path,default=ROOT/'runs/p6_train2_seed42_20261110')
    p.add_argument('--estimate-seconds',type=int,required=True)
    args=p.parse_args();args.output=args.output.resolve();args.training=args.training.resolve()
    if ROOT/'runs' not in args.output.parents:raise ValueError('New ignored run required.')
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Preserve old runs.')
    args.output.mkdir(parents=True,exist_ok=True);begin=time.perf_counter();now=datetime.now(ZONE)
    receipt=dict(pid=os.getpid(),seed=42,status='starting',started_at=now.isoformat(),queue_estimate_seconds=0,
        runtime_estimate_seconds=args.estimate_seconds,expected_completion=(now+timedelta(seconds=args.estimate_seconds)).isoformat())
    write_json(args.output/'receipt.json',receipt)
    from experiment_reproducibility import seed_everything
    receipt['reproducibility']=seed_everything(42)
    protocol=read(args.training/'protocol.json');development=Path(protocol['development']);prefixes=Path(protocol['maps'])
    # The source geometry and actual arrival frames are the unchanged v3 data.
    for name,digest in read(args.training/'source_hashes.json').items():
        if file_hash(ROOT/name)!=digest:raise ValueError('Frozen source differs: '+name)
    specs=[s for s in read(args.training/'manifest.json')['scenes'] if s['split'] in ('train','val')]
    if any(s['split']=='test' for s in specs):raise AssertionError('Test outcomes excluded.')
    write_json(args.output/'manifest.json',dict(seed=42,scenes=specs,
        counts={split:sum(s['split']==split for s in specs) for split in ('train','val')},test_outcome_reads=0))
    source=args.output/'source';source.mkdir()
    for name in ['AGENTS.md','scene.xml','.gitignore',*sorted(p.name for p in ROOT.glob('*.py'))]:shutil.copy2(ROOT/name,source/name)
    hashes={path.name:file_hash(path) for path in source.iterdir()};write_json(args.output/'source_hashes.json',hashes)
    write_json(args.output/'protocol.json',dict(seed=42,profiles=['historical',*PROFILES],
        selection='validation only: no more initial collision candidates allowed than historical; fewer fused allowed collisions, vetoed successes, extra exclusions, missed full blocks, initial vetoed successes; declared order ties',
        model_updates=0,test_outcome_reads=0,prior='only obstacle-free CAD; actual train/val reached RGBD',
        paper='https://www.arminhornung.de/Research/pub/hornung13auro.pdf',
        implementation='deterministic endpoint union and optional free-ray pruning, not OctoMap or probabilistic occupancy',
        source_training=str(args.training),development=str(development),prefixes=str(prefixes),
        auxiliary_diagnostic='Compare prior PG step0/320 on both direct and always-inspect validation branches; no training or new physical replay'))
    protected={};totals={name:{split:Counter() for split in ('train','val')} for name in ('historical',*PROFILES)}
    timings=[];perception={split:Counter() for split in ('train','val')}
    from mobile_crossed_v2 import restore_scene
    from mobile_partial_map_v5 import cad_prior
    from run_mobile_outcome_experiment import guard_scene
    receipt.update(status='calibrating',startup_seconds=time.perf_counter()-begin)
    write_json(args.output/'receipt.json',receipt)
    for index,spec in enumerate(specs):
        tick=time.perf_counter();sid=spec['scene_id'];split=spec['split']
        folder=prefixes/'data'/sid;path=folder/'maps.json';historical=read(path);protected[str(path)]=file_hash(path)
        path=folder/'reached_rgbd.npz';protected[str(path)]=file_hash(path)
        with np.load(path) as arrays:raw={k:v.copy() for k,v in arrays.items()}
        records={}
        for branch in (0,1):
            path=development/'data'/sid/str(branch)/'records.json';protected[str(path)]=file_hash(path);records[branch]=read(path)
            for r in records[branch]:
                perception[split]['candidates']+=1;perception[split]['success']+=int(r['success'])
                if r['failure_stage']=='perception':perception[split][r['result'].get('stop_reason','unknown')]+=1
        full=guard_scene(spec)['mask'];maps={'historical':historical}
        for profile in PROFILES:
            mapper=FusedPartialDoorMap(cad_prior(restore_scene(spec)),profile)
            def add(stage):mapper.add_frame({key:raw[stage+'_'+key] for key in ('rgb','depth_m','K','world_from_camera')},raw[stage+'_proprio'])
            add('initial');initial=mapper.query()
            if historical['inspection']['success']:add('post')
            fused=mapper.query();maps[profile]=dict(initial=initial,fused=fused)
        diagnostics={name:dict(counts(spec,mapping,full,records)) for name,mapping in maps.items()}
        for name in maps:totals[name][split].update(diagnostics[name])
        write_json(args.output/'data'/sid/'maps.json',dict(scene_id=sid,split=split,full_sweep_mask=full,
            profiles=maps,diagnostics=diagnostics,source_rgbd_sha256=protected[str(folder/'reached_rgbd.npz')]))
        timings.append(time.perf_counter()-tick)
        if index==0 or (index+1)%12==0:
            progress=dict(pid=os.getpid(),phase='map_calibration',completed_scenes=index+1,total_scenes=len(specs),
                last_scene_seconds=timings[-1],mean_scene_seconds=float(np.mean(timings)),elapsed_seconds=time.perf_counter()-begin)
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    baseline=totals['historical']['val'];selected='historical';best=selection_key(baseline,baseline['initial_collision_candidates_allowed'])
    for name in PROFILES:
        key=selection_key(totals[name]['val'],baseline['initial_collision_candidates_allowed'])
        if key>best:selected=name;best=key
    pg_diagnostic=validation_pg_diagnostic(args,protocol,protected)
    assert all(file_hash(path)==digest for path,digest in protected.items())
    assert all(file_hash(ROOT/name)==file_hash(source/name)==digest for name,digest in hashes.items())
    write_json(args.output/'protected_inputs.json',protected)
    selection=dict(seed=42,selected_profile=selected,selected_on='validation only',
        validation_diagnostics={name:dict(value['val']) for name,value in totals.items()},test_outcome_reads=0,
        frozen_at=datetime.now(ZONE).isoformat(),future_training_and_fresh_test_required=True)
    write_json(args.output/'selection.json',selection)
    report=dict(seed=42,totals={name:{split:dict(value) for split,value in splits.items()} for name,splits in totals.items()},
        perception={split:dict(value) for split,value in perception.items()},selected_profile=selected,
        scene_seconds=timings,startup_seconds=receipt['startup_seconds'],seconds=time.perf_counter()-begin,
        test_outcome_reads=0,model_updates=0,protected_inputs_exact=True,all_sources_exact=True,
        validation_pg_diagnostic=pg_diagnostic)
    write_json(args.output/'report.json',report)
    receipt.update(status='complete',exit_code=0,actual_seconds=report['seconds'],completed_at=datetime.now(ZONE).isoformat())
    write_json(args.output/'receipt.json',receipt)
    print(json.dumps(dict(status='complete',seconds=report['seconds'],selection=selection)),flush=True)


if __name__=='__main__':
    if os.environ.get('PYTHONHASHSEED')!='42':
        raise SystemExit(subprocess.run([sys.executable,*sys.argv],cwd=ROOT,
            env=dict(os.environ,PYTHONHASHSEED='42',CUBLAS_WORKSPACE_CONFIG=':4096:8')).returncode)
    try:main()
    except Exception as error:
        if '--output' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--output')+1])/'receipt.json'
            if path.exists():
                receipt=read(path)
                if receipt.get('pid')==os.getpid():
                    receipt.update(status='failed',exit_code=1,error=repr(error));write_json(path,receipt)
        raise
