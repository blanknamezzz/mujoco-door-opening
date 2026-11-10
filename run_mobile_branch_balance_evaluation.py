"""Frozen v8 test with explicit equivalent-policy aliases and live execution.

Aliases share evidence only after exact weights, normalization, inference
configuration and observation mode match. Distinct encoders never share caches.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime,timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from run_mobile_reward_experiment import ROOT,ZONE,write_json,file_hash
from mobile_branch_balance_v8 import ARMS
from run_mobile_inspection_regret_evaluation import validate_test_manifest

REFERENCES=('frozen_v6_learned','frozen_v6_always','frozen_v6_never')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def inference_digest(checkpoint):
    """Hash all inference tensors/config, excluding two training-only records."""
    import torch
    digest=hashlib.sha256()
    config={k:v for k,v in checkpoint['config'].items() if k not in ('branch_balance_v8','inspection_regret_v6')}
    digest.update(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    items=list(sorted(checkpoint['model'].items()))+[(key,checkpoint[key]) for key in ('state_mean','state_std')]
    for name,value in items:
        if not isinstance(value,torch.Tensor):raise ValueError('Tensor checkpoint required.')
        value=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_policies(training,reference_path):
    return {**{arm:dict(model=str((training/arm).resolve()),mode='learned',risk=1.) for arm in ARMS},
            **{name:dict(model=str(reference_path.parent.resolve()),mode=mode,risk=1.)
               for name,mode in zip(REFERENCES,('learned','always','never'))}}


def coalesce_policies(policies,signatures,primary):
    if primary not in policies:raise ValueError('Frozen primary required.')
    seen={};representatives={};aliases={}
    for name in [primary,*[key for key in policies if key!=primary]]:
        config=policies[name]
        if config['mode'] not in ('never','always','learned') or config['risk']!=1.:
            raise ValueError('Only declared risk1 neural observation modes supported.')
        key=(signatures[config['model']],config['mode'],config['risk'])
        if key not in seen:seen[key]=name;representatives[name]=config
        aliases[name]=seen[key]
    return representatives,{name:aliases[name] for name in policies}


def evaluate_unique(scenes,policies):
    import torch
    from mobile_partial_policy_v5 import load_partial_checkpoint
    from run_mobile_partial_training import predict,evaluate
    metrics={};decisions={}
    for folder in dict.fromkeys(config['model'] for config in policies.values()):
        model,source=load_partial_checkpoint(Path(folder)/'selected.pt','cuda');model.requires_grad_(False)
        # v8 may change LoRA, P and G. Cache only within this exact checkpoint.
        cached=predict(model,source,scenes)
        subset={name:config for name,config in policies.items() if config['model']==folder}
        sums,choices=evaluate(model,scenes,cached,subset);metrics.update(sums);decisions.update(choices)
        del model,source,cached;gc.collect();torch.cuda.empty_cache()
    return metrics,decisions


def checkpoint_audit(args,selection):
    import torch
    baseline=torch.load(args.initial,map_location='cpu',weights_only=True)
    fixed=[name for name in baseline['model'] if 'lora_' not in name and not name.startswith(
        ('position_policy.','grasp_policy.','inspection_policy.'))]
    assert len(fixed)==422
    audits={};signatures={}
    for arm in ARMS:
        pg=torch.load(args.training/arm/'selected_pg.pt',map_location='cpu',weights_only=True);audits[arm]={}
        for label in ('selected_pg','last_pg','selected','last'):
            cp=pg if label=='selected_pg' else torch.load(args.training/arm/(label+'.pt'),map_location='cpu',weights_only=True)
            assert all(torch.equal(baseline['model'][name],cp['model'][name]) for name in fixed)
            assert all(torch.equal(baseline[key],cp[key]) for key in ('state_mean','state_std'))
            fixed_I=all(torch.equal(value,cp['model'][name]) for name,value in pg['model'].items()
                        if not name.startswith('inspection_policy.')) if label in ('selected','last') else None
            if fixed_I is not None:assert fixed_I
            audits[arm][label]=dict(step=cp['step'],original422_and_normalization_exact=True,
                selected_pg_exact_outside_I=fixed_I)
            if label=='selected':signatures[str((args.training/arm).resolve())]=inference_digest(cp)
            if cp is not pg:del cp
            gc.collect()
        del pg;gc.collect()
    cp=torch.load(args.reference,map_location='cpu',weights_only=True)
    assert all(torch.equal(baseline['model'][name],cp['model'][name]) for name in fixed)
    assert all(torch.equal(baseline[key],cp[key]) for key in ('state_mean','state_std'))
    signatures[str(args.reference.parent.resolve())]=inference_digest(cp)
    del cp,baseline;gc.collect()
    return dict(seed=42,checkpoints=audits,original_tensor_count=422,
                all_original422_and_norms_exact=True,all_non_I_exact_during_I=True,
                checkpoint_inference_signatures=signatures),signatures


def physical_audit(args,scenes,frozen,protected,sources,representatives):
    begin=time.perf_counter()
    assert all(file_hash(path)==digest for path,digest in protected.items())
    assert all(file_hash(ROOT/name)==file_hash(args.output/'source'/name)==digest for name,digest in sources.items())
    assert file_hash(args.output/'manifest.json')==frozen['evaluation_manifest_sha256']
    groups=defaultdict(set);layouts=defaultdict(set)
    for s in read(args.output/'manifest.json')['scenes']:
        groups[s['group_id']].add(s['split']);layouts[s['group_id']].add(s['layout'])
    assert all(len(value)==1 for value in groups.values())
    assert all(value=={'front_blocked','rear_blocked','both_blocked','free'} for value in layouts.values())
    count=entries=inspections=forced=0;speed=step=0.;collisions=[];candidate_counts=Counter()
    for scene in scenes:
        sid=scene['spec']['scene_id'];full=scene['full_sweep']['mask']
        assert (args.output/'evaluation_frozen.json').stat().st_mtime < (args.output/'data'/sid/'0/observations.npz').stat().st_mtime
        for branch in (0,1):
            for r in scene['branches'][branch]['records']:
                assert r['base_reset_count']==1 and r['maximum_sampled_base_speed_m_s']<.31 and r['max_trace_step_m']<.02
                if r['opening_tested']:assert r['result']['continuous_state_at_skill_entry']
                kind='success' if r['success'] else 'collision' if r['collision'] else 'other'
                candidate_counts[('full_allowed_' if full[r['base_id']] else 'full_blocked_')+kind]+=1
    for name in representatives:
        verification=read(args.output/'policy_results'/name/'replay_verification.json')
        assert verification['count']==len(scenes) and verification['all_match']
        for scene in scenes:
            folder=args.output/'policy_results'/name/'replays'/scene['spec']['scene_id']
            item=read(folder/'result.json');r=item['record'];trace=read(folder/'trace.json');logs=read(folder/'obs.json')
            assert item['match'] and item['map_masks_match'] and r['base_reset_count']==1
            assert logs[0]['stage']=='initial' and logs[0]['map']['frames']==1
            assert all(b['time_s']>=a['time_s']-1e-8 for a,b in zip(trace,trace[1:]))
            if r['opening_tested']:assert r['result']['continuous_state_at_skill_entry'];entries+=1
            if r['inspection_attempted']:inspections+=1
            if r['inspection_success'] and not r['inspection_unsafe_abort']:
                assert all(abs(t['door_angle_deg'])<.5 for t in trace if t['phase']=='inspection')
                assert len(logs)==2 and logs[1]['map']['frames']==2
            if r['inspection_unsafe_abort']:assert not r['opening_tested'] and len(logs)==1
            if not any(logs[0]['map']['mask']):assert r['refused'] and not r['inspection_attempted'];forced+=1
            if not r['refused'] and not r['inspection_unsafe_abort']:assert logs[-1]['map']['mask'][r['base_id']]
            speed=max(speed,max(np.linalg.norm(t['base_velocity'][:2]) for t in trace));step=max(step,r['max_trace_step_m'])
            if r['collision']:collisions.append(dict(policy=name,scene_id=scene['spec']['scene_id'],failure_stage=r['failure_stage'],stop_reason=r['result'].get('stop_reason')))
            count+=1
    assert count==len(scenes)*len(representatives) and speed<.31 and step<.02
    assert sum(candidate_counts.values())==16*len(scenes)
    result=dict(seed=42,original_tensor_count=422,all16_checkpoints_original422_and_norms_exact=True,
        all_stage_I_continuations_exact=True,all_protected_sources_models_and_data_exact=True,
        primary_policy=frozen['primary_policy'],test_scenes=len(scenes),test_groups=len(scenes)//4,
        logical_policies=len(frozen['policies']),unique_executed_policies=len(representatives),
        aliases=frozen['aliases'],candidate_episodes=len(scenes)*16,physical_replays=count,
        all_replays_match=True,all_maps_from_current_frames=True,episode_reset_counts_one=True,
        opening_entry_continuity_checks=entries,actual_inspections=inspections,forced_initial_refusals=forced,
        max_sampled_base_speed_m_s=float(speed),max_recorded_step_m=float(step),
        full_map_candidate_counts=dict(candidate_counts),collisions=collisions,seconds=time.perf_counter()-begin)
    write_json(args.output/'tensor_and_physics_audit.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--training',type=Path,default=ROOT/'runs/p8_train_seed42_20261110')
    p.add_argument('--manifest',type=Path);p.add_argument('--smoke',action='store_true')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--estimate-seconds',type=int,required=True)
    args=p.parse_args()
    for name in ('output','training'):setattr(args,name,getattr(args,name).resolve())
    args.manifest=(args.manifest or args.training/'manifest.json').resolve()
    if ROOT/'runs' not in args.output.parents:raise ValueError('New ignored runs/ output required.')
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Preserve existing runs.')
    if min(args.workers,args.estimate_seconds)<1:raise ValueError('Positive budgets required.')
    args.output.mkdir(parents=True,exist_ok=True);begin=time.perf_counter();now=datetime.now(ZONE)
    receipt=dict(pid=os.getpid(),seed=42,status='starting',started_at=now.isoformat(),queue_estimate_seconds=0,
        runtime_estimate_seconds=args.estimate_seconds,expected_completion=(now+timedelta(seconds=args.estimate_seconds)).isoformat(),
        scope='pipeline smoke' if args.smoke else 'formal frozen v8 independent test; equivalent aliases share observations')
    write_json(args.output/'receipt.json',receipt)
    from experiment_reproducibility import seed_everything,worker_environment
    receipt['reproducibility']=seed_everything(42);os.environ.update(worker_environment(42))
    import torch
    torch.set_num_threads(4)
    training_protocol=read(args.training/'protocol.json');selection=read(args.training/'frozen_for_test.json')
    args.initial=Path(training_protocol['initial']);args.reference=Path(selection['reference_path'])
    assert tuple(selection['test_references'])==REFERENCES
    assert file_hash(args.training/'manifest.json')==selection['manifest_sha256']
    assert file_hash(args.initial)==training_protocol['initial_sha256']
    assert file_hash(args.reference)==selection['reference_sha256']
    protected=read(args.training/'preserved_hashes.json').copy()
    assert all(file_hash(path)==digest for path,digest in protected.items())
    for name in ('manifest.json','protocol.json','frozen_for_test.json','report.json','tensor_audit.json','source_hashes.json'):
        path=args.training/name;protected[str(path)]=file_hash(path)
    for arm in ARMS:
        for label in ('selected_pg','last_pg','selected','last'):
            path=args.training/arm/(label+'.pt');protected[str(path)]=file_hash(path)
        assert protected[str(args.training/arm/'selected.pt')]==selection['checkpoint_sha256'][arm]
    for name,digest in read(args.training/'source_hashes.json').items():
        assert file_hash(ROOT/name)==file_hash(args.training/'source'/name)==digest
    tick=time.perf_counter();weights,signatures=checkpoint_audit(args,selection);weight_seconds=time.perf_counter()-tick
    formal=read(args.training/'manifest.json');manifest=read(args.manifest)
    specs=validate_test_manifest(manifest,formal,args.smoke)
    policies=make_policies(args.training,args.reference)
    representatives,aliases=coalesce_policies(policies,signatures,selection['selected_policy'])
    import run_mobile_partial_evaluation as engine
    engine.MODEL_POLICIES=representatives;engine.POLICY_NAMES=tuple(representatives)
    longest=max(len(str(args.output/'policy_results'/name/'replays'/s['scene_id']/file))
        for name in representatives for s in specs for file in ('result.json.tmp','trace.json.tmp','obs.json.tmp'))
    if os.name=='nt' and longest>=250:raise ValueError('Choose shorter output paths.')
    write_json(args.output/'manifest.json',manifest)
    source=args.output/'source';source.mkdir()
    for name in ['AGENTS.md','scene.xml','.gitignore',*sorted(p.name for p in ROOT.glob('*.py'))]:shutil.copy2(ROOT/name,source/name)
    sources={path.name:file_hash(path) for path in source.iterdir()}
    write_json(args.output/'source_hashes.json',sources);write_json(args.output/'preserved_hashes.json',protected)
    write_json(args.output/'pretest_checkpoint_audit.json',weights)
    frozen=dict(seed=42,primary_policy=selection['selected_policy'],selected_before_test=True,
        training_selection_sha256=file_hash(args.training/'frozen_for_test.json'),
        checkpoint_sha256=selection['checkpoint_sha256'],reference_sha256=selection['reference_sha256'],
        evaluation_manifest_sha256=file_hash(args.output/'manifest.json'),policies=policies,
        representatives=representatives,aliases=aliases,inference_signatures=signatures,
        smoke=args.smoke,test_outcomes_available=False,frozen_at=datetime.now(ZONE).isoformat())
    write_json(args.output/'evaluation_frozen.json',frozen)
    write_json(args.output/'protocol.json',dict(seed=42,training=str(args.training),initial=str(args.initial),
        policies=policies,aliases=aliases,primary_policy=frozen['primary_policy'],workers=args.workers,learning_updates=0,
        physical_replays=len(specs)*len(representatives),logical_rows=len(policies),
        alias_rule='exact tensor+normalization+inference-config fingerprint and same mode/risk; primary preferred; not independent replicates',
        cache_rule='separate predictions for every distinct checkpoint; no sharing between updated encoders',
        physical_implementation='unchanged v5 candidate collection and live RGBD replay',
        information_scope='current frames and known CAD for opening; known-map navigation/arm preparation'))
    receipt.update(status='collecting',test_scenes=len(specs),unique_policies=len(representatives),
        expected_physical_replays=len(specs)*len(representatives),maximum_planned_output_path_chars=longest,
        startup_seconds=time.perf_counter()-begin)
    write_json(args.output/'receipt.json',receipt);phases={'pretest_checkpoint_audit':weight_seconds}
    phases['test_collection']=engine.collect(args,specs)
    from run_mobile_active_experiment import load_scenes
    from run_mobile_outcome_experiment import guard_scene
    scenes=load_scenes(specs,args.output)
    for scene in scenes:
        scene['maps']=read(args.output/'data'/scene['spec']['scene_id']/'maps.json');scene['full_sweep']=guard_scene(scene['spec'])
    write_json(args.output/'test_full_sweep_queries.json',{s['spec']['scene_id']:s['full_sweep'] for s in scenes})
    tick=time.perf_counter();metrics,decisions=evaluate_unique(scenes,representatives)
    phases['cached_policy_evaluation']=time.perf_counter()-tick;replays={}
    for name in representatives:
        write_json(args.output/'policy_results'/name/'test.json',dict(summary=metrics[name],decisions=decisions[name],independent_replay=True))
        replays[name]=engine.live_replay(args,name,scenes,decisions[name]);phases['replay_'+name]=replays[name]['seconds']
    for name,representative in aliases.items():
        if name==representative:continue
        metrics[name]=deepcopy(metrics[representative])
        decisions[name]=[dict(item,policy=name,execution_alias=representative) for item in decisions[representative]]
        write_json(args.output/'policy_results'/name/'test.json',dict(summary=metrics[name],decisions=decisions[name],
            independent_replay=False,execution_alias=representative))
        write_json(args.output/'policy_results'/name/'replay_verification.json',dict(logical_scenes=len(specs),
            physical_replay_count=0,all_match=True,inference_identity_verified=True,execution_alias=representative,
            source_verification=str(args.output/'policy_results'/representative/'replay_verification.json')))
    verification=physical_audit(args,scenes,frozen,protected,sources,representatives)
    phases['tensor_and_physics_audit']=verification['seconds']
    from check_git_size import audit as size_audit
    size=size_audit();write_json(args.output/'git_size_audit.json',size);assert size['passed']
    phases['startup_io_and_other']=time.perf_counter()-begin-sum(phases.values())
    report=dict(seed=42,smoke=args.smoke,primary_policy=frozen['primary_policy'],primary_selected_before_test=True,
        test={name:metrics[name] for name in policies},replays=replays,aliases=aliases,logical_policies=len(policies),
        unique_executed_policies=len(representatives),physical_replays=verification['physical_replays'],
        phase_seconds=phases,seconds=time.perf_counter()-begin,learning_updates=0,all_replays_match=True,
        all16_checkpoints_original422_and_norms_exact=True,all_stage_I_continuations_exact=True,
        frozen_weights_and_sources_exact=True,scope=receipt['scope'],
        limitation='one seed; aliases share evidence; reused development data; known-map navigation')
    write_json(args.output/'report.json',report)
    receipt.update(status='complete',exit_code=0,actual_seconds=report['seconds'],phase_seconds=phases,
                   completed_at=datetime.now(ZONE).isoformat());write_json(args.output/'receipt.json',receipt)
    print(json.dumps(dict(status='complete',seconds=report['seconds'],primary=report['primary_policy'],
        aliases=aliases,physical_replays=report['physical_replays'],test=report['test'])),flush=True)


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
