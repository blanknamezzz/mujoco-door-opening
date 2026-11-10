"""Frozen v6 objective ablation with live v5 RGB-D execution, seed 42.

Reuses the unchanged v5 physical collection and live-replay implementations.
All six policies have identical frozen P/G and encoder; only I can differ.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timedelta
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from run_mobile_reward_experiment import ROOT,ZONE,write_json,file_hash
from mobile_inspection_regret_v6 import ARMS

REFERENCES=('frozen_v5_learned','frozen_v5_always','frozen_v5_never')


def read(path):
    import json
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def make_policies(training,prior):
    return {**{arm:dict(model=str((training/arm).resolve()),mode='learned',risk=1.) for arm in ARMS},
            **{name:dict(model=str((prior/'depth_features').resolve()),mode=mode,risk=1.)
               for name,mode in zip(REFERENCES,('learned','always','never'))}}


def assert_same_continuation(baseline,checkpoint):
    import torch
    if set(baseline['model'])!=set(checkpoint['model']):raise AssertionError('Architecture changed.')
    assert all(torch.equal(value,checkpoint['model'][name]) for name,value in baseline['model'].items()
               if not name.startswith('inspection_policy.')), 'Non-I tensor changed.'
    assert all(torch.equal(baseline[key],checkpoint[key]) for key in ('state_mean','state_std'))
    base_config={k:v for k,v in baseline['config'].items() if k!='inspection_regret_v6'}
    other_config={k:v for k,v in checkpoint['config'].items() if k!='inspection_regret_v6'}
    if base_config!=other_config:raise AssertionError('Continuation configuration changed.')


def validate_test_manifest(manifest,formal,smoke):
    tests=[s for s in manifest['scenes'] if s['split']=='test']
    if not tests:raise ValueError('Empty test manifest.')
    identities={s['geometry_sha256'] for s in tests}
    if len(identities)!=len(tests):raise ValueError('Duplicate test geometry.')
    formal_ids={s['geometry_sha256'] for s in formal['scenes'] if s['split']=='test'}
    if smoke and identities & formal_ids:raise ValueError('Smoke overlaps frozen formal test.')
    if not smoke and manifest!=formal:raise ValueError('Formal test manifest must match training freeze.')
    return tests


def evaluate(args,scenes,policies):
    import torch
    from mobile_partial_policy_v5 import load_partial_checkpoint
    from run_mobile_partial_training import predict,evaluate as evaluate_partial
    metrics={};decisions={};cached=None
    for folder in dict.fromkeys(config['model'] for config in policies.values()):
        model,source=load_partial_checkpoint(Path(folder)/'selected.pt','cuda');model.requires_grad_(False)
        if cached is None:cached=predict(model,source,scenes)
        subset={name:config for name,config in policies.items() if config['model']==folder}
        sums,choices=evaluate_partial(model,scenes,cached,subset)
        metrics.update(sums);decisions.update(choices)
        del model,source;gc.collect();torch.cuda.empty_cache()
    assert set(metrics)==set(policies)
    return metrics,decisions


def audit(args,scenes,frozen,protected,sources,policies):
    import torch
    begin=time.perf_counter()
    baseline=torch.load(args.initial,map_location='cpu',weights_only=True)
    original=[n for n in baseline['model'] if 'lora_' not in n and not n.startswith(
        ('position_policy.','grasp_policy.','inspection_policy.'))]
    checkpoints={}
    for arm in ARMS:
        checkpoints[arm]={}
        for label in ('selected','last'):
            cp=torch.load(args.training/arm/(label+'.pt'),map_location='cpu',weights_only=True)
            assert_same_continuation(baseline,cp)
            checkpoints[arm][label]=dict(step=cp['step'],original_tensor_count=len(original),
                all_non_I_and_normalization_exact=True)
            del cp;gc.collect()
    cp=torch.load(args.prior/'depth_features/selected.pt',map_location='cpu',weights_only=True)
    assert_same_continuation(baseline,cp);del cp,baseline;gc.collect()
    assert all(file_hash(p)==h for p,h in protected.items())
    assert all(file_hash(ROOT/n)==h and file_hash(args.output/'source'/n)==h for n,h in sources.items())
    assert file_hash(args.output/'manifest.json')==frozen['evaluation_manifest_sha256']
    groups=defaultdict(set);layouts=defaultdict(set)
    for s in read(args.output/'manifest.json')['scenes']:
        groups[s['group_id']].add(s['split']);layouts[s['group_id']].add(s['layout'])
    assert all(len(v)==1 for v in groups.values())
    assert all(v=={'front_blocked','rear_blocked','both_blocked','free'} for v in layouts.values())
    count=entries=inspections=forced=0;maximum_speed=maximum_step=0.;collisions=[];candidate_counts=Counter()
    collection=[]
    for scene in scenes:
        sid=scene['spec']['scene_id'];mask=scene['full_sweep']['mask']
        assert (args.output/'evaluation_frozen.json').stat().st_mtime < (args.output/'data'/sid/'0/observations.npz').stat().st_mtime
        for branch in (0,1):
            for r in scene['branches'][branch]['records']:
                assert r['base_reset_count']==1
                if r['opening_tested']:assert r['result']['continuous_state_at_skill_entry']
                kind='success' if r['success'] else 'collision' if r['collision'] else 'other'
                candidate_counts[('full_allowed_' if mask[r['base_id']] else 'full_blocked_')+kind]+=1
                assert r['maximum_sampled_base_speed_m_s']<.31 and r['max_trace_step_m']<.02
                collection.append(r)
    for name in policies:
        summary=read(args.output/'policy_results'/name/'replay_verification.json')
        assert summary['count']==len(scenes) and summary['all_match']
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
            mask=logs[-1]['map']['mask']
            if not any(logs[0]['map']['mask']):assert r['refused'] and not r['inspection_attempted'];forced+=1
            if not r['refused'] and not r['inspection_unsafe_abort']:assert mask[r['base_id']]
            maximum_speed=max(maximum_speed,max(np.linalg.norm(t['base_velocity'][:2]) for t in trace))
            maximum_step=max(maximum_step,r['max_trace_step_m'])
            if r['collision']:collisions.append(dict(policy=name,scene_id=scene['spec']['scene_id'],
                failure_stage=r['failure_stage'],stop_reason=r['result'].get('stop_reason')))
            count+=1
    assert count==len(scenes)*len(policies) and len(collection)==len(scenes)*16
    assert maximum_speed<.31 and maximum_step<.02
    result=dict(seed=42,checkpoints=checkpoints,original_frozen_tensor_count=len(original),
        all6_new_checkpoints_and_v5_reference_non_I_exact=True,all_normalization_exact=True,
        all_protected_sources_manifests_and_weights_exact=True,selected_primary_before_test=frozen['primary_policy'],
        test_scenes=len(scenes),test_groups=len(scenes)//4,candidate_episodes=len(collection),replay_count=count,
        all_replays_match=True,all_replay_maps_derived_from_current_frames=True,initial_maps_have_one_frame=True,
        episode_reset_counts_one=True,opening_entry_continuity_checks=entries,actual_inspections=inspections,
        forced_initial_refusal_checks=forced,max_sampled_base_speed_m_s=float(maximum_speed),
        max_recorded_step_m=maximum_step,full_map_candidate_counts=dict(candidate_counts),collisions=collisions,
        seconds=time.perf_counter()-begin)
    write_json(args.output/'tensor_and_physics_audit.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--training',type=Path,default=ROOT/'runs/p6_train2_seed42_20261110')
    p.add_argument('--manifest',type=Path);p.add_argument('--smoke',action='store_true')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--estimate-seconds',type=int,required=True)
    args=p.parse_args()
    for name in ('output','training'):setattr(args,name,getattr(args,name).resolve())
    args.manifest=(args.manifest or args.training/'manifest.json').resolve()
    if args.workers<1 or args.estimate_seconds<1:raise ValueError('Positive budgets required.')
    if ROOT/'runs' not in args.output.parents:raise ValueError('New ignored runs/ output required.')
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Preserve existing runs.')
    args.output.mkdir(parents=True,exist_ok=True);begin=time.perf_counter();now=datetime.now(ZONE)
    receipt=dict(pid=os.getpid(),seed=42,status='starting',started_at=now.isoformat(),queue_estimate_seconds=0,
        runtime_estimate_seconds=args.estimate_seconds,expected_completion=(now+timedelta(seconds=args.estimate_seconds)).isoformat(),
        scope='pipeline smoke' if args.smoke else 'formal independent frozen v6 test; no fitting or selection')
    write_json(args.output/'receipt.json',receipt)
    from experiment_reproducibility import seed_everything,worker_environment
    receipt['reproducibility']=seed_everything(42);os.environ.update(worker_environment(42))
    import torch
    torch.set_num_threads(4)
    training_protocol=read(args.training/'protocol.json');selection=read(args.training/'frozen_for_test.json')
    args.initial=Path(training_protocol['initial']);args.prior=Path(training_protocol['prior_training'])
    if tuple(selection['test_references'])!=REFERENCES:raise ValueError('References differ from training declaration.')
    assert file_hash(args.training/'manifest.json')==selection['manifest_sha256']
    assert file_hash(args.initial)==selection['baseline_sha256']
    protected={str(args.initial):file_hash(args.initial)}
    for path,digest in read(args.training/'preserved_hashes.json').items():
        assert file_hash(path)==digest;protected[path]=digest
    for name in ('manifest.json','protocol.json','frozen_for_test.json','report.json','tensor_audit.json','source_hashes.json'):
        path=args.training/name;protected[str(path)]=file_hash(path)
    for arm in ARMS:
        for label in ('selected','last'):
            path=args.training/arm/(label+'.pt');protected[str(path)]=file_hash(path)
        assert protected[str(args.training/arm/'selected.pt')]==selection['checkpoint_sha256'][arm]
    prior_selection=read(args.prior/'frozen_for_test.json');prior_cp=args.prior/'depth_features/selected.pt'
    assert file_hash(prior_cp)==prior_selection['checkpoint_sha256']['depth_features']
    protected[str(prior_cp)]=file_hash(prior_cp)
    for name,digest in read(args.training/'source_hashes.json').items():
        assert file_hash(ROOT/name)==file_hash(args.training/'source'/name)==digest
    baseline=torch.load(args.initial,map_location='cpu',weights_only=True)
    for path in [*(args.training/arm/'selected.pt' for arm in ARMS),prior_cp]:
        cp=torch.load(path,map_location='cpu',weights_only=True);assert_same_continuation(baseline,cp);del cp
    del baseline;gc.collect()
    formal=read(args.training/'manifest.json');manifest=read(args.manifest)
    specs=validate_test_manifest(manifest,formal,args.smoke)
    policies=make_policies(args.training,args.prior)
    import run_mobile_partial_evaluation as engine
    # The imported live executor routes all policies through the original v5
    # neural path. Only checkpoint directories and declared modes differ.
    engine.MODEL_POLICIES=policies;engine.POLICY_NAMES=tuple(policies)
    longest=max(len(str(args.output/'policy_results'/name/'replays'/s['scene_id']/file))
        for name in policies for s in specs for file in ('result.json.tmp','trace.json.tmp','obs.json.tmp'))
    if os.name=='nt' and longest>=250:raise ValueError('Choose shorter output paths.')
    receipt['maximum_planned_output_path_chars']=longest
    write_json(args.output/'manifest.json',manifest)
    source=args.output/'source';source.mkdir()
    for name in ['AGENTS.md','scene.xml','.gitignore',*sorted(p.name for p in ROOT.glob('*.py'))]:shutil.copy2(ROOT/name,source/name)
    sources={path.name:file_hash(path) for path in source.iterdir()}
    write_json(args.output/'source_hashes.json',sources);write_json(args.output/'preserved_hashes.json',protected)
    frozen=dict(seed=42,primary_policy=selection['selected_policy'],selected_before_test=True,
        training_selection_sha256=file_hash(args.training/'frozen_for_test.json'),
        checkpoint_sha256=selection['checkpoint_sha256'],v5_reference_sha256=file_hash(prior_cp),
        evaluation_manifest_sha256=file_hash(args.output/'manifest.json'),policies=policies,
        test_outcomes_available=False,frozen_at=datetime.now(ZONE).isoformat(),smoke=args.smoke)
    write_json(args.output/'evaluation_frozen.json',frozen)
    write_json(args.output/'protocol.json',dict(seed=42,training=str(args.training),prior=str(args.prior),
        policies=policies,primary_policy=frozen['primary_policy'],workers=args.workers,
        scope=receipt['scope'],learning_updates=0,initial=str(args.initial),
        implementation='unchanged v5 candidate collection and real RGBD live replay; new checkpoint routing',
        information_scope='initial view/map for I; actual reached post/arrival inputs for P/G; known-map navigation'))
    phases={};receipt.update(status='collecting',test_scenes=len(specs),startup_seconds=time.perf_counter()-begin)
    write_json(args.output/'receipt.json',receipt)
    phases['test_collection']=engine.collect(args,specs)
    from run_mobile_active_experiment import load_scenes
    from run_mobile_outcome_experiment import guard_scene
    scenes=load_scenes(specs,args.output)
    for scene in scenes:
        scene['maps']=read(args.output/'data'/scene['spec']['scene_id']/'maps.json')
        scene['full_sweep']=guard_scene(scene['spec'])
    write_json(args.output/'test_full_sweep_queries.json',{s['spec']['scene_id']:s['full_sweep'] for s in scenes})
    tick=time.perf_counter();metrics,decisions=evaluate(args,scenes,policies)
    phases['cached_policy_evaluation']=time.perf_counter()-tick;replays={}
    for name in policies:
        write_json(args.output/'policy_results'/name/'test.json',dict(summary=metrics[name],decisions=decisions[name]))
        replays[name]=engine.live_replay(args,name,scenes,decisions[name]);phases['replay_'+name]=replays[name]['seconds']
    verification=audit(args,scenes,frozen,protected,sources,policies)
    phases['tensor_and_physics_audit']=verification['seconds']
    from check_git_size import audit as size_audit
    size=size_audit();write_json(args.output/'git_size_audit.json',size);assert size['passed']
    phases['startup_io_and_other']=time.perf_counter()-begin-sum(phases.values())
    report=dict(seed=42,smoke=args.smoke,primary_policy=frozen['primary_policy'],primary_selected_before_test=True,
        test=metrics,replays=replays,phase_seconds=phases,seconds=time.perf_counter()-begin,learning_updates=0,
        all_replays_match=True,all_non_I_and_original422_exact=True,frozen_weights_and_sources_exact=True,
        scope=receipt['scope'],limitation='one seed; reused development data; fixed P/G and known-map navigation')
    write_json(args.output/'report.json',report)
    receipt.update(status='complete',exit_code=0,actual_seconds=report['seconds'],phase_seconds=phases,
                   completed_at=datetime.now(ZONE).isoformat());write_json(args.output/'receipt.json',receipt)
    import json
    print(json.dumps(dict(status='complete',seconds=report['seconds'],primary=report['primary_policy'],test=metrics)),flush=True)


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
