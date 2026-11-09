"""Frozen v5 test collection and real RGB-D policy replays, without training."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
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
from run_mobile_reward_experiment import ROOT,ZONE,write_json,file_hash,worker_init
from run_mobile_partial_training import POLICIES as MODEL_POLICIES


REFERENCES={
    'geometry_full_rear':dict(map='full',station='rear_first',grasp=0,inspect='never'),
    'geometry_depth_never':dict(map='initial_depth',station='nearest',grasp=0,inspect='never'),
    'geometry_depth_rule':dict(map='current_depth',station='nearest',grasp=0,
        inspect='if chosen initial sweep sampled_free_fraction<0.4; max once'),
    'v4_always_depth':dict(map='current_depth',model='v4factorized_selected',inspect='always unless initial all-masked'),
}
POLICY_NAMES=tuple(MODEL_POLICIES)+tuple(REFERENCES)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def array_hash(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def prefix_maps(scene,arrays):
    from mobile_partial_map_v5 import PartialDoorMap,cad_prior
    mapper=PartialDoorMap(cad_prior(scene));mapper.add_frame(arrays['initial_frame'],arrays['initial_proprio'])
    initial=mapper.query()
    if arrays['post_view_available']:mapper.add_frame(arrays['post_frame'],arrays['post_proprio'])
    fused=mapper.query()
    return dict(initial=initial,fused=fused,frame_statistics=mapper.frame_stats)


def collect_scene(spec,folder):
    from mobile_crossed_v2 import restore_scene
    from mobile_partial_execution_v5 import execute_partial_candidate
    scene=restore_scene(spec);folder=Path(folder);begin=time.perf_counter();initial=None;maps=None;summaries=[]
    for branch in (0,1):
        destination=folder/str(branch);destination.mkdir(parents=True)
        arrivals=np.zeros((4,224,224,3),np.uint8);states=np.zeros((4,19),np.float32)
        commands=np.zeros((4,2,5),np.float32);records=[];post=None
        for b in range(4):
            for g in range(2):
                r,a,_,exp=execute_partial_candidate(scene,b,g,start_pose=spec['start_pose'],
                    inspection=bool(branch),initial_observation=initial)
                if initial is None:initial=(a['initial_frame'],a['initial_rgb'],a['initial_proprio'])
                if branch==0 and b==0 and g==0:
                    write_json(folder/'warmup.json',dict(pid=os.getpid(),scene_id=spec['scene_id'],
                        first_candidate_seconds=r['elapsed_wall_seconds'],elapsed_seconds=time.perf_counter()-begin,
                        integration_steps=r['integration_steps'],opening_tested=r['opening_tested']))
                candidate_post=(a['post_rgb'],a['post_proprio'],bool(a['post_view_available']))
                if post is None:post=candidate_post
                else:
                    np.testing.assert_allclose(post[1],candidate_post[1],rtol=0,atol=1e-7)
                    if post[2]!=candidate_post[2]:raise AssertionError('Candidate inspection prefixes differ.')
                if branch==1 and b==0 and g==0:
                    maps=prefix_maps(scene,a)
                    np.savez_compressed(folder/'reached_rgbd.npz',
                        **{'initial_'+k:v for k,v in a['initial_frame'].items()},
                        **{'post_'+k:v for k,v in a['post_frame'].items()},
                        initial_proprio=a['initial_proprio'],post_proprio=a['post_proprio'])
                    write_json(folder/'maps.json',maps)
                if 'arrival_rgb' in a:arrivals[b]=a['arrival_rgb'];states[b]=a['arrival_proprio']
                if 'grasp_command' in a:commands[b,g]=a['grasp_command']
                r.update(split=spec['split'],group_id=spec['group_id'],layout=spec['layout'],branch=branch,
                    maximum_sampled_base_speed_m_s=float(max(np.linalg.norm(t['base_velocity'][:2]) for t in exp.trace)))
                assert r['base_reset_count']==1
                records.append(r)
        np.savez_compressed(destination/'observations.npz',initial_rgb=initial[1],initial_proprio=initial[2],
            post_rgb=post[0],post_proprio=post[1],post_view_available=np.array(post[2]),
            arrival_rgb=arrivals,arrival_proprio=states,grasp_commands=commands)
        write_json(destination/'records.json',records)
        summaries.append(dict(branch=branch,openings=sum(r['success'] for r in records),
            inspection_success=records[0]['inspection_success']))
    result=dict(scene_id=spec['scene_id'],pid=os.getpid(),seed=42,candidates=16,
        seconds=time.perf_counter()-begin,branches=summaries,independent_full_prefix_per_candidate=True)
    write_json(folder/'receipt.json',result);return result


def collect(args,specs):
    begin=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init) as pool:
        futures=[pool.submit(collect_scene,s,args.output/'data'/s['scene_id']) for s in specs]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            progress=dict(pid=os.getpid(),phase='test_collection',completed_scenes=len(results),total_scenes=len(specs),
                elapsed_seconds=time.perf_counter()-begin,recent_scene_seconds=result['seconds'],
                mean_scene_seconds=float(np.mean([x['seconds'] for x in results])))
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    report=dict(seconds=time.perf_counter()-begin,workers=args.workers,scenes=results,
        collection_only=True,gpu_training_concurrent=False)
    write_json(args.output/'test_collection_timings.json',report);return report['seconds']


def scores_for_station(station):
    value=np.full(5,-1e6);value[4]=0.
    if station!=4:value[station]=1.
    return value


def geometry_decision(scene,name):
    from mobile_partial_execution_v5 import geometric_station,geometry_depth_action
    from run_mobile_active_experiment import choose
    a0=scene['branches'][0]['arrays'];a1=scene['branches'][1]['arrays']
    if name=='geometry_full_rear':
        b=geometric_station(scene['full_sweep']['mask'],a0['initial_proprio'],'rear_first');action=b
    else:action=geometry_depth_action(scene['maps']['initial'],a0['initial_proprio'],name=='geometry_depth_rule')
    b0=4 if action==5 else action
    b1=geometric_station(scene['maps']['fused']['mask'],a1['post_proprio'])
    g=np.tile(np.array([1.,0.]),(4,1))
    result=choose(scene,(scores_for_station(b0),np.array(1.),scores_for_station(b1),g,g),
        'always' if action==5 else 'never',name)
    result.update(geometry_rule=True,full_geometry_available=name=='geometry_full_rear',fixed_grasp_id=0)
    return result


def evaluate(args,scenes):
    import torch
    from mobile_partial_policy_v5 import load_partial_checkpoint
    from run_mobile_partial_training import predict,evaluate as evaluate_new
    from run_mobile_active_experiment import summarize,choose
    from mobile_outcome_v4 import load_outcome_checkpoint,station_values,outcome_values
    from run_mobile_outcome_experiment import predict_scenes
    metrics={};decisions={}
    for name in ('images_only','depth_features'):
        model,source=load_partial_checkpoint(args.training/name/'selected.pt','cuda');model.requires_grad_(False)
        cached=predict(model,source,scenes)
        sums,ds=evaluate_new(model,scenes,cached,{p:c for p,c in MODEL_POLICIES.items() if c['model']==name})
        metrics.update(sums);decisions.update(ds)
        del model,source,cached;gc.collect();torch.cuda.empty_cache()
    for name in ('geometry_full_rear','geometry_depth_never','geometry_depth_rule'):
        ds=[geometry_decision(s,name) for s in scenes];decisions[name]=ds;metrics[name]=summarize(ds)
    model,source=load_outcome_checkpoint(args.baseline,'cuda');model.requires_grad_(False)
    cached=predict_scenes(model,source,scenes);ds=[]
    for s in scenes:
        p=cached[s['spec']['scene_id']]
        values=(station_values(p['p0'].to('cuda'),1.,s['maps']['initial']['mask']),torch.tensor(-1e6,device='cuda'),
            station_values(p['p1'].to('cuda'),1.,s['maps']['fused']['mask']),
            outcome_values(p['g0'].to('cuda')),outcome_values(p['g1'].to('cuda')))
        values=tuple(x.detach().cpu().numpy() for x in values)
        ds.append(choose(s,values,'always' if any(s['maps']['initial']['mask']) else 'never','v4_always_depth'))
    decisions['v4_always_depth']=ds;metrics['v4_always_depth']=summarize(ds)
    del model,source,cached;gc.collect();torch.cuda.empty_cache()
    assert set(metrics)==set(POLICY_NAMES)
    return metrics,decisions


def live_replay(args,name,scenes,expected):
    import torch
    from mobile_crossed_v2 import restore_scene
    from mobile_partial_execution_v5 import execute_partial_candidate,geometric_station,geometry_depth_action
    from mobile_partial_map_v5 import PartialDoorMap,cad_prior
    from mobile_partial_policy_v5 import load_partial_checkpoint,partial_position_logits,partial_gate_features
    from mobile_outcome_v4 import load_outcome_checkpoint,position_logits,station_values,outcome_values,inspection_value
    from run_mobile_outcome_experiment import encode_front,guard_scene
    from train_hjepa import rgb_tensor,state_tensor
    new=name in MODEL_POLICIES;old=name=='v4_always_depth';neural=new or old
    if new:
        config=MODEL_POLICIES[name];model,source=load_partial_checkpoint(args.training/config['model']/'selected.pt','cuda')
    elif old:model,source=load_outcome_checkpoint(args.baseline,'cuda')
    if neural:model.eval();model.requires_grad_(False)
    begin=time.perf_counter();matches=[]
    for index,(scene,decision) in enumerate(zip(scenes,expected)):
        physical=restore_scene(scene['spec']);mapper=PartialDoorMap(cad_prior(physical));log=[];front=None
        full=guard_scene(scene['spec']) if name=='geometry_full_rear' else None
        def append_evidence(stage,arrays,evidence):
            state=arrays['initial_proprio'] if stage=='initial' else arrays['post_proprio']
            frame=arrays['initial_frame'] if stage=='initial' else arrays['post_frame']
            log.append(dict(stage=stage,map=evidence,reached_proprio=state.tolist(),
                depth_sha256=array_hash(frame['depth_m']),world_from_camera=frame['world_from_camera'].tolist(),
                policy_input_scope='current actual RGBD/proprio and known CAD; no future frame'))
        def pre(arrays):
            nonlocal front
            mapper.add_frame(arrays['initial_frame'],arrays['initial_proprio']);current=mapper.query()
            append_evidence('initial',arrays,current)
            if name=='geometry_full_rear':return geometric_station(full['mask'],arrays['initial_proprio'],'rear_first')
            if name.startswith('geometry_depth'):
                return geometry_depth_action(current,arrays['initial_proprio'],name=='geometry_depth_rule')
            if not any(current['mask']):return 4
            if old:return 5
            with torch.no_grad():
                front=encode_front(model,source,arrays['initial_rgb'],arrays['initial_proprio'])
                p=station_values(partial_position_logits(model,front,front,arrays['initial_proprio'],current['features']),1.,current['mask'])
                if config['mode']=='always':return 5
                if config['mode']=='never':return int(p.argmax())
                inspect,_=inspection_value(model,partial_gate_features(model,front,arrays['initial_proprio'],current['features'],1.),1.)
                return int(torch.cat([p,inspect]).argmax())
        def post(arrays):
            nonlocal front
            if arrays['post_view_available']:mapper.add_frame(arrays['post_frame'],arrays['post_proprio'])
            current=mapper.query();append_evidence('post',arrays,current)
            if name.startswith('geometry_'):return geometric_station(current['mask'],arrays['post_proprio'])
            with torch.no_grad():
                if front is None:front=encode_front(model,source,arrays['initial_rgb'],arrays['initial_proprio'])
                other=encode_front(model,source,arrays['post_rgb'],arrays['post_proprio']) if arrays['post_view_available'] else front
                raw=partial_position_logits(model,front,other,arrays['post_proprio'],current['features']) if new else position_logits(model,front,other,arrays['post_proprio'])
                return int(station_values(raw,1.,current['mask']).argmax())
        def grasp(rgb,state,commands):
            if not neural:return 0
            with torch.no_grad():
                z=model.encode(rgb_tensor(rgb[None],'cuda'),state_tensor(state[None],source['state_mean'],source['state_std'],'cuda'))
                logits=model.grasp_policy(torch.cat([z.expand(2,-1),torch.as_tensor(commands,device='cuda')],-1))
                return int(outcome_values(logits).argmax())
        record,arrays,trace,exp=execute_partial_candidate(physical,None,0,start_pose=scene['spec']['start_pose'],
            pre_selector=pre,post_selector=post,grasp_selector=grasp,keep_trace=True)
        match=all(record[k]==decision[k] for k in ('inspection_attempted','refused','success','collision'))
        if not record['inspection_unsafe_abort']:match=match and record['base_id']==decision['base_id']
        if record['grasp_observed']:match=match and record['grasp_id']==decision['grasp_id']
        if record['opening_tested']:match=match and record['result']['continuous_state_at_skill_entry']
        match=match and record['base_reset_count']==1
        map_masks_match=log[0]['map']['mask']==scene['maps']['initial']['mask']
        if len(log)>1:map_masks_match=map_masks_match and log[1]['map']['mask']==scene['maps']['fused']['mask']
        match=match and map_masks_match
        np.testing.assert_allclose(record['observation_cost'],decision['observation_cost'],rtol=0,atol=1e-7)
        folder=args.output/'policy_results'/name/'replays'/scene['spec']['scene_id']
        write_json(folder/'result.json',dict(match=bool(match),map_masks_match=map_masks_match,record=record,expected=decision))
        write_json(folder/'trace.json',trace);write_json(folder/'obs.json',log)
        matches.append(bool(match))
        if not match:raise AssertionError('Live policy mismatch: '+name+'/'+scene['spec']['scene_id'])
        if index==0 or (index+1)%12==0:
            progress=dict(pid=os.getpid(),phase='live_replay',policy=name,completed_scenes=index+1,total_scenes=len(scenes),
                elapsed_seconds=time.perf_counter()-begin)
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    result=dict(count=len(matches),all_match=all(matches),seconds=time.perf_counter()-begin,
                current_depth_queried_live=True,no_cached_map_used_for_actions=True)
    write_json(args.output/'policy_results'/name/'replay_verification.json',result)
    if neural:del model,source;gc.collect();torch.cuda.empty_cache()
    return result


def audit(args,scenes,frozen,protected,sources):
    import torch
    begin=time.perf_counter();torch.set_num_threads(4)
    original=torch.load(args.baseline,map_location='cpu',weights_only=True)
    fixed=[n for n in original['model'] if 'lora_' not in n and not n.startswith(('position_policy.','grasp_policy.','inspection_policy.'))]
    checkpoints={}
    for name in ('images_only','depth_features'):
        pg=torch.load(args.training/name/'selected_pg.pt',map_location='cpu',weights_only=True);checkpoints[name]={}
        for label in ('selected_pg','last_pg','selected','last'):
            cp=pg if label=='selected_pg' else torch.load(args.training/name/(label+'.pt'),map_location='cpu',weights_only=True)
            assert all(torch.equal(original['model'][n],cp['model'][n]) for n in fixed)
            assert all(torch.equal(original[k],cp[k]) for k in ('state_mean','state_std'))
            unchanged=all(torch.equal(v,cp['model'][n]) for n,v in pg['model'].items() if not n.startswith('inspection_policy.')) if label in ('selected','last') else None
            if unchanged is not None:assert unchanged
            checkpoints[name][label]=dict(step=cp['step'],original_tensor_count=len(fixed),original_and_norms_exact=True,selected_pg_exact_outside_I=unchanged)
            if cp is not pg:del cp
            gc.collect()
        del pg;gc.collect()
    del original;gc.collect()
    assert all(file_hash(p)==h for p,h in protected.items())
    assert all(file_hash(ROOT/n)==h and file_hash(args.output/'source'/n)==h for n,h in sources.items())
    assert all(file_hash(args.training/'source'/n)==h and file_hash(ROOT/n)==h
               for n,h in read(args.training/'source_hashes.json').items())
    assert file_hash(args.output/'manifest.json')==frozen['evaluation_manifest_sha256']
    groups=defaultdict(set);layouts=defaultdict(set);manifest=read(args.output/'manifest.json')
    for s in manifest['scenes']:
        groups[s['group_id']].add(s['split']);layouts[s['group_id']].add(s['layout'])
    assert all(len(v)==1 for v in groups.values())
    assert all(v=={'front_blocked','rear_blocked','both_blocked','free'} for v in layouts.values())
    assert len({s['geometry_sha256'] for s in manifest['scenes']})==len(manifest['scenes'])
    count=entries=inspections=forced=0;maximum_speed=maximum_step=0.;collisions=[];constraint_counts=Counter()
    collection=[]
    for scene in scenes:
        sid=scene['spec']['scene_id'];mask=scene['full_sweep']['mask']
        assert (args.output/'evaluation_frozen.json').stat().st_mtime < (args.output/'data'/sid/'0/observations.npz').stat().st_mtime
        for branch in (0,1):
            for r in scene['branches'][branch]['records']:
                assert r['base_reset_count']==1
                if r['opening_tested']:assert r['result']['continuous_state_at_skill_entry']
                kind='success' if r['success'] else 'collision' if r['collision'] else 'other'
                constraint_counts[('full_allowed_' if mask[r['base_id']] else 'full_blocked_')+kind]+=1
                collection.append(r)
    for name in POLICY_NAMES:
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
            if name!='geometry_full_rear':
                mask=logs[-1]['map']['mask']
                if not any(logs[0]['map']['mask']):assert r['refused'] and not r['inspection_attempted'];forced+=1
                if not r['refused'] and not r['inspection_unsafe_abort']:assert mask[r['base_id']]
            maximum_speed=max(maximum_speed,max(np.linalg.norm(t['base_velocity'][:2]) for t in trace))
            maximum_step=max(maximum_step,r['max_trace_step_m'])
            if r['collision']:collisions.append(dict(policy=name,scene_id=scene['spec']['scene_id'],failure_stage=r['failure_stage'],stop_reason=r['result'].get('stop_reason')))
            count+=1
    assert maximum_speed<.31 and maximum_step<.02
    assert count==len(scenes)*len(POLICY_NAMES) and len(collection)==len(scenes)*16
    for r in collection:
        assert r['maximum_sampled_base_speed_m_s']<.31 and r['max_trace_step_m']<.02
    result=dict(seed=42,checkpoints=checkpoints,original_frozen_tensor_count=len(fixed),all8_checkpoints_original_tensors_and_norms_exact=True,
        all_stage2_continuations_exact=True,all_protected_sources_manifests_and_weights_exact=True,
        selected_primary_before_test=frozen['primary_policy'],test_scenes=len(scenes),test_groups=len(scenes)//4,
        candidate_episodes=len(collection),replay_count=count,all_replays_match=True,
        all_replay_maps_derived_from_current_frames=True,initial_maps_have_one_frame=True,
        episode_reset_counts_one=True,opening_entry_continuity_checks=entries,actual_inspections=inspections,
        forced_initial_refusal_checks=forced,max_sampled_base_speed_m_s=float(maximum_speed),max_recorded_step_m=maximum_step,
        full_map_candidate_counts=dict(constraint_counts),collisions=collisions,seconds=time.perf_counter()-begin)
    write_json(args.output/'tensor_and_physics_audit.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--training',type=Path,default=ROOT/'runs/mobile_partial_training_seed42_20261109')
    p.add_argument('--manifest',type=Path)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--reference-declaration',type=Path,default=ROOT/'runs/mobile_partial_training_launch_seed42_20261109/receipt.json')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--estimate-seconds',type=int,required=True)
    args=p.parse_args();args.output=args.output.resolve();args.training=args.training.resolve()
    args.manifest=(args.manifest or args.training/'manifest.json').resolve()
    if args.workers<1 or args.estimate_seconds<1:raise ValueError('Positive budgets required.')
    if ROOT/'runs' not in args.output.parents:raise ValueError('Output must be ignored runs/.')
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Preserve existing runs.')
    if args.manifest!=args.training/'manifest.json' and not args.smoke:raise ValueError('Alternate manifest requires pipeline-smoke mode.')
    args.output.mkdir(parents=True);begin=time.perf_counter();now=datetime.now(ZONE)
    receipt=dict(pid=os.getpid(),seed=42,status='starting',started_at=now.isoformat(),queue_estimate_seconds=0,
        runtime_estimate_seconds=args.estimate_seconds,expected_completion=(now+timedelta(seconds=args.estimate_seconds)).isoformat(),
        scope='pipeline smoke' if args.smoke else 'formal frozen independent test; no fitting or policy selection')
    write_json(args.output/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    from experiment_reproducibility import seed_everything,worker_environment
    receipt['reproducibility']=seed_everything(42);os.environ.update(worker_environment(42))
    import torch
    torch.set_num_threads(4)
    training_protocol=read(args.training/'protocol.json');selection=read(args.training/'frozen_for_test.json')
    args.baseline=Path(training_protocol['initial']);declaration=read(args.reference_declaration)
    if declaration['references_predeclared']!=REFERENCES:raise AssertionError('Reference policy differs from its pre-test declaration.')
    if file_hash(args.training/'manifest.json')!=selection['manifest_sha256']:raise AssertionError('Frozen training/test manifest changed.')
    for name,h in selection['checkpoint_sha256'].items():
        if file_hash(args.training/name/'selected.pt')!=h:raise AssertionError('Selected checkpoint changed.')
    if file_hash(args.baseline)!=selection['baseline_sha256']:raise AssertionError('Frozen v4 baseline changed.')
    source_hashes=read(args.training/'source_hashes.json')
    if any(file_hash(ROOT/n)!=h or file_hash(args.training/'source'/n)!=h for n,h in source_hashes.items()):
        raise AssertionError('Training physics/map/model source differs from the frozen training run.')
    manifest=read(args.manifest);specs=[s for s in manifest['scenes'] if s['split']=='test']
    if not specs:raise ValueError('Empty frozen test.')
    longest_path=max(len(str(args.output/'policy_results'/name/'replays'/s['scene_id']/file))
        for name in POLICY_NAMES for s in specs for file in ('result.json.tmp','trace.json.tmp','obs.json.tmp'))
    if os.name=='nt' and longest_path>=250:
        raise ValueError('Output paths exceed the Windows path budget; choose a shorter ignored run directory: '+str(longest_path))
    receipt['maximum_planned_output_path_chars']=longest_path
    if args.smoke:
        formal={s['geometry_sha256'] for s in read(args.training/'manifest.json')['scenes'] if s['split']=='test'}
        if formal&{s['geometry_sha256'] for s in specs}:raise AssertionError('Smoke must not inspect formal test scenes.')
    write_json(args.output/'manifest.json',manifest)
    source=args.output/'source';source.mkdir()
    for name in ['AGENTS.md','scene.xml','.gitignore',*sorted(p.name for p in ROOT.glob('*.py'))]:shutil.copy2(ROOT/name,source/name)
    sources={x.name:file_hash(x) for x in source.iterdir()};write_json(args.output/'source_hashes.json',sources)
    protected={str(args.baseline):file_hash(args.baseline),str(args.reference_declaration.resolve()):file_hash(args.reference_declaration)}
    for name in ('manifest.json','protocol.json','frozen_for_test.json','tensor_audit.json','report.json','source_hashes.json'):
        path=args.training/name;protected[str(path)]=file_hash(path)
    for name in ('images_only','depth_features'):
        for label in ('selected_pg','last_pg','selected','last'):
            path=args.training/name/(label+'.pt');protected[str(path)]=file_hash(path)
    write_json(args.output/'preserved_hashes.json',protected)
    frozen=dict(seed=42,primary_policy=selection['selected_policy'],primary_selected_on_validation=True,
        training_selection_sha256=file_hash(args.training/'frozen_for_test.json'),
        selected_checkpoint_sha256=selection['checkpoint_sha256'],baseline_sha256=selection['baseline_sha256'],
        evaluation_manifest_sha256=file_hash(args.output/'manifest.json'),source_manifest_sha256=file_hash(args.manifest),
        references=REFERENCES,neural_policies=MODEL_POLICIES,all_policies=list(POLICY_NAMES),
        test_outcomes_available=False,frozen_at=datetime.now(ZONE).isoformat(),smoke=args.smoke)
    write_json(args.output/'evaluation_frozen.json',frozen)
    write_json(args.output/'protocol.json',dict(seed=42,training=str(args.training),manifest_source=str(args.manifest),
        baseline=str(args.baseline),policies=frozen['all_policies'],references=REFERENCES,workers=args.workers,
        primary_policy=frozen['primary_policy'],scope=receipt['scope'],learning_updates=0,
        known_map_scope='navigation/arm preparation unchanged; opening constraints from reached depth except full geometry reference',
        map_parameters='unchanged frozen v5 settings',source_identity='v3 physics with observation-only v5 executor fork'))
    phases={};receipt.update(status='collecting',test_scenes=len(specs),startup_seconds=time.perf_counter()-begin)
    write_json(args.output/'receipt.json',receipt)
    phases['test_collection']=collect(args,specs)
    from run_mobile_active_experiment import load_scenes
    from run_mobile_outcome_experiment import guard_scene
    scenes=load_scenes(specs,args.output)
    for scene in scenes:
        scene['maps']=read(args.output/'data'/scene['spec']['scene_id']/'maps.json')
        scene['full_sweep']=guard_scene(scene['spec'])
    write_json(args.output/'test_full_sweep_queries.json',{s['spec']['scene_id']:s['full_sweep'] for s in scenes})
    tick=time.perf_counter();metrics,decisions=evaluate(args,scenes);phases['cached_policy_evaluation']=time.perf_counter()-tick
    replays={}
    for name in POLICY_NAMES:
        write_json(args.output/'policy_results'/name/'test.json',dict(summary=metrics[name],decisions=decisions[name]))
        replays[name]=live_replay(args,name,scenes,decisions[name]);phases['replay_'+name]=replays[name]['seconds']
    verification=audit(args,scenes,frozen,protected,sources);phases['tensor_and_physics_audit']=verification['seconds']
    from check_git_size import audit as size_audit
    size=size_audit();write_json(args.output/'git_size_audit.json',size);assert size['passed']
    phases['startup_io_and_other']=time.perf_counter()-begin-sum(phases.values())
    report=dict(seed=42,smoke=args.smoke,primary_policy=frozen['primary_policy'],primary_selected_before_test=True,
        test=metrics,replays=replays,phase_seconds=phases,seconds=time.perf_counter()-begin,
        learning_updates=0,all_replays_match=verification['all_replays_match'],
        frozen_weights_and_sources_exact=True,all8_checkpoints_original422_exact=True,
        scope=receipt['scope'],limitation='one seed and known handle families; known-map navigation; opening-only depth information ablation')
    write_json(args.output/'report.json',report)
    receipt.update(status='complete',exit_code=0,actual_seconds=report['seconds'],phase_seconds=phases,completed_at=datetime.now(ZONE).isoformat())
    write_json(args.output/'receipt.json',receipt)
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
