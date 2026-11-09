"""Staged LoRA outcome supervision crossed with known-map door sweep planning.

Reuse immutable v3 train/validation physics. Freeze all selection before a fresh
grouped test collection. No previously inspected test enters training/selection.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime,timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import numpy as np

from run_mobile_reward_experiment import ROOT,ZONE,write_json,file_hash

MODELS=('scalar','factorized')
POLICIES={
    'baseline_always_plain':dict(model='baseline',mode='always',guard=False,risk=1.),
    'baseline_always_sweep':dict(model='baseline',mode='always',guard=True,risk=1.),
    **{f'{name}_{mode}_{planner}':dict(model=name,mode=mode,guard=planner=='sweep',risk=1.)
       for name in MODELS for mode in ('always','learned') for planner in ('plain','sweep')},
    'factorized_c3_learned_plain':dict(model='factorized',mode='learned',guard=False,risk=3.),
    'factorized_c3_learned_sweep':dict(model='factorized',mode='learned',guard=True,risk=3.),
}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def guard_scene(spec):
    from mobile_crossed_v2 import restore_scene
    from mobile_sweep_v4 import DoorSweepChecker
    mask,queries=DoorSweepChecker(restore_scene(spec)).mask()
    return dict(mask=mask.tolist(),queries=queries)


def make_run_manifest(development,groups,stream,output,development_groups=None):
    from mobile_crossed_v2 import make_manifest,geometry_identity
    old=read(development/'manifest.json')
    specs=[deepcopy(s) for s in old['scenes'] if s['split']!='test']
    if development_groups is not None:
        keep=set()
        for split in ('train','val'):
            available=sorted({s['group_id'] for s in specs if s['split']==split})
            if development_groups>len(available):raise ValueError('Too many development groups requested.')
            keep.update(available[:development_groups])
        specs=[s for s in specs if s['group_id'] in keep]
    new=make_manifest(dict(test=groups),stream)
    history=set()
    for name in ('mobile_reward_seed42_20261105','mobile_pipeline_smoke_seed42_20261105',
        'mobile_crossed_smoke_seed42_20261108','mobile_crossed_seed42_20261109',
        'mobile_active_smoke_seed42_20261109','mobile_active_seed42_20261109',
        'mobile_outcome_smoke_seed42_20261109','mobile_outcome_smoke2_seed42_20261109',
        'mobile_outcome_seed42_20261109'):
        path=ROOT/'runs'/name/'manifest.json'
        if path.exists() and path.parent.resolve()!=output:
            history.update(geometry_identity(s) for s in read(path)['scenes'])
    if history & {geometry_identity(s) for s in new['scenes']}:
        raise ValueError('New test overlaps historical geometry.')
    specs+=new['scenes']
    if len({geometry_identity(s) for s in specs})!=len(specs):raise ValueError('Duplicate physical scenes.')
    return dict(schema_version=4,seed=42,stream=stream,scenes=specs,
        development_source=str(development),development_manifest_sha256=file_hash(development/'manifest.json'),
        development_reused_splits=['train','val'],test_outcomes_available_at_creation=False,
        counts={split:sum(s['split']==split for s in specs) for split in ('train','val','test')},
        historical_geometries_excluded_from_test=len(history),
        scope='Reuse prior train/val, new grouped test geometry; known four handle families and known map planning.')


def attach_development(args,specs):
    """Read directly from immutable source data; hash all reused files."""
    physics=('scene.xml','simulate.py','mobile_door.py','mobile_crossed_v2.py','mobile_active_v3.py',
             'door_stress_scenes.py','perception.py','grasp_plan.py','prm.py','hjepa_scene.py')
    old=read(args.development/'source_hashes.json')
    for name in physics:
        if file_hash(ROOT/name)!=old[name]:raise ValueError('Reused physics source differs: '+name)
    protected={}
    for spec in specs:
        for branch in ('0','1'):
            folder=args.development/'data'/spec['scene_id']/branch
            for name in ('observations.npz','records.json'):
                path=folder/name;protected[str(path)]=file_hash(path)
    write_json(args.output/'reused_development_hashes.json',protected)
    from run_mobile_active_experiment import load_scenes
    return load_scenes(specs,args.development)


def encode_front(model,source,rgb,state):
    from run_mobile_active_experiment import _encode_position
    return _encode_position(model,source,rgb,state)


def grasp_logits(model,source,arrays,canonical):
    import torch
    from train_hjepa import rgb_tensor,state_tensor
    device=next(model.parameters()).device;mean,std=source['state_mean'],source['state_std']
    if canonical:
        result=[]
        for b in range(4):
            z=model.encode(rgb_tensor(arrays['arrival_rgb'][b:b+1],device),
                state_tensor(arrays['arrival_proprio'][b:b+1],mean,std,device))
            result.append(model.grasp_policy(torch.cat([z.expand(2,-1),
                torch.as_tensor(arrays['grasp_commands'][b],device=device)],-1)))
        return torch.stack(result).float()
    z=model.encode(rgb_tensor(arrays['arrival_rgb'],device),state_tensor(arrays['arrival_proprio'],mean,std,device))
    return model.grasp_policy(torch.cat([z[:,None].expand(-1,2,-1),
        torch.as_tensor(arrays['grasp_commands'],device=device)],-1)).float()


def forward_pg(model,source,scene,canonical=True):
    from mobile_outcome_v4 import position_logits
    a0=scene['branches'][0]['arrays'];a1=scene['branches'][1]['arrays']
    z=encode_front(model,source,a0['initial_rgb'],a0['initial_proprio'])
    other=encode_front(model,source,a1['post_rgb'],a1['post_proprio']) if a1['post_view_available'] else z
    return dict(p0=position_logits(model,z,z,a0['initial_proprio']),
        p1=position_logits(model,z,other,a1['post_proprio']),
        g0=grasp_logits(model,source,a0,canonical),g1=grasp_logits(model,source,a1,canonical),front=z)


def predict_scenes(model,source,scenes):
    import torch
    model.eval();predictions={}
    with torch.no_grad():
        for scene in scenes:
            item=forward_pg(model,source,scene)
            predictions[scene['spec']['scene_id']]={k:v.detach().cpu() for k,v in item.items()}
    return predictions


def scalar_prediction(model,scene,prediction,config):
    import torch
    from mobile_outcome_v4 import station_values,outcome_values,gate_features,inspection_value
    risk=config['risk'];mask=scene['sweep_mask'] if config['guard'] else None
    device=next(model.parameters()).device
    # Keep softmax/reduction on the same device as live decisions, including
    # scores close to the known refusal value0.
    p0=station_values(prediction['p0'].to(device),risk,mask);p1=station_values(prediction['p1'].to(device),risk,mask)
    g0=outcome_values(prediction['g0'].to(device),risk);g1=outcome_values(prediction['g1'].to(device),risk)
    with torch.no_grad():
        device=next(model.parameters()).device
        feature=gate_features(prediction['front'].to(device),scene['branches'][0]['arrays']['initial_proprio'],risk,config['guard'])
        value,_=inspection_value(model,feature,risk)
    return tuple(v.detach().cpu().numpy() for v in (p0,value[0],p1,g0,g1))


def decide(scene,prediction,config,name):
    from run_mobile_active_experiment import choose
    forced=config['guard'] and not any(scene['sweep_mask'])
    result=choose(scene,prediction,'never' if forced else config['mode'],name)
    result.update(sweep_guard=config['guard'],collision_risk_weight=config['risk'],
        sweep_forced_refusal=bool(forced),sweep_allowed_stations=scene['sweep_mask'] if config['guard'] else None)
    return result


def evaluate_cached(model,scenes,predictions,policies):
    from run_mobile_active_experiment import summarize
    chosen={name:[decide(s,scalar_prediction(model,s,predictions[s['spec']['scene_id']],config),config,name)
                  for s in scenes] for name,config in policies.items()}
    return {name:summarize(ds) for name,ds in chosen.items()},chosen


def pg_targets(scenes,predictions,device='cpu'):
    from mobile_outcome_v4 import outcome_values,record_outcome
    targets={}
    for scene in scenes:
        p=predictions[scene['spec']['scene_id']];one={}
        for branch in (0,1):
            ids=outcome_values(p[f'g{branch}'].to(device)).argmax(-1).cpu().numpy()
            table=np.array([[record_outcome(r) for r in scene['branches'][branch]['records'][2*b:2*b+2]]
                            for b in range(4)],np.int64)
            one[f'p{branch}']=table[np.arange(4),ids];one[f'g{branch}']=table
        one['post_mask']=not scene['branches'][1]['records'][0]['inspection']['unsafe_abort']
        targets[scene['spec']['scene_id']]=one
    return targets


def save(path,model,source,**metadata):
    from hjepa_model import save_checkpoint
    save_checkpoint(path,model,source['state_mean'],source['state_std'],seed=42,**metadata)


def train_pg(args,name,scenes):
    import torch
    from hjepa_model import load_checkpoint
    from hjepa_lora import resume_lora,freeze_core_statistics,frozen_core_digest
    from mobile_outcome_v4 import configure_outcomes,outcome_loss,load_outcome_checkpoint
    from experiment_reproducibility import seed_everything
    from run_mobile_active_experiment import validation_key
    seed_everything(42);torch.set_num_threads(4);begin=time.perf_counter()
    folder=args.output/name;folder.mkdir();model,source=load_checkpoint(args.initial,'cuda')
    configure_outcomes(model);lora=resume_lora(model,8,16.,.05);model.requires_grad_(False)
    for n,p in model.model.named_parameters():p.requires_grad_('lora_' in n)
    model.position_policy.requires_grad_(True);model.grasp_policy.requires_grad_(True)
    frozen=frozen_core_digest(model)
    training=[s for s in scenes if s['spec']['split']=='train'];validation=[s for s in scenes if s['spec']['split']=='val']
    config=dict(model=name,mode='always',guard=False,risk=1.)
    metric_name=name+'_always_plain';policies={metric_name:config}
    history=[];targets=None;refreshes=[]
    val=predict_scenes(model,source,validation);summaries,_=evaluate_cached(model,validation,val,policies)
    best=validation_key(summaries[metric_name]);selected=0;history.append(dict(step=0,summary=summaries[metric_name]))
    metadata=dict(loss_kind=name,phase='pg',initial_sha256=file_hash(args.initial),
                  manifest_sha256=file_hash(args.output/'manifest.json'))
    save(folder/'selected_pg.pt',model,source,step=0,**metadata)
    model.enable_activation_checkpointing(16)
    optimizer=torch.optim.AdamW([dict(params=[*model.position_policy.parameters(),*model.grasp_policy.parameters()],lr=3e-4),
        dict(params=[p for p in model.model.parameters() if p.requires_grad],lr=2e-5)],weight_decay=.01)
    rng=random.Random(42);timings=[]
    for step in range(1,args.pg_steps+1):
        if (step-1)%40==0:
            predictions=predict_scenes(model,source,training);targets=pg_targets(training,predictions,'cuda')
            digest=hashlib.sha256(b''.join(targets[s['spec']['scene_id']][key].tobytes()
                for s in training for key in ('p0','p1','g0','g1'))).hexdigest()
            refreshes.append(dict(step=step-1,targets_sha256=digest,train_scenes=len(training),test_arrays_opened=0))
        tick=time.perf_counter();model.eval();model.model.get_level(1).encoder.encoder.train();freeze_core_statistics(model)
        optimizer.zero_grad();losses=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for scene in rng.sample(training,args.batch_size):
                p=forward_pg(model,source,scene,False);target=targets[scene['spec']['scene_id']]
                loss=outcome_loss(p['p0'],torch.as_tensor(target['p0'],device='cuda'),name)
                if target['post_mask']:loss=loss+outcome_loss(p['p1'],torch.as_tensor(target['p1'],device='cuda'),name)
                for branch in (0,1):
                    mask=torch.as_tensor(scene['branches'][branch]['grasp_mask'],device='cuda')
                    if mask.any():loss=loss+.5*outcome_loss(p[f'g{branch}'][mask],
                        torch.as_tensor(target[f'g{branch}'],device='cuda')[mask],name)
                losses.append(loss)
            loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PG loss.')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step();torch.cuda.synchronize();timings.append(time.perf_counter()-tick)
        if step==1 or step%10==0:
            progress=dict(pid=os.getpid(),phase='pg_'+name,step=step,total_steps=args.pg_steps,loss=float(loss.detach()),
                seconds_per_step=float(np.mean(timings[-10:])),elapsed_seconds=time.perf_counter()-begin,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        if step==1 or step%40==0 or step==args.pg_steps:
            val=predict_scenes(model,source,validation);summaries,_=evaluate_cached(model,validation,val,policies)
            summary=summaries[metric_name];history.append(dict(step=step,summary=summary))
            if validation_key(summary)>best:
                best=validation_key(summary);selected=step;save(folder/'selected_pg.pt',model,source,step=step,**metadata)
    save(folder/'last_pg.pt',model,source,step=args.pg_steps,**metadata)
    if frozen_core_digest(model)!=frozen:raise AssertionError('Original model core changed.')
    report=dict(seed=42,steps=args.pg_steps,selected_step=selected,history=history,teacher_refreshes=refreshes,
        training_step_seconds=timings,lora=lora,frozen_core_unchanged=True,test_arrays_opened=0,seconds=time.perf_counter()-begin)
    write_json(folder/'pg_training.json',report)
    del model,optimizer;torch.cuda.empty_cache()
    return report


def gate_dataset(scenes,predictions,device='cpu'):
    import torch
    from mobile_outcome_v4 import gate_features,continuation_class
    features=[];labels=[];costs=[];identities=[]
    for scene in scenes:
        p=predictions[scene['spec']['scene_id']]
        for risk in (1.,3.):
            for guard in (False,True):
                features.append(gate_features(p['front'],scene['branches'][0]['arrays']['initial_proprio'],risk,guard))
                labels.append(continuation_class(scene,p['p1'].to(device),p['g1'].to(device),risk,scene['sweep_mask'] if guard else None))
                costs.append(scene['branches'][1]['records'][0]['observation_cost'])
                identities.append(dict(scene_id=scene['spec']['scene_id'],risk=risk,guard=guard))
    return torch.cat(features),torch.tensor(labels),torch.tensor(costs),identities


def train_gate(args,name,scenes):
    import torch
    from torch.nn import functional as F
    from mobile_outcome_v4 import load_outcome_checkpoint,outcome_loss
    from experiment_reproducibility import seed_everything
    from run_mobile_active_experiment import validation_key
    seed_everything(42);begin=time.perf_counter();folder=args.output/name
    model,source=load_outcome_checkpoint(folder/'selected_pg.pt','cuda');model.requires_grad_(False)
    fixed={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not n.startswith('inspection_policy.')}
    model.inspection_policy.requires_grad_(True)
    predictions=predict_scenes(model,source,scenes)
    training=[s for s in scenes if s['spec']['split']=='train'];validation=[s for s in scenes if s['spec']['split']=='val']
    features,labels,costs,identities=gate_dataset(training,predictions,'cuda')
    risks=torch.tensor([d['risk'] for d in identities],device='cuda')
    features,labels,costs=features.to('cuda'),labels.to('cuda'),costs.to('cuda')
    np.savez_compressed(folder/'gate_training_features.npz',features=features.cpu().numpy(),labels=labels.cpu().numpy(),costs=costs.cpu().numpy())
    metric_name=name+'_learned_plain';policies={metric_name:POLICIES[metric_name]}
    summaries,_=evaluate_cached(model,validation,predictions,policies);best=validation_key(summaries[metric_name])
    history=[dict(step=0,summary=summaries[metric_name])];selected=0
    metadata=dict(loss_kind=name,phase='inspection',pg_selected_sha256=file_hash(folder/'selected_pg.pt'),
        initial_sha256=file_hash(args.initial),manifest_sha256=file_hash(args.output/'manifest.json'))
    save(folder/'selected.pt',model,source,step=0,**metadata)
    optimizer=torch.optim.AdamW(model.inspection_policy.parameters(),lr=3e-4,weight_decay=.01)
    rng=random.Random(42)
    for step in range(1,args.gate_steps+1):
        ids=torch.tensor(rng.sample(range(len(features)),min(32,len(features))),device='cuda')
        model.inspection_policy.train();optimizer.zero_grad();raw=model.inspection_policy(features[ids]).float()
        # Both risks share capacity/data; the risk weight is an available input.
        parts=[]
        for risk in (1.,3.):
            mask=risks[ids]==risk
            if mask.any():parts.append(outcome_loss(raw[mask,:4],labels[ids][mask],name,risk))
        loss=torch.stack(parts).mean()+.5*F.mse_loss(F.softplus(raw[:,4]),costs[ids])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite gate loss.')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.inspection_policy.parameters(),1.);optimizer.step()
        if step==1 or step%40==0 or step==args.gate_steps:
            summaries,_=evaluate_cached(model,validation,predictions,policies);summary=summaries[metric_name]
            history.append(dict(step=step,summary=summary,loss=float(loss.detach())))
            if validation_key(summary)>best:
                best=validation_key(summary);selected=step;save(folder/'selected.pt',model,source,step=step,**metadata)
            progress=dict(pid=os.getpid(),phase='gate_'+name,step=step,total_steps=args.gate_steps,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-begin)
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    save(folder/'last.pt',model,source,step=args.gate_steps,**metadata)
    if any(not torch.equal(v,model.state_dict()[n].detach().cpu()) for n,v in fixed.items()):
        raise AssertionError('Frozen continuation changed while training observation gate.')
    result=dict(seed=42,steps=args.gate_steps,selected_step=selected,training_scenes=len(training),
        training_conditions=len(features),history=history,frozen_continuation_unchanged=True,
        target_contract='Current frozen post P/G outcome, no oracle best continuation; train only',
        test_arrays_opened=0,seconds=time.perf_counter()-begin)
    write_json(folder/'gate_training.json',result);del model,optimizer;torch.cuda.empty_cache();return result


def evaluate_model(args,name,scenes,policies):
    import torch
    from mobile_outcome_v4 import load_outcome_checkpoint
    model,source=load_outcome_checkpoint(args.output/name/'selected.pt','cuda')
    predictions=predict_scenes(model,source,scenes);result=evaluate_cached(model,scenes,predictions,policies)
    del model;torch.cuda.empty_cache();return result


def evaluate_baseline(args,scenes):
    import torch
    from hjepa_model import load_checkpoint
    from run_mobile_active_experiment import predictions,summarize
    model,source=load_checkpoint(args.initial,'cuda');decisions={n:[] for n,c in POLICIES.items() if c['model']=='baseline'}
    with torch.no_grad():
        for scene in scenes:
            pred=predictions(model,source,scene,dict(use_view=True))
            for name in decisions:
                config=POLICIES[name];p0,i,p1,g0,g1=(np.array(x,copy=True) for x in pred)
                if config['guard']:
                    p0[:4][~np.array(scene['sweep_mask'])]=-1e6;p1[:4][~np.array(scene['sweep_mask'])]=-1e6
                decisions[name].append(decide(scene,(p0,i,p1,g0,g1),config,name))
    del model;torch.cuda.empty_cache();return {n:summarize(d) for n,d in decisions.items()},decisions


def live_replay(args,name,scenes,expected):
    import torch
    from mobile_active_v3 import execute_active_candidate
    from mobile_crossed_v2 import restore_scene
    from mobile_outcome_v4 import (load_outcome_checkpoint,position_logits,station_values,
        gate_features,inspection_value,outcome_values)
    from hjepa_model import load_checkpoint
    from train_hjepa import rgb_tensor,state_tensor
    from run_mobile_active_experiment import predict_pre,predict_post
    config=POLICIES[name];baseline=config['model']=='baseline'
    loader=load_checkpoint if baseline else load_outcome_checkpoint
    model,source=loader(args.initial if baseline else args.output/config['model']/'selected.pt','cuda')
    begin=time.perf_counter();matches=[]
    for scene,decision in zip(scenes,expected):
        mask=scene['sweep_mask'] if config['guard'] else None
        forced=mask is not None and not any(mask)
        def pre(arrays):
            if forced:return 4
            with torch.no_grad():
                if baseline:
                    p,i=predict_pre(model,source,arrays['initial_rgb'],arrays['initial_proprio'])
                    if mask is not None:p[:4]=p[:4].masked_fill(~torch.tensor(mask,device='cuda'),-1e6)
                else:
                    z=encode_front(model,source,arrays['initial_rgb'],arrays['initial_proprio'])
                    p=station_values(position_logits(model,z,z,arrays['initial_proprio']),config['risk'],mask)
                    i,_=inspection_value(model,gate_features(z,arrays['initial_proprio'],config['risk'],config['guard']),config['risk'])
                    i=i[0]
                return 5 if config['mode']=='always' else int(torch.cat([p,i[None]]).argmax())
        def post(arrays):
            with torch.no_grad():
                if baseline:
                    p=predict_post(model,source,arrays,True)
                    if mask is not None:p[:4]=p[:4].masked_fill(~torch.tensor(mask,device='cuda'),-1e6)
                else:
                    z=encode_front(model,source,arrays['initial_rgb'],arrays['initial_proprio'])
                    other=encode_front(model,source,arrays['post_rgb'],arrays['post_proprio']) if arrays['post_view_available'] else z
                    p=station_values(position_logits(model,z,other,arrays['post_proprio']),config['risk'],mask)
                return int(p.argmax())
        def grasp(rgb,state,commands):
            with torch.no_grad():
                z=model.encode(rgb_tensor(rgb[None],'cuda'),state_tensor(state[None],source['state_mean'],source['state_std'],'cuda'))
                raw=model.grasp_policy(torch.cat([z.expand(2,-1),torch.as_tensor(commands,device='cuda')],-1))
                score=raw.flatten() if baseline else outcome_values(raw,config['risk'])
                return int(score.argmax())
        record,_,trace,_=execute_active_candidate(restore_scene(scene['spec']),None,0,start_pose=scene['spec']['start_pose'],
            pre_selector=pre,post_selector=post,grasp_selector=grasp,keep_trace=True)
        match=all(record[a]==decision[b] for a,b in (('inspection_attempted','inspection_attempted'),
            ('refused','refused'),('success','success'),('collision','collision')))
        if not record['inspection_unsafe_abort']:match=match and record['base_id']==decision['base_id']
        if record['grasp_observed']:match=match and record['grasp_id']==decision['grasp_id']
        if record['opening_tested']:match=match and record['result']['continuous_state_at_skill_entry']
        match=match and record['base_reset_count']==1
        record.update(sweep_guard=config['guard'],sweep_forced_refusal=forced,
                      sweep_queries=scene.get('sweep_queries') if config['guard'] else None)
        folder=args.output/'policy_results'/name/'replays'/scene['spec']['scene_id']
        write_json(folder/'result.json',dict(match=bool(match),record=record));write_json(folder/'trace.json',trace)
        matches.append(bool(match))
        if not match:raise AssertionError('Physical replay mismatch: '+name+'/'+scene['spec']['scene_id'])
    result=dict(count=len(matches),all_match=all(matches),seconds=time.perf_counter()-begin)
    write_json(args.output/'policy_results'/name/'replay_verification.json',result)
    print(json.dumps(dict(phase='replay',policy=name,**result)),flush=True)
    del model;torch.cuda.empty_cache();return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--development',type=Path,default=ROOT/'runs/mobile_active_seed42_20261109')
    parser.add_argument('--initial',type=Path,default=ROOT/'runs/mobile_active_seed42_20261109/view_free/selected.pt')
    parser.add_argument('--test-groups',type=int,default=18);parser.add_argument('--pg-steps',type=int,default=320)
    parser.add_argument('--development-groups',type=int,help='Optional whole-group subset for pipeline smoke only.')
    parser.add_argument('--gate-steps',type=int,default=800);parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--stream',default='mobile_outcome_v4_20261109')
    parser.add_argument('--estimate-seconds',type=int,default=19800)
    args=parser.parse_args();args.output=args.output.resolve();args.development=args.development.resolve();args.initial=args.initial.resolve()
    if any(v<1 for v in (args.test_groups,args.pg_steps,args.gate_steps,args.batch_size,args.workers)):raise ValueError('Positive budgets required.')
    if ROOT/'runs' not in args.output.parents:raise ValueError('Output must be ignored runs/.')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Preserve existing runs.')
    args.output.mkdir(parents=True);begin=time.perf_counter();now=datetime.now(ZONE)
    receipt=dict(pid=os.getpid(),status='starting',seed=42,started_at=now.isoformat(),queue_estimate_seconds=0,
        runtime_estimate_seconds=args.estimate_seconds,expected_completion=(now+timedelta(seconds=args.estimate_seconds)).isoformat(),
        estimate_basis='v4 corrected smoke:4test scenes295.344s ->72scenes5316.2s;48physical replays363.380s ->864replays6540.8s. Full v3 LoRA stage2394.24s/480 updates incl refresh/validation supports~4200s budget for both320-step PG and800-step frozen gates. Add600s I/O/audit and~3000s conservative load margin:19800s=5.5h, local queue0. Collection-only and training do not overlap; estimate is not measured full-run time.')
    write_json(args.output/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    from experiment_reproducibility import seed_everything,worker_environment
    receipt['reproducibility']=seed_everything(42);os.environ.update(worker_environment(42))
    if args.development_groups is not None and args.development_groups<1:raise ValueError('Positive development group subset required.')
    manifest=make_run_manifest(args.development,args.test_groups,args.stream,args.output,args.development_groups)
    write_json(args.output/'manifest.json',manifest);manifest_hash=file_hash(args.output/'manifest.json')
    write_json(args.output/'protocol.json',dict(seed=42,models=MODELS,policies=POLICIES,
        pg_steps=args.pg_steps,gate_steps=args.gate_steps,batch_scenes=args.batch_size,counts=manifest['counts'],
        initial=str(args.initial),initial_sha256=file_hash(args.initial),development=str(args.development),
        head_capacity='Both arms identical3-outcome P/G heads and5-output I head; original scalar trunks copied identically.',
        losses='scalar:expected-utility MSE; factorized:outcome cross entropy; both regress actual inspection cost.',
        stage1='Continue rank8 LoRA and P/G; selected only by always-observe no-sweep validation at risk1.',
        stage2='Freeze selected P/G and LoRA; train independent I on current frozen continuation outcome, not oracle best action.',
        gate_conditions='risk1/3 and planner plain/sweep, same4conditions per train scene in both arms; available risk and planner flag only.',
        gate_selection='learned no-sweep risk1 validation, earliest exact tie; no test access.',
        policy_selection='all declared policies by validation correct decisions, fewer collisions, openings, common net utility, path, observations; declared order ties.',
        sweep='Known full geometry; door assembly versus static obstacles and commanded base box;1degree sampling,1mm penetration tolerance; excludes moving-arm/contact dynamics.',
        sweep_semantics='Mask stations before selection; all masked -> refuse without inspection. Feasibility denominator remains unmasked physical union.',
        scope='frozen prior train/val reused; newly generated grouped test, known families and map, one seed'))
    source=args.output/'source';source.mkdir()
    for name in ['AGENTS.md','scene.xml','.gitignore',*sorted(p.name for p in ROOT.glob('*.py'))]:shutil.copy2(ROOT/name,source/name)
    hashes={p.name:file_hash(p) for p in source.iterdir()};write_json(args.output/'source_hashes.json',hashes)
    preserve={str(args.initial):file_hash(args.initial)}
    for name in ('manifest.json','report.json','frozen_for_test.json','final_verification.json'):
        p=args.development/name;preserve[str(p)]=file_hash(p)
    write_json(args.output/'preserved_hashes.json',preserve)
    dev_specs=[s for s in manifest['scenes'] if s['split']!='test'];test_specs=[s for s in manifest['scenes'] if s['split']=='test']
    phases={};tick=time.perf_counter();dev=attach_development(args,dev_specs)
    guards={s['scene_id']:guard_scene(s) for s in dev_specs};write_json(args.output/'development_sweep_queries.json',guards)
    for scene in dev:
        q=guards[scene['spec']['scene_id']];scene.update(sweep_mask=q['mask'],sweep_queries=q['queries'])
    phases['development_reuse_and_sweep']=time.perf_counter()-tick
    receipt.update(status='training',startup_seconds=time.perf_counter()-begin);write_json(args.output/'receipt.json',receipt)
    print(json.dumps(receipt),flush=True)
    training={};validation={}
    for name in MODELS:
        pg=train_pg(args,name,dev);gate=train_gate(args,name,dev)
        training[name]=dict(pg=pg,gate=gate);phases['pg_'+name]=pg['seconds'];phases['gate_'+name]=gate['seconds']
        summaries,_=evaluate_model(args,name,[s for s in dev if s['spec']['split']=='val'],
            {n:c for n,c in POLICIES.items() if c['model']==name});validation.update(summaries)
    baseline,_=evaluate_baseline(args,[s for s in dev if s['spec']['split']=='val'])
    validation={**baseline,**validation}
    from run_mobile_active_experiment import best_policy,collect,load_scenes
    selected=best_policy(validation)
    frozen=dict(selected_policy=selected,selected_before_test=True,manifest_sha256=manifest_hash,
        checkpoint_sha256={n:file_hash(args.output/n/'selected.pt') for n in MODELS},
        baseline_sha256=file_hash(args.initial),validation_metrics=validation)
    write_json(args.output/'frozen_for_test.json',frozen)
    if any((args.output/'data'/s['scene_id']).exists() for s in test_specs):raise AssertionError('Test data already existed.')
    phases['test_collection']=collect(test_specs,args,'test_collection');tests=load_scenes(test_specs,args.output)
    guards_test={s['scene_id']:guard_scene(s) for s in test_specs};write_json(args.output/'test_sweep_queries.json',guards_test)
    for scene in tests:
        q=guards_test[scene['spec']['scene_id']];scene.update(sweep_mask=q['mask'],sweep_queries=q['queries'])
    metrics,decisions=evaluate_baseline(args,tests)
    for name in MODELS:
        summaries,chosen=evaluate_model(args,name,tests,{n:c for n,c in POLICIES.items() if c['model']==name})
        metrics.update(summaries);decisions.update(chosen)
    replays={}
    for name in POLICIES:
        write_json(args.output/'policy_results'/name/'test.json',dict(summary=metrics[name],decisions=decisions[name]))
        replays[name]=live_replay(args,name,tests,decisions[name]);phases['replay_'+name]=replays[name]['seconds']
    from check_git_size import audit
    size=audit();write_json(args.output/'git_size_audit.json',size)
    if not size['passed']:raise AssertionError('Git size guard failed.')
    for path,h in preserve.items():
        if file_hash(path)!=h:raise AssertionError('Historical file changed: '+path)
    for path,h in read(args.output/'reused_development_hashes.json').items():
        if file_hash(path)!=h:raise AssertionError('Reused development data changed: '+path)
    if any(file_hash(ROOT/name)!=h for name,h in hashes.items()):raise AssertionError('Source changed during run.')
    if file_hash(args.output/'manifest.json')!=manifest_hash:raise AssertionError('Manifest changed.')
    if any(file_hash(args.output/name/'selected.pt')!=h for name,h in frozen['checkpoint_sha256'].items()):raise AssertionError('Frozen model changed.')
    from audit_mobile_outcome import audit_outcome_run
    tensor_and_physics=audit_outcome_run(args.output,ROOT)
    phases['other_startup_selection_io']=time.perf_counter()-begin-sum(phases.values())
    report=dict(seed=42,selected_policy=selected,selected_before_test=True,test=metrics,replays=replays,
        seconds=time.perf_counter()-begin,phase_seconds=phases,source_hashes_unchanged=True,preserved_hashes_unchanged=True,
        reused_development_unchanged=True,frozen_weights_unchanged=True,training_test_arrays_opened=0,
        original_frozen_tensors_exact=tensor_and_physics['all_original_tensors_and_normalizations_exact'],
        continuation_frozen_during_gate_training=tensor_and_physics['all_selected_pg_frozen_during_gate_fitting'],
        limitation='One seed, known geometry constraints and handle families; staged offline supervision, not full-model or online world-model training.')
    write_json(args.output/'report.json',report)
    receipt.update(status='complete',exit_code=0,elapsed_seconds=report['seconds'],phase_seconds=phases,completed_at=datetime.now(ZONE).isoformat())
    write_json(args.output/'receipt.json',receipt)
    print(json.dumps(dict(status='complete',seconds=report['seconds'],selected_policy=selected,test=metrics)),flush=True)


if __name__=='__main__':
    if os.environ.get('PYTHONHASHSEED')!='42':
        env=dict(os.environ,PYTHONHASHSEED='42',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        raise SystemExit(subprocess.run([sys.executable,*sys.argv],cwd=ROOT,env=env).returncode)
    try:main()
    except Exception as error:
        if '--output' in sys.argv:
            p=Path(sys.argv[sys.argv.index('--output')+1])/'receipt.json'
            if p.exists():
                receipt=read(p)
                if receipt.get('pid')==os.getpid():
                    receipt.update(status='failed',exit_code=1,error_type=type(error).__name__,error=str(error));write_json(p,receipt)
        raise
