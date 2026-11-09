"""Matched LoRA utility learning with real observation choices and paid motion.

Three models share both physically collected branches. The moved-memory control
has the same acquisition path but repeats the original frame in its second slot.
Inspection targets follow a detached current continuation policy on train data,
refreshed every40 updates. All model/policy choices freeze before new test data.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
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

from run_mobile_reward_experiment import ROOT,ZONE,write_json,file_hash,worker_init

MODELS={
    'repeat':dict(use_view=False,cost_scale=1.),
    'view_free':dict(use_view=True,cost_scale=0.),
    'view_cost':dict(use_view=True,cost_scale=1.),
}
POLICIES={
    'repeat_direct':('repeat','never'),
    'repeat_moved_memory':('repeat','always'),
    'view_free_direct':('view_free','never'),
    'view_free_always':('view_free','always'),
    'view_free_learned':('view_free','learned'),
    'view_cost_direct':('view_cost','never'),
    'view_cost_always':('view_cost','always'),
    'view_cost_learned':('view_cost','learned'),
}


def collect_scene(spec,folder):
    import numpy as np
    from mobile_active_v3 import restore_scene,execute_active_candidate
    folder=Path(folder);scene=restore_scene(spec);initial=None;begin=time.perf_counter();summaries=[]
    for branch in (0,1):
        destination=folder/str(branch);destination.mkdir(parents=True,exist_ok=True)
        arrivals=np.zeros((4,224,224,3),np.uint8);states=np.zeros((4,19),np.float32)
        commands=np.zeros((4,2,5),np.float32);records=[];post=None
        for b in range(4):
            for g in range(2):
                record,arrays,_,_=execute_active_candidate(scene,b,g,start_pose=spec['start_pose'],
                    inspection=bool(branch),initial_observation=initial)
                if initial is None:initial=(arrays['initial_rgb'],arrays['initial_proprio'])
                candidate_post=(arrays['post_rgb'],arrays['post_proprio'],bool(arrays['post_view_available']))
                if post is None:post=candidate_post
                else:
                    np.testing.assert_allclose(post[1],candidate_post[1],rtol=0,atol=1e-7)
                    if post[2]!=candidate_post[2]:raise AssertionError('Inspection prefix differs between candidate branches.')
                if arrays.get('arrival_rgb') is not None:
                    arrivals[b]=arrays['arrival_rgb'];states[b]=arrays['arrival_proprio']
                if arrays.get('grasp_command') is not None:commands[b,g]=arrays['grasp_command']
                record.update(split=spec['split'],group_id=spec['group_id'],layout=spec['layout'],branch=branch)
                records.append(record)
        np.savez_compressed(destination/'observations.npz',initial_rgb=initial[0],initial_proprio=initial[1],
            post_rgb=post[0],post_proprio=post[1],post_view_available=np.array(post[2]),
            arrival_rgb=arrivals,arrival_proprio=states,grasp_commands=commands)
        write_json(destination/'records.json',records)
        summaries.append(dict(branch=branch,openings=sum(r['success'] for r in records),
                              inspection_success=records[0]['inspection_success'],
                              observation_cost=records[0]['observation_cost']))
    result=dict(scene_id=spec['scene_id'],pid=os.getpid(),split=spec['split'],candidates=16,
                seconds=time.perf_counter()-begin,branches=summaries,independent_full_prefix_per_candidate=True)
    write_json(folder/'receipt.json',result);return result


def collect(specs,args,phase):
    started=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init) as pool:
        futures=[pool.submit(collect_scene,s,args.output/'data'/s['scene_id']) for s in specs]
        for future in as_completed(futures):
            item=future.result();results.append(item)
            progress=dict(pid=os.getpid(),phase=phase,completed_scenes=len(results),total_scenes=len(specs),
                elapsed_seconds=time.perf_counter()-started,recent_scene_seconds=item['seconds'],
                mean_scene_seconds=sum(x['seconds'] for x in results)/len(results))
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    duration=time.perf_counter()-started
    write_json(args.output/(phase+'_timings.json'),dict(seconds=duration,workers=args.workers,
        collection_only=True,gpu_training_concurrent=False,scenes=results))
    return duration


def load_scenes(specs,output):
    import numpy as np
    from mobile_active_v3 import continuation_returns
    scenes=[]
    for spec in specs:
        branches={}
        for branch in (0,1):
            path=output/'data'/spec['scene_id']/str(branch)
            records=json.loads((path/'records.json').read_text(encoding='utf-8'))
            with np.load(path/'observations.npz') as data:arrays={k:v.copy() for k,v in data.items()}
            utility,mask=continuation_returns(records)
            feasible=np.array([any(r['success'] for r in records if r['base_id']==b) for b in range(4)])
            branches[branch]=dict(records=records,arrays=arrays,utility=utility,grasp_mask=mask,feasible_bases=feasible)
        union=np.logical_or(branches[0]['feasible_bases'],branches[1]['feasible_bases'])
        scenes.append(dict(spec=spec,branches=branches,union_bases=union,
                           union_sides=np.array([union[:2].any(),union[2:].any()])))
    return scenes


def _encode_position(model,source,rgb,state):
    from train_hjepa import rgb_tensor,state_tensor
    device=next(model.parameters()).device
    return model.encode_position(rgb_tensor(rgb[None],device),
        state_tensor(state[None],source['state_mean'],source['state_std'],device))


def _position_scores(model,front,other,state):
    import torch
    from mobile_active_v3 import base_commands
    device=front.device
    # Side association comes from the commanded world station, not relative Y.
    # The relative command includes the robot's actual reached current pose.
    latents=torch.cat([front.expand(2,-1),other.expand(2,-1),(front+other)/2],0)
    raw=model.position_policy(torch.cat([latents,torch.tensor(base_commands(state),device=device)],-1)).flatten()
    return torch.cat([raw[:4],raw.new_zeros(1)]).float()  # refusal utility is known0


def predict_pre(model,source,rgb,state):
    import torch
    from mobile_active_v3 import relative_command,pose_from_state,INSPECT_POSE
    z=_encode_position(model,source,rgb,state)
    p=_position_scores(model,z,z,state)
    command=relative_command(INSPECT_POSE,pose_from_state(state))
    inspect=model.position_policy(torch.cat([z,torch.tensor(command[None],device=z.device)],-1)).flatten()[0]
    return p,inspect.float()


def predict_post(model,source,arrays,use_view):
    front=_encode_position(model,source,arrays['initial_rgb'],arrays['initial_proprio'])
    available=bool(arrays['post_view_available']) and use_view
    rgb=arrays['post_rgb'] if available else arrays['initial_rgb']
    state=arrays['post_proprio'] if available else arrays['initial_proprio']
    other=_encode_position(model,source,rgb,state)
    return _position_scores(model,front,other,arrays['post_proprio'])


def grasp_scores(model,source,arrays,canonical):
    import torch
    from train_hjepa import rgb_tensor,state_tensor
    device=next(model.parameters()).device;mean,std=source['state_mean'],source['state_std']
    if canonical:
        scores=[]
        for b in range(4):
            z=model.encode(rgb_tensor(arrays['arrival_rgb'][b:b+1],device),
                           state_tensor(arrays['arrival_proprio'][b:b+1],mean,std,device))
            scores.append(model.grasp_policy(torch.cat([z.expand(2,-1),torch.tensor(arrays['grasp_commands'][b],device=device)],-1)).flatten())
        return torch.stack(scores).float()
    z=model.encode(rgb_tensor(arrays['arrival_rgb'],device),state_tensor(arrays['arrival_proprio'],mean,std,device))
    return model.grasp_policy(torch.cat([z[:,None].expand(-1,2,-1),torch.tensor(arrays['grasp_commands'],device=device)],-1)).squeeze(-1).float()


def forward_training(model,source,scene,config):
    initial=scene['branches'][0]['arrays'];post=scene['branches'][1]['arrays']
    p0,inspect=predict_pre(model,source,initial['initial_rgb'],initial['initial_proprio'])
    p1=predict_post(model,source,post,config['use_view'])
    g0=grasp_scores(model,source,initial,False);g1=grasp_scores(model,source,post,False)
    return p0,inspect,p1,g0,g1


def predictions(model,source,scene,config):
    # The pre-selector is deliberately passed only the actually current frame.
    initial=scene['branches'][0]['arrays'];post=scene['branches'][1]['arrays']
    p0,inspect=predict_pre(model,source,initial['initial_rgb'],initial['initial_proprio'])
    p1=predict_post(model,source,post,config['use_view'])
    return tuple(x.detach().cpu().numpy() for x in (p0,inspect,p1,
        grasp_scores(model,source,initial,True),grasp_scores(model,source,post,True)))


def refresh_targets(model,source,training,config,step):
    import numpy as np
    import torch
    from mobile_active_v3 import continuation_targets
    model.eval();targets={};digest=hashlib.sha256();positive=0;negative=0;post_refusals=0
    with torch.no_grad():
        for scene in training:
            _,_,p1,g0,g1=predictions(model,source,scene,config)
            target=continuation_targets(scene,g0,g1,p1,config['cost_scale'])
            targets[scene['spec']['scene_id']]=target
            positive+=int(target['inspect']>0);negative+=int(target['inspect']<0);post_refusals+=int(p1.argmax()==4)
            digest.update(np.r_[target['p0'],target['p1'],target['inspect']].astype(np.float32).tobytes())
    return targets,dict(model_step=step,training_scenes=len(training),positive_inspection_targets=positive,
                        negative_inspection_targets=negative,teacher_post_refusals=post_refusals,
                        targets_sha256=digest.hexdigest(),test_arrays_opened=0)


def choose(scene,prediction,mode,policy_name):
    import numpy as np
    from mobile_active_v3 import observation_cost
    if mode not in ('never','always','learned'):raise ValueError('Unknown observation policy.')
    p0,inspect,p1,g0,g1=prediction
    action=5 if mode=='always' else int(p0.argmax()) if mode=='never' else int(np.r_[p0,inspect].argmax())
    branch=int(action==5);scan=scene['branches'][branch]['records'][0]['inspection']
    aborted=branch==1 and scan['unsafe_abort']
    b=None if aborted else int(p1.argmax()) if branch else action
    refused=b==4;selected=None;g=None
    if not aborted and not refused:
        scores=g1 if branch else g0;g=int(scores[b].argmax())
        selected=scene['branches'][branch]['records'][2*b+g]
    scan_cost=observation_cost(scan['path_length_m'],scan['duration_s'])
    collision=bool(scan['collision'] or selected and selected['continuation_collision'])
    success=bool(selected and selected['success']);feasible=bool(scene['union_bases'].any())
    utility=(0. if refused else 2.*success-1.)-collision
    side=None if b is None or refused else int(b>=2)
    distance=float(scan['path_length_m']) if selected is None else selected['measured_path_length_m']
    duration=float(scan['duration_s']) if selected is None else selected['episode_duration_s']
    return dict(policy=policy_name,scene_id=scene['spec']['scene_id'],group_id=scene['spec']['group_id'],
        layout=scene['spec']['layout'],branch=branch,base_id=b,grasp_id=g,refused=refused,auto_aborted=bool(aborted),
        feasible=feasible,success=success,collision=collision,nav_success=bool(selected and selected['nav_success']),
        wrong_side=bool(selected and feasible and not scene['union_sides'][side]),
        bad_base=bool(selected and scene['union_sides'][side] and not scene['union_bases'][b]),
        grasp_error=bool(selected and scene['branches'][branch]['feasible_bases'][b] and not selected['success']),
        inspection_attempted=bool(branch),inspection_success=bool(branch and scan['success']),
        inspection_failed=bool(branch and not scan['success']),observation_cost=float(scan_cost),
        path_length_m=float(distance),duration_s=float(duration),terminal_utility=float(utility),
        net_utility=float(utility-scan_cost),pre_scores=p0.tolist(),inspect_score=float(inspect),
        post_scores=p1.tolist() if branch else None)


def summarize(decisions):
    import numpy as np
    from run_mobile_reward_experiment import summarize as old_summary
    summary=old_summary(decisions)
    summary.update(inspections=sum(d['inspection_attempted'] for d in decisions),
        successful_inspections=sum(d['inspection_success'] for d in decisions),
        failed_inspections=sum(d['inspection_failed'] for d in decisions),auto_aborts=sum(d['auto_aborted'] for d in decisions),
        mean_observation_cost=float(np.mean([d['observation_cost'] for d in decisions])),
        mean_path_length_m=float(np.mean([d['path_length_m'] for d in decisions])),
        mean_duration_s=float(np.mean([d['duration_s'] for d in decisions])),
        mean_terminal_utility=float(np.mean([d['terminal_utility'] for d in decisions])),
        mean_net_utility=float(np.mean([d['net_utility'] for d in decisions])))
    return summary


def validation_key(summary):
    return (summary['correct_decisions'],-summary['collisions'],summary['openings'],summary['mean_net_utility'],
            -summary['mean_path_length_m'],-summary['inspections'])


def evaluate(model,source,scenes,model_name):
    import torch
    selected=[(p,m) for p,(name,m) in POLICIES.items() if name==model_name]
    decisions={p:[] for p,_ in selected};model.eval()
    with torch.no_grad():
        for scene in scenes:
            prediction=predictions(model,source,scene,MODELS[model_name])
            for policy,mode in selected:decisions[policy].append(choose(scene,prediction,mode,policy))
    return {p:summarize(d) for p,d in decisions.items()},decisions


def best_policy(summaries):
    winner=next(iter(summaries))
    for p in summaries:
        if validation_key(summaries[p])>validation_key(summaries[winner]):winner=p
    return winner


def train_model(args,name,scenes):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from experiment_reproducibility import seed_everything
    from hjepa_model import load_checkpoint,save_checkpoint
    from hjepa_lora import resume_lora,freeze_core_statistics,frozen_core_digest
    seed_everything(42);torch.set_num_threads(4)
    begin=time.perf_counter();folder=args.output/name;folder.mkdir();config=MODELS[name]
    model,source=load_checkpoint(args.initial,'cuda')
    if model.config.get('reward_decomposition'):raise ValueError('Scalar-head starting checkpoint required.')
    model.config.pop('position_refusal_offset',None);model.config.pop('mobile_v2',None)
    model.config.update(position_views='single',mobile_active_v3=dict(name=name,**config,
        action_features='relative goal x/y/sin delta yaw/cos delta yaw',refusal_utility=0.))
    lora=resume_lora(model,8,16.,.05);model.requires_grad_(False)
    model.position_policy.requires_grad_(True);model.grasp_policy.requires_grad_(True)
    for n,p in model.model.named_parameters():
        if 'lora_' in n:p.requires_grad_(True)
    frozen=frozen_core_digest(model)
    fixed={n:v.detach().cpu().clone() for n,v in model.state_dict().items()
           if 'lora_' not in n and not n.startswith(('position_policy.','grasp_policy.'))}
    initial_lora={n:v.detach().cpu().clone() for n,v in model.model.state_dict().items() if 'lora_' in n}
    training=[s for s in scenes if s['spec']['split']=='train'];validation=[s for s in scenes if s['spec']['split']=='val']
    summaries,_=evaluate(model,source,validation,name);winner=best_policy(summaries);best=validation_key(summaries[winner])
    selected_step=0;history=[dict(step=0,policies=summaries,best_policy=winner)]
    metadata=dict(seed=42,active_model=name,manifest_sha256=file_hash(args.output/'manifest.json'),
                  initialization_sha256=file_hash(args.initial),training_scope='existing rank8 LoRA and scalar P/G only')
    save_checkpoint(folder/'selected.pt',model,source['state_mean'],source['state_std'],step=0,**metadata)
    model.enable_activation_checkpointing(16)
    optimizer=torch.optim.AdamW([
        dict(params=[*model.position_policy.parameters(),*model.grasp_policy.parameters()],lr=3e-4),
        dict(params=[p for p in model.model.parameters() if p.requires_grad],lr=2e-5)],weight_decay=.01)
    targets,event=refresh_targets(model,source,training,config,0);refreshes=[event]
    timings=[];rng=random.Random(42)
    for step in range(1,args.steps+1):
        if step>1 and (step-1)%40==0:
            targets,event=refresh_targets(model,source,training,config,step-1);refreshes.append(event)
            write_json(folder/'target_refreshes.json',refreshes)
        tick=time.perf_counter();sample=rng.sample(training,args.batch_size)
        model.eval();model.model.get_level(1).encoder.encoder.train();freeze_core_statistics(model)
        optimizer.zero_grad();losses=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for scene in sample:
                p0,inspect,p1,g0,g1=forward_training(model,source,scene,config)
                target=targets[scene['spec']['scene_id']]
                lp=F.mse_loss(p0[:4],torch.tensor(target['p0'],device='cuda'))
                lp=lp+(F.mse_loss(p1[:4],torch.tensor(target['p1'],device='cuda')) if target['post_mask'] else p1.sum()*0.)
                lg=[]
                for branch,g in ((0,g0),(1,g1)):
                    data=scene['branches'][branch];mask=torch.tensor(data['grasp_mask'],device='cuda')
                    utility=torch.tensor(data['utility'],device='cuda')
                    lg.append(F.mse_loss(g[mask],utility[mask]) if mask.any() else g.sum()*0.)
                li=F.mse_loss(inspect,torch.tensor(target['inspect'],device='cuda'))
                losses.append(lp+.5*(lg[0]+lg[1])+li)
            loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite active utility loss.')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step();torch.cuda.synchronize();timings.append(time.perf_counter()-tick)
        if step==1 or step%10==0:
            progress=dict(pid=os.getpid(),phase='train_'+name,step=step,total_steps=args.steps,
                loss=float(loss.detach()),seconds_per_step=float(np.mean(timings[-10:])),elapsed_seconds=time.perf_counter()-begin)
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        if step==1 or step%40==0 or step==args.steps:
            summaries,_=evaluate(model,source,validation,name);winner=best_policy(summaries)
            history.append(dict(step=step,policies=summaries,best_policy=winner))
            if validation_key(summaries[winner])>best:
                best=validation_key(summaries[winner]);selected_step=step
                save_checkpoint(folder/'selected.pt',model,source['state_mean'],source['state_std'],step=step,**metadata)
            write_json(folder/'validation_history.json',history)
    save_checkpoint(folder/'last.pt',model,source['state_mean'],source['state_std'],step=args.steps,**metadata)
    if frozen_core_digest(model)!=frozen:raise AssertionError('Original core changed.')
    if any(not torch.equal(v,model.state_dict()[n].detach().cpu()) for n,v in fixed.items()):raise AssertionError('Frozen tensor changed.')
    changed=sum(not torch.equal(v,model.model.state_dict()[n].detach().cpu()) for n,v in initial_lora.items())
    del model,optimizer;torch.cuda.empty_cache();model,source_selected=load_checkpoint(folder/'selected.pt','cuda')
    validation_metrics,_=evaluate(model,source_selected,validation,name);train_metrics,_=evaluate(model,source_selected,training,name)
    report=dict(name=name,config=config,seed=42,steps=args.steps,batch_size=args.batch_size,selected_step=selected_step,
        selected_validation=validation_metrics,training_metrics=train_metrics,history=history,target_refreshes=refreshes,
        frozen_core_unchanged=True,frozen_non_policy_non_adapter_unchanged=True,changed_lora_tensors_last=changed,
        lora=lora,training_step_seconds=timings,test_arrays_opened=0,initial_sha256=file_hash(args.initial),
        selected_sha256=file_hash(folder/'selected.pt'),seconds=time.perf_counter()-begin)
    write_json(folder/'training.json',report);del model;torch.cuda.empty_cache();return report


def live_replay(args,policy,scenes,cached):
    import torch
    import numpy as np
    from hjepa_model import load_checkpoint
    from train_hjepa import rgb_tensor,state_tensor
    from mobile_active_v3 import restore_scene,execute_active_candidate
    name,mode=POLICIES[policy];config=MODELS[name];model,source=load_checkpoint(args.output/name/'selected.pt','cuda');model.eval()
    mean,std=source['state_mean'],source['state_std'];begin=time.perf_counter();matches=[]
    def pre(arrays):
        with torch.no_grad():
            p,i=predict_pre(model,source,arrays['initial_rgb'],arrays['initial_proprio'])
            return 5 if mode=='always' else int(p.argmax()) if mode=='never' else int(torch.cat([p,i[None]]).argmax())
    def post(arrays):
        with torch.no_grad():return int(predict_post(model,source,arrays,config['use_view']).argmax())
    def grasp(rgb,state,commands):
        with torch.no_grad():
            z=model.encode(rgb_tensor(rgb[None],'cuda'),state_tensor(state[None],mean,std,'cuda'))
            return int(model.grasp_policy(torch.cat([z.expand(2,-1),torch.tensor(commands,device='cuda')],-1)).argmax())
    for scene,expected in zip(scenes,cached):
        record,_,trace,_=execute_active_candidate(restore_scene(scene['spec']),None,0,start_pose=scene['spec']['start_pose'],
            pre_selector=pre,post_selector=post,grasp_selector=grasp,keep_trace=True)
        match=(record['inspection_attempted']==expected['inspection_attempted'] and record['refused']==expected['refused']
               and record['success']==expected['success'] and record['collision']==expected['collision'])
        if not record['inspection_unsafe_abort']:match=match and record['base_id']==expected['base_id']
        if record['grasp_observed']:match=match and record['grasp_id']==expected['grasp_id']
        if record['opening_tested']:match=match and record['result']['continuous_state_at_skill_entry']
        match=match and record['base_reset_count']==1
        folder=args.output/'policy_results'/policy/'replays'/scene['spec']['scene_id']
        write_json(folder/'result.json',dict(match=bool(match),record=record));write_json(folder/'trace.json',trace)
        matches.append(bool(match))
        if not match:raise AssertionError('Live observation-policy replay mismatch: '+policy+'/'+scene['spec']['scene_id'])
    result=dict(count=len(matches),all_match=all(matches),seconds=time.perf_counter()-begin)
    write_json(args.output/'policy_results'/policy/'replay_verification.json',result)
    del model;torch.cuda.empty_cache();return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initial',type=Path,default=ROOT/'runs/mobile_crossed_seed42_20261109/fixed_binary_grouped/selected.pt')
    parser.add_argument('--train-groups',type=int,default=18);parser.add_argument('--val-groups',type=int,default=9)
    parser.add_argument('--test-groups',type=int,default=9);parser.add_argument('--steps',type=int,default=480)
    parser.add_argument('--batch-size',type=int,default=2);parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--models',nargs='+',choices=list(MODELS),default=list(MODELS))
    parser.add_argument('--stream',default='mobile_active_v3_20261109');parser.add_argument('--estimate-seconds',type=int,default=21600)
    args=parser.parse_args();args.output=args.output.resolve();args.initial=args.initial.resolve()
    if any(v<1 for v in (args.train_groups,args.val_groups,args.test_groups,args.steps,args.batch_size,args.workers)):
        raise ValueError('Positive group counts/steps/batch/workers required.')
    if args.batch_size>4*args.train_groups:raise ValueError('Batch exceeds training scenes.')
    if len(set(args.models))!=len(args.models):raise ValueError('Duplicate models.')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Preserve runs; choose a new output.')
    if ROOT/'runs' not in args.output.parents:raise ValueError('Outputs must remain inside ignored runs/.')
    args.output.mkdir(parents=True);started=time.perf_counter();now=datetime.now(ZONE)
    receipt=dict(pid=os.getpid(),seed=42,status='starting',queue_estimate_seconds=0,runtime_estimate_seconds=args.estimate_seconds,
        started_at=now.isoformat(),expected_completion=(now+timedelta(seconds=args.estimate_seconds)).isoformat(),
        estimate_basis='v3 smoke:12 scenes/192 full candidates collected in940.752s at2 workers, mean worker scene132.941s, max194.409s; collection-only, no concurrent GPU training. 144 scenes extrapolate11289s wall time. Warm training steps3.11-3.38s; allow1440*3.5s plus train-only teacher refresh/validation.32 physical replays208.168s extrapolate1874s for288; add imports/I/O/conservative margin for total6h. Local launch, queue0. These extrapolations are estimates, not measured full-run durations.')
    write_json(args.output/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    from experiment_reproducibility import seed_everything,worker_environment
    receipt['reproducibility']=seed_everything(42);os.environ.update(worker_environment(42))
    from mobile_active_v3 import make_manifest,geometry_identity,OBS_PATH_WEIGHT,OBS_TIME_WEIGHT
    manifest=make_manifest(dict(train=args.train_groups,val=args.val_groups,test=args.test_groups),args.stream)
    manifest.update(schema_version=3,active_observation_version=3,observation_choices=['no inspection','physical rear inspection'])
    write_json(args.output/'manifest.json',manifest);manifest_hash=file_hash(args.output/'manifest.json')
    historical=set()
    for previous in ('mobile_reward_seed42_20261105','mobile_pipeline_smoke_seed42_20261105',
        'mobile_crossed_smoke_seed42_20261108','mobile_crossed_seed42_20261109','mobile_active_smoke_seed42_20261109',
        'mobile_active_seed42_20261109'):
        path=ROOT/'runs'/previous/'manifest.json'
        if path.exists() and path.parent.resolve()!=args.output:
            historical.update(geometry_identity(s) for s in json.loads(path.read_text(encoding='utf-8'))['scenes'])
    if historical & {s['geometry_sha256'] for s in manifest['scenes']}:raise AssertionError('Historical geometry overlap.')
    policies={p:v for p,v in POLICIES.items() if v[0] in args.models}
    write_json(args.output/'protocol.json',dict(seed=42,models={n:MODELS[n] for n in args.models},policies=policies,
        steps_per_model=args.steps,batch_scenes=args.batch_size,counts=manifest['counts'],group_counts=manifest['group_counts'],
        initial=str(args.initial),initial_sha256=file_hash(args.initial),manifest_sha256=manifest_hash,
        teacher_refresh='train-only detached current G and post P at step0 and every40 updates; no oracle-best continuation target',
        reward='task utility2*success-1-continuation_collision; known refusal0; inspection values actual chosen post-policy return minus cost',
        observation_cost=dict(path_weight_per6m=OBS_PATH_WEIGHT,time_weight_per60s=OBS_TIME_WEIGHT),
        equal_training_compute='3 position +8 arrival images per scene for every model; repeated old frame replaces second-view input in memory control',
        input_features='current RGB/history and real relative goal command; no layout/group/allowed-side/future-result feature',
        failure_contract='unsafe inspection abort stops episode; no invented post-station/G supervision; safe planning failure retains actual state and memory fallback',
        checkpoint_selection='best permitted policy on validation: correct decisions, fewer collisions, openings, net utility, shorter path, fewer inspections; earliest tie',
        policy_selection='same validation tuple, declared policy order ties; all weights/policy selection freeze before test collection',
        task_feasibility='union of measured no-inspection and inspection branches, common denominator across all policies',
        tested_historical_geometry_fingerprints=len(historical),training_scope='P/G and existing rank8 encoder LoRA; no full-core or temporal world-model training',
        previous_goal_turn='progress: six240-step LoRA arms,216 exact replays and input-alias evidence; this turn implements real observation options'))
    protect=[args.initial,ROOT/'runs/mobile_crossed_seed42_20261109/manifest.json',
             ROOT/'runs/mobile_crossed_seed42_20261109/report.json',ROOT/'runs/mobile_crossed_seed42_20261109/frozen_for_test.json']
    protected={str(p):file_hash(p) for p in protect};write_json(args.output/'preserved_hashes.json',protected)
    source=args.output/'source';source.mkdir()
    # Preserve every local Python dependency, including test helpers, rather
    # than relying on an incomplete manually maintained recovery inventory.
    for name in ['AGENTS.md','scene.xml','.gitignore',*sorted(p.name for p in ROOT.glob('*.py'))]:
        shutil.copy2(ROOT/name,source/name)
    hashes={p.name:file_hash(p) for p in source.iterdir()};write_json(args.output/'source_hashes.json',hashes)
    receipt.update(status='collecting_development',startup_seconds=time.perf_counter()-started)
    write_json(args.output/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    phases={};development=[s for s in manifest['scenes'] if s['split']!='test']
    phases['development_collection']=collect(development,args,'development_collection');dev=load_scenes(development,args.output)
    training={};validation_metrics={}
    for name in args.models:
        training[name]=train_model(args,name,dev);phases['training_'+name]=training[name]['seconds']
        validation_metrics.update(training[name]['selected_validation'])
    selected=best_policy(validation_metrics)
    frozen=dict(selected_policy=selected,selected_before_test=True,manifest_sha256=manifest_hash,
        checkpoint_sha256={n:file_hash(args.output/n/'selected.pt') for n in args.models},validation_metrics=validation_metrics)
    write_json(args.output/'frozen_for_test.json',frozen)
    tests=[s for s in manifest['scenes'] if s['split']=='test']
    if any((args.output/'data'/s['scene_id']).exists() for s in tests):raise AssertionError('Test data existed before selection freeze.')
    phases['test_collection']=collect(tests,args,'test_collection');test_data=load_scenes(tests,args.output)
    import torch
    from hjepa_model import load_checkpoint
    metrics={};decisions={};replays={}
    for name in args.models:
        model,source_model=load_checkpoint(args.output/name/'selected.pt','cuda')
        summaries,chosen=evaluate(model,source_model,test_data,name);metrics.update(summaries);decisions.update(chosen)
        del model;torch.cuda.empty_cache()
    for policy in policies:
        write_json(args.output/'policy_results'/policy/'test.json',dict(summary=metrics[policy],decisions=decisions[policy]))
        replays[policy]=live_replay(args,policy,test_data,decisions[policy]);phases['replay_'+policy]=replays[policy]['seconds']
    from check_git_size import audit
    size=audit();write_json(args.output/'git_size_audit.json',size)
    if not size['passed']:raise AssertionError('Repository size guard failed.')
    if any(file_hash(p)!=h for p,h in protected.items()):raise AssertionError('Preserved files changed.')
    if any(file_hash(ROOT/n)!=h for n,h in hashes.items()):raise AssertionError('Source changed during experiment.')
    if file_hash(args.output/'manifest.json')!=manifest_hash:raise AssertionError('Manifest changed.')
    if any(file_hash(args.output/n/'selected.pt')!=h for n,h in frozen['checkpoint_sha256'].items()):raise AssertionError('Frozen model changed.')
    phases['other_startup_io']=time.perf_counter()-started-sum(phases.values())
    report=dict(seed=42,selected_policy=selected,selected_before_test=True,test=metrics,replays=replays,
        phase_seconds=phases,seconds=time.perf_counter()-started,preserved_hashes_unchanged=True,
        source_hashes_unchanged=True,manifest_hash_unchanged=True,frozen_weights_unchanged=True,
        original_core_unchanged=True,training_test_arrays_opened=0,
        limitation='one seed; known geometry navigation and known handle families; offline detached continuation utility supervision, not online RL or temporal world-model training')
    write_json(args.output/'report.json',report)
    receipt.update(status='complete',elapsed_seconds=report['seconds'],phase_seconds=phases,exit_code=0,
        completed_at=datetime.now(ZONE).isoformat());write_json(args.output/'receipt.json',receipt)
    print(json.dumps(dict(status='complete',seconds=report['seconds'],selected_policy=selected,test=metrics)),flush=True)


if __name__=='__main__':
    if os.environ.get('PYTHONHASHSEED')!='42':
        env=dict(os.environ,PYTHONHASHSEED='42',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        raise SystemExit(subprocess.run([sys.executable,*sys.argv],cwd=ROOT,env=env).returncode)
    try:main()
    except Exception as error:
        if '--output' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--output')+1])/'receipt.json'
            if path.exists():
                receipt=json.loads(path.read_text(encoding='utf-8'))
                if receipt.get('pid')==os.getpid():
                    receipt.update(status='failed',exit_code=1,error_type=type(error).__name__,error=str(error))
                    write_json(path,receipt)
        raise
