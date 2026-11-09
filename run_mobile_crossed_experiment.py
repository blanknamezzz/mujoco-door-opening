"""Continue H-JEPA LoRA across reward, scene batching and opening planners.

Six matched arms share a group-disjoint crossed cohort. Both motion planners
are measured before training; their union defines common evaluation feasibility.
Training labels use only the current planner's measured outcomes. Test outcome
collection starts only after all validation choices and weight hashes freeze.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timedelta
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from run_mobile_reward_experiment import ROOT,ZONE,write_json,file_hash,conditions,worker_init

ARMS={
    'fixed_stage_iid':dict(planner='fixed',objective='stage',sampler='iid'),
    'fixed_stage_grouped':dict(planner='fixed',objective='stage',sampler='grouped'),
    'fixed_binary_grouped':dict(planner='fixed',objective='binary',sampler='grouped'),
    'orbit_stage_iid':dict(planner='orbit',objective='stage',sampler='iid'),
    'orbit_stage_grouped':dict(planner='orbit',objective='stage',sampler='grouped'),
    'orbit_binary_grouped':dict(planner='orbit',objective='binary',sampler='grouped'),
}


def collect_scene(spec,folder):
    import numpy as np
    from mobile_crossed_v2 import restore_scene,execute_candidate_v2,PLANNERS
    folder=Path(folder);scene=restore_scene(spec);initial=None;results=[];begin=time.perf_counter()
    for planner in PLANNERS:
        destination=folder/planner;destination.mkdir(parents=True,exist_ok=True)
        arrivals=np.zeros((4,224,224,3),np.uint8);states=np.zeros((4,19),np.float32)
        commands=np.zeros((4,2,5),np.float32);records=[]
        for b in range(4):
            for g in range(2):
                record,arrays,_,_=execute_candidate_v2(scene,b,g,start_pose=spec['start_pose'],
                                                       initial_observation=initial,planner=planner)
                if initial is None:initial=(arrays['initial_rgb'],arrays['initial_proprio'])
                if arrays.get('arrival_rgb') is not None:
                    arrivals[b]=arrays['arrival_rgb'];states[b]=arrays['arrival_proprio']
                if arrays.get('grasp_command') is not None:commands[b,g]=arrays['grasp_command']
                record.update(split=spec['split'],group_id=spec['group_id'],layout=spec['layout'])
                records.append(record)
        np.savez_compressed(destination/'observations.npz',initial_rgb=initial[0],initial_proprio=initial[1],
                            arrival_rgb=arrivals,arrival_proprio=states,grasp_commands=commands)
        write_json(destination/'records.json',records)
        results.append(dict(planner=planner,successes=sum(r['success'] for r in records),
                            tested=sum(r['opening_tested'] for r in records),
                            collisions=sum(r['collision'] for r in records)))
    result=dict(scene_id=spec['scene_id'],pid=os.getpid(),split=spec['split'],candidates=16,
                seconds=time.perf_counter()-begin,planners=results)
    write_json(folder/'receipt.json',result)
    return result


def collect(specs,args,phase):
    started=time.perf_counter();completed=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init) as pool:
        futures=[pool.submit(collect_scene,s,args.output/'data'/s['scene_id']) for s in specs]
        for future in as_completed(futures):
            item=future.result();completed.append(item)
            progress=dict(pid=os.getpid(),phase=phase,completed_scenes=len(completed),total_scenes=len(specs),
                          elapsed_seconds=time.perf_counter()-started,recent_scene_seconds=item['seconds'],
                          mean_scene_seconds=sum(r['seconds'] for r in completed)/len(completed))
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    duration=time.perf_counter()-started
    write_json(args.output/(phase+'_timings.json'),dict(seconds=duration,workers=args.workers,
        collection_only=True,gpu_training_concurrent=False,scenes=completed))
    return duration


def load_scenes(specs,output):
    import numpy as np
    from mobile_crossed_v2 import PLANNERS,factorized_targets
    from staged_mobile_reward import credit_targets
    scenes=[]
    for spec in specs:
        planner_data={}
        for planner in PLANNERS:
            folder=output/'data'/spec['scene_id']/planner
            records=json.loads((folder/'records.json').read_text(encoding='utf-8'))
            with np.load(folder/'observations.npz') as data:arrays={k:v.copy() for k,v in data.items()}
            planner_data[planner]=dict(records=records,arrays=arrays,credit=credit_targets(records),
                                       binary=factorized_targets(records))
        union_bases=np.logical_or.reduce([d['credit']['feasible_bases'] for d in planner_data.values()])
        union_sides=np.array([union_bases[:2].any(),union_bases[2:].any()])
        scenes.append(dict(spec=spec,planners=planner_data,union_bases=union_bases,union_sides=union_sides))
    return scenes


def scene_view(scene,planner):
    return dict(spec=scene['spec'],**scene['planners'][planner])


def policy_logits(model,source,data,canonical=False):
    if not canonical:
        from run_mobile_reward_experiment import logits
        return logits(model,source,data)
    import torch
    from train_hjepa import rgb_tensor,state_tensor
    arrays=data['arrays'];device=next(model.parameters()).device
    mean,std=source['state_mean'],source['state_std']
    z=model.encode_position(rgb_tensor(arrays['initial_rgb'][None],device),
                            state_tensor(arrays['initial_proprio'][None],mean,std,device))
    p=model.position_policy(torch.cat([z.expand(5,-1),torch.tensor(conditions(),device=device)],-1)).flatten()
    grasp=[]
    for b in range(4):
        z=model.encode(rgb_tensor(arrays['arrival_rgb'][b:b+1],device),
                       state_tensor(arrays['arrival_proprio'][b:b+1],mean,std,device))
        g=model.grasp_policy(torch.cat([z.expand(2,-1),torch.tensor(arrays['grasp_commands'][b],device=device)],-1)).flatten()
        grasp.append(g)
    return p.float(),torch.stack(grasp).float()


def decision(scene,planner,b,g):
    data=scene['planners'][planner];refused=b==4
    record=None if refused else data['records'][2*b+g]
    feasible=bool(scene['union_bases'].any());conditional=bool(data['credit']['feasible_bases'].any())
    side=None if refused else int(b>=2)
    return dict(scene_id=scene['spec']['scene_id'],group_id=scene['spec']['group_id'],layout=scene['spec']['layout'],
        base_id=b,grasp_id=g,refused=refused,feasible=feasible,conditional_feasible=conditional,
        success=bool(record and record['success']),collision=bool(record and record['collision']),
        nav_success=bool(record and record['nav_success']),
        wrong_side=bool(not refused and feasible and not scene['union_sides'][side]),
        bad_base=bool(not refused and scene['union_sides'][side] and not scene['union_bases'][b]),
        planner_limitation=bool(not refused and scene['union_bases'][b] and not data['credit']['feasible_bases'][b]),
        grasp_error=bool(not refused and data['credit']['feasible_bases'][b] and not record['success']))


def summarize(decisions):
    from run_mobile_reward_experiment import summarize as original_summary
    result=original_summary(decisions)
    result.update(planner_feasible_scenes=sum(d['conditional_feasible'] for d in decisions),
                  planner_limitations=sum(d['planner_limitation'] for d in decisions),
                  conditional_correct_refusals=sum(d['refused'] and not d['conditional_feasible'] for d in decisions),
                  conditional_false_refusals=sum(d['refused'] and d['conditional_feasible'] for d in decisions))
    return result


def evaluate(model,source,scenes,planner):
    import torch
    model.eval();decisions=[]
    with torch.no_grad():
        for scene in scenes:
            p,g=policy_logits(model,source,scene_view(scene,planner),True)
            b=int(p.argmax());gid=None if b==4 else int(g[b].argmax())
            decisions.append(decision(scene,planner,b,gid))
    return summarize(decisions),decisions


def validation_key(summary):
    return (summary['correct_decisions'],-summary['collisions'],summary['openings'],
            -summary['false_refusals'],-summary['navigation_failures'])


def sample_training(training,sampler,rng):
    if sampler=='iid':return rng.sample(training,4)
    if sampler!='grouped':raise ValueError('Unknown sampler.')
    groups={}
    for scene in training:groups.setdefault(scene['spec']['group_id'],[]).append(scene)
    group=groups[rng.choice(sorted(groups))]
    if len(group)!=4 or len({s['spec']['layout'] for s in group})!=4:raise ValueError('Complete four-layout training group required.')
    return group


def train_arm(args,arm,scenes):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from experiment_reproducibility import seed_everything
    from hjepa_model import load_checkpoint,save_checkpoint
    from hjepa_lora import resume_lora,freeze_core_statistics,frozen_core_digest
    from mobile_crossed_v2 import factorized_loss
    seed_everything(42);torch.set_num_threads(4)
    config=ARMS[arm];folder=args.output/arm;folder.mkdir()
    begin=time.perf_counter();model,source=load_checkpoint(args.initial,'cuda')
    if model.config.get('reward_decomposition'):raise ValueError('Initialize from scalar-head LoRA.')
    model.config.update(position_views='single',mobile_v2=dict(**config,seed=42))
    model.config.pop('position_refusal_offset',None)
    lora=resume_lora(model,8,16.,.05)
    model.requires_grad_(False);model.position_policy.requires_grad_(True);model.grasp_policy.requires_grad_(True)
    for n,p in model.model.named_parameters():
        if 'lora_' in n:p.requires_grad_(True)
    frozen=frozen_core_digest(model)
    immutable={n:v.detach().cpu().clone() for n,v in model.state_dict().items()
               if not n.startswith(('position_policy.','grasp_policy.')) and 'lora_' not in n}
    initial_lora={n:v.detach().cpu().clone() for n,v in model.model.state_dict().items() if 'lora_' in n}
    training=[s for s in scenes if s['spec']['split']=='train'];validation=[s for s in scenes if s['spec']['split']=='val']
    summary,_=evaluate(model,source,validation,config['planner']);best=validation_key(summary)
    history=[dict(step=0,validation=summary)];selected_step=0
    metadata=dict(seed=42,mobile_v2_arm=arm,initial_checkpoint_sha256=file_hash(args.initial),
                  manifest_sha256=file_hash(args.output/'manifest.json'),training_scope='P/G and existing encoder LoRA only')
    save_checkpoint(folder/'selected.pt',model,source['state_mean'],source['state_std'],step=0,**metadata)
    model.enable_activation_checkpointing(16)
    optimizer=torch.optim.AdamW([
        dict(params=[*model.position_policy.parameters(),*model.grasp_policy.parameters()],lr=3e-4),
        dict(params=[p for p in model.model.parameters() if p.requires_grad],lr=2e-5)],weight_decay=.01)
    rng=random.Random(42);timings=[]
    for step in range(1,args.steps+1):
        tick=time.perf_counter();sample=sample_training(training,config['sampler'],rng)
        model.eval();model.model.get_level(1).encoder.encoder.train();freeze_core_statistics(model)
        optimizer.zero_grad();losses=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for scene in sample:
                data=scene_view(scene,config['planner']);p,g=policy_logits(model,source,data)
                if config['objective']=='binary':losses.append(factorized_loss(p,g,data['binary']))
                else:
                    pt=torch.as_tensor(data['credit']['position'],device='cuda')
                    gt=torch.as_tensor(data['credit']['grasp'],device='cuda')
                    mask=torch.as_tensor(data['credit']['grasp_training_mask'],device='cuda').all(-1)
                    lp=F.kl_div((p/.2).log_softmax(-1),(pt/.3).softmax(-1),reduction='sum')
                    lg=F.kl_div((g[mask]/.2).log_softmax(-1),(gt[mask]/.3).softmax(-1),reduction='batchmean') if mask.any() else p.sum()*0.
                    losses.append(lp+lg)
            loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite crossed loss.')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step();torch.cuda.synchronize();timings.append(time.perf_counter()-tick)
        if step==1 or step%10==0:
            progress=dict(pid=os.getpid(),phase='train_'+arm,step=step,total_steps=args.steps,
                          loss=float(loss.detach()),seconds_per_step=float(np.mean(timings[-10:])),
                          elapsed_seconds=time.perf_counter()-begin)
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        if step==1 or step%40==0 or step==args.steps:
            summary,_=evaluate(model,source,validation,config['planner']);history.append(dict(step=step,validation=summary))
            if validation_key(summary)>best:
                best=validation_key(summary);selected_step=step
                save_checkpoint(folder/'selected.pt',model,source['state_mean'],source['state_std'],step=step,**metadata)
            write_json(folder/'validation_history.json',history)
    save_checkpoint(folder/'last.pt',model,source['state_mean'],source['state_std'],step=args.steps,**metadata)
    if frozen_core_digest(model)!=frozen:raise AssertionError('Original core changed.')
    if any(not torch.equal(v,model.state_dict()[n].detach().cpu()) for n,v in immutable.items()):
        raise AssertionError('Frozen non-P/G non-adapter tensor changed.')
    changed=sum(not torch.equal(v,model.model.state_dict()[n].detach().cpu()) for n,v in initial_lora.items())
    del model,optimizer;torch.cuda.empty_cache()
    model,selected_source=load_checkpoint(folder/'selected.pt','cuda')
    selected_validation,_=evaluate(model,selected_source,validation,config['planner'])
    train_summary,_=evaluate(model,selected_source,training,config['planner'])
    report=dict(arm=arm,config=config,seed=42,steps=args.steps,selected_step=selected_step,
                selected_validation=selected_validation,training_summary=train_summary,history=history,
                lora=lora,changed_lora_tensors_last=changed,frozen_core_unchanged=True,
                frozen_non_policy_non_adapter_unchanged=True,test_arrays_opened=0,
                selected_sha256=file_hash(folder/'selected.pt'),initial_sha256=file_hash(args.initial),
                training_step_seconds=timings,seconds=time.perf_counter()-begin)
    write_json(folder/'training.json',report)
    del model;torch.cuda.empty_cache()
    return report


def live_replay(args,arm,scenes,cached):
    import numpy as np
    import torch
    from hjepa_model import load_checkpoint
    from train_hjepa import rgb_tensor,state_tensor
    from mobile_crossed_v2 import execute_candidate_v2,restore_scene
    config=ARMS[arm];model,source=load_checkpoint(args.output/arm/'selected.pt','cuda');model.eval()
    begin=time.perf_counter();mean,std=source['state_mean'],source['state_std'];all_results=[]
    def base_selector(rgb,state):
        with torch.no_grad():
            z=model.encode_position(rgb_tensor(rgb[None],'cuda'),state_tensor(state[None],mean,std,'cuda'))
            return int(model.position_policy(torch.cat([z.expand(5,-1),torch.tensor(conditions(),device='cuda')],-1)).argmax())
    def grasp_selector(rgb,state,commands):
        with torch.no_grad():
            z=model.encode(rgb_tensor(rgb[None],'cuda'),state_tensor(state[None],mean,std,'cuda'))
            return int(model.grasp_policy(torch.cat([z.expand(2,-1),torch.tensor(commands,device='cuda')],-1)).argmax())
    for scene,expected in zip(scenes,cached):
        record,_,trace,_=execute_candidate_v2(restore_scene(scene['spec']),None,0,
            start_pose=scene['spec']['start_pose'],base_selector=base_selector,grasp_selector=grasp_selector,
            planner=config['planner'],keep_trace=True)
        match=(record['base_id']==expected['base_id'] and record['success']==expected['success'] and record['collision']==expected['collision'])
        if record.get('grasp_observed'):match=match and record['grasp_id']==expected['grasp_id']
        if record.get('opening_tested'):match=match and record['result']['continuous_state_at_skill_entry']
        match=match and record['base_reset_count']==1
        folder=args.output/arm/'replays'/scene['spec']['scene_id']
        write_json(folder/'result.json',dict(match=bool(match),record=record))
        write_json(folder/'trace.json',trace);all_results.append(bool(match))
        if not match:raise AssertionError('Frozen live replay mismatch: '+arm+'/'+scene['spec']['scene_id'])
    report=dict(count=len(all_results),all_match=all(all_results),seconds=time.perf_counter()-begin)
    write_json(args.output/arm/'replay_verification.json',report)
    del model;torch.cuda.empty_cache()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initial',type=Path,default=ROOT/'runs/mobile_reward_seed42_20261105/decomposed/selected.pt')
    parser.add_argument('--train-groups',type=int,default=18);parser.add_argument('--val-groups',type=int,default=9)
    parser.add_argument('--test-groups',type=int,default=9);parser.add_argument('--steps',type=int,default=240)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--estimate-seconds',type=int,default=18000)
    parser.add_argument('--stream',default='mobile_crossed_v2_20261108')
    parser.add_argument('--arms',nargs='+',choices=list(ARMS),default=list(ARMS))
    args=parser.parse_args();args.output=args.output.resolve();args.initial=args.initial.resolve()
    if any(v<1 for v in (args.train_groups,args.val_groups,args.test_groups,args.steps,args.workers)):
        raise ValueError('Positive group counts/steps/workers required.')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Choose a new output directory.')
    if ROOT/'runs' not in args.output.parents:raise ValueError('Keep experiment outputs in ignored runs/.')
    if len(set(args.arms))!=len(args.arms):raise ValueError('Duplicate arms are not allowed.')
    args.output.mkdir(parents=True);started=time.perf_counter();now=datetime.now(ZONE)
    receipt=dict(pid=os.getpid(),seed=42,status='starting',queue_estimate_seconds=0,
                 runtime_estimate_seconds=args.estimate_seconds,started_at=now.isoformat(),
                 expected_completion=(now+timedelta(seconds=args.estimate_seconds)).isoformat(),
                 estimate_basis=f'v1 collection-only1033s/48 scenes at2 workers. New cohort{4*(args.train_groups+args.val_groups+args.test_groups)} scenes x2 planners; allow15% additional collection margin. v1 training3s/update batch3 => estimated4s/update batch4 x{args.steps} x{len(args.arms)} arms. Replay allowance14.3s x{4*args.test_groups} cases x{len(args.arms)} arms. Declared duration is a conservative estimate, not measured execution time.')
    comparable=ROOT/'runs/mobile_crossed_smoke_seed42_20261108/report.json'
    if comparable.exists():
        calibration=json.loads(comparable.read_text(encoding='utf-8'))
        duration=calibration['phase_seconds']
        measured_collection=duration['development_collection']+duration['test_collection']
        replay_seconds=sum(v['seconds'] for v in calibration['replays'].values())
        replay_cases=sum(v['count'] for v in calibration['replays'].values())
        receipt['measured_comparable']=dict(path=str(comparable),sha256=file_hash(comparable),
            collection_scenes=12,collection_seconds=measured_collection,workers=2,collection_only=True,
            replay_cases=replay_cases,replay_seconds=replay_seconds,
            source='independent-stream integration smoke; not a performance estimate or final test')
        receipt['estimate_basis']=f'Measured v2 collection-only {measured_collection:.2f}s/12 scenes,2 workers; scale to {4*(args.train_groups+args.val_groups+args.test_groups)} scenes. Allow4s/update atbatch4 x{args.steps} x{len(args.arms)} arms versus measured2.34-2.90s/update in smoke. Replay measured{replay_seconds/replay_cases:.2f}s/case; allow9s x{4*args.test_groups} x{len(args.arms)}. Add startup/validation/I/O margin. Declared whole-job duration is conservative, not measured time.'
    write_json(args.output/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    from experiment_reproducibility import seed_everything,worker_environment
    receipt['reproducibility']=seed_everything(42);os.environ.update(worker_environment(42))
    from mobile_crossed_v2 import make_manifest,geometry_identity
    manifest=make_manifest(dict(train=args.train_groups,val=args.val_groups,test=args.test_groups),args.stream)
    write_json(args.output/'manifest.json',manifest);manifest_hash=file_hash(args.output/'manifest.json')
    historical=set()
    for previous in ('mobile_reward_seed42_20261105','mobile_pipeline_smoke_seed42_20261105',
                     'mobile_crossed_smoke_seed42_20261108'):
        old=json.loads((ROOT/'runs'/previous/'manifest.json').read_text(encoding='utf-8'))
        historical.update(geometry_identity(s) for s in old['scenes'])
    if historical & {s['geometry_sha256'] for s in manifest['scenes']}:raise AssertionError('Historical geometry overlap.')
    write_json(args.output/'protocol.json',dict(seed=42,arms={a:ARMS[a] for a in args.arms},
        steps_per_arm=args.steps,batch_scenes=4,counts=manifest['counts'],group_counts=manifest['group_counts'],
        initial=str(args.initial),initial_sha256=file_hash(args.initial),manifest_sha256=manifest_hash,
        grouped_sampler='one complete four-layout group per update; group-disjoint train/val/test',
        iid_sampler='four independently sampled scenes from the identical training cohort',
        stage_objective='v1 stage-credit KL at P and visible G; P temperature0.2, target0.3',
        binary_objective='side BCE relative to refusal, positive-side gradient pooled over measured feasible bases only; negative side all bases; +0.25 base BCE +collision-weighted visible G success BCE',
        motion='fixed base versus base pose following the hand hinge arc, identical navigation, preparation, grip and passive-door physics',
        evaluation_feasibility='common union of both planners; conditional feasibility also reported; training labels are planner-specific',
        checkpoint_rule='validation common-union correct decisions, fewer collisions, openings, fewer false refusals/navigation failures; earliest tie',
        arm_tie_order=args.arms,test_used_for_selection=False,training_scope='existing rank8 LoRA and P/G only',
        inference_inputs='reached current single-front RGB and base commands, then reached arrival RGB and grasp commands; no group/side/outcome labels',
        historical_geometry_fingerprints_checked=len(historical),previous_goal_turn='progress: implemented and physically validated mobile v1; this turn starts new evidence-driven axes'))
    preserve=[args.initial,ROOT/'runs/hjepa/refusal_model_seed42/selected.pt',
              ROOT/'runs/mobile_reward_seed42_20261105/manifest.json',ROOT/'runs/mobile_reward_seed42_20261105/report.json',
              ROOT/'runs/mobile_crossed_smoke_seed42_20261108/manifest.json',
              ROOT/'runs/mobile_crossed_smoke_seed42_20261108/report.json']
    protected={str(p):file_hash(p) for p in preserve};write_json(args.output/'preserved_hashes.json',protected)
    source=args.output/'source';source.mkdir()
    for name in ('AGENTS.md','mobile_crossed_v2.py','run_mobile_crossed_experiment.py','mobile_door.py',
                 'run_mobile_reward_experiment.py','staged_mobile_reward.py','door_stress_scenes.py','simulate.py',
                 'hjepa_model.py','hjepa_lora.py','experiment_reproducibility.py','scene.xml','perception.py',
                 'prm.py','grasp_plan.py','hjepa_scene.py','train_hjepa.py','hjepa_runtime.py','test_mobile_crossed_v2.py'):
        shutil.copy2(ROOT/name,source/name)
    hashes={p.name:file_hash(p) for p in source.iterdir()};write_json(args.output/'source_hashes.json',hashes)
    receipt.update(status='collecting_development',startup_seconds=time.perf_counter()-started)
    write_json(args.output/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    phases={};development=[s for s in manifest['scenes'] if s['split']!='test']
    phases['development_collection']=collect(development,args,'development_collection')
    dev=load_scenes(development,args.output);training={}
    for arm in args.arms:
        training[arm]=train_arm(args,arm,dev);phases['training_'+arm]=training[arm]['seconds']
    selected=args.arms[0]
    for arm in args.arms[1:]:
        if validation_key(training[arm]['selected_validation'])>validation_key(training[selected]['selected_validation']):selected=arm
    frozen=dict(selected_arm=selected,selected_before_test=True,manifest_sha256=manifest_hash,
                checkpoint_sha256={a:file_hash(args.output/a/'selected.pt') for a in args.arms},
                selected_validation={a:training[a]['selected_validation'] for a in args.arms})
    write_json(args.output/'frozen_for_test.json',frozen)
    tests=[s for s in manifest['scenes'] if s['split']=='test']
    if any((args.output/'data'/s['scene_id']).exists() for s in tests):raise AssertionError('Test outcomes existed before freeze.')
    phases['test_collection']=collect(tests,args,'test_collection');test_data=load_scenes(tests,args.output)
    import torch
    from hjepa_model import load_checkpoint
    metrics={};replays={};decisions={}
    for arm in args.arms:
        model,source_model=load_checkpoint(args.output/arm/'selected.pt','cuda')
        metrics[arm],decisions[arm]=evaluate(model,source_model,test_data,ARMS[arm]['planner'])
        write_json(args.output/arm/'test.json',dict(summary=metrics[arm],decisions=decisions[arm]))
        del model;torch.cuda.empty_cache()
        replays[arm]=live_replay(args,arm,test_data,decisions[arm]);phases['replay_'+arm]=replays[arm]['seconds']
    from check_git_size import audit
    size=audit();write_json(args.output/'git_size_audit.json',size)
    if not size['passed']:raise AssertionError('Repository size guard failed.')
    if any(file_hash(p)!=digest for p,digest in protected.items()):raise AssertionError('Preserved file changed.')
    if any(file_hash(ROOT/name)!=digest for name,digest in hashes.items()):raise AssertionError('Source changed during experiment.')
    if file_hash(args.output/'manifest.json')!=manifest_hash:raise AssertionError('Manifest changed.')
    if any(file_hash(args.output/a/'selected.pt')!=d for a,d in frozen['checkpoint_sha256'].items()):raise AssertionError('Frozen weights changed.')
    import numpy as np
    coverage={p:sum(s['planners'][p]['credit']['feasible_bases'].any() for s in test_data) for p in ('fixed','orbit')}
    coverage={p:int(v) for p,v in coverage.items()}
    phases['other_startup_io']=time.perf_counter()-started-sum(phases.values())
    report=dict(seed=42,selected_arm=selected,selected_before_test=True,test=metrics,replays=replays,
                candidate_coverage=coverage,union_feasible_scenes=int(sum(s['union_bases'].any() for s in test_data)),
                phase_seconds=phases,seconds=time.perf_counter()-started,source_hashes_unchanged=True,
                protected_hashes_unchanged=True,manifest_hash_unchanged=True,frozen_weights_unchanged=True,
                original_core_unchanged=True,training_test_arrays_opened=0,
                limitation='one seed; group-crossed known handle families; known-geometry navigation; no learned exploration or temporal world-model update')
    write_json(args.output/'report.json',report)
    receipt.update(status='complete',elapsed_seconds=report['seconds'],phase_seconds=phases,exit_code=0,
                   completed_at=datetime.now(ZONE).isoformat());write_json(args.output/'receipt.json',receipt)
    print(json.dumps(dict(status='complete',seconds=report['seconds'],selected_arm=selected,test=metrics)),flush=True)


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
