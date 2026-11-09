"""Matched endpoint LoRA study with continuous mobile physical counterfactuals.

Train/validation collection and checkpoint selection precede all test outcome
collection. The frozen test manifest is written at launch. No full-core or
temporal world-model training occurs. Navigation is a known-geometry baseline.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
ZONE=timezone(timedelta(hours=8))


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def generator(split,stream):
    import numpy as np
    key=int.from_bytes(hashlib.sha256((stream+'/'+split).encode()).digest()[:4],'little')
    return np.random.default_rng(np.random.SeedSequence([42,key]))


def make_manifest(counts,stream='mobile_credit_v1_20261105'):
    from door_stress_scenes import scenario_suite,Obstacle
    templates=[s for s in scenario_suite() if s.handle_style in ('lever','t-bar','d-pull','offset-bar')]
    specs=[]
    for split,count in counts.items():
        rng=generator(split,stream)
        for i in range(count):
            scene=templates[i%len(templates)]
            obstacles=tuple(replace(o,pos=(o.pos[0]+float(rng.uniform(-.015,.015)),
                                           o.pos[1]+float(rng.uniform(-.015,.015)),o.pos[2])) for o in scene.obstacles)
            both_blocked=i%4==3
            if both_blocked:
                obstacles += tuple(replace(o,name='obstacle_mirrored_'+o.name,
                                            pos=(o.pos[0],-o.pos[1],o.pos[2]),
                                            euler_deg=(o.euler_deg[0],o.euler_deg[1],-o.euler_deg[2])) for o in obstacles)
            scene=replace(scene,scene_id=f'{stream}_{split}_{i:03d}',obstacles=obstacles,
                          handle_x=scene.handle_x+float(rng.uniform(-.018,.018)),
                          handle_z=scene.handle_z+float(rng.uniform(-.009,.009)),
                          handle_friction=float(rng.uniform(.085,.3)),
                          hinge_resistance=float(rng.uniform(.9,1.4)))
            body=asdict(scene)
            body.update(split=split,template_id=templates[i%len(templates)].scene_id,
                        both_sides_obstructed_design=both_blocked,
                        start_pose=[.6+float(rng.uniform(-.035,.035)),-1.65,math.pi/2])
            identity={k:v for k,v in body.items() if k not in ('scene_id','split','description')}
            body['geometry_sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            specs.append(body)
    if len({s['geometry_sha256'] for s in specs})!=len(specs):raise AssertionError('Repeated geometry.')
    return dict(schema_version=1,seed=42,stream=stream,
                scope='new continuous-geometry variants of known four supported handle profiles; not unseen-family generalization',
                scenes=specs,counts=counts,test_outcomes_collected=False)


def restore_scene(spec):
    from door_stress_scenes import StressDoorScene,Obstacle
    fields={k:v for k,v in spec.items() if k not in (
        'split','template_id','both_sides_obstructed_design','start_pose','geometry_sha256')}
    fields['obstacles']=tuple(Obstacle(**o) for o in fields['obstacles'])
    fields['handle_faces']=tuple(fields['handle_faces'])
    return StressDoorScene(**fields)


def worker_init():
    from experiment_reproducibility import seed_everything
    seed_everything(42)
    import torch
    torch.set_num_threads(1)


def collect_scene(spec,folder):
    import numpy as np
    from mobile_door import execute_candidate
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    scene=restore_scene(spec);records=[];initial=None
    arrival=np.zeros((4,224,224,3),np.uint8)
    arrival_state=np.zeros((4,19),np.float32)
    commands=np.zeros((4,2,5),np.float32)
    begin=time.perf_counter()
    for b in range(4):
        for g in range(2):
            record,arrays,_,_=execute_candidate(scene,b,g,start_pose=spec['start_pose'],initial_observation=initial)
            if initial is None:initial=(arrays['initial_rgb'],arrays['initial_proprio'])
            if arrays.get('arrival_rgb') is not None:
                arrival[b]=arrays['arrival_rgb'];arrival_state[b]=arrays['arrival_proprio']
            if arrays.get('grasp_command') is not None:commands[b,g]=arrays['grasp_command']
            record['split']=spec['split'];records.append(record)
    np.savez_compressed(folder/'observations.npz',initial_rgb=initial[0],initial_proprio=initial[1],
                        arrival_rgb=arrival,arrival_proprio=arrival_state,grasp_commands=commands)
    write_json(folder/'records.json',records)
    result=dict(scene_id=scene.scene_id,split=spec['split'],candidates=len(records),
                seconds=time.perf_counter()-begin,openings=sum(r['success'] for r in records),
                nav_arrivals=sum(r['nav_success'] for r in records),
                opening_attempts=sum(r['opening_tested'] for r in records),
                scene_complete=True)
    write_json(folder/'receipt.json',result)
    return result


def collect(specs,output,workers,phase):
    start=time.perf_counter();results=[]
    # No training is concurrent with collection. Separate processes isolate the
    # native renderer and use CPU MuJoCo; throughput is measured per phase.
    with ProcessPoolExecutor(max_workers=workers,initializer=worker_init) as pool:
        futures={pool.submit(collect_scene,s,output/'data'/s['scene_id']):s for s in specs}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            progress=dict(phase=phase,pid=os.getpid(),completed_scenes=len(results),total_scenes=len(specs),
                          candidates=sum(r['candidates'] for r in results),elapsed_seconds=time.perf_counter()-start,
                          recent_scene_seconds=result['seconds'],mean_scene_seconds=sum(r['seconds'] for r in results)/len(results))
            write_json(output/'progress.json',progress)
            print(json.dumps(progress),flush=True)
    write_json(output/(phase+'_timings.json'),dict(seconds=time.perf_counter()-start,workers=workers,
        collection_only=True,gpu_training_concurrent=False,scenes=results))
    return time.perf_counter()-start


def load_scenes(specs,output):
    import numpy as np
    from staged_mobile_reward import credit_targets
    result=[]
    for spec in specs:
        folder=output/'data'/spec['scene_id']
        records=json.loads((folder/'records.json').read_text(encoding='utf-8'))
        with np.load(folder/'observations.npz') as data:arrays={k:v.copy() for k,v in data.items()}
        result.append(dict(spec=spec,records=records,arrays=arrays,credit=credit_targets(records)))
    return result


def conditions():
    import numpy as np
    from mobile_door import BASE_POSES
    return np.array([[x,y,math.sin(yaw),math.cos(yaw)] for x,y,yaw in BASE_POSES]+[[0.,0.,0.,0.]],np.float32)


def logits(model,source,scene,grad=False):
    import numpy as np
    import torch
    from train_hjepa import rgb_tensor,state_tensor
    arrays=scene['arrays'];device=next(model.parameters()).device
    mean,std=source['state_mean'],source['state_std']
    initial=model.encode_position(rgb_tensor(arrays['initial_rgb'][None],device),
                                  state_tensor(arrays['initial_proprio'][None],mean,std,device))
    cmd=torch.tensor(conditions(),device=device)
    p=model.position_policy(torch.cat([initial.expand(5,-1),cmd],-1)).flatten()
    arrival=model.encode(rgb_tensor(arrays['arrival_rgb'],device),
                         state_tensor(arrays['arrival_proprio'],mean,std,device))
    features=torch.tensor(arrays['grasp_commands'],device=device)
    g=model.grasp_policy(torch.cat([arrival[:,None].expand(-1,2,-1),features],-1)).squeeze(-1)
    return p.float(),g.float()


def selection_key(summary):
    return (summary['correct_decisions'],-summary['collisions'],summary['openings'],-summary['navigation_failures'])


def summarize(decisions):
    import numpy as np
    return dict(scenes=len(decisions),feasible_scenes=sum(d['feasible'] for d in decisions),
                openings=sum(d['success'] for d in decisions),correct_refusals=sum(d['refused'] and not d['feasible'] for d in decisions),
                false_refusals=sum(d['refused'] and d['feasible'] for d in decisions),
                correct_decisions=sum(d['success'] or d['refused'] and not d['feasible'] for d in decisions),
                collisions=sum(d['collision'] for d in decisions),
                navigation_failures=sum(not d['refused'] and not d['nav_success'] for d in decisions),
                wrong_side_errors=sum(d['wrong_side'] for d in decisions),
                within_side_base_errors=sum(d['bad_base'] for d in decisions),
                grasp_errors=sum(d['grasp_error'] for d in decisions),
                mean_terminal_utility=float(np.mean([0. if d['refused'] else 2.*d['success']-1.-d['collision'] for d in decisions])))


def evaluate(model,source,scenes,fixed=None):
    import torch
    decisions=[]
    model.eval()
    with torch.no_grad():
        for scene in scenes:
            if fixed is None:
                p,g=logits(model,source,scene)
                b=int(p.argmax());gid=None if b==4 else int(g[b].argmax())
            else:b,gid=fixed
            credit=scene['credit'];refused=b==4;feasible=bool(credit['feasible_bases'].any())
            selected=None if refused else scene['records'][2*b+gid]
            side=None if refused else (0 if b<2 else 1)
            decisions.append(dict(scene_id=scene['spec']['scene_id'],base_id=b,grasp_id=gid,refused=refused,
                feasible=feasible,success=bool(selected and selected['success']),collision=bool(selected and selected['collision']),
                nav_success=bool(selected and selected['nav_success']),
                wrong_side=bool(not refused and feasible and not credit['feasible_sides'][side]),
                bad_base=bool(not refused and credit['feasible_sides'][side] and not credit['feasible_bases'][b]),
                grasp_error=bool(not refused and credit['feasible_bases'][b] and not selected['success'])))
    return summarize(decisions),decisions


def train_arm(args,arm,scenes):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from experiment_reproducibility import seed_everything
    from hjepa_model import load_checkpoint,save_checkpoint
    from hjepa_lora import resume_lora,freeze_core_statistics,frozen_core_digest
    from staged_mobile_reward import shared_targets
    seed_everything(42);torch.set_num_threads(4)
    folder=args.output/arm;folder.mkdir()
    begin=time.perf_counter()
    model,source=load_checkpoint(args.initial,'cuda')
    if model.config.get('reward_decomposition'):raise ValueError('Initialize both arms from the same scalar policy.')
    model.config['position_views']='single'
    model.config.pop('position_refusal_offset',None)
    lora=resume_lora(model,8,16.,.05)
    model.requires_grad_(False)
    model.position_policy.requires_grad_(True);model.grasp_policy.requires_grad_(True)
    for n,p in model.model.named_parameters():
        if 'lora_' in n:p.requires_grad_(True)
    original=frozen_core_digest(model)
    frozen_task={n:v.detach().cpu().clone() for n,v in model.state_dict().items()
                 if not n.startswith(('position_policy.','grasp_policy.')) and 'lora_' not in n}
    initial_lora={n:v.detach().cpu().clone() for n,v in model.model.state_dict().items() if 'lora_' in n}
    train=[s for s in scenes if s['spec']['split']=='train'];val=[s for s in scenes if s['spec']['split']=='val']
    # The common control's G distribution is frozen before either arm trains.
    # This computation opens train/validation arrays only.
    model.eval()
    with torch.no_grad():
        for scene in train:
            _,g=logits(model,source,scene)
            scene['shared']=shared_targets(scene['records'],(g/.2).softmax(-1).cpu().numpy())
    baseline,_=evaluate(model,source,val)
    best=selection_key(baseline);selected=0;history=[dict(step=0,validation=baseline)]
    metadata=dict(seed=42,mobile_reward_arm=arm,training_scope='endpoint P/G and encoder LoRA; no temporal world-model training',
                  scene_manifest_sha256=file_hash(args.output/'manifest.json'))
    save_checkpoint(folder/'selected.pt',model,source['state_mean'],source['state_std'],step=0,**metadata)
    model.enable_activation_checkpointing(16)
    groups=[dict(params=[*model.position_policy.parameters(),*model.grasp_policy.parameters()],lr=3e-4),
            dict(params=[p for p in model.model.parameters() if p.requires_grad],lr=2e-5)]
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    rng=random.Random(42);timings=[]
    for step in range(1,args.steps+1):
        tick=time.perf_counter();sample=rng.sample(train,min(args.batch_size,len(train)))
        model.eval();model.model.get_level(1).encoder.encoder.train();freeze_core_statistics(model)
        optimizer.zero_grad();losses=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for scene in sample:
                p,g=logits(model,source,scene,True)
                target=scene['credit'] if arm=='decomposed' else scene['shared']
                pt=torch.tensor(target['position'],device='cuda')
                gt=torch.tensor(target['grasp'],device='cuda')
                mask=torch.tensor(scene['credit']['grasp_training_mask'],device='cuda').all(-1)
                lp=F.kl_div((p/.2).log_softmax(-1),(pt/.3).softmax(-1),reduction='sum')
                lg=(F.kl_div((g[mask]/.2).log_softmax(-1),(gt[mask]/.3).softmax(-1),reduction='batchmean')
                    if mask.any() else p.sum()*0.)
                losses.append(lp+lg)
            loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite mobile reward loss.')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step();torch.cuda.synchronize();timings.append(time.perf_counter()-tick)
        if step==1 or step%10==0:
            progress=dict(phase='training_'+arm,pid=os.getpid(),step=step,total_steps=args.steps,
                          loss=float(loss.detach()),seconds_per_step=float(np.mean(timings[-10:])),elapsed_seconds=time.perf_counter()-begin)
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        if step==1 or step%20==0 or step==args.steps:
            summary,_=evaluate(model,source,val);history.append(dict(step=step,validation=summary))
            if selection_key(summary)>best:
                best=selection_key(summary);selected=step
                save_checkpoint(folder/'selected.pt',model,source['state_mean'],source['state_std'],step=step,**metadata)
            write_json(folder/'validation_history.json',history)
    save_checkpoint(folder/'last.pt',model,source['state_mean'],source['state_std'],step=args.steps,**metadata)
    if frozen_core_digest(model)!=original:raise AssertionError('Original H-JEPA core changed.')
    if any(not torch.equal(v,model.state_dict()[n].detach().cpu()) for n,v in frozen_task.items()):
        raise AssertionError('Frozen non-P/G non-adapter tensor changed.')
    changed=sum(not torch.equal(v,model.model.state_dict()[n].detach().cpu()) for n,v in initial_lora.items())
    del optimizer,model;torch.cuda.empty_cache()
    model,source=load_checkpoint(folder/'selected.pt','cuda')
    val_result,_=evaluate(model,source,val);train_result,_=evaluate(model,source,train)
    report=dict(arm=arm,seed=42,steps=args.steps,selected_step=selected,baseline_validation=baseline,
                selected_validation=val_result,training_summary=train_result,history=history,
                frozen_core_unchanged=True,frozen_non_policy_non_adapter_unchanged=True,
                changed_lora_tensors_last=changed,lora=lora,training_step_seconds=timings,
                seconds=time.perf_counter()-begin,initial_sha256=file_hash(args.initial),test_arrays_opened=0,
                selected_sha256=file_hash(folder/'selected.pt'),precision='BF16 training; FP32 deterministic evaluation')
    write_json(folder/'training.json',report)
    del model;torch.cuda.empty_cache()
    return report


def live_replays(args,arm,scenes,cached):
    import numpy as np
    import torch
    from hjepa_model import load_checkpoint
    from mobile_door import execute_candidate
    from train_hjepa import rgb_tensor,state_tensor
    model,source=load_checkpoint(args.output/arm/'selected.pt','cuda');model.eval()
    mean,std=source['state_mean'],source['state_std'];results=[];begin=time.perf_counter()
    def choose_base(rgb,state):
        with torch.no_grad():
            z=model.encode_position(rgb_tensor(rgb[None],'cuda'),state_tensor(state[None],mean,std,'cuda'))
            command=torch.tensor(conditions(),device='cuda')
            return int(model.position_policy(torch.cat([z.expand(5,-1),command],-1)).argmax())
    def choose_grasp(rgb,state,commands):
        with torch.no_grad():
            z=model.encode(rgb_tensor(rgb[None],'cuda'),state_tensor(state[None],mean,std,'cuda'))
            return int(model.grasp_policy(torch.cat([z.expand(2,-1),torch.tensor(commands,device='cuda')],-1)).argmax())
    for scene,decision in zip(scenes,cached):
        record,_,trace,_=execute_candidate(restore_scene(scene['spec']),None,0,start_pose=scene['spec']['start_pose'],
                                          keep_trace=True,base_selector=choose_base,grasp_selector=choose_grasp)
        refusal=record['base_id']==4
        match=(record['base_id']==decision['base_id'] and record['success']==decision['success']
               and record['collision']==decision['collision'])
        if not refusal and record.get('grasp_observed'):
            match=match and record['grasp_id']==decision['grasp_id']
        if record.get('opening_tested'):
            match=match and record['result']['continuous_state_at_skill_entry'] and record['base_reset_count']==1
        item=dict(scene_id=record['scene_id'],match=bool(match),record=record)
        results.append(item)
        folder=args.output/arm/'replays'/record['scene_id']
        write_json(folder/'result.json',item);write_json(folder/'trace.json',trace)
        if not match:raise AssertionError(f'Live physical replay mismatch: {arm}/{record["scene_id"]}')
    report=dict(count=len(results),all_match=all(r['match'] for r in results),seconds=time.perf_counter()-begin)
    write_json(args.output/arm/'replay_verification.json',report)
    del model;torch.cuda.empty_cache()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initial',type=Path,default=ROOT/'runs/hjepa/refusal_model_seed42/selected.pt')
    parser.add_argument('--train',type=int,default=24);parser.add_argument('--val',type=int,default=8)
    parser.add_argument('--test',type=int,default=16);parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--batch-size',type=int,default=3);parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--estimate-seconds',type=int,default=3600)
    parser.add_argument('--stream',default='mobile_credit_v1_20261105')
    args=parser.parse_args();args.output=args.output.resolve();args.initial=args.initial.resolve()
    if any(v<1 for v in (args.train,args.val,args.test,args.steps,args.batch_size,args.workers)):
        raise ValueError('Positive scene, step, batch and worker counts required.')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Use a new ignored runs directory.')
    if ROOT/'runs' not in args.output.parents:raise ValueError('Experiment outputs must remain inside ignored runs/.')
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();now=datetime.now(ZONE)
    receipt=dict(pid=os.getpid(),seed=42,status='starting',queue_estimate_seconds=0,
                 runtime_estimate_seconds=args.estimate_seconds,started_at=now.isoformat(),
                 expected_completion=(now+timedelta(seconds=args.estimate_seconds)).isoformat(),
                 estimate_basis=f'Preflight collection-only: 6 scenes, ~45.4s/scene at 2 workers; extrapolate {args.train+args.val+args.test} scenes with margin. Preflight LoRA ~0.9s/step at batch1; scale to batch{args.batch_size} x {args.steps} steps x 2 arms. Allow 32 replay episodes x14.3s plus startup/validation/I/O. Conservative whole-job estimate, not measured execution time.')
    write_json(args.output/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    from experiment_reproducibility import seed_everything,worker_environment
    receipt['reproducibility']=seed_everything(42)
    os.environ.update(worker_environment(42))
    from staged_mobile_reward import reward_contract
    from mobile_door import MobileSettings
    from check_git_size import audit
    manifest=make_manifest(dict(train=args.train,val=args.val,test=args.test),args.stream)
    write_json(args.output/'manifest.json',manifest)
    manifest_hash=file_hash(args.output/'manifest.json')
    protocol=dict(seed=42,counts=manifest['counts'],steps_per_arm=args.steps,batch_size=args.batch_size,
        initialization=str(args.initial),initial_sha256=file_hash(args.initial),
        frozen_test_manifest_sha256=manifest_hash,reward_contract=reward_contract(),
        mobile_settings=asdict(MobileSettings()),base_footprint='circumscribed circle +15mm; arm/environment clearance8mm',
        position_commands=conditions().tolist(),grasp_angles_deg=[-15.,22.],grip_force_N=6.,
        stopping_rule=f'fixed {args.steps} steps per arm; validation every 20 steps plus step0/1; no test-driven changes',
        checkpoint_selection='validation correct decisions, fewer collisions, openings, fewer navigation failures; earliest exact tie',
        arm_selection='same validation tuple; exact cross-arm tie favors shared control',
        scope='single reached front RGB at start; reached RGB at arrival for G; no alternate future or hidden scene features',
        training='P/G heads and existing rank-8 LoRA only; original core/value heads/statistics frozen',
        controls='same source, native 224px RGB, 19-D state contract, optimizer, batch/steps, scene/action tables and seed',
        calibration_cases_excluded=['oak-lever-wall nominal pilot'],test_outcomes_available_to_training=False)
    write_json(args.output/'protocol.json',protocol)
    preserve=[args.initial,ROOT/'runs/hjepa/lora_continued_model/selected.pt',
              ROOT/'runs/hjepa/side_reward_test20261029/manifest.json',ROOT/'runs/door_scene_matrix_seed42/summary.json']
    original_hashes={str(p):file_hash(p) for p in preserve}
    write_json(args.output/'preserved_hashes.json',original_hashes)
    source_dir=args.output/'source';source_dir.mkdir()
    import shutil
    for name in ('AGENTS.md','mobile_door.py','staged_mobile_reward.py','run_mobile_reward_experiment.py',
                 'test_mobile_door.py','simulate.py','hjepa_model.py','hjepa_lora.py','experiment_reproducibility.py'):
        shutil.copy2(ROOT/name,source_dir/name)
    source_hashes={p.name:file_hash(p) for p in source_dir.iterdir()}
    write_json(args.output/'source_hashes.json',source_hashes)
    receipt.update(status='collecting_train_validation',startup_seconds=time.perf_counter()-started)
    write_json(args.output/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    phases={}
    development=[s for s in manifest['scenes'] if s['split']!='test']
    phases['train_validation_collection']=collect(development,args.output,args.workers,'train_validation_collection')
    development_data=load_scenes(development,args.output)
    training={arm:train_arm(args,arm,development_data) for arm in ('shared','decomposed')}
    for arm in training:phases['training_'+arm]=training[arm]['seconds']
    selected=max(('shared','decomposed'),key=lambda a:(selection_key(training[a]['selected_validation']),a=='shared'))
    frozen=dict(selected_arm=selected,selected_before_test=True,
                checkpoint_sha256={a:file_hash(args.output/a/'selected.pt') for a in training},
                test_manifest_sha256=manifest_hash,selected_validation={a:training[a]['selected_validation'] for a in training})
    write_json(args.output/'frozen_for_test.json',frozen)
    if any((args.output/'data'/s['scene_id']).exists() for s in manifest['scenes'] if s['split']=='test'):
        raise AssertionError('Test data existed before model freeze.')
    tests=[s for s in manifest['scenes'] if s['split']=='test']
    phases['test_collection']=collect(tests,args.output,args.workers,'test_collection')
    test_data=load_scenes(tests,args.output)
    import torch
    from hjepa_model import load_checkpoint
    evaluations={};decisions={};replays={}
    for arm in ('shared','decomposed'):
        model,source=load_checkpoint(args.output/arm/'selected.pt','cuda')
        evaluations[arm],decisions[arm]=evaluate(model,source,test_data)
        write_json(args.output/arm/'test.json',dict(summary=evaluations[arm],decisions=decisions[arm]))
        del model;torch.cuda.empty_cache()
        replays[arm]=live_replays(args,arm,test_data,decisions[arm]);phases['replay_'+arm]=replays[arm]['seconds']
    # A fixed command is chosen on validation only, then tested as a reference.
    model,source=load_checkpoint(args.initial,'cuda');model.config['position_views']='single'
    val=[s for s in development_data if s['spec']['split']=='val']
    fixed=max([(b,g) for b in range(4) for g in range(2)]+[(4,None)],key=lambda cmd:selection_key(evaluate(model,source,val,cmd)[0]))
    evaluations['validation_selected_fixed_command']=evaluate(model,source,test_data,fixed)[0]
    evaluations['frozen_initial']=evaluate(model,source,test_data)[0]
    del model;torch.cuda.empty_cache()
    size=audit();write_json(args.output/'git_size_audit.json',size)
    if not size['passed']:raise AssertionError('Repository size guard failed.')
    if file_hash(args.output/'manifest.json')!=manifest_hash:raise AssertionError('Frozen manifest changed.')
    preserved=all(file_hash(p)==digest for p,digest in original_hashes.items())
    if not preserved:raise AssertionError('A preserved checkpoint/dataset changed.')
    unchanged=all(file_hash(args.output/a/'selected.pt')==digest for a,digest in frozen['checkpoint_sha256'].items())
    if not unchanged:raise AssertionError('Selected weights changed after freeze.')
    source_unchanged=all(file_hash(ROOT/name)==digest for name,digest in source_hashes.items())
    if not source_unchanged:raise AssertionError('Experiment source changed during execution.')
    paired=dict(shared_only_correct=0,decomposed_only_correct=0,both_correct=0,both_wrong=0)
    for left,right in zip(decisions['shared'],decisions['decomposed']):
        l=left['success'] or left['refused'] and not left['feasible']
        r=right['success'] or right['refused'] and not right['feasible']
        paired['both_correct' if l and r else 'shared_only_correct' if l else 'decomposed_only_correct' if r else 'both_wrong']+=1
    phases['other_startup_evaluation_io']=time.perf_counter()-started-sum(phases.values())
    report=dict(seed=42,selected_arm=selected,selected_before_test=True,test=evaluations,
                fixed_command=list(fixed),replays=replays,paired_comparison=paired,phase_seconds=phases,
                seconds=time.perf_counter()-started,frozen_test_manifest_unchanged=True,
                selected_weights_unchanged=True,preserved_original_files_unchanged=True,
                source_hashes_unchanged=source_unchanged,
                original_core_frozen_both_arms=True,training_test_arrays_opened=0,
                test_scope=f'{args.test} newly frozen continuous-geometry variants; one training seed, known templates, finite action library',
                limitation='Geometry navigation; no wheel model; no exploration policy, retry-policy training or dense temporal world-model update.')
    write_json(args.output/'report.json',report)
    receipt.update(status='complete',elapsed_seconds=report['seconds'],phase_seconds=phases,exit_code=0,
                   completed_at=datetime.now(ZONE).isoformat())
    write_json(args.output/'receipt.json',receipt)
    print(json.dumps(dict(status='complete',seconds=report['seconds'],selected_arm=selected,test=evaluations)),flush=True)


if __name__=='__main__':
    if os.environ.get('PYTHONHASHSEED')!='42':
        environment=dict(os.environ,PYTHONHASHSEED='42',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        raise SystemExit(subprocess.run([sys.executable,*sys.argv],cwd=ROOT,env=environment).returncode)
    try:
        main()
    except Exception as error:
        if '--output' in sys.argv:
            receipt_path=Path(sys.argv[sys.argv.index('--output')+1])/'receipt.json'
            if receipt_path.exists():
                failed=json.loads(receipt_path.read_text(encoding='utf-8'))
                if failed.get('pid')==os.getpid():
                    failed.update(status='failed',error_type=type(error).__name__,error=str(error),exit_code=1,
                                  failed_at=datetime.now(ZONE).isoformat())
                    write_json(receipt_path,failed)
        raise
