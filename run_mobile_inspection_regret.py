"""Matched frozen-P/G observation training; train/validation only, seed 42."""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta
import gc
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
from run_mobile_reward_experiment import ROOT,ZONE,write_json,file_hash
from mobile_inspection_regret_v6 import ARMS,paired_teacher,inspection_loss,TEMPERATURE


def read(path):
    import json
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(args,model,source,scenes):
    import torch
    from run_mobile_partial_training import predict
    from mobile_partial_policy_v5 import partial_gate_features
    cached=predict(model,source,scenes);rows=[];features=[];score_cache={}
    from mobile_outcome_v4 import station_values,outcome_values
    for scene in scenes:
        sid=scene['spec']['scene_id'];p={k:v.to('cuda') for k,v in cached[sid].items()}
        for risk in (1.,3.):
            target=paired_teacher(scene,p,risk)
            features.append(partial_gate_features(model,p['front'],
                scene['branches'][0]['arrays']['initial_proprio'],scene['maps']['initial']['features'],risk))
            rows.append(dict(scene_id=sid,split=scene['spec']['split'],**target))
        score_cache[sid]=tuple(x.detach().cpu().numpy() for x in (
            station_values(p['p0'],1.,scene['maps']['initial']['mask']),
            station_values(p['p1'],1.,scene['maps']['fused']['mask']),
            outcome_values(p['g0'],1.),outcome_values(p['g1'],1.)))
    tensor=torch.cat(features).detach()
    labels=torch.tensor([r['outcome_class'] for r in rows],device='cuda')
    arrays={key:np.array([r[key] for r in rows]) for key in (
        'cost','risk','direct_predicted_value','advantage','eligible')}
    np.savez_compressed(args.output/'features.npz',features=tensor.cpu().numpy(),
                        labels=labels.cpu().numpy(),**arrays)
    write_json(args.output/'training_and_validation_labels.json',rows)
    tensors={key:torch.as_tensor(value,device='cuda',dtype=torch.bool if key=='eligible' else torch.float32)
             for key,value in arrays.items()}
    return dict(features=tensor,labels=labels,rows=rows,scores=score_cache,**tensors)


def evaluate(model,scenes,data,arm):
    import torch
    from mobile_outcome_v4 import inspection_value
    from run_mobile_active_experiment import choose,summarize
    decisions=[];indices={r['scene_id']:i for i,r in enumerate(data['rows']) if r['risk']==1.}
    model.eval()
    with torch.no_grad():
        for scene in scenes:
            sid=scene['spec']['scene_id'];index=indices[sid]
            value,_=inspection_value(model,data['features'][index:index+1],1.)
            p0,p1,g0,g1=data['scores'][sid]
            scores=(p0,value[0].detach().cpu().numpy(),p1,g0,g1)
            decisions.append(choose(scene,scores,'learned' if data['rows'][index]['eligible'] else 'never',arm))
    return summarize(decisions),decisions


def train_arm(args,model,source,scenes,data,arm,initial_i):
    import torch
    from experiment_reproducibility import seed_everything
    from run_mobile_active_experiment import validation_key
    from hjepa_model import save_checkpoint
    seed_everything(42);begin=time.perf_counter();folder=args.output/arm;folder.mkdir()
    model.inspection_policy.load_state_dict(initial_i)
    model.config['inspection_regret_v6']=dict(arm=arm,temperature=TEMPERATURE,
        inference='unchanged v5 expected inspect utility minus cost versus best initial P/refusal',
        continuation='all P/G, existing LoRA, original core and normalization frozen',
        input='initial RGB feature, relative goal, declared risk, initial depth features only')
    model.requires_grad_(False);model.inspection_policy.requires_grad_(True)
    train=[i for i,r in enumerate(data['rows']) if r['split']=='train']
    validation=[s for s in scenes if s['spec']['split']=='val']
    summary,decisions=evaluate(model,validation,data,arm)
    best=validation_key(summary);selected=0;history=[dict(step=0,summary=summary)]
    metadata=dict(experiment='inspection_regret_v6',phase='inspection',seed=42,
        initial_sha256=file_hash(args.initial),manifest_sha256=file_hash(args.output/'manifest.json'))
    def save(label,step):
        save_checkpoint(folder/(label+'.pt'),model,source['state_mean'],source['state_std'],step=step,**metadata)
    save('selected',0)
    optimizer=torch.optim.AdamW(model.inspection_policy.parameters(),lr=3e-4,weight_decay=.01)
    rng=random.Random(42);timings=[]
    for step in range(1,args.steps+1):
        tick=time.perf_counter();sample=torch.tensor(rng.sample(train,min(32,len(train))),device='cuda')
        model.inspection_policy.train();optimizer.zero_grad()
        raw=model.inspection_policy(data['features'][sample]).float()
        result=inspection_loss(raw,data['labels'][sample],data['cost'][sample],data['risk'][sample],
            data['direct_predicted_value'][sample],data['advantage'][sample],data['eligible'][sample],arm)
        loss=result['loss']
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite inspection regret loss.')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.inspection_policy.parameters(),1.)
        optimizer.step();torch.cuda.synchronize();timings.append(time.perf_counter()-tick)
        if step==1 or step%40==0 or step==args.steps:
            summary,decisions=evaluate(model,validation,data,arm)
            history.append(dict(step=step,summary=summary,loss=float(loss.detach()),
                outcome_loss=float(result['outcome'].detach()),cost_loss=float(result['cost'].detach()),
                decision_regret=float(result['regret'].detach())))
            if validation_key(summary)>best:
                best=validation_key(summary);selected=step;save('selected',step)
            progress=dict(pid=os.getpid(),phase='gate_'+arm,step=step,total_steps=args.steps,
                seconds_per_update=float(np.mean(timings[-40:])),elapsed_seconds=time.perf_counter()-begin,
                validation_correct=summary['correct_decisions'],validation_inspections=summary['inspections'])
            write_json(args.output/'progress.json',progress)
            import json
            print(json.dumps(progress),flush=True)
    save('last',args.steps)
    checkpoint=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
    model.inspection_policy.load_state_dict({n.removeprefix('inspection_policy.'):v for n,v in checkpoint['model'].items()
                                            if n.startswith('inspection_policy.')})
    summary,decisions=evaluate(model,validation,data,arm)
    write_json(folder/'validation.json',dict(summary=summary,decisions=decisions))
    result=dict(seed=42,steps=args.steps,selected_step=selected,history=history,
                training_examples=len(train),validation=summary,seconds=time.perf_counter()-begin,
                mean_update_seconds=float(np.mean(timings)),test_outcome_reads=0)
    write_json(folder/'training.json',result)
    del optimizer,checkpoint;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--initial',type=Path,default=ROOT/'runs/mobile_partial_training_seed42_20261109/depth_features/selected_pg.pt')
    p.add_argument('--prior-training',type=Path,default=ROOT/'runs/mobile_partial_training_seed42_20261109')
    p.add_argument('--history-root',type=Path,default=ROOT/'runs')
    p.add_argument('--development-groups',type=int)
    p.add_argument('--test-groups',type=int,default=18)
    p.add_argument('--stream',default='mobile_regret_v6_20261110')
    p.add_argument('--steps',type=int,default=1600)
    p.add_argument('--estimate-seconds',type=int,required=True)
    args=p.parse_args()
    for name in ('output','initial','prior_training','history_root'):setattr(args,name,getattr(args,name).resolve())
    if ROOT/'runs' not in args.output.parents:raise ValueError('New ignored runs/ output required.')
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Preserve existing runs.')
    if min(args.steps,args.test_groups,args.estimate_seconds)<1:raise ValueError('Positive budgets required.')
    args.output.mkdir(parents=True,exist_ok=True);begin=time.perf_counter();now=datetime.now(ZONE)
    receipt=dict(pid=os.getpid(),seed=42,status='starting',started_at=now.isoformat(),queue_estimate_seconds=0,
        runtime_estimate_seconds=args.estimate_seconds,expected_completion=(now+timedelta(seconds=args.estimate_seconds)).isoformat(),
        scope='three matched observation objectives; fixed existing LoRA/P/G; no test outcome reads')
    write_json(args.output/'receipt.json',receipt)
    from experiment_reproducibility import seed_everything
    receipt['reproducibility']=seed_everything(42)
    import torch
    torch.set_num_threads(4)
    from mobile_partial_policy_v5 import load_partial_checkpoint
    previous=read(args.prior_training/'protocol.json')
    args.development=Path(previous['development']);args.maps=Path(previous['maps'])
    source_hashes=read(args.prior_training/'source_hashes.json')
    if any(file_hash(ROOT/n)!=h for n,h in source_hashes.items()):
        raise ValueError('Prior frozen physics/model/map source differs.')
    protected={str(args.initial):file_hash(args.initial)}
    for name in ('protocol.json','manifest.json','frozen_for_test.json','source_hashes.json','report.json'):
        path=args.prior_training/name;protected[str(path)]=file_hash(path)
    from run_mobile_outcome_experiment import make_run_manifest,attach_development
    from mobile_crossed_v2 import geometry_identity
    manifest=make_run_manifest(args.development,args.test_groups,args.stream,args.output,args.development_groups)
    history=set();history_files=[]
    for path in args.history_root.glob('*/manifest.json'):
        if path.parent.resolve()==args.output:continue
        history.update(geometry_identity(s) for s in read(path).get('scenes',[])
                       if 'start_pose' in s and 'obstacles' in s)
        history_files.append(str(path))
    tests=[s for s in manifest['scenes'] if s['split']=='test']
    if history & {geometry_identity(s) for s in tests}:raise ValueError('Fresh tests overlap history.')
    manifest.update(schema_version=6,historical_geometries_excluded_from_test=len(history),history_manifests=history_files)
    write_json(args.output/'manifest.json',manifest)
    specs=[s for s in manifest['scenes'] if s['split'] in ('train','val')]
    scenes=attach_development(args,specs)
    protected.update(read(args.output/'reused_development_hashes.json'))
    for scene in scenes:
        folder=args.maps/'data'/scene['spec']['scene_id'];scene['maps']=read(folder/'maps.json')
        if not scene['maps']['original_candidate_prefix_states_match']:raise AssertionError('Unverified prefix.')
        for name in ('maps.json','reached_rgbd.npz'):protected[str(folder/name)]=file_hash(folder/name)
    source=args.output/'source';source.mkdir()
    for name in ['AGENTS.md','scene.xml','.gitignore',*sorted(p.name for p in ROOT.glob('*.py'))]:shutil.copy2(ROOT/name,source/name)
    hashes={path.name:file_hash(path) for path in source.iterdir()};write_json(args.output/'source_hashes.json',hashes)
    protocol=dict(seed=42,arms=ARMS,steps_per_arm=args.steps,initial=str(args.initial),initial_sha256=file_hash(args.initial),
        prior_training=str(args.prior_training),development=str(args.development),maps=str(args.maps),
        counts=manifest['counts'],lr=3e-4,weight_decay=.01,batch_examples=32,risks=[1.,3.],temperature=TEMPERATURE,
        outcome_loss='CE of actual frozen post-P/G selected outcome +0.5 cost MSE',
        regret_loss='abs(actual inspect net utility - actual direct utility) weighted BCE of (predicted inspect net utility - max frozen P0)/0.2',
        arms_definition=dict(outcome='outcome loss',joint_regret='outcome loss + regret loss',
                             decision_regret='regret loss +0.5 cost MSE'),
        target_scope='fixed greedy P/G selected actions on each real physical branch; never oracle-best actions',
        input_scope='same v5 initial RGB/proprio/CAD/depth features; no post image/map or actual outcome at inference',
        architecture='identical input/hidden/output dimensions and initialization; deployed v5 comparison unchanged',
        encoder='original core and already-LoRA-adapted encoder frozen; only I updated to isolate objective',
        selection='validation correct, fewer collisions, openings, net utility, path, inspections; earliest exact tie',
        policy_order=list(ARMS),test_scope='fresh grouped test manifest frozen; outcomes not collected in this job',
        limitation='same small reused train/validation sets; fixed continuation; known-map navigation')
    write_json(args.output/'protocol.json',protocol);write_json(args.output/'preserved_hashes.json',protected)
    model,checkpoint=load_partial_checkpoint(args.initial,'cuda');model.requires_grad_(False)
    initial_i={n:v.detach().cpu().clone() for n,v in model.inspection_policy.state_dict().items()}
    fixed={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not n.startswith('inspection_policy.')}
    tick=time.perf_counter();data=prepare(args,model,checkpoint,scenes)
    phases={'shared_feature_and_teacher_cache':time.perf_counter()-tick}
    receipt.update(status='training',startup_seconds=time.perf_counter()-begin)
    write_json(args.output/'receipt.json',receipt)
    training={};validation={}
    for arm in ARMS:
        training[arm]=train_arm(args,model,checkpoint,scenes,data,arm,initial_i)
        validation[arm]=training[arm]['validation'];phases['gate_'+arm]=training[arm]['seconds']
    from run_mobile_active_experiment import best_policy
    selected=best_policy(validation)
    from hjepa_lora import frozen_core_digest
    audits={};baseline=torch.load(args.initial,map_location='cpu',weights_only=True)
    original=[n for n in baseline['model'] if 'lora_' not in n and not n.startswith(
        ('position_policy.','grasp_policy.','inspection_policy.'))]
    for arm in ARMS:
        audits[arm]={}
        for label in ('selected','last'):
            cp=torch.load(args.output/arm/(label+'.pt'),map_location='cpu',weights_only=True)
            assert all(torch.equal(value,cp['model'][name]) for name,value in fixed.items())
            assert all(torch.equal(baseline[key],cp[key]) for key in ('state_mean','state_std'))
            audits[arm][label]=dict(step=cp['step'],all_non_I_exact=True,original_tensor_count=len(original),
                changed_I_tensors=sum(not torch.equal(value,cp['model']['inspection_policy.'+name]) for name,value in initial_i.items()))
            del cp;gc.collect()
    assert all(file_hash(path)==digest for path,digest in protected.items())
    assert all(file_hash(ROOT/name)==digest and file_hash(source/name)==digest for name,digest in hashes.items())
    audit=dict(seed=42,checkpoints=audits,all_non_I_exact=True,original_tensor_count=len(original),
               all_protected_sources_models_and_inputs_exact=True,test_outcome_reads=0)
    write_json(args.output/'tensor_audit.json',audit)
    frozen=dict(seed=42,selected_policy=selected,selected_before_test=True,
        checkpoint_sha256={arm:file_hash(args.output/arm/'selected.pt') for arm in ARMS},
        baseline_sha256=file_hash(args.initial),manifest_sha256=file_hash(args.output/'manifest.json'),
        validation_metrics=validation,test_outcomes_collected=False,frozen_at=datetime.now(ZONE).isoformat(),
        test_references=['frozen_v5_learned','frozen_v5_always','frozen_v5_never'])
    write_json(args.output/'frozen_for_test.json',frozen)
    phases['startup_audit_and_io']=time.perf_counter()-begin-sum(phases.values())
    report=dict(seed=42,selected_policy=selected,training=training,validation=validation,phase_seconds=phases,
        seconds=time.perf_counter()-begin,test_outcome_reads=0,all_non_I_exact=True,
        status='training_complete_test_pending')
    write_json(args.output/'report.json',report)
    receipt.update(status='complete',exit_code=0,actual_seconds=report['seconds'],
                   completed_at=datetime.now(ZONE).isoformat(),phase_seconds=phases)
    write_json(args.output/'receipt.json',receipt)
    import json
    print(json.dumps(dict(status='complete',selected=selected,seconds=report['seconds'],validation=validation)),flush=True)


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
