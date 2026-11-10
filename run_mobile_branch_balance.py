"""Four matched LoRA continuations with branch-aware validation, seed42."""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta
import gc
import hashlib
import json
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
from mobile_branch_balance_v8 import ARMS,ScenePairSampler,weighted_outcome_loss,pg_selection_key


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,model,source,**metadata):
    from hjepa_model import save_checkpoint
    save_checkpoint(path,model,source['state_mean'],source['state_std'],seed=42,experiment='branch_balance_v8',**metadata)


def validate_pg(model,source,validation):
    from run_mobile_partial_training import predict,evaluate
    policies={mode:dict(mode=mode,risk=1.) for mode in ('never','always')}
    return evaluate(model,validation,predict(model,source,validation),policies)[0]


def train_pg(args,arm,scenes):
    import torch
    from experiment_reproducibility import seed_everything
    from mobile_partial_policy_v5 import load_partial_checkpoint
    from hjepa_lora import resume_lora,freeze_core_statistics,frozen_core_digest
    from run_mobile_outcome_experiment import pg_targets
    from run_mobile_partial_training import forward,predict
    seed_everything(42);begin=time.perf_counter();folder=args.output/arm;folder.mkdir()
    model,source=load_partial_checkpoint(args.initial,'cuda');lora=resume_lora(model,8,16.,.05)
    model.requires_grad_(False)
    for name,p in model.model.named_parameters():p.requires_grad_('lora_' in name)
    model.position_policy.requires_grad_(True);model.grasp_policy.requires_grad_(True)
    model.config['branch_balance_v8']=dict(arm=arm,**ARMS[arm],pg_selection='safety-first both branches',
        inference='same current-depth v5 execution; actual risk1 terminal values unchanged')
    frozen=frozen_core_digest(model)
    training=[s for s in scenes if s['spec']['split']=='train'];validation=[s for s in scenes if s['spec']['split']=='val']
    sampler=ScenePairSampler(training,ARMS[arm]['sampler']);rng=random.Random(42)
    summaries=validate_pg(model,source,validation);best=pg_selection_key(summaries['never'],summaries['always'])
    history=[dict(step=0,summary=summaries)];selected=0;timings=[];refreshes=[]
    metadata=dict(phase='pg',initial_sha256=file_hash(args.initial),manifest_sha256=file_hash(args.output/'manifest.json'))
    save(folder/'selected_pg.pt',model,source,step=0,**metadata)
    model.enable_activation_checkpointing(16)
    optimizer=torch.optim.AdamW([dict(params=[*model.position_policy.parameters(),*model.grasp_policy.parameters()],lr=3e-4),
        dict(params=[p for p in model.model.parameters() if p.requires_grad],lr=2e-5)],weight_decay=.01)
    for step in range(1,args.pg_steps+1):
        if (step-1)%40==0:
            targets=pg_targets(training,predict(model,source,training),'cuda')
            digest=hashlib.sha256(b''.join(targets[s['spec']['scene_id']][key].tobytes()
                for s in training for key in ('p0','p1','g0','g1'))).hexdigest()
            refreshes.append(dict(step=step-1,training_scenes=len(training),targets_sha256=digest,test_outcome_reads=0))
        tick=time.perf_counter();model.eval();model.model.get_level(1).encoder.encoder.train();freeze_core_statistics(model)
        optimizer.zero_grad();losses=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for scene in sampler.sample(rng):
                prediction=forward(model,source,scene,False);target=targets[scene['spec']['scene_id']]
                def loss(logits,labels):return weighted_outcome_loss(logits,torch.as_tensor(labels,device='cuda'),ARMS[arm]['collision_weight'])
                value=loss(prediction['p0'],target['p0'])
                if target['post_mask']:value=value+loss(prediction['p1'],target['p1'])
                for branch in (0,1):
                    mask=torch.as_tensor(scene['branches'][branch]['grasp_mask'],device='cuda')
                    if mask.any():value=value+.5*loss(prediction[f'g{branch}'][mask],target[f'g{branch}'][mask.cpu().numpy()])
                losses.append(value)
            value=torch.stack(losses).mean()
        if not torch.isfinite(value):raise FloatingPointError('Nonfinite branch-balanced PG loss.')
        value.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step();torch.cuda.synchronize();timings.append(time.perf_counter()-tick)
        if step==1 or step%10==0:
            progress=dict(pid=os.getpid(),phase='pg_'+arm,step=step,total_steps=args.pg_steps,loss=float(value.detach()),
                seconds_per_update=float(np.mean(timings[-10:])),elapsed_seconds=time.perf_counter()-begin)
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        if step==1 or step%40==0 or step==args.pg_steps:
            summaries=validate_pg(model,source,validation);history.append(dict(step=step,summary=summaries))
            key=pg_selection_key(summaries['never'],summaries['always'])
            if key>best:best=key;selected=step;save(folder/'selected_pg.pt',model,source,step=step,**metadata)
    save(folder/'last_pg.pt',model,source,step=args.pg_steps,**metadata)
    assert frozen_core_digest(model)==frozen
    result=dict(seed=42,steps=args.pg_steps,selected_step=selected,history=history,teacher_refreshes=refreshes,
        lora=lora,training_step_seconds=timings,seconds=time.perf_counter()-begin,original_core_exact=True,test_outcome_reads=0)
    write_json(folder/'pg_training.json',result)
    del model,source,optimizer;gc.collect();torch.cuda.empty_cache();return result


def train_gate(args,arm,scenes):
    import torch
    from experiment_reproducibility import seed_everything
    from mobile_partial_policy_v5 import load_partial_checkpoint
    from run_mobile_inspection_regret import prepare,evaluate
    from mobile_inspection_regret_v6 import inspection_loss
    from run_mobile_active_experiment import validation_key
    seed_everything(42);begin=time.perf_counter();folder=args.output/arm
    model,source=load_partial_checkpoint(folder/'selected_pg.pt','cuda');model.requires_grad_(False)
    fixed={name:value.detach().cpu().clone() for name,value in model.state_dict().items() if not name.startswith('inspection_policy.')}
    initial_i={name:value.detach().cpu().clone() for name,value in model.inspection_policy.state_dict().items()}
    data=prepare(SimpleNamespace(output=folder),model,source,scenes)
    validation=[s for s in scenes if s['spec']['split']=='val'];training=[i for i,row in enumerate(data['rows']) if row['split']=='train']
    model.config['inspection_regret_v6']=dict(arm='joint_regret',temperature=.2,
        continuation='selected PG and LoRA frozen; current fixed selected-action counterfactual targets',
        input='initial RGB, relative goal, declared risk, initial depth features only')
    model.inspection_policy.requires_grad_(True);rng=random.Random(42)
    summary,decisions=evaluate(model,validation,data,arm);best=validation_key(summary);selected=0
    history=[dict(step=0,summary=summary)]
    metadata=dict(phase='inspection',pg_selected_sha256=file_hash(folder/'selected_pg.pt'),
                  manifest_sha256=file_hash(args.output/'manifest.json'))
    save(folder/'selected.pt',model,source,step=0,**metadata)
    optimizer=torch.optim.AdamW(model.inspection_policy.parameters(),lr=3e-4,weight_decay=.01)
    for step in range(1,args.gate_steps+1):
        sample=torch.tensor(rng.sample(training,min(32,len(training))),device='cuda')
        model.inspection_policy.train();optimizer.zero_grad();raw=model.inspection_policy(data['features'][sample]).float()
        result=inspection_loss(raw,data['labels'][sample],data['cost'][sample],data['risk'][sample],
            data['direct_predicted_value'][sample],data['advantage'][sample],data['eligible'][sample],'joint_regret')
        if not torch.isfinite(result['loss']):raise FloatingPointError('Nonfinite I loss.')
        result['loss'].backward();torch.nn.utils.clip_grad_norm_(model.inspection_policy.parameters(),1.);optimizer.step()
        if step==1 or step%40==0 or step==args.gate_steps:
            summary,decisions=evaluate(model,validation,data,arm)
            history.append(dict(step=step,summary=summary,loss=float(result['loss'].detach())))
            if validation_key(summary)>best:best=validation_key(summary);selected=step;save(folder/'selected.pt',model,source,step=step,**metadata)
            progress=dict(pid=os.getpid(),phase='gate_'+arm,step=step,total_steps=args.gate_steps,elapsed_seconds=time.perf_counter()-begin)
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    save(folder/'last.pt',model,source,step=args.gate_steps,**metadata)
    assert all(torch.equal(value,model.state_dict()[name].detach().cpu()) for name,value in fixed.items())
    cp=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
    model.inspection_policy.load_state_dict({name.removeprefix('inspection_policy.'):value for name,value in cp['model'].items()
                                            if name.startswith('inspection_policy.')})
    summary,decisions=evaluate(model,validation,data,arm)
    write_json(folder/'validation.json',dict(summary=summary,decisions=decisions))
    result=dict(seed=42,steps=args.gate_steps,selected_step=selected,history=history,validation=summary,
                all_non_I_exact=True,test_outcome_reads=0,seconds=time.perf_counter()-begin)
    write_json(folder/'gate_training.json',result)
    del model,source,optimizer,fixed,initial_i,cp,data;gc.collect();torch.cuda.empty_cache();return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prior',type=Path,default=ROOT/'runs/p6_train2_seed42_20261110')
    p.add_argument('--history-root',type=Path,default=ROOT/'runs')
    p.add_argument('--development-groups',type=int);p.add_argument('--test-groups',type=int,default=18)
    p.add_argument('--pg-steps',type=int,default=320);p.add_argument('--gate-steps',type=int,default=1600)
    p.add_argument('--stream',default='mobile_branch_v8_20261110');p.add_argument('--estimate-seconds',type=int,required=True)
    args=p.parse_args()
    for name in ('output','prior','history_root'):setattr(args,name,getattr(args,name).resolve())
    if ROOT/'runs' not in args.output.parents:raise ValueError('New ignored run required.')
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Preserve old runs.')
    if min(args.pg_steps,args.gate_steps,args.test_groups,args.estimate_seconds)<1:raise ValueError('Positive budgets required.')
    args.output.mkdir(parents=True,exist_ok=True);begin=time.perf_counter();now=datetime.now(ZONE)
    receipt=dict(pid=os.getpid(),seed=42,status='starting',started_at=now.isoformat(),queue_estimate_seconds=0,
        runtime_estimate_seconds=args.estimate_seconds,expected_completion=(now+timedelta(seconds=args.estimate_seconds)).isoformat())
    write_json(args.output/'receipt.json',receipt)
    from experiment_reproducibility import seed_everything
    receipt['reproducibility']=seed_everything(42)
    import torch
    torch.set_num_threads(4)
    previous=read(args.prior/'protocol.json');args.initial=Path(previous['initial']);args.development=Path(previous['development']);args.maps=Path(previous['maps'])
    assert file_hash(args.initial)==previous['initial_sha256']
    protected=read(args.prior/'preserved_hashes.json').copy()
    assert all(file_hash(path)==digest for path,digest in protected.items())
    protected[str(args.initial)]=file_hash(args.initial)
    old_selection=read(args.prior/'frozen_for_test.json')
    reference_path=args.prior/old_selection['selected_policy']/'selected.pt'
    assert file_hash(reference_path)==old_selection['checkpoint_sha256'][old_selection['selected_policy']]
    protected[str(reference_path)]=file_hash(reference_path)
    for name in ('protocol.json','manifest.json','frozen_for_test.json','source_hashes.json','report.json'):
        path=args.prior/name;protected[str(path)]=file_hash(path)
    for name,h in read(args.prior/'source_hashes.json').items():
        if file_hash(ROOT/name)!=h:raise ValueError('Frozen source differs: '+name)
    from run_mobile_outcome_experiment import make_run_manifest,attach_development
    from mobile_crossed_v2 import geometry_identity
    manifest=make_run_manifest(args.development,args.test_groups,args.stream,args.output,args.development_groups)
    history=set()
    for path in args.history_root.glob('*/manifest.json'):
        if path.parent.resolve()==args.output:continue
        history.update(geometry_identity(s) for s in read(path).get('scenes',[]) if 'start_pose' in s and 'obstacles' in s)
    if history & {geometry_identity(s) for s in manifest['scenes'] if s['split']=='test'}:raise ValueError('Fresh test overlaps history.')
    manifest.update(schema_version=8,historical_geometries_excluded_from_test=len(history))
    write_json(args.output/'manifest.json',manifest)
    scenes=attach_development(args,[s for s in manifest['scenes'] if s['split'] in ('train','val')])
    protected.update(read(args.output/'reused_development_hashes.json'))
    for scene in scenes:
        folder=args.maps/'data'/scene['spec']['scene_id'];scene['maps']=read(folder/'maps.json')
        for name in ('maps.json','reached_rgbd.npz'):protected[str(folder/name)]=file_hash(folder/name)
    source=args.output/'source';source.mkdir()
    for name in ['AGENTS.md','scene.xml','.gitignore',*sorted(p.name for p in ROOT.glob('*.py'))]:shutil.copy2(ROOT/name,source/name)
    hashes={path.name:file_hash(path) for path in source.iterdir()};write_json(args.output/'source_hashes.json',hashes)
    protocol=dict(seed=42,arms=ARMS,initial=str(args.initial),initial_sha256=file_hash(args.initial),
        prior=str(args.prior),development=str(args.development),maps=str(args.maps),counts=manifest['counts'],
        pg_steps=args.pg_steps,gate_steps=args.gate_steps,batch_scenes=2,encoder='continue rank8 LoRA; original core frozen',
        pg_selection='fewer collisions summed across both modes; always correct; direct correct; openings; net; path; earliest ties',
        pg_losses='CE with collision class weight1 or4; same actual selected-G outcome targets refreshed every40 training steps',
        evaluation_utility='unchanged success1 ordinaryFailure-1 collision-2 refusal0 and actual observation cost',
        gate='same1600-step joint-regret I after freezing selected PG and LoRA',
        sampler='random2 scenes versus2 different layouts from one intact randomly selected training group',
        checkpoint_warning='weighted CE scores are not calibrated class posteriors',
        map='unchanged historical v5: new fusion calibration did not improve validation',
        test_references=['frozen_v6_learned','frozen_v6_always','frozen_v6_never'],
        no_new_test_if_all_selected_tensors_identical_to_frozen_v6=True,test_outcome_reads=0)
    write_json(args.output/'protocol.json',protocol);write_json(args.output/'preserved_hashes.json',protected)
    training={};validation={};phases={}
    receipt.update(status='training',startup_seconds=time.perf_counter()-begin);write_json(args.output/'receipt.json',receipt)
    for arm in ARMS:
        pg=train_pg(args,arm,scenes);gate=train_gate(args,arm,scenes)
        training[arm]=dict(pg=pg,gate=gate);validation[arm]=gate['validation']
        phases['pg_'+arm]=pg['seconds'];phases['gate_'+arm]=gate['seconds']
    baseline=torch.load(args.initial,map_location='cpu',weights_only=True)
    original=[name for name in baseline['model'] if 'lora_' not in name and not name.startswith(('position_policy.','grasp_policy.','inspection_policy.'))]
    audits={}
    for arm in ARMS:
        pg=torch.load(args.output/arm/'selected_pg.pt',map_location='cpu',weights_only=True);audits[arm]={}
        for label in ('selected_pg','last_pg','selected','last'):
            cp=pg if label=='selected_pg' else torch.load(args.output/arm/(label+'.pt'),map_location='cpu',weights_only=True)
            assert all(torch.equal(baseline['model'][name],cp['model'][name]) for name in original)
            assert all(torch.equal(baseline[key],cp[key]) for key in ('state_mean','state_std'))
            fixed=all(torch.equal(value,cp['model'][name]) for name,value in pg['model'].items() if not name.startswith('inspection_policy.')) if label in ('selected','last') else None
            if fixed is not None:assert fixed
            changed=sum(not torch.equal(value,cp['model'][name]) for name,value in baseline['model'].items() if 'lora_' in name)
            if label=='last_pg':assert changed>0
            audits[arm][label]=dict(step=cp['step'],original_tensor_count=len(original),original_and_norms_exact=True,
                changed_lora_tensors=changed,selected_pg_exact_outside_I=fixed)
            if cp is not pg:del cp
            gc.collect()
        del pg;gc.collect()
    reference=read(args.prior/'frozen_for_test.json')['selected_policy'];reference_path=args.prior/reference/'selected.pt'
    frozen_reference=torch.load(reference_path,map_location='cpu',weights_only=True);reference_hash=file_hash(reference_path)
    identity={}
    for arm in ARMS:
        cp=torch.load(args.output/arm/'selected.pt',map_location='cpu',weights_only=True)
        identity[arm]=all(torch.equal(value,cp['model'][name]) for name,value in frozen_reference['model'].items())
        del cp;gc.collect()
    assert all(file_hash(path)==digest for path,digest in protected.items())
    assert all(file_hash(ROOT/name)==file_hash(source/name)==digest for name,digest in hashes.items())
    audit=dict(seed=42,checkpoints=audits,all_original422_and_norms_exact=True,all_non_I_frozen_during_gate=True,
        selected_tensor_identity_to_frozen_v6=identity,test_outcome_reads=0)
    write_json(args.output/'tensor_audit.json',audit)
    from run_mobile_active_experiment import best_policy
    selected=best_policy(validation)
    frozen=dict(seed=42,selected_policy=selected,selected_before_test=True,manifest_sha256=file_hash(args.output/'manifest.json'),
        checkpoint_sha256={arm:file_hash(args.output/arm/'selected.pt') for arm in ARMS},
        reference_path=str(reference_path),reference_sha256=reference_hash,validation_metrics=validation,
        selected_tensor_identity_to_frozen_v6=identity,new_test_required=not all(identity.values()),
        test_references=protocol['test_references'],test_outcomes_collected=False,frozen_at=datetime.now(ZONE).isoformat())
    write_json(args.output/'frozen_for_test.json',frozen)
    phases['startup_io_selection_audit']=time.perf_counter()-begin-sum(phases.values())
    report=dict(seed=42,selected_policy=selected,training=training,validation=validation,phase_seconds=phases,
        seconds=time.perf_counter()-begin,test_outcome_reads=0,all_original422_and_norms_exact=True,
        selected_tensor_identity_to_frozen_v6=identity,new_test_required=frozen['new_test_required'],
        status='training_complete_fresh_test_pending' if frozen['new_test_required'] else 'training_complete_same_frozen_policy_no_new_test')
    write_json(args.output/'report.json',report)
    receipt.update(status='complete',exit_code=0,actual_seconds=report['seconds'],completed_at=datetime.now(ZONE).isoformat(),phase_seconds=phases)
    write_json(args.output/'receipt.json',receipt)
    print(json.dumps(dict(status=report['status'],seconds=report['seconds'],selected=selected,validation=validation,
                         tensor_identity_to_reference=identity)),flush=True)


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
