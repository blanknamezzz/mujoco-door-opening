"""Continue LoRA with matched current-depth / image-only P/I conditioning.

Only frozen prior train/validation outcomes and newly reached RGB-D prefixes are
read. Fresh test geometry is frozen now; its outcomes are collected separately
after training, model selection and this run's tensor audit have completed.
"""
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
import numpy as np
from run_mobile_reward_experiment import ROOT,ZONE,write_json,file_hash

MODELS=('images_only','depth_features')
POLICIES={f'{name}_{mode}':dict(model=name,mode=mode,risk=1.)
          for name in MODELS for mode in ('never','always','learned')}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def forward(model,source,scene,canonical=True):
    from run_mobile_outcome_experiment import encode_front,grasp_logits
    from mobile_partial_policy_v5 import partial_position_logits
    a0=scene['branches'][0]['arrays'];a1=scene['branches'][1]['arrays']
    z=encode_front(model,source,a0['initial_rgb'],a0['initial_proprio'])
    other=encode_front(model,source,a1['post_rgb'],a1['post_proprio']) if a1['post_view_available'] else z
    return dict(p0=partial_position_logits(model,z,z,a0['initial_proprio'],scene['maps']['initial']['features']),
        p1=partial_position_logits(model,z,other,a1['post_proprio'],scene['maps']['fused']['features']),
        g0=grasp_logits(model,source,a0,canonical),g1=grasp_logits(model,source,a1,canonical),front=z)


def predict(model,source,scenes):
    import torch
    model.eval();result={}
    with torch.no_grad():
        for scene in scenes:
            result[scene['spec']['scene_id']]={k:v.detach().cpu() for k,v in forward(model,source,scene).items()}
    return result


def scalar_scores(model,scene,prediction,risk=1.):
    import torch
    from mobile_outcome_v4 import station_values,outcome_values,inspection_value
    from mobile_partial_policy_v5 import partial_gate_features
    device=next(model.parameters()).device
    p0=station_values(prediction['p0'].to(device),risk,scene['maps']['initial']['mask'])
    p1=station_values(prediction['p1'].to(device),risk,scene['maps']['fused']['mask'])
    with torch.no_grad():
        x=partial_gate_features(model,prediction['front'].to(device),
            scene['branches'][0]['arrays']['initial_proprio'],scene['maps']['initial']['features'],risk)
        inspect,_=inspection_value(model,x,risk)
    return tuple(x.detach().cpu().numpy() for x in (p0,inspect[0],p1,
        outcome_values(prediction['g0'].to(device),risk),outcome_values(prediction['g1'].to(device),risk)))


def evaluate(model,scenes,cached,policies):
    from run_mobile_active_experiment import choose,summarize
    choices={}
    for name,config in policies.items():
        decisions=[]
        for scene in scenes:
            forced=not any(scene['maps']['initial']['mask'])
            scores=scalar_scores(model,scene,cached[scene['spec']['scene_id']],config['risk'])
            d=choose(scene,scores,'never' if forced else config['mode'],name)
            d.update(partial_map=True,map_features=model.config['mobile_partial_v5']['use_map_features'],
                initial_mask=scene['maps']['initial']['mask'],fused_mask_used=scene['maps']['fused']['mask'] if d['branch'] else None,
                initial_map_forced_refusal=forced)
            decisions.append(d)
        choices[name]=decisions
    return {n:summarize(ds) for n,ds in choices.items()},choices


def save(path,model,source,**metadata):
    from hjepa_model import save_checkpoint
    save_checkpoint(path,model,source['state_mean'],source['state_std'],seed=42,**metadata)


def train_pg(args,name,scenes):
    import torch
    from mobile_outcome_v4 import load_outcome_checkpoint,outcome_loss
    from mobile_partial_policy_v5 import configure_partial_policy
    from hjepa_lora import resume_lora,freeze_core_statistics,frozen_core_digest
    from experiment_reproducibility import seed_everything
    from run_mobile_outcome_experiment import pg_targets
    from run_mobile_active_experiment import validation_key
    seed_everything(42);torch.set_num_threads(4);begin=time.perf_counter()
    folder=args.output/name;folder.mkdir()
    model,source=load_outcome_checkpoint(args.initial,'cuda')
    configure_partial_policy(model,name=='depth_features')
    lora=resume_lora(model,8,16.,.05);model.requires_grad_(False)
    for n,p in model.model.named_parameters():p.requires_grad_('lora_' in n)
    model.position_policy.requires_grad_(True);model.grasp_policy.requires_grad_(True)
    frozen=frozen_core_digest(model)
    train=[s for s in scenes if s['spec']['split']=='train'];val=[s for s in scenes if s['spec']['split']=='val']
    key=name+'_always';policies={key:POLICIES[key]};history=[];refreshes=[];timings=[]
    summary=evaluate(model,val,predict(model,source,val),policies)[0][key]
    best=validation_key(summary);selected=0;history.append(dict(step=0,summary=summary))
    metadata=dict(phase='pg',experiment='partial_map_v5',loss_kind='factorized',initial_sha256=file_hash(args.initial),
                  manifest_sha256=file_hash(args.output/'manifest.json'))
    save(folder/'selected_pg.pt',model,source,step=0,**metadata)
    model.enable_activation_checkpointing(16)
    optimizer=torch.optim.AdamW([dict(params=[*model.position_policy.parameters(),*model.grasp_policy.parameters()],lr=3e-4),
        dict(params=[p for p in model.model.parameters() if p.requires_grad],lr=2e-5)],weight_decay=.01)
    rng=random.Random(42)
    for step in range(1,args.pg_steps+1):
        if (step-1)%40==0:
            targets=pg_targets(train,predict(model,source,train),'cuda')
            digest=hashlib.sha256(b''.join(targets[s['spec']['scene_id']][k].tobytes()
                for s in train for k in ('p0','p1','g0','g1'))).hexdigest()
            refreshes.append(dict(step=step-1,targets_sha256=digest,training_scenes=len(train),test_outcome_reads=0))
        tick=time.perf_counter();model.eval();model.model.get_level(1).encoder.encoder.train();freeze_core_statistics(model)
        optimizer.zero_grad();losses=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for scene in rng.sample(train,args.batch_size):
                p=forward(model,source,scene,False);t=targets[scene['spec']['scene_id']]
                loss=outcome_loss(p['p0'],torch.as_tensor(t['p0'],device='cuda'),'factorized')
                if t['post_mask']:loss=loss+outcome_loss(p['p1'],torch.as_tensor(t['p1'],device='cuda'),'factorized')
                for branch in (0,1):
                    mask=torch.as_tensor(scene['branches'][branch]['grasp_mask'],device='cuda')
                    if mask.any():loss=loss+.5*outcome_loss(p[f'g{branch}'][mask],
                        torch.as_tensor(t[f'g{branch}'],device='cuda')[mask],'factorized')
                losses.append(loss)
            loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite partial-map PG loss.')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step();torch.cuda.synchronize();timings.append(time.perf_counter()-tick)
        if step==1 or step%10==0:
            progress=dict(pid=os.getpid(),phase='pg_'+name,step=step,total_steps=args.pg_steps,loss=float(loss.detach()),
                seconds_per_step=float(np.mean(timings[-10:])),elapsed_seconds=time.perf_counter()-begin)
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        if step==1 or step%40==0 or step==args.pg_steps:
            summary=evaluate(model,val,predict(model,source,val),policies)[0][key]
            history.append(dict(step=step,summary=summary))
            if validation_key(summary)>best:
                best=validation_key(summary);selected=step;save(folder/'selected_pg.pt',model,source,step=step,**metadata)
    save(folder/'last_pg.pt',model,source,step=args.pg_steps,**metadata)
    if frozen_core_digest(model)!=frozen:raise AssertionError('Original H-JEPA core changed.')
    result=dict(seed=42,steps=args.pg_steps,selected_step=selected,history=history,teacher_refreshes=refreshes,
        training_step_seconds=timings,lora=lora,original_core_exact=True,test_outcome_reads=0,seconds=time.perf_counter()-begin)
    write_json(folder/'pg_training.json',result)
    del model,source,optimizer;gc.collect();torch.cuda.empty_cache();return result


def gate_data(model,scenes,cached,device):
    import torch
    from mobile_outcome_v4 import continuation_class
    from mobile_partial_policy_v5 import partial_gate_features
    features=[];labels=[];costs=[];risks=[];ids=[]
    for scene in scenes:
        p=cached[scene['spec']['scene_id']]
        for risk in (1.,3.):
            features.append(partial_gate_features(model,p['front'].to(device),
                scene['branches'][0]['arrays']['initial_proprio'],scene['maps']['initial']['features'],risk))
            labels.append(continuation_class(scene,p['p1'].to(device),p['g1'].to(device),risk,scene['maps']['fused']['mask']))
            costs.append(scene['branches'][1]['records'][0]['observation_cost']);risks.append(risk)
            ids.append(dict(scene_id=scene['spec']['scene_id'],risk=risk))
    return torch.cat(features),torch.tensor(labels,device=device),torch.tensor(costs,device=device),torch.tensor(risks,device=device),ids


def train_gate(args,name,scenes):
    import torch
    from torch.nn import functional as F
    from mobile_outcome_v4 import outcome_loss
    from mobile_partial_policy_v5 import load_partial_checkpoint
    from experiment_reproducibility import seed_everything
    from run_mobile_active_experiment import validation_key
    seed_everything(42);begin=time.perf_counter();folder=args.output/name
    model,source=load_partial_checkpoint(folder/'selected_pg.pt','cuda');model.requires_grad_(False)
    fixed={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not n.startswith('inspection_policy.')}
    model.inspection_policy.requires_grad_(True)
    cached=predict(model,source,scenes)
    training=[s for s in scenes if s['spec']['split']=='train'];validation=[s for s in scenes if s['spec']['split']=='val']
    features,labels,costs,risks,identities=gate_data(model,training,cached,'cuda')
    np.savez_compressed(folder/'gate_training_features.npz',features=features.detach().cpu().numpy(),labels=labels.cpu().numpy(),costs=costs.cpu().numpy())
    write_json(folder/'gate_training_identities.json',identities)
    key=name+'_learned';policies={key:POLICIES[key]}
    summary=evaluate(model,validation,cached,policies)[0][key]
    best=validation_key(summary);selected=0;history=[dict(step=0,summary=summary)]
    metadata=dict(phase='inspection',experiment='partial_map_v5',pg_selected_sha256=file_hash(folder/'selected_pg.pt'),
        initial_sha256=file_hash(args.initial),manifest_sha256=file_hash(args.output/'manifest.json'))
    save(folder/'selected.pt',model,source,step=0,**metadata)
    optimizer=torch.optim.AdamW(model.inspection_policy.parameters(),lr=3e-4,weight_decay=.01);rng=random.Random(42)
    for step in range(1,args.gate_steps+1):
        sample=torch.tensor(rng.sample(range(len(features)),min(32,len(features))),device='cuda')
        model.inspection_policy.train();optimizer.zero_grad();raw=model.inspection_policy(features[sample]).float()
        losses=[]
        for risk in (1.,3.):
            mask=risks[sample]==risk
            if mask.any():losses.append(outcome_loss(raw[mask,:4],labels[sample][mask],'factorized',risk))
        loss=torch.stack(losses).mean()+.5*F.mse_loss(F.softplus(raw[:,4]),costs[sample])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite partial-map gate loss.')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.inspection_policy.parameters(),1.);optimizer.step()
        if step==1 or step%40==0 or step==args.gate_steps:
            summary=evaluate(model,validation,cached,policies)[0][key]
            history.append(dict(step=step,summary=summary,loss=float(loss.detach())))
            if validation_key(summary)>best:
                best=validation_key(summary);selected=step;save(folder/'selected.pt',model,source,step=step,**metadata)
            progress=dict(pid=os.getpid(),phase='gate_'+name,step=step,total_steps=args.gate_steps,elapsed_seconds=time.perf_counter()-begin)
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    save(folder/'last.pt',model,source,step=args.gate_steps,**metadata)
    if any(not torch.equal(v,model.state_dict()[n].detach().cpu()) for n,v in fixed.items()):
        raise AssertionError('Frozen continuation changed while training I.')
    result=dict(seed=42,steps=args.gate_steps,selected_step=selected,history=history,
        training_examples=len(features),target='current frozen post P/G outcome under fused depth mask; no oracle best action',
        gate_input='initial RGB feature, current relative inspection goal, declared risk, initial depth features only',
        all_non_inspection_tensors_exact=True,test_outcome_reads=0,seconds=time.perf_counter()-begin)
    write_json(folder/'gate_training.json',result)
    del model,source,optimizer,fixed;gc.collect();torch.cuda.empty_cache();return result


def audit_tensors(args):
    import torch
    original=torch.load(args.initial,map_location='cpu',weights_only=True)
    names=[n for n in original['model'] if 'lora_' not in n and not n.startswith(('position_policy.','grasp_policy.','inspection_policy.'))]
    audits={}
    for name in MODELS:
        pg=torch.load(args.output/name/'selected_pg.pt',map_location='cpu',weights_only=True);audits[name]={}
        for label in ('selected_pg','last_pg','selected','last'):
            cp=pg if label=='selected_pg' else torch.load(args.output/name/(label+'.pt'),map_location='cpu',weights_only=True)
            assert all(torch.equal(original['model'][n],cp['model'][n]) for n in names)
            assert all(torch.equal(original[k],cp[k]) for k in ('state_mean','state_std'))
            changed=sum(not torch.equal(v,cp['model'][n]) for n,v in original['model'].items() if 'lora_' in n)
            if label=='last_pg':assert changed>0
            fixed=all(torch.equal(v,cp['model'][n]) for n,v in pg['model'].items() if not n.startswith('inspection_policy.')) if label in ('selected','last') else None
            if fixed is not None:assert fixed
            audits[name][label]=dict(step=cp['step'],original_tensor_count=len(names),original_tensors_exact=True,
                normalization_exact=True,changed_lora_tensors=changed,selected_pg_exact_outside_I=fixed)
            if cp is not pg:del cp
            gc.collect()
        del pg;gc.collect()
    result=dict(seed=42,checkpoints=audits,original_tensor_count=len(names),all_original_tensors_exact=True,all_stage2_continuations_exact=True)
    write_json(args.output/'tensor_audit.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--development',type=Path,default=ROOT/'runs/mobile_active_seed42_20261109')
    p.add_argument('--maps',type=Path,default=ROOT/'runs/mobile_partial_development_seed42_20261109')
    p.add_argument('--initial',type=Path,default=ROOT/'runs/mobile_outcome_seed42_20261109/factorized/selected.pt')
    p.add_argument('--history-root',type=Path,default=ROOT/'runs')
    p.add_argument('--development-groups',type=int)
    p.add_argument('--pg-steps',type=int,default=320);p.add_argument('--gate-steps',type=int,default=800)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--test-groups',type=int,default=18)
    p.add_argument('--stream',default='mobile_partial_v5_20261109')
    p.add_argument('--estimate-seconds',type=int,required=True)
    args=p.parse_args()
    for key in ('output','development','maps','initial','history_root'):setattr(args,key,getattr(args,key).resolve())
    if ROOT/'runs' not in args.output.parents:raise ValueError('Output must be ignored runs/.')
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Preserve existing runs.')
    if min(args.pg_steps,args.gate_steps,args.batch_size,args.test_groups)<1:raise ValueError('Positive budgets required.')
    args.output.mkdir(parents=True);begin=time.perf_counter();now=datetime.now(ZONE)
    receipt=dict(pid=os.getpid(),seed=42,status='starting',started_at=now.isoformat(),queue_estimate_seconds=0,
        runtime_estimate_seconds=args.estimate_seconds,expected_completion=(now+timedelta(seconds=args.estimate_seconds)).isoformat(),
        scope='train/validation LoRA continuation and pre-test freeze only; no final test outcomes')
    write_json(args.output/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    from experiment_reproducibility import seed_everything
    receipt['reproducibility']=seed_everything(42)
    from run_mobile_outcome_experiment import make_run_manifest,attach_development
    from mobile_crossed_v2 import geometry_identity
    manifest=make_run_manifest(args.development,args.test_groups,args.stream,args.output,args.development_groups)
    history=set();history_files=[]
    for path in args.history_root.glob('*/manifest.json'):
        if path.parent.resolve()==args.output:continue
        document=read(path)
        for spec in document.get('scenes',[]):
            if 'start_pose' in spec and 'obstacles' in spec:history.add(geometry_identity(spec))
        history_files.append(str(path))
    tests=[s for s in manifest['scenes'] if s['split']=='test']
    if any(geometry_identity(s) in history for s in tests):raise AssertionError('Fresh test overlaps historical scene geometry.')
    manifest.update(schema_version=5,historical_geometries_excluded_from_test=len(history),history_manifests=history_files)
    write_json(args.output/'manifest.json',manifest)
    source=args.output/'source';source.mkdir()
    for name in ['AGENTS.md','scene.xml','.gitignore',*sorted(p.name for p in ROOT.glob('*.py'))]:shutil.copy2(ROOT/name,source/name)
    hashes={x.name:file_hash(x) for x in source.iterdir()};write_json(args.output/'source_hashes.json',hashes)
    specs=[s for s in manifest['scenes'] if s['split'] in ('train','val')];scenes=attach_development(args,specs)
    map_hashes={}
    for scene in scenes:
        folder=args.maps/'data'/scene['spec']['scene_id'];path=folder/'maps.json';maps=read(path)
        if not maps['original_candidate_prefix_states_match'] or not maps['queries_preserve_live_state']:raise AssertionError('Invalid prefix.')
        scene['maps']=maps
        for path in (path,folder/'reached_rgbd.npz'):map_hashes[str(path)]=file_hash(path)
    if file_hash(ROOT/'mobile_partial_map_v5.py')!=read(args.maps/'source_hashes.json')['mobile_partial_map_v5.py']:
        raise AssertionError('Map implementation differs from frozen acquisition.')
    write_json(args.output/'reused_map_hashes.json',map_hashes)
    initial_hash=file_hash(args.initial)
    protocol=dict(seed=42,models=MODELS,policies=POLICIES,initial=str(args.initial),initial_sha256=initial_hash,
        maps=str(args.maps),development=str(args.development),pg_steps=args.pg_steps,gate_steps=args.gate_steps,
        batch_scenes=args.batch_size,counts=manifest['counts'],loss='3-class outcome CE; terminal values1,-1,-1-c; refusal0; actual inspection cost',
        map_comparison='matched-capacity actual current map features vs zeroed features; same partial depth hard masks',
        stage1='original core frozen; existing rank8 LoRA and P/G train; val always-observe at risk1 selects PG',
        stage2='all non-I tensors frozen; I fits current greedy post P/G outcomes, using initial view/map only',
        checkpoint_selection='validation correct decisions, fewer collisions, openings, net utility, path, observations; earliest exact checkpoint tie',
        policy_selection='six declared model policies at risk1 by same validation key; declared order exact ties',
        test_scope='fresh grouped manifest frozen; no test data collection in this training job',
        known_map_scope='navigation/arm preparation held fixed; depth limitation applies to opening sweep planning')
    write_json(args.output/'protocol.json',protocol)
    receipt.update(status='training',startup_seconds=time.perf_counter()-begin);write_json(args.output/'receipt.json',receipt)
    training={};validation={};phases={}
    for name in MODELS:
        pg=train_pg(args,name,scenes);gate=train_gate(args,name,scenes)
        training[name]=dict(pg=pg,gate=gate);phases['pg_'+name]=pg['seconds'];phases['gate_'+name]=gate['seconds']
        from mobile_partial_policy_v5 import load_partial_checkpoint
        import torch
        model,checkpoint=load_partial_checkpoint(args.output/name/'selected.pt','cuda')
        val=[s for s in scenes if s['spec']['split']=='val'];cached=predict(model,checkpoint,val)
        summaries,decisions=evaluate(model,val,cached,{n:c for n,c in POLICIES.items() if c['model']==name})
        validation.update(summaries);write_json(args.output/name/'validation_decisions.json',decisions)
        del model,checkpoint,cached;gc.collect();torch.cuda.empty_cache()
    from run_mobile_active_experiment import best_policy
    selected=best_policy(validation)
    audit=audit_tensors(args)
    for path,h in {**map_hashes,**read(args.output/'reused_development_hashes.json')}.items():
        if file_hash(path)!=h:raise AssertionError('Historical development input changed: '+path)
    assert file_hash(args.initial)==initial_hash
    assert all(file_hash(ROOT/n)==h and file_hash(args.output/'source'/n)==h for n,h in hashes.items())
    frozen=dict(seed=42,selected_policy=selected,selected_before_test=True,manifest_sha256=file_hash(args.output/'manifest.json'),
        checkpoint_sha256={n:file_hash(args.output/n/'selected.pt') for n in MODELS},baseline_sha256=initial_hash,
        validation_metrics=validation,training_source_hashes=hashes,test_outcomes_collected=False,
        frozen_at=datetime.now(ZONE).isoformat(),final_evaluation_required=True)
    write_json(args.output/'frozen_for_test.json',frozen)
    from check_git_size import audit as size_audit
    size=size_audit();write_json(args.output/'git_size_audit.json',size);assert size['passed']
    phases['startup_selection_audit_io']=time.perf_counter()-begin-sum(phases.values())
    report=dict(seed=42,selected_policy=selected,validation=validation,training=training,phase_seconds=phases,
        seconds=time.perf_counter()-begin,all_original_tensors_exact=audit['all_original_tensors_exact'],
        all_continuations_frozen_during_I=True,all_sources_and_historical_inputs_exact=True,
        test_outcome_files_opened=0,status='training_complete_test_pending',physical_replays_in_this_job=0)
    write_json(args.output/'report.json',report)
    receipt.update(status='complete',phase='training_and_pretest_freeze',exit_code=0,actual_seconds=report['seconds'],
        completed_at=datetime.now(ZONE).isoformat(),phase_seconds=phases)
    write_json(args.output/'receipt.json',receipt)
    print(json.dumps(dict(status='training_complete_test_pending',seconds=report['seconds'],selected=selected,validation=validation)),flush=True)


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
