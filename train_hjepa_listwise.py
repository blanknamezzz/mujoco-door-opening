"""Position-policy adaptation: door-side reward or successful-set control.

Uses full native official encoder at 224px, not temporal world-model training.
All physical success labels are training targets; inference still sees initial
RGB and candidate base commands only. G stays fixed while P is optimized.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import time
import hjepa_runtime
import numpy as np
import torch
from torch.nn import functional as F
from hjepa_model import load_checkpoint, save_checkpoint
from hjepa_lora import adapter_state, frozen_core_digest, resume_lora, freeze_core_statistics
from train_hjepa import read_data, rgb_tensor, state_tensor
from evaluate_structural_training import evaluate_records
from run_hjepa_lora_experiment import hash_file, write_json
from door_side_reward import side_reward_targets, door_side_loss, side_feasibility_loss, reward_contract, validation_key
from position_observation import encode_position_views,position_features
from experiment_reproducibility import seed_everything


def successful_set_loss(scores, successful, temperature=.2):
    """Reward total probability of any correct action, with no arbitrary tie."""
    if scores.shape != successful.shape or not successful.any(-1).all():
        raise ValueError('Each scene needs a nonempty correct-action set.')
    logits = scores/temperature
    return (torch.logsumexp(logits, -1)-torch.logsumexp(
        logits.masked_fill(~successful, -torch.inf), -1)).mean()


def correct_action_mask(successes, refusal):
    successes = torch.as_tensor(successes, dtype=torch.bool).clone()
    refusal = torch.as_tensor(refusal, dtype=torch.bool)
    if successes.shape != refusal.shape or refusal.sum() != 1:
        raise ValueError('Exactly one explicit refusal required per scene.')
    successes[refusal] = False
    if not successes.any():
        successes[refusal] = True
    return successes


def train(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'training.json').exists() or (args.output/'best.pt').exists():
        raise FileExistsError('Use a fresh output, preserve previous runs.')
    reproducibility=seed_everything(args.seed)
    torch.set_num_threads(4)
    manifest, records = read_data(args.data, required_splits=('train','val'), include_splits=('train','val'),policy_only=True)
    unique_arrays={id(v):v for _,x in records for v in x.values() if isinstance(v,np.ndarray)}
    data_memory_bytes=sum(v.nbytes for v in unique_arrays.values())
    if any(s['split']=='test' for s in manifest['scenes']):
        raise ValueError('Dedicated train/validation manifest required.')
    model, source = load_checkpoint(args.initial, 'cuda')
    if not model.is_official or model.config.get('reward_decomposition'):
        raise ValueError('Warm-start the scalar official LoRA policy.')
    model.config['position_reward_objective']=args.objective
    side_objective=args.objective in ('door-side','side-feasibility')
    model.config['position_reward']=reward_contract(args.objective) if side_objective else None
    # Threshold selection is a later frozen phase, not a hidden inherited
    # decision parameter inside a new objective comparison.
    model.config.pop('position_refusal_offset',None)
    model.config['position_views']=args.position_views
    position_encoder=getattr(args,'position_encoder',None)
    if position_encoder is not None:
        model.config['position_encoder']=position_encoder
    if args.position_views=='paired' and not manifest.get('position_rear_files'):
        raise ValueError('Paired position policy requires a prepared rear-camera dataset.')
    lora = resume_lora(model, 8, 16., .05)
    model.requires_grad_(False)
    model.position_policy.requires_grad_(True)
    if not args.head_only:
        for n,p in model.model.named_parameters():
            if 'lora_' in n: p.requires_grad_(True)
    mean, std = source['state_mean'], source['state_std']
    budgets = {k:source[k] for k in ('base_budget','grasp_budget','positioning_abstention',
        'candidate_base_poses','abstention_reward','adapter_base_checkpoint')}
    budgets.update(seed=args.seed,reproducibility=reproducibility)
    frozen = frozen_core_digest(model)
    original_grasp = {n:v.clone() for n,v in model.grasp_policy.state_dict().items()}
    original_lora = {n:v.clone() for n,v in model.model.state_dict().items() if 'lora_' in n}
    tables = defaultdict(lambda:defaultdict(list))
    for e,x in records: tables[e['scene_id']][e['base_id']].append((e,x))
    cached=[]
    model.eval()
    with torch.inference_mode():
        for table in tables.values():
            groups=[g for _,g in sorted(table.items())]
            first=groups[0][0][1]
            rear=first.get('initial_rear_rgb')
            latent=encode_position_views(model,rgb_tensor(first['initial_rgb'][None],'cuda'),
                state_tensor(first['initial_proprio'][None],mean,std,'cuda'),
                rgb_tensor(rear[None],'cuda') if rear is not None else None)
            conditions=torch.tensor(np.stack([g[0][1]['base'] for g in groups]),device='cuda')
            old_scores=model.position_policy(position_features(latent,conditions[None])[0]).flatten()
            successes=[]
            for group in groups:
                e,x=group[0]
                if e.get('positioning_only') or not e.get('grasp_observed',True):
                    successes.append(False); continue
                z=model.encode(rgb_tensor(x['arrival_rgb'][None],'cuda'),state_tensor(x['arrival_proprio'][None],mean,std,'cuda'))
                g_conditions=torch.tensor(np.stack([a['grasp'] for _,a in group]),device='cuda')
                scores=model.grasp_policy(torch.cat([z.expand(len(group),-1),g_conditions],-1)).flatten()
                eligible=[i for i,(e,_) in enumerate(group) if e.get('current_reachable',True)]
                selected=max(eligible,key=lambda i:float(scores[i])) if eligible else int(scores.argmax())
                successes.append(group[selected][0]['success'])
            mask=correct_action_mask(successes,[g[0][0].get('positioning_only',False) for g in groups])
            side_targets=(side_reward_targets([[e for e,_ in g] for g in groups],source['grasp_budget'])
                          if side_objective else {})
            cached.append(dict(split=groups[0][0][0]['split'],scene_id=groups[0][0][0]['scene_id'],
                rgb=first['initial_rgb'],state=first['initial_proprio'],conditions=conditions.cpu().numpy(),
                rear=rear,latent=latent.cpu().numpy()[0],teacher=old_scores.cpu().numpy(),mask=mask.numpy(),**side_targets))
    training=[x for x in cached if x['split']=='train']
    model.enable_activation_checkpointing()
    groups=[dict(params=list(model.position_policy.parameters()),lr=3e-4)]
    if not args.head_only:
        groups.append(dict(params=[p for p in model.model.parameters() if p.requires_grad],lr=2e-5))
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    def validation():
        model.eval()
        return evaluate_records(model,source,records,manifest,split='val')[0]
    baseline=validation()
    def key(s): return validation_key(s,args.objective)
    best_key=key(baseline); selected_step=0
    save_checkpoint(args.output/'best.pt',model,mean.cpu(),std.cpu(),step=0,**budgets)
    torch.save(dict(adapter=adapter_state(model),config=model.config,base_checkpoint=budgets['adapter_base_checkpoint'],
        state_mean=mean.cpu(),state_std=std.cpu(),metadata=dict(step=0,**budgets)),args.output/'initial_adapter.pt')
    history=[]; timings=[]
    write_json(args.output/'adapter_contract.json',dict(lora=lora,original_core_frozen=True,
        frozen_core_sha256=frozen,head_only=args.head_only,grasp_head_frozen=True,
        trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],test_arrays_opened=0,
        target=('measured side feasibility and within-side base quality, independent of G ranking'
            if side_objective else 'successful actions under initial fixed G, refusal only if none succeeds'),
        objective=args.objective,door_side_reward=reward_contract(args.objective) if side_objective else None,
        position_views=args.position_views,
        position_encoder=model.config.get('position_encoder','projected'),
        reproducibility=reproducibility,policy_data_unique_array_bytes=data_memory_bytes,
        optimizer='native endpoint encoder supervised; no temporal update or smaller model'))
    print(json.dumps(dict(pid=os.getpid(),stage='training',head_only=args.head_only,
        baseline_validation=baseline,cache_seconds=time.monotonic()-started)),flush=True)
    for step in range(1,args.steps+1):
        tick=time.monotonic()
        sample=random.choices(training,k=args.batch_size)
        model.eval()
        if args.head_only:
            latent=torch.tensor(np.stack([x['latent'] for x in sample]),device='cuda')
        else:
            # Activate checkpointing/dropout only for the native ViT. Frozen
            # projector BatchNorm and JEPA mutable queues never enter train mode.
            model.model.get_level(1).encoder.encoder.train()
            freeze_core_statistics(model)
        optimizer.zero_grad()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if not args.head_only:
                latent=encode_position_views(model,rgb_tensor(np.stack([x['rgb'] for x in sample]),'cuda'),
                    state_tensor(np.stack([x['state'] for x in sample]),mean,std,'cuda'),
                    rgb_tensor(np.stack([x['rear'] for x in sample]),'cuda') if args.position_views=='paired' else None)
            conditions=torch.tensor(np.stack([x['conditions'] for x in sample]),device='cuda')
            features=position_features(latent,conditions)
            scores=model.position_policy(features).squeeze(-1).float()
            mask=torch.tensor(np.stack([x['mask'] for x in sample]),device='cuda')
            side_parts={}
            if side_objective:
                loss_function=side_feasibility_loss if args.objective=='side-feasibility' else door_side_loss
                primary,side_parts=loss_function(scores,
                    torch.tensor(np.stack([x['sides'] for x in sample]),device='cuda'),
                    torch.tensor(np.stack([x['side_rewards'] for x in sample]),device='cuda'),
                    torch.tensor(np.stack([x['base_rewards'] for x in sample]),device='cuda'))
            else:
                primary=successful_set_loss(scores,mask)
            teacher=torch.tensor(np.stack([x['teacher'] for x in sample]),device='cuda')
            anchor=F.kl_div((scores/.2).log_softmax(-1),(teacher/.2).softmax(-1),reduction='batchmean')
            loss=primary+.02*anchor
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite listwise loss.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step()
        timings.append(time.monotonic()-tick)
        if step==1 or step%10==0 or step==args.steps:
            progress=dict(step=step,total_steps=args.steps,loss=float(loss.detach()),
                primary_loss=float(primary.detach()),objective=args.objective,
                reward_losses={k:float(v.detach()) for k,v in side_parts.items()},anchor=float(anchor.detach()),
                recent_seconds_per_step=float(np.mean(timings[-30:])),elapsed_seconds=time.monotonic()-started,pid=os.getpid())
            write_json(args.output/'progress.json',progress); print(json.dumps(progress),flush=True)
        if step==1 or step%40==0 or step==args.steps:
            summary=validation()
            entry=dict(step=step,train=float(loss.detach()),val=0.,policy_validation=summary)
            history.append(entry)
            if key(summary)>best_key:
                best_key=key(summary); selected_step=step
                save_checkpoint(args.output/'best.pt',model,mean.cpu(),std.cpu(),step=step,**budgets)
            print(json.dumps(dict(validation=summary,selected_step=selected_step)),flush=True)
    if frozen_core_digest(model)!=frozen: raise AssertionError('Original core changed.')
    if any(not torch.equal(v,model.grasp_policy.state_dict()[n]) for n,v in original_grasp.items()):
        raise AssertionError('Frozen grasp head changed.')
    changed=sum(not torch.equal(v,model.model.state_dict()[n]) for n,v in original_lora.items())
    if args.head_only and changed: raise AssertionError('Head control changed LoRA.')
    save_checkpoint(args.output/'last.pt',model,mean.cpu(),std.cpu(),step=args.steps,**budgets)
    selected,selected_source=load_checkpoint(args.output/'best.pt','cuda')
    save_checkpoint(args.output/'selected.pt',selected,mean.cpu(),std.cpu(),step=selected_step,**budgets)
    torch.save(dict(adapter=adapter_state(selected),config=selected.config,base_checkpoint=budgets['adapter_base_checkpoint'],
        state_mean=mean.cpu(),state_std=std.cpu(),metadata=dict(step=selected_step,**budgets)),args.output/'adapter.pt')
    selected_validation=evaluate_records(selected,selected_source,records,manifest,split='val')[0]
    train_summary=evaluate_records(selected,selected_source,records,manifest,split='train')[0]
    report=dict(history=history,unadapted_validation=baseline,steps=args.steps,selected_world_model_step=selected_step,
        selected_policy_updates=0,selected_method_adopted=selected_step>0,listwise_position=True,head_only=args.head_only,
        selected_validation=selected_validation,training_summary=train_summary,changed_lora_tensors=changed,
        initialization=dict(path=str(args.initial),sha256=hash_file(args.initial),step=source.get('step')),
        objective=args.objective,door_side_reward=reward_contract(args.objective) if side_objective else None,
        position_views=args.position_views,
        position_encoder=model.config.get('position_encoder','projected'),
        reproducibility=reproducibility,seed=args.seed,policy_data_unique_array_bytes=data_memory_bytes,
        adapter_base_checkpoint=budgets['adapter_base_checkpoint'],lora=lora,scenes=manifest['scenes'],
        test_arrays_opened=0,frozen_core_unchanged=True,grasp_head_frozen=True,selected_world_model_unchanged=True,
        checkpoint_selection=('validation correct side decisions, then correct task decisions, openings and terminal utility; step-zero fallback'
            if side_objective else 'validation correct decisions, openings, then terminal utility; step-zero fallback'),
        batch_size=args.batch_size,precision='bf16-mixed',training_step_seconds=timings,
        elapsed_seconds=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    write_json(args.output/'training.json',report)
    print(json.dumps(dict(complete=True,selected_step=selected_step,validation=selected_validation,
        elapsed_seconds=report['elapsed_seconds'],changed_lora_tensors=changed)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('runs/hjepa/diverse_lora_trainval'))
    p.add_argument('--initial',type=Path,default=Path('runs/hjepa/lora_continued_model/selected.pt'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=240)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--head-only',action='store_true')
    p.add_argument('--objective',choices=('door-side','side-feasibility','successful-set'),default='door-side')
    p.add_argument('--position-views',choices=('single','paired','repeat-front'),default='single')
    p.add_argument('--position-encoder',choices=('projected','backbone_cls','backbone_patch_mean'),default=None,
        help='Optional positioning-only pooling override; dynamics and grasp latents stay native.')
    args=p.parse_args()
    if args.steps<1 or args.batch_size<2:p.error('Positive steps and batch size >=2 required.')
    train(args)
