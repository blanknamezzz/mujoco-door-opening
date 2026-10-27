"""LoRA H-JEPA: measured progress, outcome components and within-scene preferences.

This is finite-candidate supervised/offline policy improvement, not PPO or a
reproduction of the cited generalist VLM reward models.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import hjepa_runtime
import torch
from torch.nn import functional as F
from decomposed_reward import (COMPONENTS, VERSION, configure_decomposition,
                               component_loss, outcome_target, preference_loss, terminal_utility)
from hjepa_lora import adapter_state, freeze_core_statistics, frozen_core_digest, resume_lora
from hjepa_model import load_checkpoint, save_checkpoint
from train_hjepa import read_data, rgb_tensor, state_tensor
from run_hjepa_lora_experiment import hash_file, write_json


def group_records(records):
    groups = defaultdict(lambda: defaultdict(list))
    for episode, arrays in records:
        groups[episode['scene_id']][episode['base_id']].append((episode, arrays))
    return [list(sorted(bases.items())) for bases in groups.values()]


def policy_objective(model, scenes, importance, mean, std, device):
    """Position sees only initial RGB; arrivals are G inputs / training targets."""
    initial_arrays = [bases[0][1][0][1] for bases in scenes]
    initial = model.encode_position(
        rgb_tensor(np.stack([x['initial_rgb'] for x in initial_arrays]), device),
        state_tensor(np.stack([x['initial_proprio'] for x in initial_arrays]), mean, std, device))
    arrival_groups, group_index = [], {}
    for si, bases in enumerate(scenes):
        for bi, group in bases:
            ep, x = group[0]
            if not ep.get('positioning_only') and ep.get('grasp_observed', True):
                group_index[(si, bi)] = len(arrival_groups)
                arrival_groups.append(group)
    arrivals = None
    if arrival_groups:
        arrivals = model.encode(rgb_tensor(np.stack([g[0][1]['arrival_rgb'] for g in arrival_groups]), device),
            state_tensor(np.stack([g[0][1]['arrival_proprio'] for g in arrival_groups]), mean, std, device))
    components, preferences = [], []
    for si, bases in enumerate(scenes):
        targets, g_component, g_preference = [], [], []
        for bi, group in bases:
            ep, x = group[0]
            if ep.get('positioning_only'):
                targets.append(None)
                continue
            labels = torch.tensor(np.stack([outcome_target(e) for e, _ in group]), device=device)
            if (si, bi) in group_index:
                latent = arrivals[group_index[(si, bi)]].expand(len(group), -1)
                conditions = torch.tensor(np.stack([a['grasp'] for _, a in group]), device=device)
                features = torch.cat([latent, conditions], -1)
                logits = model.grasp_policy.component_logits(features)
                scores = model.grasp_policy(features).flatten()
                g_component.append(component_loss(logits, labels))
                actual_utilities = 2*labels[:, 0]-1-labels[:, 1]
                g_preference.append(preference_loss(scores, actual_utilities))
                eligible = torch.tensor([e.get('current_reachable', True) for e, _ in group], device=device)
                if not eligible.any():
                    eligible = torch.ones_like(eligible)
                # G is held fixed while constructing P's expected downstream
                # labels. Future outcomes never enter position input features.
                weights = (scores.detach().masked_fill(~eligible, -torch.inf)/.2).softmax(0)
                targets.append((weights[:, None]*labels).sum(0))
            else:
                targets.append(labels.mean(0))  # no observed grasp: negative P example only
        conditions = torch.tensor(np.stack([g[0][1]['base'] for _, g in bases]), device=device)
        features = torch.cat([initial[si:si+1].expand(len(bases), -1), conditions], -1)
        physical = [i for i, target in enumerate(targets) if target is not None]
        labels = torch.stack([targets[i] for i in physical])
        p_component = component_loss(model.position_policy.component_logits(features[physical]), labels)
        utilities = torch.stack([initial.new_tensor(0.) if t is None else 2*t[0]-1-t[1] for t in targets])
        p_preference = preference_loss(model.position_policy(features), utilities)
        zero = initial.sum()*0
        components.append(importance[si]*(p_component+(torch.stack(g_component).mean() if g_component else zero)))
        preferences.append(importance[si]*(p_preference+(torch.stack(g_preference).mean() if g_preference else zero)))
    return torch.stack(components).mean(), torch.stack(preferences).mean()


def train(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output/f).exists() for f in ('best.pt', 'selected.pt', 'training.json')):
        raise FileExistsError('Preserve existing runs; choose a new output directory.')
    print(json.dumps(dict(process_id=os.getpid(), stage='loading_trainval', steps=args.steps)), flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    manifest, records = read_data(args.data, required_splits=('train', 'val'), include_splits=('train', 'val'))
    if any(s['split'] == 'test' for s in manifest['scenes']):
        raise ValueError('Use the dedicated train/validation manifest, not a combined split.')
    model, source = load_checkpoint(args.initial, 'cuda')
    if not model.is_official:
        raise ValueError('This experiment keeps the complete official architecture.')
    initialization = dict(path=str(args.initial), sha256=hash_file(args.initial), step=source['step'], optimizer_resumed=False)
    base = source.get('adapter_base_checkpoint')
    if not base:
        previous = json.loads((args.initial.parent/'training.json').read_text(encoding='utf-8'))
        base = previous.get('adapter_base_checkpoint') or previous['initialization']
    if hash_file(Path(base['path'])) != base['sha256']:
        raise ValueError('Original base hash mismatch.')
    mean, std = source['state_mean'].to('cuda'), source['state_std'].to('cuda')
    lora = resume_lora(model, 8, 16., .05)
    configure_decomposition(model, dict(version=VERSION, blend=0., component_names=list(COMPONENTS)))
    lora['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.enable_activation_checkpointing()
    frozen = frozen_core_digest(model)
    budgets = {k: source[k] for k in ('base_budget', 'grasp_budget', 'positioning_abstention', 'candidate_base_poses', 'abstention_reward')}
    budgets['adapter_base_checkpoint'] = base
    del source
    training_records = [(e, x) for e, x in records if e['split'] == 'train']
    scenes = group_records(training_records)
    cohorts = [[s for s in scenes if any(e['success'] for _, g in s for e, _ in g) == feasible] for feasible in (False, True)]
    if not all(cohorts):
        raise ValueError('Need both solvable and unsolvable training scenes.')
    windows = defaultdict(list)
    raw_length = model.config['raw_clip_frames']
    for e, x in records:
        windows[e['split']].extend((e, x, start) for start in range(len(x['pixels'])-raw_length+1))
    by_scene = defaultdict(list)
    for clip in windows['train']:
        by_scene[clip[0]['scene_id']].append(clip)
    clip_groups = list(by_scene.values())
    rng_val = random.Random(args.seed+100)
    val_clips = rng_val.choices(windows['val'], k=128)
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in model.model.parameters() if p.requires_grad], lr=5e-5),
        dict(params=[p for n, p in model.named_parameters() if not n.startswith('model.') and p.requires_grad], lr=3e-4)], weight_decay=.01)
    warmup = min(30, max(1, args.steps//10))
    def scale(step):
        if step < warmup:
            return (step+1)/warmup
        return .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1, args.steps-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    contract = dict(**lora, frozen_core_sha256=frozen, original_core_trainable=False,
        trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],
        components=list(COMPONENTS), progress_target='measured signed door qpos[0] / (pi/2), native hierarchical target aliases',
        loss_weights=dict(world=1., components=2., preferences=1., progress=.25, terminal_value=.25),
        p_target='expected measured G outcomes under detached current reachable G distribution',
        balanced_scene_cohorts=True, component_importance_corrected=True, test_arrays_opened=0)
    write_json(args.output/'adapter_contract.json', contract)

    def temporal_loss(clips, stage):
        skip, length = model.config['frame_skip'], model.config['clip_frames']
        q = np.stack([np.clip(x['proprio'][s:s+raw_length:skip, 0]*e['opening_direction']/(np.pi/2), 0, 1)
                      for e, x, s in clips])
        batch = dict(pixels_level1=rgb_tensor(np.stack([x['pixels'][s:s+raw_length:skip] for _,x,s in clips]), 'cuda', scale=False),
            proprio_level1=state_tensor(np.stack([x['proprio'][s:s+raw_length:skip] for _,x,s in clips]), mean, std, 'cuda'),
            action_level1=torch.tensor(np.stack([x['action'][s:s+raw_length].reshape(length,-1) for _,x,s in clips]),device='cuda'),
            door_progress_level1=torch.tensor(q[:,:,None], device='cuda', dtype=torch.float32))
        output = model.predictive_loss(batch, stage)
        utility = torch.tensor([terminal_utility(e['success'], e['collision']) for e,_,_ in clips], device='cuda')[:,None,None]
        value, progress = output['loss']*0, output['loss']*0
        for level, value_head, progress_head in ((1,model.value_low,model.progress_low),(2,model.value_high,model.progress_high)):
            latent = output[f'embed_{level}']
            target = output[f'door_progress_level{level}']
            if target.shape[:2] != latent.shape[:2]:
                raise AssertionError('Native temporal progress alignment was lost.')
            value = value+F.mse_loss(value_head(latent.detach()), utility.expand(-1,latent.size(1),-1))
            progress = progress+F.mse_loss(progress_head(latent).sigmoid(), target)
        return output['loss'], value, progress

    from evaluate_structural_training import evaluate_records
    def validation():
        model.eval()
        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
            torch.manual_seed(args.seed+10000)
            summary, _ = evaluate_records(model, dict(state_mean=mean,state_std=std,**budgets), records, manifest, split='val')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                world, value, progress = temporal_loss(val_clips,'val')
        return summary, float(world+ .25*value+.25*progress)
    baseline, baseline_loss = validation()
    best_key = (baseline['correct_task_decisions'], baseline['mean_terminal_utility_v2'], -baseline_loss)
    selected_step = 0
    save_checkpoint(args.output/'best.pt', model, mean.cpu(), std.cpu(), step=0, seed=args.seed, **budgets)
    print(json.dumps(dict(unadapted_validation=baseline, validation_loss=baseline_loss)),flush=True)
    model.config['reward_decomposition']['blend'] = 1.
    model.position_policy.blend = model.grasp_policy.blend = 1.
    history, timings = [], []
    for step in range(1,args.steps+1):
        tick=time.monotonic()
        model.train()
        freeze_core_statistics(model)
        clips=[random.choice(random.choice(clip_groups)) for _ in range(128)]
        sampled=[random.choice(cohorts[i%2]) for i in range(args.scene_batch)]
        importance=[2*len(cohorts[i%2])/len(scenes) for i in range(args.scene_batch)]
        optimizer.zero_grad()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            world,value,progress=temporal_loss(clips,'train')
            model.eval()  # frozen normalization for endpoint encodings, gradients stay enabled
            components,preferences=policy_objective(model,sampled,importance,mean,std,'cuda')
            loss=world+.25*value+.25*progress+2*components+preferences
        # ViT activation recomputation must see the same dropout mode as its
        # original temporal forward, even though endpoint encodings used eval.
        model.train()
        freeze_core_statistics(model)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite decomposed loss.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step()
        scheduler.step()
        timings.append(time.monotonic()-tick)
        if step==1 or step%10==0 or step==args.steps:
            record=dict(step=step,total_steps=args.steps,elapsed_seconds=time.monotonic()-started,
                recent_seconds_per_step=float(np.mean(timings[-50:])),loss=float(loss.detach()),
                predictive=float(world.detach()),components=float(components.detach()),preferences=float(preferences.detach()),
                progress=float(progress.detach()),value=float(value.detach()),process_id=os.getpid())
            write_json(args.output/'progress.json',record)
            print(json.dumps(record),flush=True)
        if step==1 or step%50==0 or step==args.steps:
            summary,val_loss=validation()
            entry=dict(step=step,train=float(loss.detach()),val=val_loss,policy_validation=summary,
                       elapsed_seconds=time.monotonic()-started)
            history.append(entry)
            key=(summary['correct_task_decisions'],summary['mean_terminal_utility_v2'],-val_loss)
            if key>best_key:
                best_key,selected_step=key,step
                save_checkpoint(args.output/'best.pt',model,mean.cpu(),std.cpu(),step=step,seed=args.seed,**budgets)
            print(json.dumps(dict(validation=entry,selected_step=selected_step)),flush=True)
    if frozen_core_digest(model)!=frozen:
        raise AssertionError('Original core weights or buffers changed.')
    save_checkpoint(args.output/'last.pt',model,mean.cpu(),std.cpu(),step=args.steps,seed=args.seed,**budgets)
    selected, selected_source=load_checkpoint(args.output/'best.pt','cuda')
    save_checkpoint(args.output/'selected.pt',selected,mean.cpu(),std.cpu(),step=selected_step,seed=args.seed,**budgets)
    torch.save(dict(adapter=adapter_state(selected),config=selected.config,base_checkpoint=base,
        state_mean=mean.cpu(),state_std=std.cpu(),metadata=dict(step=selected_step,**budgets)),args.output/'adapter.pt')
    report=dict(history=history,unadapted_validation=baseline,steps=args.steps,selected_world_model_step=selected_step,
        selected_policy_updates=0,stage_policy_selection=dict(selected_update=0,updates_executed=0),
        initialization=initialization,adapter_base_checkpoint=base,lora=lora,reward_decomposition=selected.config['reward_decomposition'],
        scenes=manifest['scenes'],data_access_splits=['train','val'],test_arrays_opened=0,
        frozen_core_unchanged=True,selected_world_model_unchanged=True,selected_checkpoint='selected.pt',
        checkpoint_selection='validation correct task decisions, terminal utility v2, then fixed temporal validation loss',
        elapsed_seconds=time.monotonic()-started,training_step_seconds=timings,batch_size=128,precision='bf16-mixed',
        parameters=sum(p.numel() for p in selected.parameters()),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        selected_method_adopted=selected.config['reward_decomposition']['blend']==1.,
        note='No extra unconstrained scalar-head optimization; components and preference ranks are trained jointly with LoRA.')
    write_json(args.output/'training.json',report)
    write_json(args.output/'resolved_config.json',selected.config)
    print(json.dumps(dict(complete=True,selected_step=selected_step,elapsed_seconds=report['elapsed_seconds'],original_core_unchanged=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('runs/hjepa/diverse_lora_trainval'))
    p.add_argument('--initial',type=Path,default=Path('runs/hjepa/lora_continued_model/selected.pt'))
    p.add_argument('--output',type=Path,default=Path('runs/hjepa/reward_v2_model20261026'))
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--scene-batch',type=int,default=8)
    p.add_argument('--seed',type=int,default=43)
    args=p.parse_args()
    if args.steps<1 or args.scene_batch<2 or args.scene_batch%2:
        p.error('Use positive steps and an even scene batch >=2.')
    train(args)
