"""Train official H-JEPA dynamics and warm-start the two door stage policies."""
import argparse
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import hjepa_runtime  # process-local CUDA environment fix
import torch
from torch import nn
from hjepa_model import DoorHJEPA, compact_config, default_config, load_checkpoint, resolve_device, save_checkpoint


def read_data(directory):
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    scene_splits = {}
    records = []
    for ep in manifest['episodes']:
        previous = scene_splits.setdefault(ep['scene_id'], ep['split'])
        if previous != ep['split']:
            raise ValueError('Scene leakage between dataset splits.')
        with np.load(directory/ep['file'], allow_pickle=False) as arrays:
            record = {k: arrays[k].copy() for k in arrays.files}
        if ep.get('pixels_file'):
            record['pixels'] = np.load(directory/ep['pixels_file'], mmap_mode='r', allow_pickle=False)
        records.append((ep, record))
    for split in ['train', 'val', 'test']:
        if not any(ep['split'] == split for ep, _ in records):
            raise ValueError(f'Missing {split} scenes.')
    return manifest, records


def rgb_tensor(array, device, scale=True):
    pixels = torch.as_tensor(np.moveaxis(array, -1, -3).copy(), device=device)
    return pixels.float()/255. if scale else pixels


def state_tensor(array, mean, std, device):
    return ((torch.as_tensor(array, device=device).float()-mean)/std).clamp(-10, 10)


def train(data, output, steps=None, batch_size=None, seed=42, device='cpu', policy_updates=0,
          profile='official', epochs=None, activation_checkpointing=True, init_checkpoint=None,
          lora_rank=None, lora_alpha=16., lora_dropout=.05):
    started = time.perf_counter()
    if any((output/name).exists() for name in ['best.pt', 'last.pt', 'selected.pt', 'training.json']):
        raise FileExistsError('Training output already contains a run; choose a fresh --output.')
    if (steps is not None and steps < 1) or (batch_size is not None and batch_size < 2):
        raise ValueError('Use positive steps and batch_size >= 2 for SIGReg.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Small models are much faster without large CPU thread pools.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    manifest, records = read_data(data)
    budgets = dict(base_budget=manifest.get('base_budget', max(ep['base_id'] for ep, _ in records)+1),
                   grasp_budget=manifest.get('grasp_budget', max(ep['grasp_id'] for ep, _ in records)+1))
    if manifest.get('positioning_abstention'):
        budgets.update(positioning_abstention=True, candidate_base_poses=manifest['candidate_base_poses'],
                       abstention_reward=manifest['abstention_reward'])
    training = [(ep, x) for ep, x in records if ep['split'] == 'train']
    state_arrays = [x['proprio'] for _, x in training if len(x['proprio'])]
    state_arrays += [x['initial_proprio'][None] for _, x in training]
    state_arrays += [x['arrival_proprio'][None] for _, x in training]
    states = np.concatenate(state_arrays)
    mean = torch.tensor(states.mean(0), dtype=torch.float32, device=device)
    std = torch.tensor(np.maximum(states.std(0), .01), dtype=torch.float32, device=device)
    config = default_config(states.shape[-1]) if profile == 'official' else compact_config(states.shape[-1])
    if manifest['image_size'] != config['image_size']:
        raise ValueError(f"{profile} requires native {config['image_size']}x{config['image_size']} data; recollect, do not upscale old frames.")
    raw_interval = config.get('raw_observation_interval_s', config['observation_interval_s'])
    if abs(manifest['observation_interval_s']-raw_interval) > 1e-6:
        raise ValueError(f'{profile} requires {raw_interval}s raw observation intervals.')
    initialization = None
    if init_checkpoint:
        model, source = load_checkpoint(init_checkpoint, device)
        if model.config.get('profile') != profile:
            raise ValueError('Initialization checkpoint profile does not match requested architecture.')
        config = model.config
        import hashlib
        with init_checkpoint.open('rb') as f:
            initialization = dict(path=str(init_checkpoint), sha256=hashlib.file_digest(f, 'sha256').hexdigest(),
                                  step=source.get('step'), optimizer_resumed=False)
        del source
    else:
        model = DoorHJEPA(config).to(device)
    lora_info, frozen_digest = None, None
    if lora_rank is not None:
        if not init_checkpoint or not model.is_official:
            raise ValueError('LoRA requires an existing official checkpoint.')
        from hjepa_lora import install_lora, frozen_core_digest
        lora_info = install_lora(model, lora_rank, lora_alpha, lora_dropout)
        frozen_digest = frozen_core_digest(model)
    if model.is_official and lora_rank is None:
        actions = np.concatenate([x['action'] for _, x in training if len(x['action'])])
        model.action_mean.copy_(torch.tensor(actions.mean(0), device=device))
        model.action_std.copy_(torch.tensor(actions.std(0, ddof=1), device=device))
    if profile == 'official' and activation_checkpointing:
        model.enable_activation_checkpointing()
    batch_size = batch_size or (config['loader']['batch_size'] if profile == 'official' else 16)
    if lora_rank is not None:
        # Frozen parameters never enter the optimizer. Moderate adapter LR and
        # ordinary task heads; do not resume upstream epoch-scheduler state.
        groups = [dict(params=[p for p in model.model.parameters() if p.requires_grad], lr=1e-4),
                  dict(params=[p for n, p in model.named_parameters() if not n.startswith('model.') and p.requires_grad], lr=3e-4)]
        optim = torch.optim.AdamW(groups, weight_decay=.01)
    elif profile == 'official':
        groups = [{'params': model.model.get_level(i).parameters(), 'lr': config[f'level{i}']['lr']}
                  for i in (1, 2)]
        groups.append({'params': [p for n, p in model.named_parameters() if not n.startswith('model.')],
                       'lr': config['level1']['lr']})
        optim = torch.optim.AdamW(groups, weight_decay=config['optimizer']['weight_decay'])
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    mixed_precision = profile == 'official' and device == 'cuda'
    if mixed_precision and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Official bf16-mixed training requires BF16 support on this GPU.')
    # Position warm start uses best observed downstream grasp return for a base.
    position_targets = {}
    for ep, _ in records:
        key = (ep['scene_id'], ep['base_id'])
        position_targets[key] = max(position_targets.get(key, -float('inf')), ep['reward'])
    windows = {}
    for split in ['train', 'val', 'test']:
        windows[split] = [(ep, x, start) for ep, x in records if ep['split'] == split
                          for start in range(len(x['pixels'])-model.config['raw_clip_frames']+1)]
        if not windows[split]:
            raise ValueError(f"No full {model.config['raw_clip_frames']}-frame raw clips in {split}; collect reachable trajectories.")
    steps_per_epoch = len(windows['train'])//batch_size
    if steps_per_epoch < 1:
        raise ValueError('Fewer training clips than one complete batch.')
    epochs = epochs or (config['trainer']['max_epochs'] if profile == 'official' else 1)
    steps = steps or (epochs*steps_per_epoch if profile == 'official' else 1000)
    scheduler = None
    scheduler_info = None
    if lora_rank is not None:
        warmup_steps = min(50, max(1, steps//10))
        def adapter_scale(step):
            if step < warmup_steps:
                return (step+1)/warmup_steps
            return .1+.9*.5*(1+math.cos(math.pi*(step-warmup_steps)/max(1, steps-warmup_steps)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, adapter_scale)
        scheduler_info = dict(type='adapter warmup cosine', interval='step', warmup_steps=warmup_steps, max_steps=steps)
    elif profile == 'official':
        # Match main_hjepa.py and stable-pretraining 0.1.7's scheduler defaults,
        # including the upstream epoch interval (not an invented step schedule).
        warmup_steps = max(1, int(.01*steps))
        def lr_scale(epoch):
            if epoch < warmup_steps:
                return epoch/warmup_steps
            return .5*(1+math.cos(math.pi*(epoch-warmup_steps)/max(1, steps-warmup_steps)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_scale)
        scheduler_info = dict(type='LinearWarmupCosineAnnealingLR', source='stable-pretraining==0.1.7',
                              interval='epoch', warmup_steps=warmup_steps, max_steps=steps)
    output.mkdir(parents=True, exist_ok=True)
    (output/'resolved_config.json').write_text(json.dumps(model.config, indent=2), encoding='utf-8')
    history = []
    # Fixed validation clips make checkpoint comparisons reproducible. Test
    # clips are never evaluated during adaptation or policy selection.
    validation_clips = random.Random(seed+1).choices(windows['val'], k=batch_size)
    balanced_windows = {}
    for clip in windows['train']:
        balanced_windows.setdefault(clip[0]['scene_id'], []).append(clip)
    balanced_pools = list(balanced_windows.values())
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    if lora_info:
        (output/'adapter_contract.json').write_text(json.dumps(dict(**lora_info,
            frozen_core_sha256=frozen_digest, trainable_names=trainable_names,
            base_checkpoint=initialization, policy_encoder_gradients=True,
            scene_balanced_sampling=True), indent=2), encoding='utf-8')

    def loss_for(split, count, policies=True, clips=None):
        clips = clips or random.choices(windows[split], k=count)
        length = model.config['clip_frames']
        skip = model.config.get('frame_skip', 1)
        span = length*skip
        batch = dict(
            pixels_level1=rgb_tensor(np.stack([x['pixels'][s:s+span:skip] for _, x, s in clips]), device,
                                     scale=profile != 'official'),
            proprio_level1=state_tensor(np.stack([x['proprio'][s:s+span:skip] for _, x, s in clips]), mean, std, device),
            action_level1=torch.tensor(np.stack([x['action'][s:s+span].reshape(length, -1) for _, x, s in clips]), device=device))
        out = model.predictive_loss(batch, split)
        returns = torch.tensor([ep['reward']/15. for ep, _, _ in clips], device=device)[:, None, None]
        values = (nn.functional.mse_loss(model.value_low(out['embed_1'].detach()), returns.expand(-1, out['embed_1'].size(1), -1))
                  + nn.functional.mse_loss(model.value_high(out['embed_2'].detach()), returns.expand(-1, out['embed_2'].size(1), -1)))
        policy_loss = torch.zeros((), device=device)
        if policies:
            pool = [(ep, x) for ep, x in records if ep['split'] == split]
            for stage, head, feature in [('initial', model.position_policy, 'base'),
                                         ('arrival', model.grasp_policy, 'grasp')]:
                if stage == 'initial':
                    # One initial example per base, so a single refusal record
                    # has equal weight to a base with four grasp records.
                    stage_pool = list({(ep['scene_id'], ep['base_id']): (ep, x) for ep, x in pool}.values())
                else:
                    stage_pool = [(ep, x) for ep, x in pool if ep.get('grasp_observed', True)]
                if not stage_pool:
                    continue
                examples = random.choices(stage_pool, k=min(count, 16) if lora_info else count)
                was_training = model.training
                model.eval()
                # LoRA must see end-task supervision: a predictive-only encoder
                # can ignore static obstacles while its frozen heads cannot
                # recover discarded information. Base weights remain frozen.
                with torch.set_grad_enabled(lora_info is not None and split == 'train'):
                    encoder = model.encode_position if stage == 'initial' else model.encode
                    latent = encoder(
                        rgb_tensor(np.stack([x[f'{stage}_rgb'] for _, x in examples]), device),
                        state_tensor(np.stack([x[f'{stage}_proprio'] for _, x in examples]), mean, std, device))
                model.train(was_training)
                if lora_info:
                    from hjepa_lora import freeze_core_statistics
                    freeze_core_statistics(model)
                condition = torch.tensor(np.stack([x[feature] for _, x in examples]), device=device)
                targets = [(position_targets[(ep['scene_id'], ep['base_id'])] if stage == 'initial'
                            else ep['reward'])/15. for ep, _ in examples]
                target = torch.tensor(targets, device=device)[:, None]
                policy_loss = policy_loss+nn.functional.mse_loss(head(torch.cat([latent, condition], -1)), target)
        total = out['loss']+values+policy_loss
        return total, dict(predictive=float(out['loss'].detach()), value=float(values.detach()),
                           policy=float(policy_loss.detach()))

    best = float('inf')
    best_return = -float('inf')
    baseline_validation = None
    if lora_info:
        from evaluate_structural_training import evaluate_records
        model.eval()
        with torch.inference_mode():
            baseline_validation, _ = evaluate_records(model, dict(state_mean=mean, state_std=std, **budgets),
                                                      records, manifest, split='val')
        best_return = baseline_validation['mean_reward']
        save_checkpoint(output/'best.pt', model, mean.cpu(), std.cpu(), step=0, seed=seed, **budgets)
        print(json.dumps(dict(unadapted_validation=baseline_validation)), flush=True)
    print(json.dumps(dict(profile=profile, parameters=sum(p.numel() for p in model.parameters()),
        batch_size=batch_size, steps=steps, steps_per_epoch=steps_per_epoch,
        precision='bf16-mixed' if mixed_precision else 'float32',
        activation_checkpointing=model.config.get('activation_checkpointing', False), lora=lora_info)), flush=True)
    for step in range(steps):
        model.train()
        if lora_info:
            from hjepa_lora import freeze_core_statistics
            freeze_core_statistics(model)
        if step % steps_per_epoch == 0:
            epoch_order = random.sample(windows['train'], len(windows['train']))
        start = (step % steps_per_epoch)*batch_size
        clips = epoch_order[start:start+batch_size]
        if lora_info:
            clips = [random.choice(random.choice(balanced_pools)) for _ in range(batch_size)]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=mixed_precision):
            loss, parts = loss_for('train', batch_size, clips=clips)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss.')
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optim.step()
        if scheduler is not None and (lora_info or (step+1) % steps_per_epoch == 0):
            scheduler.step()
        if step == 0 or (step+1) % max(1, min(50, steps)) == 0 or step+1 == steps:
            model.eval()
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=mixed_precision):
                val, val_parts = loss_for('val', batch_size, clips=validation_clips if lora_info else None)
            policy_validation = None
            if lora_info:
                from evaluate_structural_training import evaluate_records
                with torch.inference_mode():
                    policy_validation, _ = evaluate_records(model, dict(state_mean=mean, state_std=std, **budgets),
                                                            records, manifest, split='val')
            entry = dict(step=step+1, epoch=step//steps_per_epoch+1,
                         train=float(loss.detach()), val=float(val),
                         learning_rates=[g['lr'] for g in optim.param_groups],
                         elapsed_seconds=time.perf_counter()-started, **parts,
                         val_components=val_parts, policy_validation=policy_validation)
            history.append(entry)
            print(json.dumps(entry), flush=True)
            validation_return = policy_validation['mean_reward'] if policy_validation else None
            improved = (validation_return > best_return+1e-8 or
                        (abs(validation_return-best_return) <= 1e-8 and float(val) < best)) if lora_info else float(val) < best
            if improved:
                best = float(val)
                if lora_info:
                    best_return = validation_return
                save_checkpoint(output/'best.pt', model, mean.cpu(), std.cpu(), step=step+1, seed=seed, **budgets)
    if lora_info:
        from hjepa_lora import frozen_core_digest
        if frozen_core_digest(model) != frozen_digest:
            raise AssertionError('LoRA training changed frozen core tensors or running statistics.')
    if policy_updates:
        alternate_policy_updates(model, training, mean, std, device, policy_updates)
    save_checkpoint(output/'last.pt', model, mean.cpu(), std.cpu(), step=steps, seed=seed, **budgets)
    report = dict(history=history, upstream_commit=__import__('hjepa_model').UPSTREAM_COMMIT,
                  scenes=manifest['scenes'], train_windows=len(windows['train']),
                  val_windows=len(windows['val']), test_windows=len(windows['test']),
                  policy_updates=policy_updates, profile=profile, steps=steps, batch_size=batch_size,
                  steps_per_epoch=steps_per_epoch, epochs_completed=steps/steps_per_epoch,
                  precision='bf16-mixed' if mixed_precision else 'float32',
                  parameters=sum(p.numel() for p in model.parameters()),
                  scheduler=scheduler_info, elapsed_seconds=time.perf_counter()-started,
                  initialization=initialization, task_metadata=budgets, lora=lora_info,
                  frozen_core_unchanged=True if lora_info else None,
                  checkpoint_selection='validation task return, predictive loss breaks ties' if lora_info else 'validation loss',
                  unadapted_validation=baseline_validation,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device == 'cuda' else None,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device == 'cuda' else None,
                  note=('Validation task return selects best.pt (unadapted fallback included); base core tensors and statistics remain exact. '
                        if lora_info else 'Validation loss selects best.pt; ')+'last.pt includes optional finite-pool policy updates. '
                       'No test data used for training or checkpoint selection. Small runs are pipeline checks.')
    (output/'training.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


def alternate_policy_updates(model, records, mean, std, device, updates, validation=None, select_best=False):
    """Exact finite-pool policy gradients using measured counterfactual returns.

    Alternate the position and grasp heads with frozen world-model encoders.
    This uses cached simulator rewards; it is not online GRPO/PPO exploration.
    Position receives expected grasp+opening return under the FIXED grasp policy.
    Grasp receives opening returns weighted by the FIXED positioning policy.
    """
    model.eval()
    groups = {}
    for ep, x in records:
        groups.setdefault(ep['scene_id'], {}).setdefault(ep['base_id'], []).append((ep, x))
    pos_optim = torch.optim.Adam(model.position_policy.parameters(), lr=1e-3)
    grasp_optim = torch.optim.Adam(model.grasp_policy.parameters(), lr=1e-3)
    # The encoder is frozen throughout these updates. Cache only training-set
    # latents, and never encode a fictitious arrival for the refusal action.
    def cache(source):
        cached = []
        for base_groups in source.values():
            all_base = list(base_groups.values())
            first = all_base[0][0][1]
            with torch.no_grad():
                initial = getattr(model, 'encode_position', model.encode)(rgb_tensor(first['initial_rgb'][None], device),
                                       state_tensor(first['initial_proprio'][None], mean, std, device))
                arrivals = [None if g[0][0].get('positioning_only') else model.encode(
                    rgb_tensor(g[0][1]['arrival_rgb'][None], device),
                    state_tensor(g[0][1]['arrival_proprio'][None], mean, std, device)) for g in all_base]
            cached.append((all_base, initial, arrivals))
        return cached
    cached = cache(groups)
    validation_groups = {}
    for ep, x in validation or []:
        validation_groups.setdefault(ep['scene_id'], {}).setdefault(ep['base_id'], []).append((ep, x))
    val_cached = cache(validation_groups)
    if select_best and not val_cached:
        raise ValueError('Validation records required for stage-policy checkpoint selection.')
    selections = []
    best_policy = None
    best_val = -float('inf')
    def select_policy(iteration):
        nonlocal best_val, best_policy
        with torch.no_grad():
            rewards = []
            for all_base, initial, arrivals in val_cached:
                features = torch.tensor(np.stack([g[0][1]['base'] for g in all_base]), device=device)
                logits = model.position_policy(torch.cat([initial.expand(len(all_base), -1), features], -1)).flatten()
                bid = int(logits.argmax())
                g, latent = all_base[bid], arrivals[bid]
                if latent is None:
                    rewards.append(g[0][0]['reward'])
                    continue
                features = torch.tensor(np.stack([x['grasp'] for _, x in g]), device=device)
                scores = model.grasp_policy(torch.cat([latent.expand(len(g), -1), features], -1)).flatten()
                eligible = [i for i, (ep, _) in enumerate(g) if ep.get('current_reachable', True)]
                gid = max(eligible, key=lambda i: float(scores[i])) if eligible else int(scores.argmax())
                rewards.append(g[gid][0]['reward'])
        score = float(np.mean(rewards))
        selections.append(dict(update=iteration, validation_mean_reward=score))
        print(json.dumps(dict(stage_policy_validation=selections[-1])), flush=True)
        if score > best_val+1e-8:
            best_val = score
            best_policy = (iteration, {k: v.detach().cpu().clone() for k, v in model.position_policy.state_dict().items()},
                           {k: v.detach().cpu().clone() for k, v in model.grasp_policy.state_dict().items()})
    if select_best:
        select_policy(0)
    for iteration in range(updates):
        all_base, initial, arrivals = random.choice(cached)
        base_conditions = torch.tensor(np.stack([g[0][1]['base'] for g in all_base]), device=device)
        position_logits = model.position_policy(torch.cat([initial.expand(len(all_base), -1), base_conditions], -1)).flatten()
        values, grasp_losses = [], []
        for g, latent in zip(all_base, arrivals):
            if latent is None:
                values.append(torch.tensor(g[0][0]['reward']/15., device=device))
                grasp_losses.append(torch.zeros((), device=device))
                continue
            conditions = torch.tensor(np.stack([x['grasp'] for _, x in g]), device=device)
            logits = model.grasp_policy(torch.cat([latent.expand(len(g), -1), conditions], -1)).flatten()
            returns = torch.tensor([ep['reward']/15. for ep, _ in g], device=device)
            values.append((logits.detach().softmax(0)*returns).sum())
            # Centering preserves exact expected-return gradient, no sampled ratio.
            grasp_losses.append(-(logits.softmax(0)*(returns-returns.mean())).sum())
        if iteration % 2 == 0:
            values = torch.stack(values)
            loss = -(position_logits.softmax(0)*(values-values.mean())).sum()
            pos_optim.zero_grad()
            loss.backward()
            pos_optim.step()
        else:
            weights = position_logits.detach().softmax(0)
            loss = sum(weight*term for weight, term in zip(weights, grasp_losses))
            grasp_optim.zero_grad()
            loss.backward()
            grasp_optim.step()
        if (iteration+1) % 200 == 0:
            print(json.dumps(dict(stage_policy_update=iteration+1, updates=updates)), flush=True)
        if select_best and ((iteration+1) % 100 == 0 or iteration+1 == updates):
            select_policy(iteration+1)
    if select_best:
        iteration, position, grasp = best_policy
        model.position_policy.load_state_dict(position)
        model.grasp_policy.load_state_dict(grasp)
        return dict(selected_update=iteration, updates_executed=updates, validation=selections)
    return dict(selected_update=updates, updates_executed=updates)


def finalize_selected_checkpoint(data, output, policy_updates=100, seed=42, device='cpu'):
    """Select the predictor by validation, THEN train its frozen stage policies.

    Keeps best.pt and last.pt for audit. No test outcomes enter these updates.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    model, source = load_checkpoint(output/'best.pt', device)
    _, records = read_data(data)
    training = [(ep, x) for ep, x in records if ep['split'] == 'train']
    validation = [(ep, x) for ep, x in records if ep['split'] == 'val']
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    mean, std = source['state_mean'].to(device), source['state_std'].to(device)
    selection = alternate_policy_updates(model, training, mean, std, device, policy_updates,
                                         validation=validation, select_best=bool(source.get('positioning_abstention')))
    # Verify that policy fitting did not alter ANY world-model weight or buffer.
    unchanged = all(torch.equal(value, source['model']['model.'+key])
                    for key, value in model.model.state_dict().items())
    if not unchanged:
        raise AssertionError('Stage policy updates changed the selected world model.')
    report_path = output/'training.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    save_checkpoint(output/'selected.pt', model, mean.cpu(), std.cpu(),
                    step=source['step'], training_steps=report['steps'], seed=seed,
                    base_budget=source['base_budget'], grasp_budget=source['grasp_budget'],
                    policy_updates=selection['selected_update'], policy_updates_executed=policy_updates, world_model_unchanged=True,
                    **{k: source[k] for k in ['positioning_abstention', 'candidate_base_poses', 'abstention_reward'] if k in source},
                    selection='minimum validation loss, followed by train-only stage policy updates')
    report.update(selected_checkpoint='selected.pt', selected_world_model_step=source['step'],
                  selected_policy_updates=selection['selected_update'], stage_policy_selection=selection,
                  selected_world_model_unchanged=True)
    if model.config.get('lora'):
        from hjepa_lora import adapter_state
        torch.save(dict(adapter=adapter_state(model), config=model.config,
                        base_checkpoint=report['initialization'], state_mean=mean.cpu(), state_std=std.cpu(),
                        metadata={**{k: source[k] for k in ['base_budget', 'grasp_budget', 'positioning_abstention',
                                                          'candidate_base_poses', 'abstention_reward'] if k in source},
                                                          'step': source['step'], 'policy_updates': selection['selected_update']}),
                   output/'adapter.pt')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(selected='selected.pt', world_model_step=source['step'],
                          policy_updates=selection['selected_update'], policy_updates_executed=policy_updates,
                          world_model_unchanged=True)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('runs/hjepa/official_data'))
    p.add_argument('--output', type=Path, default=Path('runs/hjepa/official_model'))
    p.add_argument('--profile', choices=['official', 'compact'], default='official')
    p.add_argument('--steps', type=int, default=None, help='Override the official epoch budget for a short check')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None, help='Default: official 128, compact 16')
    p.add_argument('--no-activation-checkpointing', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--policy-updates', type=int, default=100)
    p.add_argument('--init-checkpoint', type=Path, help='Continue full-model weights with a fresh optimizer on new data')
    p.add_argument('--lora-rank', type=int, help='Freeze original core and train reversible low-rank adapters only')
    p.add_argument('--lora-alpha', type=float, default=16.)
    p.add_argument('--lora-dropout', type=float, default=.05)
    a = p.parse_args()
    device = resolve_device(a.device)
    train(a.data, a.output, a.steps, a.batch_size, a.seed, device, a.policy_updates if a.profile != 'official' else 0,
          profile=a.profile, epochs=a.epochs, activation_checkpointing=not a.no_activation_checkpointing,
          init_checkpoint=a.init_checkpoint, lora_rank=a.lora_rank, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout)
    if a.profile == 'official':
        finalize_selected_checkpoint(a.data, a.output, a.policy_updates, a.seed, device)
