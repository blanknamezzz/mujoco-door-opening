"""Validation-selected downstream probe with the full frozen official ViT.

Only training returns supervise the positioning head. Feature standardization
is folded into its existing first affine layer, preserving parameter shapes.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import numpy as np
import hjepa_runtime
import torch
from hjepa_model import load_checkpoint, save_checkpoint
from train_hjepa import read_data, rgb_tensor, state_tensor
from evaluate_structural_training import evaluate_records


def fold_standardization(head, mean, std):
    fitted = deepcopy(head)
    linear = fitted.net[0]
    width = len(mean)
    with torch.no_grad():
        linear.weight[:, :width].div_(std)
        linear.bias.sub_(linear.weight[:, :width]@mean)
    return fitted


def refit(data, output, steps=300):
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    manifest, records = read_data(data)
    model, checkpoint = load_checkpoint(output/'selected.pt', 'cpu')
    if checkpoint.get('position_refit'):
        print('Selected checkpoint already includes positioning refit; preserving it.', flush=True)
        return json.loads((output/'position_refit.json').read_text())
    original_world = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith('position_policy.')}
    baseline_summary, _ = evaluate_records(model, checkpoint, records, manifest, split='val')
    model.config['position_encoder'] = 'backbone_cls'
    groups = {}
    for ep, x in records:
        if ep['split'] in ('train', 'val'):
            groups.setdefault(ep['split'], {}).setdefault(ep['scene_id'], {}).setdefault(ep['base_id'], []).append((ep, x))
    cache = {}
    for split, scenes in groups.items():
        inputs, targets, realized = [], [], []
        for bases in scenes.values():
            candidates = list(bases.values())
            first = candidates[0][0][1]
            with torch.no_grad():
                latent = model.encode_position(rgb_tensor(first['initial_rgb'][None], 'cpu'),
                    state_tensor(first['initial_proprio'][None], checkpoint['state_mean'], checkpoint['state_std'], 'cpu'))
            features = torch.tensor(np.stack([g[0][1]['base'] for g in candidates]))
            inputs.append(torch.cat([latent.expand(len(candidates), -1), features], -1))
            targets.append(torch.tensor([max(ep['reward'] for ep, _ in g)/15. for g in candidates]))
            returns = []
            with torch.no_grad():
                for group in candidates:
                    ep, x = group[0]
                    if ep.get('positioning_only') or not ep.get('grasp_observed', True):
                        returns.append(ep['reward'])
                        continue
                    z = model.encode(rgb_tensor(x['arrival_rgb'][None], 'cpu'),
                                     state_tensor(x['arrival_proprio'][None], checkpoint['state_mean'], checkpoint['state_std'], 'cpu'))
                    f = torch.tensor(np.stack([x['grasp'] for _, x in group]))
                    scores = model.grasp_policy(torch.cat([z.expand(len(group), -1), f], -1)).flatten()
                    eligible = [i for i, (e, _) in enumerate(group) if e.get('current_reachable', True)]
                    gid = max(eligible, key=lambda i: float(scores[i])) if eligible else int(scores.argmax())
                    returns.append(group[gid][0]['reward'])
            realized.append(torch.tensor(returns))
        cache[split] = (torch.stack(inputs), torch.stack(targets), torch.stack(realized))
    x, y, _ = cache['train']
    v, _, returns = cache['val']
    width = x.size(-1)-4
    mean, std = x[:, 0, :width].mean(0), x[:, 0, :width].std(0).clamp_min(.1)
    x, v = x.clone(), v.clone()
    x[..., :width] = (x[..., :width]-mean)/std
    v[..., :width] = (v[..., :width]-mean)/std
    head = deepcopy(model.position_policy)
    head.apply(lambda m: m.reset_parameters() if isinstance(m, (torch.nn.Linear, torch.nn.LayerNorm)) else None)
    optimizer = torch.optim.Adam(head.parameters(), lr=.001)
    best_score, best_head, best_step = -float('inf'), None, None
    history = []
    for step in range(steps+1):
        if step % 10 == 0 or step == steps:
            with torch.no_grad():
                choices = head(v).squeeze(-1).argmax(-1)
                score = returns[torch.arange(len(v)), choices].mean().item()
            history.append(dict(step=step, validation_mean_reward=score))
            if score > best_score+1e-8:
                best_score, best_step, best_head = score, step, deepcopy(head.state_dict())
        if step == steps:
            break
        # Smooth reward ranking: temperature is in normalized return units.
        loss = -(torch.softmax(y/.03, -1)*torch.log_softmax(head(x).squeeze(-1), -1)).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    head.load_state_dict(best_head)
    model.position_policy = fold_standardization(head, mean, std)
    with torch.no_grad():
        original_v = cache['val'][0]
        torch.testing.assert_close(model.position_policy(original_v), head(v), atol=2e-4, rtol=2e-4)
    new_summary, _ = evaluate_records(model, checkpoint, records, manifest, split='val')
    accepted = new_summary['mean_reward'] > baseline_summary['mean_reward']+.01
    details = dict(accepted=accepted, steps_executed=steps, selected_step=best_step,
                   objective='train-return soft-label cross-entropy, temperature .03',
                   position_encoder='same official ViT, CLS before JEPA projector',
                   validation_before=baseline_summary, validation_after=new_summary, history=history,
                   feature_statistics_split='train only; affine normalization folded into existing head',
                   minimum_validation_return_improvement=.01,
                   test_evaluated=False)
    if accepted:
        for key, value in original_world.items():
            if not torch.equal(value, model.state_dict()[key]):
                raise AssertionError(f'Position probe changed a frozen tensor: {key}')
        shutil.copyfile(output/'selected.pt', output/'selected_before_position_refit.pt')
        metadata = {k: value for k, value in checkpoint.items() if k not in
                    ['model', 'config', 'state_mean', 'state_std', 'upstream_commit']}
        metadata['position_refit'] = dict(selected_step=best_step, steps_executed=steps)
        save_checkpoint(output/'selected.pt', model, checkpoint['state_mean'], checkpoint['state_std'], **metadata)
    else:
        details['note'] = 'Original selected checkpoint retained; validation did not improve.'
    report_path = output/'training.json'
    report = json.loads(report_path.read_text())
    report['position_policy_refit'] = details
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output/'position_refit.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in details.items() if k != 'history'}, indent=2), flush=True)
    return details


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('runs/hjepa/structural_data'))
    p.add_argument('--model', type=Path, default=Path('runs/hjepa/structural_model'))
    a = p.parse_args()
    refit(a.data, a.model)
