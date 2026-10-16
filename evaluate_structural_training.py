"""Evaluate frozen structural checkpoints on scene-held-out physical tables.

Selectors see recorded current observations and candidate commands only.
Measured returns are looked up after selection, including refusal correctness.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import hjepa_runtime
import torch
from collect_hjepa import small_rgb
from hjepa_model import load_checkpoint, resolve_device
from hjepa_scene import action_at, lookahead_steps
from train_hjepa import read_data, rgb_tensor, state_tensor
from evaluate_hjepa_stages import write_json


@torch.inference_mode()
def choose(model, checkpoint, groups, method='direct', horizon=1.):
    device = next(model.parameters()).device
    def encode(x, stage):
        encoder = getattr(model, 'encode_position', model.encode) if stage == 'initial' else model.encode
        return encoder(rgb_tensor(x[f'{stage}_rgb'][None], device),
            state_tensor(x[f'{stage}_proprio'][None], checkpoint['state_mean'], checkpoint['state_std'], device))
    all_base = [g for g in groups.values() if checkpoint.get('positioning_abstention') or
                not g[0][0].get('positioning_only')]
    first = all_base[0][0][1]
    latent = encode(first, 'initial')
    features = torch.tensor(np.stack([g[0][1]['base'] for g in all_base]), device=device)
    logits = model.position_policy(torch.cat([latent.expand(len(all_base), -1), features], -1)).flatten()
    selected = int(logits.argmax())
    group = all_base[selected]
    if group[0][0].get('positioning_only'):
        return group[0][0], dict(position_scores=logits.cpu().tolist(), grasp_scores=None)
    latent = encode(group[0][1], 'arrival')
    features = torch.tensor(np.stack([x['grasp'] for _, x in group]), device=device)
    scores = model.grasp_policy(torch.cat([latent.expand(len(group), -1), features], -1)).flatten()
    if not group[0][0].get('grasp_observed', True):
        return group[0][0], dict(position_scores=logits.cpu().tolist(), grasp_scores=None)
    if method == 'world-model':
        skip, raw = model.config['frame_skip'], model.config['raw_observation_interval_s']
        actions = torch.tensor(np.stack([np.stack([
            np.concatenate([action_at(i*.5+j*raw, np.asarray(ep['point_world_m']), ep['yaw_offset_rad'],
                interval=raw, opening_direction=ep['opening_direction']) for j in range(skip)])
            for i in range(lookahead_steps(horizon))]) for ep, _ in group]), device=device)
        low = model.imagine(latent.expand(len(group), -1), actions, 1)
        high = model.imagine(latent.expand(len(group), -1), actions, 2)
        scores = ((model.value_low(low)+model.value_high(high))/2).flatten()
    eligible = [i for i, (ep, _) in enumerate(group) if ep.get('current_reachable', True)]
    gid = max(eligible, key=lambda i: float(scores[i])) if eligible else int(scores.argmax())
    return group[gid][0], dict(position_scores=logits.cpu().tolist(), grasp_scores=scores.cpu().tolist())


def evaluate_records(model, checkpoint, records, manifest, split='test', method='direct', horizon=1.):
    groups = {}
    for ep, x in records:
        if ep['split'] == split:
            groups.setdefault(ep['scene_id'], {}).setdefault(ep['base_id'], []).append((ep, x))
    scene_specs = {s['scene_id']: s for s in manifest['scenes']}
    decisions = []
    for scene_id, table in groups.items():
        selected, scores = choose(model, checkpoint, table, method, horizon)
        # Only AFTER the action has been selected, query evaluation outcomes.
        feasible = any(ep['success'] for g in table.values() for ep, _ in g if not ep.get('positioning_only'))
        rejected = selected.get('positioning_only', False)
        decisions.append(dict(scene_id=scene_id, family=scene_specs[scene_id]['family'],
            base_id=selected['base_id'], grasp_id=selected['grasp_id'],
            side=selected.get('operating_side'), abstained=rejected, actually_feasible=feasible,
            success=selected['success'], correct_rejection=bool(rejected and not feasible),
            false_rejection=bool(rejected and feasible),
            task_decision_correct=bool(selected['success'] or (rejected and not feasible)),
            result=selected['result'], reward=selected['reward'], **scores))
    return summarize(decisions), decisions


def summarize(decisions):
    return dict(scenes=len(decisions), feasible_scenes=sum(d['actually_feasible'] for d in decisions),
                opening_successes=sum(d['success'] for d in decisions),
                correct_rejections=sum(d['correct_rejection'] for d in decisions),
                false_rejections=sum(d['false_rejection'] for d in decisions),
                correct_task_decisions=sum(d['task_decision_correct'] for d in decisions),
                collisions=sum(d['result'].get('unexpected_contact_samples', 0) > 0 for d in decisions),
                mean_reward=float(np.mean([d['reward'] for d in decisions])))


def evaluate(data, model_dir, output, baseline, device='cuda'):
    manifest, records = read_data(data)
    training = json.loads((model_dir/'training.json').read_text())
    seen = {s['scene_id'] for s in training['scenes'] if s['split'] != 'test'}
    tests = {s['scene_id'] for s in manifest['scenes'] if s['split'] == 'test'}
    if seen & tests or not tests:
        raise ValueError('Unexpected or overlapping held-out scene split.')
    output.mkdir(parents=True, exist_ok=True)
    summaries, all_decisions = {}, {}
    old, old_ckpt = load_checkpoint(baseline, device)
    summaries['old_checkpoint'], all_decisions['old_checkpoint'] = evaluate_records(old, old_ckpt, records, manifest)
    del old, old_ckpt
    before, pre = load_checkpoint(model_dir/'best.pt', device)
    after, post = load_checkpoint(model_dir/'selected.pt', device)
    from evaluate_hjepa_stages import verify_frozen_world
    verify_frozen_world(before, after)
    originals = {name: deepcopy(getattr(after, name).state_dict()) for name in ['position_policy', 'grasp_policy']}
    position_encoder_after = after.config.get('position_encoder', 'projected')
    for p in ['pre', 'post']:
        for g in ['pre', 'post']:
            after.position_policy.load_state_dict(before.position_policy.state_dict() if p == 'pre' else originals['position_policy'])
            after.config['position_encoder'] = (before.config.get('position_encoder', 'projected') if p == 'pre'
                                                else position_encoder_after)
            after.grasp_policy.load_state_dict(before.grasp_policy.state_dict() if g == 'pre' else originals['grasp_policy'])
            name = f'P_{p}/G_{g}'
            summaries[name], all_decisions[name] = evaluate_records(after, post, records, manifest)
    for name, state in originals.items():
        getattr(after, name).load_state_dict(state)
    after.config['position_encoder'] = position_encoder_after
    no_refusal = dict(post, positioning_abstention=False)
    summaries['trained_four_actions'], all_decisions['trained_four_actions'] = evaluate_records(
        after, no_refusal, records, manifest)
    for h in [.25, .5, 1.]:
        name = f'world_h{h}'
        summaries[name], all_decisions[name] = evaluate_records(after, post, records, manifest, method='world-model', horizon=h)
    report = dict(summaries=summaries, decisions=all_decisions,
                  selection='World checkpoint chosen by validation loss; stage policies chosen by validation return. '
                            'This test report was evaluated after both were frozen.',
                  world_frozen_during_stage_updates=True, split='test', test_scene_count=len(tests))
    write_json(output/'heldout.json', report)
    print(json.dumps(summaries, indent=2), flush=True)
    # Replay the final policy on the six fresh test scenes using actual current
    # rendering/contact dynamics, including no-execution refusal decisions.
    from structural_scene import structural_scene_from_spec
    from run_hjepa import select_and_execute
    from simulate import save_trial
    replays = []
    for spec in manifest['scenes']:
        if spec['split'] != 'test':
            continue
        plan, rows, states = select_and_execute(after, post, structural_scene_from_spec(spec), method='direct')
        cached = next(d for d in all_decisions['P_post/G_post'] if d['scene_id'] == spec['scene_id'])
        actual_reject = plan['result']['outcome'] == 'E_policy_abstained'
        if actual_reject != cached['abstained'] or (not actual_reject and
            (plan['base_id'], plan['grasp_id']) != (cached['base_id'], cached['grasp_id'])):
            raise AssertionError('Live replay selection differs from matched held-out observations.')
        if plan['metrics']['success'] != cached['success']:
            raise AssertionError('Live physical outcome differs from collected counterfactual.')
        save_trial('replay', plan, rows, states, output/spec['scene_id'])
        replays.append(dict(scene_id=spec['scene_id'], selection_matches=True, outcome_matches=True, result=plan['result']))
    write_json(output/'replay.json', replays)
    # Previously inspected canonical scenes are exploratory checks, separate
    # from the new held-out variants above. Hidden hinge pairs are flagged.
    from structural_scene import structural_scenes
    canonical = []
    for name, scene in structural_scenes().items():
        plan, rows, states = select_and_execute(after, post, scene, method='direct')
        old_table = Path('runs/hjepa/structural_evaluation')/f'{name}.json'
        feasible = None
        if old_table.exists():
            table = json.loads(old_table.read_text())
            feasible = any(t['metrics']['success'] for b in table['bases'] for t in b['trials'])
        abstained = plan['result']['outcome'] == 'E_policy_abstained'
        save_trial('replay', plan, rows, states, output/'canonical'/name)
        canonical.append(dict(scene_id=name, hidden_direction_control=name.startswith('one-way-'),
            abstained=abstained, actually_feasible=feasible, success=plan['metrics']['success'],
            correct_task_decision=(plan['metrics']['success'] or (abstained and not feasible)) if feasible is not None else None,
            side=plan.get('operating_side'), result=plan['result']))
    write_json(output/'canonical.json', canonical)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('runs/hjepa/structural_data'))
    p.add_argument('--model', type=Path, default=Path('runs/hjepa/structural_model'))
    p.add_argument('--output', type=Path, default=Path('runs/hjepa/structural_trained_evaluation'))
    p.add_argument('--baseline', type=Path, default=Path('runs/hjepa/official_model/selected.pt'))
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a = p.parse_args()
    torch.set_num_threads(4)
    evaluate(a.data, a.model, a.output, a.baseline, resolve_device(a.device))
