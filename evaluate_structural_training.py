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
from decomposed_reward import terminal_utility
from door_side_reward import decision_side_metrics
from position_observation import encode_position_views,position_features,position_decision_logits


@torch.inference_mode()
def choose(model, checkpoint, groups, method='direct', horizon=1., force_position_execution=False):
    device = next(model.parameters()).device
    def encode(x, stage):
        encoder = getattr(model, 'encode_position', model.encode) if stage == 'initial' else model.encode
        return encoder(rgb_tensor(x[f'{stage}_rgb'][None], device),
            state_tensor(x[f'{stage}_proprio'][None], checkpoint['state_mean'], checkpoint['state_std'], device))
    all_base = [g for g in groups.values() if checkpoint.get('positioning_abstention') or
                not g[0][0].get('positioning_only')]
    first = all_base[0][0][1]
    state=state_tensor(first['initial_proprio'][None],checkpoint['state_mean'],checkpoint['state_std'],device)
    rear=first.get('initial_rear_rgb')
    latent = encode_position_views(model,rgb_tensor(first['initial_rgb'][None],device),state,
        rgb_tensor(rear[None],device) if rear is not None else None)
    features = torch.tensor(np.stack([g[0][1]['base'] for g in all_base]), device=device)
    initial_features=position_features(latent,features[None])[0]
    logits = position_decision_logits(model,initial_features).flatten()
    position_components = None
    if hasattr(model.position_policy, 'component_logits'):
        predicted = model.position_policy.component_logits(initial_features).sigmoid().cpu().tolist()
        position_components = [None if g[0][0].get('positioning_only') else values for g, values in zip(all_base, predicted)]
    selection_logits=logits
    if force_position_execution:
        # Used only to fit/evaluate a global refusal decision. Preserve the
        # identical full candidate forward and return its original scores.
        physical=torch.tensor([not g[0][0].get('positioning_only',False) for g in all_base],device=device)
        if not physical.any():raise ValueError('Physical candidate required.')
        selection_logits=logits.masked_fill(~physical,-torch.inf)
    selected = int(selection_logits.argmax())
    group = all_base[selected]
    if group[0][0].get('positioning_only'):
        return group[0][0], dict(position_scores=logits.cpu().tolist(), grasp_scores=None,
                                position_components=position_components, grasp_components=None)
    latent = encode(group[0][1], 'arrival')
    features = torch.tensor(np.stack([x['grasp'] for _, x in group]), device=device)
    scores = model.grasp_policy(torch.cat([latent.expand(len(group), -1), features], -1)).flatten()
    if not group[0][0].get('grasp_observed', True):
        return group[0][0], dict(position_scores=logits.cpu().tolist(), grasp_scores=None,
                                position_components=position_components, grasp_components=None)
    grasp_components = (model.grasp_policy.component_logits(torch.cat([latent.expand(len(group), -1), features], -1)).sigmoid().cpu().tolist()
                        if hasattr(model.grasp_policy, 'component_logits') else None)
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
    return group[gid][0], dict(position_scores=logits.cpu().tolist(), grasp_scores=scores.cpu().tolist(),
                             position_components=position_components, grasp_components=grasp_components)


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
        successful_bases = [base for base, candidates in table.items()
                            if any(ep['success'] for ep, _ in candidates if not ep.get('positioning_only'))]
        rejected = selected.get('positioning_only', False)
        chosen_base_feasible = selected['base_id'] in successful_bases
        decisions.append(dict(scene_id=scene_id, family=scene_specs[scene_id]['family'],
            base_id=selected['base_id'], grasp_id=selected['grasp_id'],
            side=selected.get('operating_side'), abstained=rejected, actually_feasible=feasible,
            success=selected['success'], correct_rejection=bool(rejected and not feasible),
            false_rejection=bool(rejected and feasible),
            chosen_base_has_successful_grasp=chosen_base_feasible,
            positioning_error=bool(feasible and not rejected and not chosen_base_feasible),
            grasp_selection_error=bool(chosen_base_feasible and not selected['success']),
            task_decision_correct=bool(selected['success'] or (rejected and not feasible)),
            terminal_utility_v2=terminal_utility(selected['success'], selected.get('collision', False), rejected),
            result=selected['result'], reward=selected['reward'],
            **decision_side_metrics(selected,[e for g in table.values() for e,_ in g]),**scores))
    return summarize(decisions), decisions


def summarize(decisions):
    return dict(scenes=len(decisions), feasible_scenes=sum(d['actually_feasible'] for d in decisions),
                opening_successes=sum(d['success'] for d in decisions),
                correct_rejections=sum(d['correct_rejection'] for d in decisions),
                false_rejections=sum(d['false_rejection'] for d in decisions),
                correct_task_decisions=sum(d['task_decision_correct'] for d in decisions),
                collisions=sum(d['result'].get('unexpected_contact_samples', 0) > 0 for d in decisions),
                positioning_errors=sum(d.get('positioning_error', False) for d in decisions),
                grasp_selection_errors=sum(d.get('grasp_selection_error', False) for d in decisions),
                side_evaluated_scenes=sum(d.get('side_metrics_available',False) for d in decisions),
                correct_side_decisions=sum(d.get('side_decision_correct',False) for d in decisions),
                wrong_side_errors=sum(d.get('wrong_side_error',False) for d in decisions),
                within_side_base_errors=sum(d.get('within_side_base_error',False) for d in decisions),
                mean_side_reward=(float(np.mean([d['position_side_reward'] for d in decisions if d.get('side_metrics_available')]))
                    if any(d.get('side_metrics_available') for d in decisions) else None),
                mean_terminal_utility_v2=float(np.mean([d.get('terminal_utility_v2',
                    terminal_utility(d['success'], d['result'].get('unexpected_contact_samples', 0)>0, d['abstained'])) for d in decisions])),
                mean_reward=float(np.mean([d['reward'] for d in decisions])))


def assert_replay_matches(plan, cached):
    """An absent grasp after perception failure is not a selected grasp #0."""
    actual = plan['result']
    rejected = actual['outcome'] == 'E_policy_abstained'
    if rejected != cached['abstained']:
        raise AssertionError('Live replay refusal differs from held-out selection.')
    if not rejected:
        if plan['base_id'] != cached['base_id']:
            raise AssertionError('Live replay base differs from held-out selection.')
        if actual['outcome'] == 'B_perception_failed':
            if cached['result']['outcome'] != 'B_perception_failed' or plan['grasp_id'] is not None:
                raise AssertionError('Live replay perception failure differs from held-out result.')
        elif plan['grasp_id'] != cached['grasp_id']:
            raise AssertionError('Live replay grasp differs from held-out selection.')
    if (actual['outcome'], plan['metrics']['success']) != (cached['result']['outcome'], cached['success']):
        raise AssertionError('Live physical outcome differs from collected counterfactual.')


def evaluate(data, model_dir, output, baseline, device='cuda', references=None, canonical=True):
    from experiment_reproducibility import seed_everything
    reproducibility=seed_everything(42)
    manifest, records = read_data(data, required_splits=('test',), include_splits=('test',),policy_only=True)
    training = json.loads((model_dir/'training.json').read_text())
    seen = {s['scene_id'] for s in training['scenes'] if s['split'] != 'test'}
    tests = {s['scene_id'] for s in manifest['scenes'] if s['split'] == 'test'}
    if seen & tests or not tests:
        raise ValueError('Unexpected or overlapping held-out scene split.')
    from prepare_hjepa_splits import assert_scene_disjoint
    assert_scene_disjoint([s for s in training['scenes'] if s['split'] != 'test'],
                          [s for s in manifest['scenes'] if s['split'] == 'test'])
    output.mkdir(parents=True, exist_ok=True)
    summaries, all_decisions = {}, {}
    old, old_ckpt = load_checkpoint(baseline, device)
    summaries['old_checkpoint'], all_decisions['old_checkpoint'] = evaluate_records(old, old_ckpt, records, manifest)
    del old, old_ckpt
    for index, path in enumerate(references or []):
        reference, source = load_checkpoint(path, device)
        key = f'reference_{index}'
        summaries[key], all_decisions[key] = evaluate_records(reference, source, records, manifest)
        del reference, source
    before, pre = load_checkpoint(model_dir/'best.pt', device)
    after, post = load_checkpoint(model_dir/'selected.pt', device)
    from evaluate_hjepa_stages import verify_frozen_world
    verify_frozen_world(before, after)
    originals = {name: deepcopy(getattr(after, name).state_dict()) for name in ['position_policy', 'grasp_policy']}
    position_encoder_after = after.config.get('position_encoder', 'projected')
    refusal_offset_after = after.config.get('position_refusal_offset',0.)
    for p in ['pre', 'post']:
        for g in ['pre', 'post']:
            after.position_policy.load_state_dict(before.position_policy.state_dict() if p == 'pre' else originals['position_policy'])
            after.config['position_encoder'] = (before.config.get('position_encoder', 'projected') if p == 'pre'
                                                else position_encoder_after)
            after.config['position_refusal_offset']=(before.config.get('position_refusal_offset',0.) if p=='pre'
                                                    else refusal_offset_after)
            after.grasp_policy.load_state_dict(before.grasp_policy.state_dict() if g == 'pre' else originals['grasp_policy'])
            name = f'P_{p}/G_{g}'
            summaries[name], all_decisions[name] = evaluate_records(after, post, records, manifest)
    for name, state in originals.items():
        getattr(after, name).load_state_dict(state)
    after.config['position_encoder'] = position_encoder_after
    after.config['position_refusal_offset'] = refusal_offset_after
    no_refusal = dict(post, positioning_abstention=False)
    summaries['trained_four_actions'], all_decisions['trained_four_actions'] = evaluate_records(
        after, no_refusal, records, manifest)
    for h in [.25, .5, 1.]:
        name = f'world_h{h}'
        summaries[name], all_decisions[name] = evaluate_records(after, post, records, manifest, method='world-model', horizon=h)
    families = {name: {family: summarize([d for d in decisions if d['family'] == family])
                      for family in sorted({d['family'] for d in decisions})}
                for name, decisions in all_decisions.items()}
    report = dict(summaries=summaries, decisions=all_decisions, by_family=families,
                  dataset=str(data), baseline=str(baseline), references=[str(p) for p in references or []],
                  selection=training.get('checkpoint_selection', 'validation loss')+'; stage policies chosen by validation return. '
                            'This test report was evaluated after both were frozen.',
                  world_frozen_during_stage_updates=True, split='test', test_scene_count=len(tests),reproducibility=reproducibility)
    write_json(output/'heldout.json', report)
    print(json.dumps(summaries, indent=2), flush=True)
    # Replay the final policy on all held-out test scenes using actual current
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
        assert_replay_matches(plan, cached)
        save_trial('replay', plan, rows, states, output/spec['scene_id'])
        replays.append(dict(scene_id=spec['scene_id'], selection_matches=True, outcome_matches=True, result=plan['result']))
    write_json(output/'replay.json', replays)
    if not canonical:
        return report
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
    p.add_argument('--reference', type=Path, action='append', help='Additional frozen reference checkpoint')
    p.add_argument('--skip-canonical', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    evaluate(a.data, a.model, a.output, a.baseline, resolve_device(a.device), a.reference, not a.skip_canonical)
