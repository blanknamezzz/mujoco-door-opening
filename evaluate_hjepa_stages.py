"""Frozen-model scene stress tests and matched two-policy interventions.

Physics counterfactuals are evaluation labels only: selectors never receive
them. Every grasp starts from the same observed arrival state for its base.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import hjepa_runtime  # process-local CUDA environment fix, before torch
import torch

from collect_hjepa import small_rgb
from hjepa_model import load_checkpoint, resolve_device
from hjepa_scene import (BASE_CANDIDATES, GRASP_OFFSETS, DoorScene, action_at,
                         base_features, grasp_features, grasps, lookahead_steps,
                         proprio, task_reward)
from perception import observe, estimate_frame
from train_hjepa import rgb_tensor, state_tensor


@dataclass(frozen=True)
class Case:
    scene: DoorScene
    family: str
    mode: str = 'nominal'


def make_suite(seed=20261017, random_scenes=8):
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(random_scenes):
        scene = DoorScene(f'new_{seed}_{i:02d}', float(rng.uniform(.73, .84)),
                          float(rng.uniform(.82, .88)), float(rng.uniform(.5, .9)),
                          float(rng.uniform(.8, 1.6)))
        cases.append(Case(scene, 'new_in_range'))
    for name, kwargs in [
        ('low_friction_005', dict(friction=.005)),
        ('low_friction_05', dict(friction=.05)),
        ('low_friction_20', dict(friction=.2)),
        ('hinge_4', dict(hinge_resistance=4.)),
        ('hinge_8', dict(hinge_resistance=8.)),
        ('handle_near_hinge', dict(handle_x=.65)),
        ('handle_far_hinge', dict(handle_x=.9)),
        ('handle_low', dict(handle_z=.76)),
        ('handle_high', dict(handle_z=.94)),
    ]:
        cases.append(Case(DoorScene(name, **kwargs), 'stress'))
    cases.append(Case(DoorScene('no_grip_control'), 'negative_control', 'no_grip'))
    return cases


def stage_metrics(result, rows, target=90.):
    metrics = task_reward(result, rows, full_angle=target)
    positioned, aligned, grasped = metrics['milestones']
    # A failed visual estimate is a positioning/observation boundary failure,
    # not evidence of a feasible placement or completed alignment.
    positioned = positioned and result['outcome'] != 'B_perception_failed'
    aligned = aligned and positioned
    grasped = grasped and aligned
    success = metrics['success']
    if success:
        failure = None
    elif result['outcome'] == 'E_policy_abstained':
        failure = 'policy_abstention'
    elif result['outcome'] == 'B_perception_failed':
        failure = 'arrival_perception'
    elif not positioned:
        failure = 'position_unreachable'
    elif not aligned:
        failure = 'alignment'
    elif not grasped:
        failure = 'grasp'
    elif result.get('max_angle_deg', 0.) >= target-2 and result.get('stop_reason') == 'time_limit':
        failure = 'hold'
    else:
        failure = 'opening'
    return dict(**metrics, stages=[bool(positioned), bool(aligned), bool(grasped), bool(success)],
                failure_stage=failure)


def summarize(trials):
    n = len(trials)
    if not n:
        return dict(episodes=0)
    stages = np.asarray([t['metrics']['stages'] for t in trials], dtype=int)
    counts = stages.sum(0).tolist()
    failures, reasons = {}, {}
    for t in trials:
        failure = t['metrics']['failure_stage']
        if failure:
            failures[failure] = failures.get(failure, 0)+1
            reason = t['result'].get('stop_reason', t['result']['outcome'])
            reasons[reason] = reasons.get(reason, 0)+1
    return dict(episodes=n, stage_counts=dict(zip(['position', 'align', 'grasp', 'open_hold'], counts)),
                conditional_rates={name: (num/den if den else None) for name, num, den in
                    zip(['position/all', 'align/position', 'grasp/align', 'open_hold/grasp'],
                        counts, [n, *counts[:-1]])},
                success_rate=counts[-1]/n,
                mean_safe_opening_fraction=float(np.mean([t['metrics']['safe_opening_fraction'] for t in trials])),
                mean_final_angle_deg=float(np.mean([t['result'].get('final_angle_deg', 0.) for t in trials])),
                mean_reward=float(np.mean([t['metrics']['reward'] for t in trials])),
                collision_rate=float(np.mean([t['metrics']['collision'] for t in trials])),
                slip_or_loss_rate=float(np.mean([t['result'].get('slip_detected', False) or
                                                t['result'].get('contact_lost', False) for t in trials])),
                failure_stages=failures, failure_reasons=reasons)


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def verify_frozen_world(pre, post):
    for name, value in pre.state_dict().items():
        if not name.startswith(('position_policy.', 'grasp_policy.')):
            if not torch.equal(value, post.state_dict()[name]):
                raise ValueError(f'Unmatched non-policy weight/buffer: {name}')


class Evaluator:
    def __init__(self, before, after, device):
        self.pre, self.pre_ckpt = load_checkpoint(before, device)
        self.post, self.ckpt = load_checkpoint(after, device)
        verify_frozen_world(self.pre, self.post)
        if self.pre.config.get('position_encoder', 'projected') != self.post.config.get('position_encoder', 'projected'):
            raise ValueError('Position feature interfaces differ; use evaluate_structural_training.py for this comparison.')
        for name in ['state_mean', 'state_std']:
            if not torch.equal(self.pre_ckpt[name], self.ckpt[name]):
                raise ValueError(f'Unmatched input normalization: {name}')
        self.device = device

    @torch.inference_mode()
    def encode(self, exp, frame):
        return self.post.encode(
            rgb_tensor(small_rgb(frame, self.post.config['image_size'], True)[None], self.device),
            state_tensor(proprio(exp)[None], self.ckpt['state_mean'], self.ckpt['state_std'], self.device))

    @torch.inference_mode()
    def policy_scores(self, latent, candidates, stage):
        features = np.stack([base_features(x) for x in candidates] if stage == 'position'
                            else [grasp_features(*x) for x in candidates])
        inputs = torch.cat([latent.expand(len(candidates), -1),
                            torch.tensor(features, device=self.device)], -1)
        return {label: getattr(model, f'{stage}_policy')(inputs).flatten().cpu().tolist()
                for label, model in [('pre', self.pre), ('post', self.post)]}

    @torch.inference_mode()
    def world_scores(self, latent, candidates, fraction, opening_direction=1):
        skip = self.post.config['frame_skip']
        raw_interval = self.post.config['raw_observation_interval_s']
        steps = lookahead_steps(fraction)
        actions = torch.tensor(np.stack([
            np.stack([np.concatenate([action_at(i*.5+j*raw_interval, point, yaw, interval=raw_interval,
                                               opening_direction=opening_direction)
                                     for j in range(skip)]) for i in range(steps)])
            for point, yaw in candidates]), device=self.device)
        low = self.post.imagine(latent.expand(len(candidates), -1), actions, 1)
        high = self.post.imagine(latent.expand(len(candidates), -1), actions, 2)
        return ((self.post.value_low(low)+self.post.value_high(high))/2).flatten().cpu().tolist()

    def execute(self, scene, base, point, yaw, mode='nominal', target=90.):
        exp = scene.experiment(base)
        # Constructor gives exactly the observed arrival state, independently
        # for each candidate; never reset to its grasp-specific pre-position.
        result, rows, _ = exp.trial(point, yaw, mode=mode, reset_before_trial=False,
                                    opening_angle_deg=target)
        return dict(result=result, metrics=stage_metrics(result, rows, target))

    def counterfactuals(self, case):
        scene = case.scene
        initial = scene.experiment()
        initial_latent = self.encode(initial, observe(initial))
        scores = self.policy_scores(initial_latent, BASE_CANDIDATES, 'position')
        base_trials = []
        for bid, base in enumerate(BASE_CANDIDATES):
            exp = scene.experiment(base)
            arrival = observe(exp)
            try:
                estimate = estimate_frame(arrival)
            except ValueError as error:
                result = dict(outcome='B_perception_failed', stop_reason='perception_failure',
                              max_angle_deg=0., final_angle_deg=0., perception_error=str(error))
                trials = [dict(result=result, metrics=stage_metrics(result, [])) for _ in GRASP_OFFSETS]
                base_trials.append(dict(base_id=bid, base=list(base), feasible=[False]*4,
                                        direct_scores=None, world_scores=None, geometry_scores=None,
                                        trials=trials))
                continue
            candidates = grasps(estimate)
            latent = self.encode(exp, arrival)
            direct = self.policy_scores(latent, candidates, 'grasp')
            worlds = {str(h): self.world_scores(latent, candidates, h) for h in [.25, .5, 1.]}
            from planning import choose_grasp
            geom = choose_grasp(exp, estimate, candidate_offsets=[(dz, round(math.degrees(y), 6)) for dz, y in GRASP_OFFSETS])
            geometry_scores = [next(c['score'] for c in geom['candidates']
                                    if np.allclose(c['point_world_m'], p) and
                                    abs(c['yaw_offset_deg']-math.degrees(y)) < 1e-6)
                               for p, y in candidates]
            feasible, trials = [], []
            for gid, (point, yaw) in enumerate(candidates):
                yaw0 = math.pi/2+yaw
                pre = point-.16*np.array([math.cos(yaw0), math.sin(yaw0), 0.])
                feasible.append(exp.ik(point, yaw0) is not None and exp.ik(pre, yaw0) is not None)
                trial = self.execute(scene, base, point, yaw, case.mode)
                trials.append(dict(grasp_id=gid, point_world_m=point.tolist(), yaw_offset_rad=yaw, **trial))
            base_trials.append(dict(base_id=bid, base=list(base), feasible=feasible,
                                    direct_scores=direct, world_scores=worlds,
                                    geometry_scores=geometry_scores, trials=trials))
            print(f'{scene.scene_id}: base {bid+1}/5; successes {sum(t["metrics"]["success"] for t in trials)}/4', flush=True)
        return dict(case=asdict(case), position_scores=scores, bases=base_trials)


def selected_grasp(base, scores):
    eligible = np.flatnonzero(base['feasible'])
    return int(max(eligible, key=lambda i: scores[i])) if len(eligible) else int(np.argmax(scores))


def select(counterfactual, position='post', grasp='post', method='direct', horizon=1.):
    # This function deliberately never accesses trial outcomes when ranking.
    bid = 0 if position == 'fixed' else int(np.argmax(counterfactual['position_scores'][position]))
    base = counterfactual['bases'][bid]
    if base['direct_scores'] is None:
        gid = 0
    else:
        scores = (base['direct_scores'][grasp] if method == 'direct' else
                  base['world_scores'][str(horizon)] if method == 'world-model' else base['geometry_scores'])
        gid = selected_grasp(base, scores)
    return dict(scene_id=counterfactual['case']['scene']['scene_id'],
                family=counterfactual['case']['family'], base_id=bid, grasp_id=gid,
                result=base['trials'][gid]['result'], metrics=base['trials'][gid]['metrics'])


def aggregate(tables):
    conditions = {}
    for p in ['pre', 'post']:
        for g in ['pre', 'post']:
            conditions[f'P_{p}/G_{g}'] = [select(t, p, g) for t in tables]
    conditions['fixed/G_post'] = [select(t, 'fixed') for t in tables]
    for p in ['fixed', 'pre', 'post']:
        for h in [.25, .5, 1.]:
            conditions[f'P_{p}/world/h{h}'] = [select(t, p, method='world-model', horizon=h) for t in tables]
        conditions[f'P_{p}/geometry'] = [select(t, p, method='geometry') for t in tables]
    # Conditional grasp comparison: same scene, base, reached image and state.
    matched_grasps = {'pre': [], 'post': []}
    for table in tables:
        for base in table['bases']:
            for g in matched_grasps:
                gid = (selected_grasp(base, base['direct_scores'][g])
                       if base['direct_scores'] is not None else 0)
                matched_grasps[g].append(base['trials'][gid])
    result = dict(conditions={name: summarize(trials) for name, trials in conditions.items()},
                  by_family={family: {name: summarize([t for t in trials if t['family'] == family])
                                      for name, trials in conditions.items()}
                             for family in sorted({t['case']['family'] for t in tables})},
                  matched_arrivals={g: summarize(trials) for g, trials in matched_grasps.items()},
                  all_candidates=summarize([trial for t in tables for b in t['bases'] for trial in b['trials']]),
                  decisions=conditions)
    for name, baseline in [('position_update_with_G_pre', 'P_pre/G_pre'),
                           ('grasp_update_with_P_pre', 'P_pre/G_pre'),
                           ('position_update_with_G_post', 'P_pre/G_post'),
                           ('grasp_update_with_P_post', 'P_post/G_pre')]:
        changed = {'position_update_with_G_pre': 'P_post/G_pre',
                   'grasp_update_with_P_pre': 'P_pre/G_post',
                   'position_update_with_G_post': 'P_post/G_post',
                   'grasp_update_with_P_post': 'P_post/G_post'}[name]
        pairs = list(zip(conditions[baseline], conditions[changed]))
        result.setdefault('paired_effects', {})[name] = dict(
            success_delta=result['conditions'][changed]['success_rate']-result['conditions'][baseline]['success_rate'],
            safe_progress_delta=result['conditions'][changed]['mean_safe_opening_fraction']-result['conditions'][baseline]['mean_safe_opening_fraction'],
            improved=sum(b['metrics']['success'] and not a['metrics']['success'] for a, b in pairs),
            regressed=sum(a['metrics']['success'] and not b['metrics']['success'] for a, b in pairs),
            choice_changed=sum((a['base_id'], a['grasp_id']) != (b['base_id'], b['grasp_id']) for a, b in pairs))
    return result


def angle_sweeps(evaluator, cases, output):
    # These interventions test controllers outside the trained goal/candidate
    # distribution. They do not imply a goal-conditioned learned policy.
    path = output/'angles.json'
    trials = json.loads(path.read_text()) if path.exists() else []
    done = {t['key'] for t in trials}
    for case in cases:
        jobs = []
        for offset in [-45, -30, -15, 0, 15, 30, 45]:
            jobs.append(('base_yaw', offset, (.45, -1.05, math.pi/2+math.radians(offset)), 0., 90.))
        for yaw in [-30, -15, 0, 15, 30]:
            jobs.append(('grasp_yaw', yaw, BASE_CANDIDATES[3], math.radians(yaw), 90.))
        for bid in [0, 1, 3]:
            for target in [30., 60., 90.]:
                jobs.append((f'target_base{bid}', target, BASE_CANDIDATES[bid], 0., target))
        for kind, value, base, yaw, target in jobs:
            key = f'{case.scene.scene_id}/{kind}/{value}'
            if key in done:
                continue
            exp = case.scene.experiment(base)
            try:
                estimate = estimate_frame(observe(exp))
                point = np.asarray(estimate['center_world_m'])
                trial = evaluator.execute(case.scene, base, point, yaw, case.mode, target)
            except ValueError as error:
                # Only perception errors are expected; execution/config errors
                # should not be swallowed as a negative physical result.
                if 'Handle detection failed' not in str(error):
                    raise
                result = dict(outcome='B_perception_failed', stop_reason='perception_failure',
                              final_angle_deg=0., perception_error=str(error))
                trial = dict(result=result, metrics=stage_metrics(result, [], target))
            trials.append(dict(key=key, scene_id=case.scene.scene_id, kind=kind, value=value,
                               base=list(base), yaw_offset_deg=math.degrees(yaw), target_angle_deg=target, **trial))
            write_json(path, trials)
        print(f'Angle sweep: {case.scene.scene_id} complete ({len(trials)} total)', flush=True)
    return trials


def evaluate(before, after, output, seed=20261017, random_scenes=8, device='cuda'):
    start = time.perf_counter()
    cases = make_suite(seed, random_scenes)
    training = json.loads((after.parent/'training.json').read_text(encoding='utf-8'))
    # Check physical parameters too: distinct IDs alone do not prove novelty.
    fields = ['handle_x', 'handle_z', 'friction', 'hinge_resistance']
    signatures = {tuple(s[f] for f in fields) for s in training['scenes']}
    for case in cases:
        if case.family == 'new_in_range' and tuple(getattr(case.scene, f) for f in fields) in signatures:
            raise ValueError('Fresh test scene physically overlaps collection.')
    spec = dict(schema=1, seed=seed, cases=[asdict(c) for c in cases],
                before_sha256=sha256(before), after_sha256=sha256(after),
                source_sha256={name: sha256(Path(name)) for name in
                    ['evaluate_hjepa_stages.py', 'simulate.py', 'hjepa_scene.py', 'perception.py',
                     'planning.py', 'hjepa_model.py', 'collect_hjepa.py']},
                candidate_bases=BASE_CANDIDATES, candidate_grasps=GRASP_OFFSETS,
                note='Frozen checkpoints; no tuning. Counterfactual physics labels are evaluation-only. '
                     'Stress controls reuse nominal geometry; only new_in_range scenes are fresh random samples. '
                     'Partial target angles are controller tests, not goal-conditioned H-JEPA tests.')
    output.mkdir(parents=True, exist_ok=True)
    spec = json.loads(json.dumps(spec))  # normalize tuples for resume comparison
    if (output/'suite.json').exists() and json.loads((output/'suite.json').read_text()) != spec:
        raise ValueError('Output belongs to a different suite/source/checkpoint; use a fresh --output.')
    write_json(output/'suite.json', spec)
    evaluator = Evaluator(before, after, device)
    tables = []
    for case in cases:
        path = output/f'{case.scene.scene_id}.json'
        if path.exists():
            table = json.loads(path.read_text())
        else:
            table = evaluator.counterfactuals(case)
            write_json(path, table)
        tables.append(table)
        metrics = aggregate(tables)
        write_json(output/'metrics.json', metrics)
    # Two fresh scenes plus physical failure and handle-height stress controls.
    angle_cases = [c for c in cases if c.family == 'new_in_range'][:2]
    angle_cases += [c for c in cases if c.scene.scene_id in ['low_friction_005', 'handle_high']]
    angles = angle_sweeps(evaluator, angle_cases, output)
    metrics['angles'] = {kind: {str(v): summarize([t for t in angles if t['kind'] == kind and t['value'] == v])
                               for v in sorted({t['value'] for t in angles if t['kind'] == kind})}
                         for kind in sorted({t['kind'] for t in angles})}
    metrics['metadata'] = dict(world_and_value_weights_identical=True, updated_policy_steps=100,
                               scene_count=len(cases), random_scene_count=random_scenes,
                               counterfactual_trials=20*len(cases), angle_trials=len(angles),
                               total_physical_trials=20*len(cases)+len(angles),
                               invocation_seconds=time.perf_counter()-start, device=str(device))
    write_json(output/'metrics.json', metrics)
    print(json.dumps(dict(metadata=metrics['metadata'], paired_effects=metrics['paired_effects'],
                          conditions=metrics['conditions']), indent=2), flush=True)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, default=Path('runs/hjepa/official_model/best.pt'))
    parser.add_argument('--after', type=Path, default=Path('runs/hjepa/official_model/selected.pt'))
    parser.add_argument('--output', type=Path, default=Path('runs/hjepa/stage_evaluation'))
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--random-scenes', type=int, default=8)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.before, args.after, args.output, args.seed, args.random_scenes, resolve_device(args.device))
