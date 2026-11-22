"""Matched structural-scene tests with a frozen official predictor.

The old front-only checkpoints are evaluated on an explicitly new two-sided
candidate pool. Counterfactual physics never enters learned/geometric ranking.
"""
import argparse
from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import mujoco
import hjepa_runtime
import torch
from evaluate_hjepa_stages import (Evaluator, selected_grasp, sha256, stage_metrics,
                                   summarize, write_json)
from hjepa_model import resolve_device
from hjepa_scene import grasps, GRASP_OFFSETS
from perception import observe, estimate_frame
from planning import choose_grasp
from structural_scene import STRUCTURAL_BASES, structural_scenes


def assess_scene(evaluator, scene):
    initial = scene.experiment()
    latent = evaluator.encode(initial, observe(initial))
    position_scores = evaluator.policy_scores(latent, STRUCTURAL_BASES, 'position')
    bases = []
    for bid, base in enumerate(STRUCTURAL_BASES):
        exp = scene.experiment(base)
        arrival = observe(exp)
        try:
            handle = estimate_frame(arrival)
        except ValueError as error:
            result = dict(outcome='B_perception_failed', stop_reason='perception_failure',
                          final_angle_deg=0., perception_error=str(error))
            trials = [dict(grasp_id=g, executor_called=False, integrated=False,
                           result=result, metrics=stage_metrics(result, [])) for g in range(4)]
            bases.append(dict(base_id=bid, side=exp.operating_side, base=list(base),
                              feasible=[False]*4, direct_scores=None, world_scores=None,
                              geometry=None, trials=trials))
            print(f'{scene.scene_id} / {exp.operating_side} base {bid}: arrival perception failed', flush=True)
            continue
        candidates = grasps(handle, exp.grasp_yaw_offset, exp.grasp_yaw_sign)
        latent = evaluator.encode(exp, arrival)
        direct = evaluator.policy_scores(latent, candidates, 'grasp')
        worlds = {str(h): evaluator.world_scores(latent, candidates, h, exp.opening_direction)
                  for h in [.25, .5, 1.]}
        geom = choose_grasp(exp, handle, candidate_offsets=[(dz, round(math.degrees(y), 6)) for dz, y in GRASP_OFFSETS])
        geometry = [next(c for c in geom['candidates'] if np.allclose(c['point_world_m'], p) and
                         abs(c['yaw_offset_deg']-math.degrees(y)) < 1e-6) for p, y in candidates]
        feasible, trials = [], []
        for gid, (point, yaw) in enumerate(candidates):
            yaw0 = math.pi/2+yaw
            pre = point-.16*np.array([math.cos(yaw0), math.sin(yaw0), 0.])
            feasible.append(exp.ik(point, yaw0) is not None and exp.ik(pre, yaw0) is not None)
            reached = scene.experiment(base)
            snapshots = {}

            def record(e, terminal):
                if terminal:
                    snapshots['terminal'] = e.d.qpos.tolist()
                elif any(abs(e.d.time-t) < .002 for t in [0., 2.5, 3.5]):
                    snapshots[f't{e.d.time:.1f}'] = e.d.qpos.tolist()

            result, rows, _ = reached.trial(point, yaw, reset_before_trial=False,
                observer=record if gid == 0 else None)
            trials.append(dict(grasp_id=gid, executor_called=True, integrated=bool(rows),
                               point_world_m=point.tolist(), yaw_offset_rad=yaw,
                               result=result, metrics=stage_metrics(result, rows),
                               stage_qpos=snapshots if gid == 0 else None))
        bases.append(dict(base_id=bid, side=exp.operating_side, base=list(base),
                          feasible=feasible, direct_scores=direct, world_scores=worlds,
                          geometry=geometry, trials=trials))
        print(f'{scene.scene_id} / {exp.operating_side} base {bid}: '
              f'{sum(t["metrics"]["success"] for t in trials)}/4 open+hold; '
              f'{trials[0]["result"]["stop_reason"]}', flush=True)
    return dict(scene=asdict(scene), position_scores=position_scores, bases=bases)


def decision(table, bid, gid):
    base = table['bases'][bid]
    trial = base['trials'][gid]
    return dict(scene_id=table['scene']['scene_id'], base_id=bid, side=base['side'],
                grasp_id=gid, result=trial['result'], metrics=trial['metrics'])


def choose(table, position='post', grasp='post', horizon=None):
    bid = (0 if position == 'front' else 2 if position == 'back'
           else int(np.argmax(table['position_scores'][position])))
    base = table['bases'][bid]
    if base['direct_scores'] is None:
        gid = 0
    else:
        scores = base['direct_scores'][grasp] if horizon is None else base['world_scores'][str(horizon)]
        gid = selected_grasp(base, scores)
    return decision(table, bid, gid)


def choose_geometry(table):
    candidates = [(g['score'], b['base_id'], gid, g['geometric_open_deg'])
                  for b in table['bases'] if b['geometry'] is not None
                  for gid, g in enumerate(b['geometry'])]
    if not candidates:
        return None, dict(abstained=True, reason='no_observable_grasp', predicted_max_angle_deg=None)
    _, bid, gid, angle = max(candidates, key=lambda x: x[0])
    return decision(table, bid, gid), dict(abstained=angle < 90.,
        reason='no_full_geometric_path' if angle < 90 else 'full_geometric_path',
        predicted_max_angle_deg=angle)


def aggregate(tables):
    conditions = {f'P_{p}/G_{g}': [choose(t, p, g) for t in tables]
                  for p in ['pre', 'post'] for g in ['pre', 'post']}
    for side in ['front', 'back']:
        conditions[f'fixed-{side}/G_post'] = [choose(t, side) for t in tables]
    for h in [.25, .5, 1.]:
        conditions[f'P_post/world/h{h}'] = [choose(t, horizon=h) for t in tables]
    geometric_decisions, guards, coverage = [], [], []
    for table in tables:
        selected, guard = choose_geometry(table)
        feasible_sides = sorted({b['side'] for b in table['bases'] if any(t['metrics']['success'] for t in b['trials'])})
        coverage.append(dict(scene_id=table['scene']['scene_id'], feasible_sides=feasible_sides))
        if selected is not None:
            geometric_decisions.append(selected)
        guards.append(dict(scene_id=table['scene']['scene_id'], **guard,
                           actually_feasible=bool(feasible_sides),
                           decision=selected if not guard['abstained'] else None))
    conditions['geometry-side'] = geometric_decisions
    matched = {'pre': [], 'post': []}
    for table in tables:
        for base in table['bases']:
            for g in matched:
                gid = selected_grasp(base, base['direct_scores'][g]) if base['direct_scores'] is not None else 0
                matched[g].append(base['trials'][gid])
    trials = [t for table in tables for b in table['bases'] for t in b['trials']]
    return dict(conditions={k: summarize(v) for k, v in conditions.items()},
                decisions=conditions, matched_arrivals={k: summarize(v) for k, v in matched.items()},
                physical_coverage=coverage, geometry_guard=guards,
                metadata=dict(scene_count=len(tables), candidate_assessments=len(trials),
                              executor_calls=sum(t['executor_called'] for t in trials),
                              integrated_trials=sum(t['integrated'] for t in trials),
                              frozen_world_and_values=True, new_pool_out_of_training_distribution=True,
                              learned_abstention_available=False),
                geometry_guard_summary=dict(attempts=sum(not g['abstained'] for g in guards),
                    successes=sum(g['decision']['metrics']['success'] for g in guards if g['decision']),
                    rejections=sum(g['abstained'] for g in guards),
                    correct_rejections=sum(g['abstained'] and not g['actually_feasible'] for g in guards),
                    false_rejections=sum(g['abstained'] and g['actually_feasible'] for g in guards)))


def render_report(tables, metrics, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen = ['pillar-front', 'pillar-both', 'one-way-positive', 'front-access-only']
    fig, axes = plt.subplots(4, 2, figsize=(10, 12), layout='constrained')
    scenes = structural_scenes()
    for row, name in enumerate(chosen):
        table = next(t for t in tables if t['scene']['scene_id'] == name)
        for col, bid in enumerate([0, 2]):
            exp = scenes[name].experiment(STRUCTURAL_BASES[bid])
            trial = table['bases'][bid]['trials'][0]
            if trial.get('stage_qpos') and trial['stage_qpos'].get('terminal'):
                exp.d.qpos[:] = trial['stage_qpos']['terminal']
                mujoco.mj_forward(exp.m, exp.d)
            # Same fixed top view for both sides; neutral-grasp physical terminal
            # states, not imagined frames or selected-policy success illustrations.
            cid = exp.m.camera('overview').id
            exp.m.cam_pos[cid] = [.45, 0., 3.8]
            exp.m.cam_quat[cid] = [1., 0., 0., 0.]
            mujoco.mj_forward(exp.m, exp.d)
            with mujoco.Renderer(exp.m, height=400, width=500) as renderer:
                renderer.update_scene(exp.d, camera='overview')
                picture = renderer.render().copy()
            ax = axes[row, col]
            ax.imshow(picture)
            result = trial['result']
            ax.set_title(f'{name} | {table["bases"][bid]["side"]}\n'
                         f'{result.get("final_signed_angle_deg", 0.):.1f} deg; {result["stop_reason"]}', fontsize=10)
            ax.axis('off')
    figure_path = Path('docs/hjepa_structural_results.png')
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)
    rows = ['# Structural door-constraint experiments (2026-10-17)', '',
            'Scenes add physical pillars, one-/two-way hinges, and front/back handles. '
            'MuJoCo contact blocks the panel without scene-name failure rules. '
            'Front pulling gives positive hinge angles, back pulling negative; base stays fixed while opening.', '',
            '**Changing stations cannot bypass the same blocked door arc.** Two-way doors can turn oppositely; '
            'if a pillar blocks the only permitted direction, the current library has no complete-opening candidate.', '',
            '![Physical scene terminal states](hjepa_structural_results.png)', '',
            'Top views show blue arms, brown doors, and gray pillars. '
            'Panels use neutral-grasp terminal states, or reached states where no handle is visible.', '',
            '## Frozen-checkpoint measurements', '',
            'Use complete official-model pre-update `best.pt` and post-update `selected.pt`. '
            'Shared world/value/normalization tensors match without training or test tuning. '
            'Candidates include two bases per side and four grasps per base: four bases total. '
            '**Back-side bases, reverse actions, and new structures are outside old training data**; these are capability checks.', '',
            '| Method | Success /9 | Position -> Align -> Grasp -> Open/Hold |', '|---|---:|---|']
    for name, m in metrics['conditions'].items():
        rows.append(f'| {name} | {m["stage_counts"]["open_hold"]}/{m["episodes"]} | '
                    +' → '.join(map(str, m['stage_counts'].values()))+' |')
    rows += ['', '## Physically feasible sides', '',
             '| Scene | Candidate-verified feasible side | Old P/G | Updated P/G | Known-geometry selection |',
             '|---|---|---|---|---|']
    def label(t):
        return f'{t["side"]}: '+('success' if t['metrics']['success'] else t['result']['stop_reason'])
    for i, c in enumerate(metrics['physical_coverage']):
        rows.append(f'| {c["scene_id"]} | {", ".join(c["feasible_sides"]) or "No complete-opening candidate"} | '
                    + ' | '.join(label(metrics['decisions'][k][i]) for k in
                                 ['P_pre/G_pre', 'P_post/G_post', 'geometry-side'])+' |')
    guard = metrics['geometry_guard_summary']
    rows += ['', f'Known-geometry full-path screening executes {guard["attempts"]} scenes with {guard["successes"]} successes; '
             f'{guard["rejections"]} refusals include {guard["correct_rejections"]} correct and {guard["false_rejections"]} false. '
             'Refusal is not opening success. This privileged reference reads geometry/hinge ranges '
             'unavailable to learned policy or latent rollout.', '',
             '## Physical interpretation and limitations', '',
             '- `pillar-front/back`: contact near 40 degrees blocks one turn; the opposite reaches 90.',
             '- `pillar-both`: both turns are blocked; changing only base/grasp does not complete opening.',
             '- `one-way-positive/negative`: handles on both sides but only one allowed turn. '
             'Wrong-side pulling can grasp but slips without moving the door. '
             'This does not exclude a different pushing skill; the library supports pulling only.',
             '- `front/back-access-only`: only one face has a handle. '
             'Geometry checks confirm collision when reaching through the panel; no perception-failure fallback uses them.',
             '- `front-only-blocked`: one-sided handle, one-way hinge, and pillar leave no full-opening candidate.',
             '- Learned selection always chooses an action, without trained refusal/side switching or calibrated feasibility. '
             'Collision stops are executor protection rather than advance predictions.',
             '- Arrival switches the calibrated camera to the operating face. Side changes still reset simulation; '
             'navigation around the door and automatic regrasp are not implemented.', '',
             '## Records and reproduction', '',
             f'{metrics["metadata"]["candidate_assessments"]} candidate evaluations, '
             f'{metrics["metadata"]["executor_calls"]} executor calls, '
             f'{metrics["metadata"]["integrated_trials"]} dynamic integrations. '
             'Classify by reached vision, IK, approach, stable grasp, and opening/hold; '
             'save neutral-grasp joints at 0/2.5/3.5 s and termination.', '',
             '```powershell', '.\.venv\Scripts\python.exe evaluate_hjepa_structure.py --device cuda',
             '.\.venv\Scripts\python.exe run_hjepa.py --scenario pillar-front --base-mode fixed --fixed-base-id 2 --method geometry --output runs/hjepa/structural_demo',
             '.\.venv\Scripts\python.exe -m unittest test_structural_scene test_hjepa test_hjepa_stages test_git_size -v',
             '```', '',
             'JSON, candidate CSV, and source/checkpoint SHA256 records are in ignored '
             '`runs/hjepa/structural_evaluation/`. Older reports bind historical source; '
             'reruns with changed hashes need a new `--output` rather than mixed caches.', '']
    Path('docs/hjepa_structural_results.md').write_text('\n'.join(rows), encoding='utf-8')
    with (output/'candidates.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['scene', 'base', 'side', 'grasp', 'executor_called', 'integrated',
                         'success', 'signed_angle_deg', 'stop_reason', 'failure_stage'])
        for table in tables:
            for base in table['bases']:
                for t in base['trials']:
                    writer.writerow([table['scene']['scene_id'], base['base_id'], base['side'], t['grasp_id'],
                        t['executor_called'], t['integrated'], t['metrics']['success'],
                        t['result'].get('final_signed_angle_deg', 0.), t['result'].get('stop_reason'), t['metrics']['failure_stage']])


def evaluate(before, after, output, device='cuda'):
    scenes = structural_scenes()
    spec = dict(schema=1, scenes={n: asdict(s) for n, s in scenes.items()},
                bases=STRUCTURAL_BASES, grasps=GRASP_OFFSETS,
                before_sha256=sha256(before), after_sha256=sha256(after),
                source_sha256={name: sha256(Path(name)) for name in ['evaluate_hjepa_structure.py',
                    'structural_scene.py', 'scene.xml', 'simulate.py', 'planning.py', 'hjepa_scene.py',
                    'evaluate_hjepa_stages.py', 'perception.py', 'hjepa_model.py', 'collect_hjepa.py']})
    spec = json.loads(json.dumps(spec))
    output.mkdir(parents=True, exist_ok=True)
    if (output/'suite.json').exists() and json.loads((output/'suite.json').read_text()) != spec:
        raise ValueError('Different structural suite/source/checkpoint; choose a fresh --output.')
    write_json(output/'suite.json', spec)
    evaluator = Evaluator(before, after, device)
    if evaluator.ckpt.get('base_budget', 5) < 4 or evaluator.ckpt.get('grasp_budget', 4) < 4:
        raise ValueError('Checkpoint candidate budget is too small for this matched suite.')
    start = time.perf_counter()
    tables = []
    for name, scene in scenes.items():
        path = output/f'{name}.json'
        table = json.loads(path.read_text()) if path.exists() else assess_scene(evaluator, scene)
        write_json(path, table)
        tables.append(table)
        metrics = aggregate(tables)
        write_json(output/'metrics.json', metrics)
    metrics['metadata']['invocation_seconds'] = time.perf_counter()-start
    write_json(output/'metrics.json', metrics)
    render_report(tables, metrics, output)
    print(json.dumps(dict(metadata=metrics['metadata'],
                          results={k: v['stage_counts'] for k, v in metrics['conditions'].items()},
                          geometry_guard=metrics['geometry_guard_summary']), indent=2), flush=True)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, default=Path('runs/hjepa/official_model/best.pt'))
    parser.add_argument('--after', type=Path, default=Path('runs/hjepa/official_model/selected.pt'))
    parser.add_argument('--output', type=Path, default=Path('runs/hjepa/structural_evaluation'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.before, args.after, args.output, resolve_device(args.device))
