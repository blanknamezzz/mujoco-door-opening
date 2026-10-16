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
    rows = ['# 结构性门约束实验（2026-10-17）', '',
            '这些场景新增实体柱子、双向/单向铰链，以及正反面独立把手。'
            '柱子与门板通过 MuJoCo 接触产生阻挡；没有按场景名硬编码失败。'
            '正面拉开对应正铰链角，反面拉开对应负铰链角，底座在开门时固定。', '',
            '**换站位不能绕过同一条受阻的门轨迹。** 双向门能改用反向转动；'
            '单向门若唯一允许的转动路径也被柱子挡住，则当前候选池没有完整开门解。', '',
            '![实体场景终止状态](hjepa_structural_results.png)', '',
            '图中统一为顶视图，蓝色机械臂、棕色门板、灰色实体柱子。'
            '每格展示对应侧的中性抓取终止状态；无可见把手时展示到达状态。', '',
            '## 冻结 checkpoint 的实测', '',
            '沿用官方完整模型的更新前 `best.pt` 与更新后 `selected.pt`。'
            '共享世界模型、价值头及归一化参数逐张量相同；此次未训练或按测试结果调参。'
            '候选池为两侧各两个底座、每个底座四个抓取，共四个底座。'
            '**反面底座、反向动作和新结构都在旧训练分布之外**，这些是能力检查，不是已完成适配的模型。', '',
            '| 方法 | 成功/9 | Position → Align → Grasp → Open/Hold |', '|---|---:|---|']
    for name, m in metrics['conditions'].items():
        rows.append(f'| {name} | {m["stage_counts"]["open_hold"]}/{m["episodes"]} | '
                    +' → '.join(map(str, m['stage_counts'].values()))+' |')
    rows += ['', '## 哪一侧实际可行', '',
             '| 场景 | 候选执行验证的可行侧 | 旧 P/G 结果 | 更新后 P/G 结果 | 已知几何选侧结果 |',
             '|---|---|---|---|---|']
    def label(t):
        return f'{t["side"]}: '+('成功' if t['metrics']['success'] else t['result']['stop_reason'])
    for i, c in enumerate(metrics['physical_coverage']):
        rows.append(f'| {c["scene_id"]} | {", ".join(c["feasible_sides"]) or "无完整开门解"} | '
                    + ' | '.join(label(metrics['decisions'][k][i]) for k in
                                 ['P_pre/G_pre', 'P_post/G_post', 'geometry-side'])+' |')
    guard = metrics['geometry_guard_summary']
    rows += ['', f'已知几何的完整路径筛选：执行 {guard["attempts"]} 个场景、成功 {guard["successes"]} 个；'
             f'拒绝 {guard["rejections"]} 个，其中正确拒绝 {guard["correct_rejections"]} 个，误拒绝 {guard["false_rejections"]} 个。'
             '拒绝不计为开门成功。该参考读取已知几何/铰链范围，属于显式特权基线；'
             '这些信息未送入 learned policy 或 latent rollout。', '',
             '## 物理含义和限制', '',
             '- `pillar-front/back`：门板运动到约 40° 时接触柱子；只有另一转向可完成 90°。',
             '- `pillar-both`：两个转向都受阻，单纯重新选底座或抓取不能完成。',
             '- `one-way-positive/negative`：两侧都有把手，铰链限位只允许一个转向。'
             '错误侧的固定“拉门”技能可以夹住，但拉不动并出现滑移。'
             '这不代表从该侧使用另一种推门技能也绝对不可能；当前动作库只有拉门。',
             '- `front/back-access-only`：只在指定面安装把手；另一面看不到把手。'
             '额外的几何诊断确认从反面穿过门板去抓真实把手会碰撞，此诊断没有作为感知失败的回退。',
             '- `front-only-blocked`：单面把手、单向铰链与允许方向上的柱子共同构成无完整开门解的场景。',
             '- 现有 learned selector 始终选一个动作，没有训练过拒绝/换侧策略，也没有校准可行性阈值。'
             '真实碰撞停止属于执行器保护，不应当作模型提前预测了阻挡。',
             '- 到达后传感器切换到对应操作面，并刷新标定 RGB-D。移动到对侧仍是模拟器重置；'
             '没有实现绕门导航或自动重抓。', '',
             '## 记录与复现', '',
             f'{metrics["metadata"]["candidate_assessments"]} 条候选评估，'
             f'{metrics["metadata"]["executor_calls"]} 次执行器调用，'
             f'{metrics["metadata"]["integrated_trials"]} 次实际动力学积分。'
             '失败分类使用到达观察、当前 IK、接近、稳定夹持和最终开门/保持状态；'
             '同时保存了中性抓取在 0/2.5/3.5 s 和终止时的关节状态。', '',
             '```powershell', '.\.venv\Scripts\python.exe evaluate_hjepa_structure.py --device cuda',
             '.\.venv\Scripts\python.exe run_hjepa.py --scenario pillar-front --base-mode fixed --fixed-base-id 2 --method geometry --output runs/hjepa/structural_demo',
             '.\.venv\Scripts\python.exe -m unittest test_structural_scene test_hjepa test_hjepa_stages test_git_size -v',
             '```', '',
             'JSON、逐候选 CSV 与 source/checkpoint SHA256 位于忽略目录 '
             '`runs/hjepa/structural_evaluation/`。旧参数/角度报告是其当时源码版本的历史结果；'
             '重跑旧套件时如源码校验变化，应指定新的 `--output`，避免混用缓存。', '']
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
