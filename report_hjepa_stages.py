"""Export small reviewable reports/figures from completed frozen-policy tests."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(stages, views, destination):
    metrics = json.loads((stages/'metrics.json').read_text())
    view_metrics = json.loads((views/'metrics.json').read_text())
    suite = json.loads((stages/'suite.json').read_text())
    tables = [json.loads((stages/f'{c["scene"]["scene_id"]}.json').read_text()) for c in suite['cases']]
    angles = json.loads((stages/'angles.json').read_text())
    view_trials = json.loads((views/'trials.json').read_text())
    if 'metadata' not in metrics or len(tables) != metrics['metadata']['scene_count']:
        raise ValueError('Stage evaluation is incomplete.')
    candidates = [t for table in tables for b in table['bases'] for t in b['trials']]
    all_trials = candidates+angles+view_trials
    count = len(all_trials)
    execution_calls = sum(t['result']['outcome'] != 'B_perception_failed' for t in all_trials)
    integrated = sum(t['result']['outcome'] not in ['B_perception_failed', 'A_initial_unreachable'] for t in all_trials)
    oracle_successes = sum(any(t['metrics']['success'] for b in table['bases'] for t in b['trials']) for table in tables)
    destination.parent.mkdir(parents=True, exist_ok=True)

    names = ['P_pre/G_pre', 'P_post/G_pre', 'P_pre/G_post', 'P_post/G_post']
    labels = ['P pre / G pre', 'P post / G pre', 'P pre / G post', 'P post / G post']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for i, (name, label) in enumerate(zip(names, labels)):
        values = list(metrics['conditions'][name]['stage_counts'].values())
        axes[0, 0].plot(range(4), values, marker='o', linestyle='-' if i < 2 else '--',
                       linewidth=3 if i < 2 else 1.5, markersize=7 if i < 2 else 4, label=label)
    axes[0, 0].set(xticks=range(4), xticklabels=['Position', 'Align', 'Grasp', 'Open + hold'],
                    ylabel='Completed scenes', title=f'Matched policy interventions (N={len(tables)})',
                    ylim=(-.5, len(tables)+.5))
    axes[0, 0].legend(fontsize=8)
    heat_names = names+['P_post/world/h1.0', 'P_post/geometry']
    heat = np.array([[int(t['metrics']['success']) for t in metrics['decisions'][name]] for name in heat_names]).T
    axes[0, 1].imshow(heat, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    scene_labels = [c['scene']['scene_id'].replace('new_20261017_', 'New ') for c in suite['cases']]
    axes[0, 1].set(yticks=range(len(tables)), yticklabels=scene_labels, xticks=range(6),
                    xticklabels=['pre/pre', 'post/pre', 'pre/post', 'post/post', 'WM full', 'Geometry'],
                    title='Full-open success: green=yes, red=no')
    axes[0, 1].tick_params(axis='both', labelsize=8)
    for kind, label in [('base_yaw', 'Base yaw offset'), ('grasp_yaw', 'Grasp yaw offset')]:
        group = metrics['angles'][kind]
        ordered = sorted(group, key=float)
        axes[1, 0].plot([float(k) for k in ordered], [100*group[k]['success_rate'] for k in ordered], 'o-', label=label)
    axes[1, 0].set(xlabel='Offset (degrees)', ylabel='Success (%)', ylim=(-5, 105),
                    title='Angle sensitivity (4 repeated scenes; target=90 deg)')
    axes[1, 0].legend(fontsize=8)
    for bid in [0, 1, 3]:
        group = metrics['angles'][f'target_base{bid}']
        ordered = sorted(group, key=float)
        axes[1, 1].plot([float(k) for k in ordered], [100*group[k]['success_rate'] for k in ordered],
                       marker='o', linestyle='--' if bid == 3 else '-', linewidth=3 if bid == 0 else 1.5,
                       markersize=7 if bid == 0 else 4, label=f'Base {bid}')
    axes[1, 1].set(xlabel='Controller target (degrees)', ylabel='Target + hold success (%)',
                    ylim=(-5, 105), xticks=[30, 60, 90], title='Goal-angle intervention (4 repeated scenes)')
    axes[1, 1].legend(fontsize=8)
    for ax in [axes[0, 0], axes[1, 0], axes[1, 1]]:
        ax.grid(alpha=.2)
    fig.savefig(destination.with_suffix('.png'), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
    for ax, method in zip(axes, ['direct', 'world-model']):
        yaw, elevation = [-30, -15, 0, 15, 30], [-10, 0, 10]
        values = [[100*view_metrics['by_view'][f'yaw{y}/elev{e}'][method]['success_rate'] for y in yaw] for e in elevation]
        ax.imshow(values, vmin=0, vmax=100, cmap='RdYlGn', aspect='auto')
        for r in range(3):
            for c in range(5):
                ax.text(c, r, f'{values[r][c]:.0f}%', ha='center', va='center')
        ax.set(xticks=range(5), xticklabels=yaw, yticks=range(3), yticklabels=elevation,
               xlabel='Camera yaw (deg)', ylabel='Camera elevation (deg)', title=f'{method} (2 scenes per cell)')
    fig.savefig(destination.with_name(destination.stem+'_views.png'), dpi=150)
    plt.close(fig)

    lines = ['# H-JEPA 分阶段与角度实验（2026-10-17）', '',
             f'官方完整模型保持冻结，共 **{count} 条评估记录**：{len(candidates)} 条场景候选、'
             f'{len(angles)} 条执行角度/目标角度、{len(view_trials)} 条相机视角记录。'
             f'其中 {execution_calls} 次调用执行器，{integrated} 次实际积分动力学；'
             '其余在视觉或初始 IK 阶段被拒绝，仍计为失败，不伪装成完成过的动力学执行。', '',
             f'新随机场景 8 个，seed={suite["seed"]}；另有 9 个物理参数压力场景及 1 个无夹持对照。'
             '相机实验复用其中两个新场景，角度扫描复用四个场景。记录之间共享场景，不能作为独立场景数。', '',
             '定位 P 与抓取 G 各有更新前/后两个版本：`best.pt` 的监督预热策略，以及 `selected.pt` 的 '
             '100 次交替离线更新策略。逐张量检查确认世界模型、价值头及归一化参数完全一致。'
             '候选执行结果仅用于事后评估，未用于测试时排序、训练或调参。', '',
             '![分阶段与角度实验](hjepa_stage_results.png)', '',
             '## 两个 policy 的独立作用', '',
             '| 组合 | Position | Align | Grasp | Open + hold | 成功率 | 安全开度比例 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in names+['fixed/G_post', 'P_post/world/h0.25', 'P_post/world/h0.5', 'P_post/world/h1.0', 'P_post/geometry']:
        m = metrics['conditions'][name]
        counts = m['stage_counts']
        lines.append(f'| {name} | '+ ' | '.join(str(v) for v in counts.values())+
                     f' | {100*m["success_rate"]:.1f}% | {m["mean_safe_opening_fraction"]:.3f} |')
    lines += ['', '上表分母均为 18 个场景；Position 指当前抓点和预抓点 IK 可达，'
              'Align 指 2.5 s 接近完成且此前没有碰撞，Grasp 指稳定双侧夹持，Open + hold 指达标并稳定保持一秒。'
              '成功率不能用“成功夹住的次数”替代。', '',
              '| 场景组 | P_pre/G_pre | P_post/G_pre | P_pre/G_post | P_post/G_post |',
              '|---|---:|---:|---:|---:|']
    for family, group in metrics['by_family'].items():
        lines.append('| '+family+' | '+' | '.join(f'{group[n]["stage_counts"]["open_hold"]}/{group[n]["episodes"]}' for n in names)+' |')
    lines += ['', '固定每个到达状态、画面和底座后，抓取 policy 的对照如下（18×5=90 个到达状态，包含上游失败）：', '',
              '| 抓取版本 | Position | Align | Grasp | Open + hold |', '|---|---:|---:|---:|---:|']
    for g, m in metrics['matched_arrivals'].items():
        lines.append(f'| {g} | '+' | '.join(str(v) for v in m['stage_counts'].values())+' |')
    lines += ['', '配对效应（一次只替换一个 policy）：', '',
              '| 替换 | 成功率变化（百分点） | 改善/退化场景 | 决策改变场景 |', '|---|---:|---:|---:|']
    for name, e in metrics['paired_effects'].items():
        lines.append(f'| {name} | {100*e["success_delta"]:+.1f} | {e["improved"]}/{e["regressed"]} | {e["choice_changed"]} |')
    post_bases = sorted({t['base_id'] for t in metrics['decisions']['P_post/G_post']})
    lines += ['', f'更新后的定位 policy 在这 18 个场景中均选择底座 {post_bases}。'
              '固定名义底座 0 同样成功 14/18；因此本次改善说明修正了坏站位偏好，'
              '尚未证明优于固定好站位，或具备广泛的自适应定位能力。', '',
              f'仅用物理结果事后检查候选覆盖率：{oracle_successes}/18 个场景至少有一个候选能成功。'
              '这不是可部署的 oracle policy，也不参与选动作。', '',
              '## 更新后仍然失败的场景', '', '| 场景 | 停止原因 | 最终角度 | 失败阶段 |', '|---|---|---:|---|']
    for t in metrics['decisions']['P_post/G_post']:
        if not t['metrics']['success']:
            lines.append(f'| {t["scene_id"]} | {t["result"].get("stop_reason")} | '
                         f'{t["result"].get("final_angle_deg", 0.):.3f}° | {t["metrics"]["failure_stage"]} |')
    lines += ['', '## 角度与视觉', '',
              '底座 yaw 偏移 -45°～45°，抓取 yaw 偏移 -30°～30°；目标开度 30°、60°、90°。'
              '部分开度测试修改实体控制轨迹，要求最后一秒角度误差不超过 2°、持续双侧接触及低角速度。'
              '它们只评估控制器；当前 H-JEPA 没有训练为目标角度条件模型。', '',
              '| 扫描 | 角度 | 成功/记录 | 平均最终角度 |', '|---|---:|---:|---:|']
    for kind, group in metrics['angles'].items():
        for value in sorted(group, key=float):
            m = group[value]
            lines.append(f'| {kind} | {float(value):g}° | {m["stage_counts"]["open_hold"]}/{m["episodes"]} | '
                         f'{m["mean_final_angle_deg"]:.2f}° |')
    lines += ['', '![相机角度成功率](hjepa_stage_results_views.png)', '',
              '| 相机测试方法 | 成功/记录 | 失败阶段 |', '|---|---:|---|']
    for method, m in view_metrics['by_method'].items():
        lines.append(f'| {method} | {m["stage_counts"]["open_hold"]}/{m["episodes"]} | {m["failure_stages"]} |')
    lines += ['', '## 结论范围与复现', '',
              '这些结果支持定位更新改善下游可行性；本次新随机场景中，单独更新抓取 policy 没有提高完整开门率。'
              '固定底座、direct 与 geometry 也是必要参照；当前结果不能据此宣称世界模型优于这些方法。'
              '摩擦与铰链阻力在初始 RGB 中不可见，当前单帧观察没有接触后的材料辨识信息；'
              '相机测试失败也应按视觉边界失败计入，不能记为抓取或开门能力。', '',
              '没有用这批结果调整权重或重新训练。只有一个训练种子、八个新随机场景，压力场景有意构造，'
              '尚不足以给出通用泛化或统计显著性结论。下一轮改模型后应使用新的测试种子。', '',
              '```powershell', '.\.venv\Scripts\python.exe evaluate_hjepa_stages.py --device cuda',
              '.\.venv\Scripts\python.exe evaluate_hjepa_views.py --device cuda',
              '.\.venv\Scripts\python.exe report_hjepa_stages.py',
              '.\.venv\Scripts\python.exe check_git_size.py', '```', '',
              '详细 JSON、逐候选物理结果、checkpoint/source SHA256 均位于忽略目录 '
              '`runs/hjepa/{stage_evaluation,view_evaluation}/`。'
              '已完成 14 项代码测试和原始基线 8 项回归检查。'
              'Git 检查默认拒绝超过 10 MiB 的可见文件及暂存 blob，模型/数据/视频输出不进入版本库。', '']
    destination.write_text('\n'.join(lines), encoding='utf-8')
    with (stages/'conditions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'episodes', 'position', 'align', 'grasp', 'open_hold', 'success_rate', 'safe_fraction'])
        for name, m in metrics['conditions'].items():
            writer.writerow([name, m['episodes'], *m['stage_counts'].values(), m['success_rate'], m['mean_safe_opening_fraction']])
    print(json.dumps(dict(records=count, execution_calls=execution_calls, integrated_trials=integrated,
                          candidate_coverage=f'{oracle_successes}/{len(tables)}', report=str(destination)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages', type=Path, default=Path('runs/hjepa/stage_evaluation'))
    parser.add_argument('--views', type=Path, default=Path('runs/hjepa/view_evaluation'))
    parser.add_argument('--output', type=Path, default=Path('docs/hjepa_stage_results.md'))
    args = parser.parse_args()
    report(args.stages, args.views, args.output)
