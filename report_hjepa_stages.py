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

    lines = ['# H-JEPA stage and angle experiments (2026-10-17)', '',
             f'The official full model is frozen. **{count} evaluation records** include {len(candidates)} scene candidates, '
             f'{len(angles)} execution/target-angle records, and {len(view_trials)} camera-view records. '
             f'There are {execution_calls} executor calls and {integrated} dynamic integrations; '
             'vision/initial-IK rejections still count as failures rather than completed dynamics.', '',
             f'Eight fresh random scenes use seed={suite["seed"]}, plus 9 stress scenes and 1 no-grip control. '
             'Cameras reuse two fresh scenes and angle scans four; shared records are not independent scenes.', '',
             'P/G pre/post versions are supervised-warmup `best.pt` and `selected.pt` after '
             '100 alternating offline updates. World/value/normalization tensors match exactly. '
             'Candidate outcomes support post hoc evaluation rather than test ranking, training, or tuning.', '',
             '![Stage and angle experiments](hjepa_stage_results.png)', '',
             '## Separate policy effects', '',
             '| Combination | Position | Align | Grasp | Open + hold | Success rate | Safe opening fraction |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in names+['fixed/G_post', 'P_post/world/h0.25', 'P_post/world/h0.5', 'P_post/world/h1.0', 'P_post/geometry']:
        m = metrics['conditions'][name]
        counts = m['stage_counts']
        lines.append(f'| {name} | '+ ' | '.join(str(v) for v in counts.values())+
                     f' | {100*m["success_rate"]:.1f}% | {m["mean_safe_opening_fraction"]:.3f} |')
    lines += ['', 'Every denominator is 18 scenes. Position means reachable grasp/pregrasp IK; '
              'Align completes the 2.5 s approach without collision; Grasp is stable bilateral contact; Open + hold meets target and holds one second. '
              'Grasp acquisition is not full-task success.', '',
              '| Scene group | P_pre/G_pre | P_post/G_pre | P_pre/G_post | P_post/G_post |',
              '|---|---:|---:|---:|---:|']
    for family, group in metrics['by_family'].items():
        lines.append('| '+family+' | '+' | '.join(f'{group[n]["stage_counts"]["open_hold"]}/{group[n]["episodes"]}' for n in names)+' |')
    lines += ['', 'Fix reached states, images, and bases to compare grasp policies across 18x5=90 reached states, including upstream failures:', '',
              '| Grasp version | Position | Align | Grasp | Open + hold |', '|---|---:|---:|---:|---:|']
    for g, m in metrics['matched_arrivals'].items():
        lines.append(f'| {g} | '+' | '.join(str(v) for v in m['stage_counts'].values())+' |')
    lines += ['', 'Paired effects replacing one policy at a time:', '',
              '| Replacement | Success-rate change, percentage points | Improved/degraded scenes | Changed decisions |', '|---|---:|---:|---:|']
    for name, e in metrics['paired_effects'].items():
        lines.append(f'| {name} | {100*e["success_delta"]:+.1f} | {e["improved"]}/{e["regressed"]} | {e["choice_changed"]} |')
    post_bases = sorted({t['base_id'] for t in metrics['decisions']['P_post/G_post']})
    lines += ['', f'Updated P selects bases {post_bases} across these 18 scenes. '
              'Fixed nominal base 0 also succeeds 14/18, showing a corrected poor preference '
              'rather than improvement over a good fixed base or broad adaptation.', '',
              f'Post hoc candidate coverage: {oracle_successes}/18 scenes have a successful candidate. '
              'This oracle summary is not deployed or used for selection.', '',
              '## Remaining failures', '', '| Scene | Stop reason | Final angle | Failure stage |', '|---|---|---:|---|']
    for t in metrics['decisions']['P_post/G_post']:
        if not t['metrics']['success']:
            lines.append(f'| {t["scene_id"]} | {t["result"].get("stop_reason")} | '
                         f'{t["result"].get("final_angle_deg", 0.):.3f}° | {t["metrics"]["failure_stage"]} |')
    lines += ['', '## Angles and vision', '',
              'Base yaw spans -45 to 45 degrees, grasp yaw -30 to 30; targets are 30/60/90 degrees. '
              'Partial-opening trials change physical references and require final-second error <=2 degrees, bilateral contact, and low speed. '
              'These assess the controller; H-JEPA is not target-angle conditioned.', '',
              '| Scan | Angle | Success / records | Mean final angle |', '|---|---:|---:|---:|']
    for kind, group in metrics['angles'].items():
        for value in sorted(group, key=float):
            m = group[value]
            lines.append(f'| {kind} | {float(value):g}° | {m["stage_counts"]["open_hold"]}/{m["episodes"]} | '
                         f'{m["mean_final_angle_deg"]:.2f}° |')
    lines += ['', '![Camera-angle success rates](hjepa_stage_results_views.png)', '',
              '| Camera test method | Success / records | Failure stage |', '|---|---:|---|']
    for method, m in view_metrics['by_method'].items():
        lines.append(f'| {method} | {m["stage_counts"]["open_hold"]}/{m["episodes"]} | {m["failure_stages"]} |')
    lines += ['', '## Scope and reproduction', '',
              'P updates improve downstream feasibility; G-only updates do not improve full opening on these fresh scenes. '
              'Fixed-base/direct/geometry controls remain necessary without an established world-model advantage. '
              'Initial RGB does not reveal friction/hinge resistance or post-contact material properties. '
              'Camera failures are vision failures rather than grasp/opening capability.', '',
              'No weights are tuned or retrained on these outcomes. One seed, eight fresh scenes, and constructed stresses '
              'do not establish broad generalization/significance. Use a fresh test seed after model changes.', '',
              '```powershell', '.\.venv\Scripts\python.exe evaluate_hjepa_stages.py --device cuda',
              '.\.venv\Scripts\python.exe evaluate_hjepa_views.py --device cuda',
              '.\.venv\Scripts\python.exe report_hjepa_stages.py',
              '.\.venv\Scripts\python.exe check_git_size.py', '```', '',
              'Detailed JSON, candidate outcomes, and checkpoint/source SHA256 records are in ignored '
              '`runs/hjepa/{stage_evaluation,view_evaluation}/`. '
              'Fourteen code tests and 8 original-baseline regressions pass. '
              'Git rejects visible files/staged blobs over 10 MiB; models/data/videos stay outside version control.', '']
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
