"""Plot independent scene-level task decisions, never training accuracy."""
import argparse
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def interval(successes, total):
    if not total:
        return 0., 0.
    z = 1.959963984540054
    p = successes/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return center-half, center+half


def report(evaluation, training, output):
    result = json.loads((evaluation/'heldout.json').read_text(encoding='utf-8'))
    replay = json.loads((evaluation/'replay.json').read_text(encoding='utf-8'))
    if len(replay) != result['test_scene_count'] or not all(r['selection_matches'] and r['outcome_matches'] for r in replay):
        raise ValueError('Complete matching physical test replays required before reporting.')
    history = json.loads(training.read_text(encoding='utf-8'))
    keys = (['reference_0', 'old_checkpoint', 'P_post/G_post'] if result.get('references') else
            ['old_checkpoint', 'P_post/G_post'])
    labels = (['Original base', 'Previous LoRA', 'Continued LoRA'] if len(keys) == 3 else ['Original base', 'LoRA 1200'])
    if history.get('reward_decomposition'):
        labels[-1]='Decomposed LoRA' if history.get('selected_method_adopted') else 'Baseline fallback'
    if history.get('decision_calibration') and history.get('selected_method_adopted'):
        labels[-1]='Calibrated LoRA'
    if history.get('listwise_position'):
        if history.get('arm_summaries'):
            arms=history.get('comparison_arms') or [dict(name='head_only',label='Head-only',reference='reference_2'),
                dict(name='position_lora',label='P+LoRA',reference='reference_3')]
            keys=['old_checkpoint','reference_1']+[a['reference'] for a in arms]
            labels=['Scalar LoRA','Decomp.']+[a['label'] for a in arms]
            for index,arm in enumerate(arms,2):
                arm_report=json.loads((training.parent/arm['name']/'training.json').read_text())
                if not arm_report.get('selected_method_adopted'):
                    labels[index]+=' fallback'
        else:
            labels[-1]=('Baseline fallback' if not history.get('selected_method_adopted') else
                        'Listwise head control' if history.get('head_only') else 'Listwise LoRA')
    summaries = result['summaries']
    count = result['test_scene_count']
    families = sorted({d['family'] for d in result['decisions'][keys[-1]]})
    matrix = np.array([[np.mean([d['task_decision_correct'] for d in result['decisions'][key] if d['family'] == family])
                        for key in keys] for family in families])
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 6.4), gridspec_kw={'width_ratios': [1.05, 1.05, 1.25]})
    curve_metric = 'correct_task_decisions' if history.get('reward_decomposition') or history.get('listwise_position') else 'mean_reward'
    if history.get('door_side_reward'):
        curve_metric='correct_side_decisions'
    trace = [(h['step'], h['policy_validation'][curve_metric]) for h in history['history'] if h.get('policy_validation')]
    if history.get('unadapted_validation'):
        trace.insert(0, (0, history['unadapted_validation'][curve_metric]))
    axes[0].plot(*np.asarray(trace).T, color='#31688e', linewidth=1.8, marker='.', markersize=5)
    if history.get('listwise_position') and history.get('arm_summaries'):
        axes[0].lines[0].remove()
        for arm,color in zip(arms,('#31688e','#b55a30')):
            arm_report=json.loads((training.parent/arm['name']/'training.json').read_text())
            points=[(0,arm_report['unadapted_validation'][curve_metric])]+[
                (h['step'],h['policy_validation'][curve_metric]) for h in arm_report['history']]
            axes[0].plot(*np.asarray(points).T,color=color,linewidth=1.8,marker='.',markersize=5,label=arm['label'])
    axes[0].axvline(history['selected_world_model_step'], color='#b55a30', linestyle='--', label='Selected on validation')
    axes[0].set(title='Validation selects weights', xlabel='LoRA update step',
                ylabel='Correct validation decisions' if curve_metric=='correct_task_decisions' else 'Mean task reward')
    if history.get('listwise_position'):
        axes[0].set_xlabel('Position-policy update step')
    if history.get('door_side_reward'):
        axes[0].set_ylabel('Correct side decisions on validation')
    if history.get('decision_calibration'):
        axes[0].set_title('Development selection\n(includes threshold fit)',fontsize=10)
    axes[0].legend(loc='lower right', fontsize=8)
    axes[0].grid(alpha=.18)
    percentages = [summaries[k]['correct_task_decisions']/count*100 for k in keys]
    bounds = [interval(summaries[k]['correct_task_decisions'], count) for k in keys]
    errors = np.array([[p-lo*100, hi*100-p] for p, (lo, hi) in zip(percentages, bounds)]).T
    bars = axes[1].bar(range(len(keys)), percentages, yerr=errors, capsize=4,
                       color=['#999fa5', '#4b88aa', '#399280', '#9971ab'][-len(keys):], width=.6)
    for bar, key in zip(bars, keys):
        axes[1].text(bar.get_x()+bar.get_width()/2, 4,
            f"{summaries[key]['correct_task_decisions']}/{count}", ha='center', va='bottom', color='white', weight='bold')
    axes[1].set(xticks=range(len(keys)), xticklabels=[x.replace(' ', '\n', 1) for x in labels],
                ylim=(0, 100), ylabel='Correct task decisions (%)')
    axes[1].set_title(f'{count} independent test scenes', pad=34)
    axes[1].text(.5, 1.015, 'Open feasible doors or refuse impossible cases\n95% Wilson intervals',
                 transform=axes[1].transAxes, ha='center', fontsize=8)
    axes[2].imshow(matrix, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    axes[2].set(xticks=range(len(keys)), xticklabels=[x.replace(' ', '\n', 1) for x in labels],
                yticks=range(len(families)), yticklabels=families, title='Correct decisions by family')
    for row, family in enumerate(families):
        for col, key in enumerate(keys):
            decisions = [d for d in result['decisions'][key] if d['family'] == family]
            numerator = sum(d['task_decision_correct'] for d in decisions)
            axes[2].text(col, row, f'{numerator}/{len(decisions)}', ha='center', va='center',
                         color='white' if matrix[row, col] > .6 else '#1c3449', fontsize=9)
    title='Frozen checkpoint evaluation - held-out scene geometry and camera views'
    if history.get('sensor_ablation'):
        selected_label=next(a['label'] for a in arms if a['name']==history['selected_arm'])
        title='Fixed-camera ablation - validation selected '+selected_label+'; no test reselection'
    if history.get('coverage_ablation'):
        selected_label=next(a['label'] for a in arms if a['name']==history['selected_arm'])
        title='Training coverage ablation - validation selected '+selected_label+'; seed 42'
    if history.get('pooling_ablation'):
        selected_label=next(a['label'] for a in arms if a['name']==history['selected_arm'])
        title='Native ViT pooling - validation selected '+selected_label+'; seed 42'
    figure.suptitle(title, fontsize=14, weight='bold')
    figure.subplots_adjust(left=.055, right=.985, bottom=.14, top=.86, wspace=.65)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor='white')
    plt.close(figure)
    previous = {d['scene_id']: d for d in result['decisions']['old_checkpoint']}
    current = result['decisions']['P_post/G_post']
    comparison = dict(
        newly_correct=sum(d['task_decision_correct'] and not previous[d['scene_id']]['task_decision_correct'] for d in current),
        newly_incorrect=sum(not d['task_decision_correct'] and previous[d['scene_id']]['task_decision_correct'] for d in current))
    summary = dict(test_scenes=count, dataset=result['dataset'], physical_replays=len(replay),
                   summaries={label: summaries[key] for key, label in zip(keys, labels)}, paired_comparison=comparison,
                   train_arrays_include_test=False if history.get('test_arrays_opened') == 0 else None,
                   image=str(output))
    if history.get('reward_decomposition'):
        summary['reward_decomposition']=history['reward_decomposition']
        summary['method_adopted']=history.get('selected_method_adopted')
    if history.get('decision_calibration'):
        summary['decision_calibration']=history['decision_calibration']
    (evaluation/'report_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, default=Path('runs/hjepa/lora_continued_model/independent_test'))
    p.add_argument('--training', type=Path, default=Path('runs/hjepa/lora_continued_model/training.json'))
    p.add_argument('--output', type=Path, default=Path('docs/hjepa_lora_independent.png'))
    a = p.parse_args()
    report(a.evaluation, a.training, a.output)
