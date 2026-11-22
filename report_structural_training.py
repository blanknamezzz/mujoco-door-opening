"""Small reproducible report of completed full-model structural fine-tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def report(data, model, evaluation, output):
    manifest = json.loads((data/'manifest.json').read_text())
    training = json.loads((model/'training.json').read_text())
    heldout = json.loads((evaluation/'heldout.json').read_text())
    replays = json.loads((evaluation/'replay.json').read_text())
    validation = training['stage_policy_selection']['validation']
    history = training['history']
    names = ['old_checkpoint', 'P_pre/G_pre', 'P_post/G_post', 'trained_four_actions', 'world_h1.0']
    labels = ['Old', 'Fine-tuned', '+ Policy updates', 'No refusal', 'WM full']
    decisions = heldout['decisions']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout='constrained')
    axes[0].plot([h['step'] for h in history], [h['train'] for h in history], label='Train sampled loss')
    axes[0].plot([h['step'] for h in history], [h['val'] for h in history], label='Validation sampled loss')
    axes[0].set(xlabel='Full-model optimizer step', ylabel='Loss', title='Official full-model fine-tuning')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    axes[1].plot([v['update'] for v in validation], [v['validation_mean_reward'] for v in validation], 'o-')
    selected = training['selected_policy_updates']
    axes[1].axvline(selected, color='tab:orange', linestyle='--', label=f'Selected {selected}')
    refit = training.get('position_policy_refit')
    if refit and refit['accepted']:
        offset = training['stage_policy_selection']['updates_executed']
        axes[1].plot([offset+v['step'] for v in refit['history']],
                     [v['validation_mean_reward'] for v in refit['history']], label='Backbone position probe')
        axes[1].axvline(offset+refit['selected_step'], color='tab:green', linestyle=':', label='Selected probe')
    axes[1].set(xlabel='Alternating policy updates', ylabel='Validation mean return', title='Frozen-world policy selection')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.2)
    values = []
    for i in range(heldout['test_scene_count']):
        row = []
        for name in names:
            d = decisions[name][i]
            row.append(1 if d['success'] else 2 if d['correct_rejection'] else 3 if d['false_rejection'] else 0)
        values.append(row)
    axes[2].imshow(values, cmap=ListedColormap(['#c0392b', '#229954', '#2980b9', '#f39c12']),
                   vmin=0, vmax=3, aspect='auto', interpolation='nearest')
    families = [d['family'] for d in decisions[names[0]]]
    axes[2].set(yticks=range(len(families)), yticklabels=families, xticks=range(len(names)), xticklabels=labels,
                title='Held-out decisions\nGreen: open; blue: correct refusal\nRed: failure; orange: false refusal')
    axes[2].tick_params(axis='x', rotation=30, labelsize=8)
    axes[2].tick_params(axis='y', labelsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.png'), dpi=150)
    plt.close(fig)
    size = sum(p.stat().st_size for p in data.glob('*.npy'))/2**30
    lines = ['# Full H-JEPA structural-scene training (2026-10-17)', '',
        f'Continue all world-model weights from the official full checkpoint for **{training["epochs_completed"]:g} epochs / '
        f'{training["steps"]} steps**. Parameters {training["parameters"]:,}, batch {training["batch_size"]}, '
        f'{training["precision"]}, 224x224 inputs, two official levels, and ViT recomputation.', '',
        '**Held-out results: old/new models open 3/5 and 4/5 feasible scenes. '
        'One single-pillar scene still selects the wrong side and the double-pillar case is not refused; lookahead does not surpass direct.**', '',
        f'Training takes {training["elapsed_seconds"]/60:.1f} minutes; PyTorch peak allocation '
        f'{training["peak_allocated_mib"]:.1f} MiB, reservation {training["peak_reserved_mib"]:.1f} MiB. '
        'Exclude collection and subsequent policy optimization with frozen world model.', '',
        f'Data: 24 fresh scenes (12 train / 6 val / 6 test), {len(manifest["episodes"])} records, '
        f'{sum(e["frames"] for e in manifest["episodes"]):,} frames, about {size:.2f} GiB native images. '
        'Scenes include two-way doors, single/double pillars, and one-sided handles; split complete scenes. '
        'Tests do not enter training, normalization, checkpoint selection, or policy updates.', '',
        '![Training and testing](hjepa_structural_training.png)', '',
        '## Six held-out scenes', '',
        '| Method | Complete opening | Correct refusal | False refusal | Correct decisions | Collisions |', '|---|---:|---:|---:|---:|---:|']
    for name, m in heldout['summaries'].items():
        lines.append(f'| {name} | {m["opening_successes"]}/{m["feasible_scenes"]} | {m["correct_rejections"]} | '
                     f'{m["false_rejections"]} | {m["correct_task_decisions"]}/{m["scenes"]} | {m["collisions"]} |')
    lines += ['', 'Opening denominators count scenes with successful candidates; correctness includes infeasible refusals. '
              'Refusal is not opening success. Old has four candidates; new policies add zero-feature refusal without added parameters. '
              '`trained_four_actions` removes refusal for matching four-action comparisons.', '',
              '| Scene | Old model | New policy | Full world-model prediction horizon |', '|---|---|---|---|']
    def label(d):
        if d['abstained']:
            return 'correct refusal' if d['correct_rejection'] else 'false refusal'
        return f'{d["side"]}: '+('success' if d['success'] else d['result']['stop_reason'])
    for i, family in enumerate(families):
        lines.append(f'| {family} | '+ ' | '.join(label(decisions[n][i]) for n in
                     ['old_checkpoint', 'P_post/G_post', 'world_h1.0'])+' |')
    lines += ['', '## Training and selection', '',
        f'Validation loss selects world-model step {training["selected_world_model_step"]}. '
        f'Then run {training["stage_policy_selection"]["updates_executed"]} alternating offline policy updates, '
        f'selecting both policies at update {selected} by validation return. '
        'Evaluate every 100 updates including initialization, without test-based selection. '
        'Tensor audits confirm the world model is frozen during policy optimization.', '',
        'World-model training uses official prediction loss plus task value/policy supervision on physical trajectories. '
        'P uses best measured training-base downstream returns; G uses actual grasp/opening returns. '
        'Refusal is always 0 without feasibility truth, invented trajectories, or reached-grasp observations. '
        'Additional optimization is finite-candidate offline policy gradient on frozen encodings/cached returns, not online PPO/GRPO.', '',
        f'Replay all {len(replays)} fresh test scenes with live RGB-D/contact dynamics; '
        'assert chosen actions/outcomes match held-out candidates, using future returns only for post-selection scoring.', '',
        '## Reproduction and limitations', '',
        'One seed and six scenes do not establish broad generalization or significant geometry superiority. '
        'Compare pre/post policies on the same world model separately from old-model baselines to preserve representation identity. '
        'Hidden one-way hinges lack visual cues and are excluded from main training; identical closed views cannot reveal their limits. '
        'Retain fixed pulling, simulator base placement, and calibrated gold-handle perception.', '',
        '```powershell',
        '.\.venv\Scripts\python.exe run_hjepa.py --checkpoint runs/hjepa/structural_model/selected.pt --scenario pillar-front --output runs/hjepa/structural_trained_demo',
        '.\.venv\Scripts\python.exe evaluate_structural_training.py --device cuda',
        '.\.venv\Scripts\python.exe report_structural_training.py', '```', '',
        'Models/data/logs remain ignored in `runs/hjepa/structural_{data,model,trained_evaluation}/`. '
        'Model directories preserve initialization SHA256, launch snapshots, and manifest hashes. '
        'Git stores small source/docs/figures rather than weights/arrays.', '']
    if refit:
        lines += ['## Validation-based positioning representation refit', '',
            f'Validation return before refit: {refit["validation_before"]["mean_reward"]:.3f}. '
            f'After fitting P on the same ViT pre-projection CLS: {refit["validation_after"]["mean_reward"]:.3f}; '
            f'accepted: {refit["accepted"]}.', '',
            'Refit updates the original P MLP only, preserving core/predictors/value/G tensors. '
            'Use temperature-0.03 soft-label ranking from training returns, folding feature normalization into its first affine layer; '
            'add no parameters or geometry truth. '
            f'Fit {refit["steps_executed"]} steps and select {refit["selected_step"]} on validation without test involvement.', '',
            ('Pre/post P comparisons change the chosen representation interface (projected versus pre-projection CLS); '
             'G and latent rollout retain original projected features. ' if refit['accepted'] else
             'No validation improvement; retain projected features and matching pre/post interfaces. ')+
            'Development also tries frozen-head return regression, ranking, and overhead observation rehearsal; '
            'validation selection excludes unsuccessful extra-camera variants. '
            'Small-set model selection still needs more independent scenes.', '',
            '```powershell', '.\.venv\Scripts\python.exe refit_structural_position.py', '```', '']
    if (evaluation/'canonical.json').exists():
        canonical = json.loads((evaluation/'canonical.json').read_text())
        lines += ['## Supplementary checks on earlier structures', '',
            'Earlier debugging cases are supplemental, outside the six fresh held-out metrics. '
            'Two hidden one-way cases have identical RGB-D without visible turn cues.', '',
            '| Scene | New-policy decision | Outcome |', '|---|---|---|']
        for d in canonical:
            lines.append(f'| {d["scene_id"]} | {"refuse" if d["abstained"] else d["side"]} | '
                         f'{d["result"]["stop_reason"]} |')
        lines += ['']
    output.write_text('\n'.join(lines), encoding='utf-8')
    print(str(output))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('runs/hjepa/structural_data'))
    p.add_argument('--model', type=Path, default=Path('runs/hjepa/structural_model'))
    p.add_argument('--evaluation', type=Path, default=Path('runs/hjepa/structural_trained_evaluation'))
    p.add_argument('--output', type=Path, default=Path('docs/hjepa_structural_training.md'))
    a = p.parse_args()
    report(a.data, a.model, a.evaluation, a.output)
