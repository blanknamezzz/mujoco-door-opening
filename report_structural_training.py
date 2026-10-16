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
    lines = ['# 完整 H-JEPA 结构场景训练结果（2026-10-17）', '',
        f'从原官方完整 checkpoint 继续训练全部世界模型权重，完成 **{training["epochs_completed"]:g} 个 epoch / '
        f'{training["steps"]} 步**。参数 {training["parameters"]:,}，batch {training["batch_size"]}，'
        f'{training["precision"]}，224×224 输入，保持两层官方结构与 ViT 重计算。', '',
        '**保留测试结果：5 个可行场景中，旧模型成功 3 个，新模型成功 4 个。'
        '仍有一个单侧柱子场景选错侧，双侧受阻场景也未学会拒绝；当前没有证明 lookahead 优于 direct。**', '',
        f'耗时 {training["elapsed_seconds"]/60:.1f} 分钟；PyTorch 峰值分配 '
        f'{training["peak_allocated_mib"]:.1f} MiB，保留 {training["peak_reserved_mib"]:.1f} MiB。'
        '此耗时不包含数据采集和随后冻结世界模型的策略更新。', '',
        f'数据为 24 个新场景（12 train / 6 val / 6 test），共 {len(manifest["episodes"])} 条记录、'
        f'{sum(e["frames"] for e in manifest["episodes"]):,} 帧，原生图像约 {size:.2f} GiB。'
        '场景包含双向门、单侧/双侧柱子和单侧把手；同一场景的所有底座/抓取均留在同一 split。'
        '测试场景未用于训练、归一化、checkpoint 选择或 policy 更新。', '',
        '![训练与测试](hjepa_structural_training.png)', '',
        '## 六个保留测试场景', '',
        '| 方法 | 完整开门 | 正确拒绝 | 误拒绝 | 决策正确 | 碰撞 |', '|---|---:|---:|---:|---:|---:|']
    for name, m in heldout['summaries'].items():
        lines.append(f'| {name} | {m["opening_successes"]}/{m["feasible_scenes"]} | {m["correct_rejections"]} | '
                     f'{m["false_rejections"]} | {m["correct_task_decisions"]}/{m["scenes"]} | {m["collisions"]} |')
    lines += ['', '“完整开门”分母为有成功候选的场景数；“决策正确”同时计入完成开门和对无解场景的正确拒绝。'
              '拒绝不计为开门成功。Old 仅有四个执行候选；新策略增加零特征的拒绝候选，网络参数量不变。'
              '`trained_four_actions` 去掉新模型的拒绝候选，用于比较相同四个执行候选下的选侧能力。', '',
              '| 场景 | 旧模型 | 新策略 | world-model 完整预测时域 |', '|---|---|---|---|']
    def label(d):
        if d['abstained']:
            return '正确拒绝' if d['correct_rejection'] else '误拒绝'
        return f'{d["side"]}: '+('成功' if d['success'] else d['result']['stop_reason'])
    for i, family in enumerate(families):
        lines.append(f'| {family} | '+ ' | '.join(label(decisions[n][i]) for n in
                     ['old_checkpoint', 'P_post/G_post', 'world_h1.0'])+' |')
    lines += ['', '## 如何训练与选择', '',
        f'世界模型由验证 loss 选择第 {training["selected_world_model_step"]} 步。'
        f'随后执行 {training["stage_policy_selection"]["updates_executed"]} 次交替离线策略更新，'
        f'用验证场景回报选择第 {selected} 次更新的两个 policy。'
        '每 100 次评估一次，并保留更新前候选；没有在测试集上挑选轮数。'
        '逐张量检查确认该阶段世界模型完全冻结。', '',
        '世界模型阶段使用全部真实轨迹做官方预测损失，以及本任务的价值/策略监督。'
        '定位监督使用训练场景各底座的最佳已测下游回报；抓取监督使用实际抓取/开门回报。'
        '拒绝动作始终回报 0，不读取是否有解的真值，且不生成虚构轨迹或抓取到达观察。'
        '两 policy 的追加优化使用冻结编码和缓存的训练回报，是有限候选的离线策略梯度，不是在线 PPO/GRPO。', '',
        f'全部 {len(replays)} 个新测试场景已用实时 RGB-D 和接触动力学重放；'
        '逐例断言选择与实际结果和保留候选表一致。候选表的未来回报只在选动作后用于计分。', '',
        '## 复现与限制', '',
        '这是单训练种子的六场景保留测试，不能据此宣称广泛泛化或显著优于几何方法。'
        '同一世界模型上的 policy 前后组合，与旧世界模型基线分开计算，避免混用不同编码空间。'
        '没有可见线索的单向铰链未纳入主训练集；仅靠关闭时相同的图像无法识别隐藏转向限制。'
        '当前仍使用固定拉门技能、模拟器底座放置和已标定的金色把手视觉。', '',
        '```powershell',
        '.\.venv\Scripts\python.exe run_hjepa.py --checkpoint runs/hjepa/structural_model/selected.pt --scenario pillar-front --output runs/hjepa/structural_trained_demo',
        '.\.venv\Scripts\python.exe evaluate_structural_training.py --device cuda',
        '.\.venv\Scripts\python.exe report_structural_training.py', '```', '',
        '模型、数据与原始日志位于忽略目录 `runs/hjepa/structural_{data,model,trained_evaluation}/`。'
        '模型目录保存初始化 checkpoint SHA256、训练源码快照和数据 manifest 哈希。'
        '仓库只保存小型源码、说明与图表；权重与训练数组不纳入 Git。', '']
    if refit:
        lines += ['## 定位表示的验证修正', '',
            f'原冻结策略追加优化的验证平均回报为 {refit["validation_before"]["mean_reward"]:.3f}。'
            f'用同一 ViT 投影前 CLS 拟合定位策略后的验证回报为 {refit["validation_after"]["mean_reward"]:.3f}；'
            f'本次是否采用：{refit["accepted"]}。', '',
            '该修正只更新原有定位 MLP，核心 H-JEPA、预测器、价值头和抓取策略逐张量不变。'
            '用训练回报构建温度 0.03 的 soft-label 排序监督，训练集特征归一化折叠进原 MLP 的首个仿射层；'
            '没有增加参数或引入几何真值。'
            f'额外拟合 {refit["steps_executed"]} 步，验证回报选中第 {refit["selected_step"]} 步，测试集未参与选择。', '',
            ('定位 P 的前/后比较包含所选视觉表示接口的变化（投影后表示与投影前 CLS）；'
             '抓取 G 和 latent rollout 继续使用原来的投影后表示。' if refit['accepted'] else
             '该候选未改善验证结果，最终保留原投影后表示；policy 前后比较使用相同的编码接口。')+
            '验证期也试过冻结头回报回归、排序损失及俯视观察预演；最终方案按验证回报选择，'
            '没有将未改善的额外相机方案接入最终策略。'
            '这属于小验证集上的模型选择，仍需更多独立场景验证。', '',
            '```powershell', '.\.venv\Scripts\python.exe refit_structural_position.py', '```', '']
    if (evaluation/'canonical.json').exists():
        canonical = json.loads((evaluation/'canonical.json').read_text())
        lines += ['## 原有结构案例的补充检查', '',
            '这些案例曾用于调试，属于补充检查，不合并进六个新保留场景的指标。'
            '两个隐藏单向限位案例有相同初始 RGB-D，没有给模型可见方向线索。', '',
            '| 场景 | 新策略决策 | 结果 |', '|---|---|---|']
        for d in canonical:
            lines.append(f'| {d["scene_id"]} | {"拒绝" if d["abstained"] else d["side"]} | '
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
