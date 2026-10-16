# 完整 H-JEPA 结构场景训练结果（2026-10-17）

从原官方完整 checkpoint 继续训练全部世界模型权重，完成 **9 个 epoch / 1053 步**。参数 23,324,422，batch 128，bf16-mixed，224×224 输入，保持两层官方结构与 ViT 重计算。

**保留测试结果：5 个可行场景中，旧模型成功 3 个，新模型成功 4 个。仍有一个单侧柱子场景选错侧，双侧受阻场景也未学会拒绝；当前没有证明 lookahead 优于 direct。**

耗时 43.2 分钟；PyTorch 峰值分配 2848.3 MiB，保留 3514.0 MiB。此耗时不包含数据采集和随后冻结世界模型的策略更新。

数据为 24 个新场景（12 train / 6 val / 6 test），共 408 条记录、40,518 帧，原生图像约 5.68 GiB。场景包含双向门、单侧/双侧柱子和单侧把手；同一场景的所有底座/抓取均留在同一 split。测试场景未用于训练、归一化、checkpoint 选择或 policy 更新。

![训练与测试](hjepa_structural_training.png)

## 六个保留测试场景

| 方法 | 完整开门 | 正确拒绝 | 误拒绝 | 决策正确 | 碰撞 |
|---|---:|---:|---:|---:|---:|
| old_checkpoint | 3/5 | 0 | 0 | 3/6 | 2 |
| P_pre/G_pre | 3/5 | 0 | 0 | 3/6 | 2 |
| P_pre/G_post | 3/5 | 0 | 0 | 3/6 | 2 |
| P_post/G_pre | 4/5 | 0 | 0 | 4/6 | 2 |
| P_post/G_post | 4/5 | 0 | 0 | 4/6 | 2 |
| trained_four_actions | 4/5 | 0 | 0 | 4/6 | 2 |
| world_h0.25 | 4/5 | 0 | 0 | 4/6 | 2 |
| world_h0.5 | 4/5 | 0 | 0 | 4/6 | 2 |
| world_h1.0 | 4/5 | 0 | 0 | 4/6 | 2 |

“完整开门”分母为有成功候选的场景数；“决策正确”同时计入完成开门和对无解场景的正确拒绝。拒绝不计为开门成功。Old 仅有四个执行候选；新策略增加零特征的拒绝候选，网络参数量不变。`trained_four_actions` 去掉新模型的拒绝候选，用于比较相同四个执行候选下的选侧能力。

| 场景 | 旧模型 | 新策略 | world-model 完整预测时域 |
|---|---|---|---|
| free-two-way | front: 成功 | front: 成功 | front: 成功 |
| pillar-front | front: door_obstructed | front: door_obstructed | front: door_obstructed |
| pillar-back | front: 成功 | front: 成功 | front: 成功 |
| pillar-both | front: door_obstructed | back: door_obstructed | back: door_obstructed |
| front-access-only | front: 成功 | front: 成功 | front: 成功 |
| back-access-only | front: perception_failure | back: 成功 | back: 成功 |

## 如何训练与选择

世界模型由验证 loss 选择第 1000 步。随后执行 1000 次交替离线策略更新，用验证场景回报选择第 100 次更新的两个 policy。每 100 次评估一次，并保留更新前候选；没有在测试集上挑选轮数。逐张量检查确认该阶段世界模型完全冻结。

世界模型阶段使用全部真实轨迹做官方预测损失，以及本任务的价值/策略监督。定位监督使用训练场景各底座的最佳已测下游回报；抓取监督使用实际抓取/开门回报。拒绝动作始终回报 0，不读取是否有解的真值，且不生成虚构轨迹或抓取到达观察。两 policy 的追加优化使用冻结编码和缓存的训练回报，是有限候选的离线策略梯度，不是在线 PPO/GRPO。

全部 6 个新测试场景已用实时 RGB-D 和接触动力学重放；逐例断言选择与实际结果和保留候选表一致。候选表的未来回报只在选动作后用于计分。

## 复现与限制

这是单训练种子的六场景保留测试，不能据此宣称广泛泛化或显著优于几何方法。同一世界模型上的 policy 前后组合，与旧世界模型基线分开计算，避免混用不同编码空间。没有可见线索的单向铰链未纳入主训练集；仅靠关闭时相同的图像无法识别隐藏转向限制。当前仍使用固定拉门技能、模拟器底座放置和已标定的金色把手视觉。

```powershell
.\.venv\Scripts\python.exe run_hjepa.py --checkpoint runs/hjepa/structural_model/selected.pt --scenario pillar-front --output runs/hjepa/structural_trained_demo
.\.venv\Scripts\python.exe evaluate_structural_training.py --device cuda
.\.venv\Scripts\python.exe report_structural_training.py
```

模型、数据与原始日志位于忽略目录 `runs/hjepa/structural_{data,model,trained_evaluation}/`。模型目录保存初始化 checkpoint SHA256、训练源码快照和数据 manifest 哈希。仓库只保存小型源码、说明与图表；权重与训练数组不纳入 Git。

## 定位表示的验证修正

原冻结策略追加优化的验证平均回报为 12.056。用同一 ViT 投影前 CLS 拟合定位策略后的验证回报为 9.165；本次是否采用：False。

该修正只更新原有定位 MLP，核心 H-JEPA、预测器、价值头和抓取策略逐张量不变。用训练回报构建温度 0.03 的 soft-label 排序监督，训练集特征归一化折叠进原 MLP 的首个仿射层；没有增加参数或引入几何真值。额外拟合 300 步，验证回报选中第 60 步，测试集未参与选择。

该候选未改善验证结果，最终保留原投影后表示；policy 前后比较使用相同的编码接口。验证期也试过冻结头回报回归、排序损失及俯视观察预演；最终方案按验证回报选择，没有将未改善的额外相机方案接入最终策略。这属于小验证集上的模型选择，仍需更多独立场景验证。

```powershell
.\.venv\Scripts\python.exe refit_structural_position.py
```

## 原有结构案例的补充检查

这些案例曾用于调试，属于补充检查，不合并进六个新保留场景的指标。两个隐藏单向限位案例有相同初始 RGB-D，没有给模型可见方向线索。

| 场景 | 新策略决策 | 结果 |
|---|---|---|
| free-two-way | front | full_open_and_hold |
| pillar-front | back | full_open_and_hold |
| pillar-back | front | full_open_and_hold |
| pillar-both | back | door_obstructed |
| one-way-positive | front | full_open_and_hold |
| one-way-negative | front | slip_or_contact_loss |
| front-access-only | front | full_open_and_hold |
| back-access-only | back | full_open_and_hold |
| front-only-blocked | back | perception_failure |
