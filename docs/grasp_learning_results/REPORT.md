# 监督学习与强化学习：新站位实测对照

结果解释、失败案例与下一步建议见 [ANALYSIS.md](ANALYSIS.md)。本次 SL 未明显优于 RL，训练集选出的固定 +18° 同样达到 10/24 成功。

结论仅适用于这批场景与实现。新场景指同一扇门周围未参与训练的底座位置和朝向，不代表新门型、未知摩擦或真实机器人。

## 数据与公平性

- 训练 / 验证 / 测试场景数：36 / 12 / 24。所有角度按场景分组，划分先于仿真。
- 每场景 21 个角度：−20° 至 +20°、间隔 2°。物理仿真产生标签，无人工指定最佳角度。
- 每种主方法每个种子使用 756 次训练交互。训练种子：[42, 43, 44]。
- 同一网络规模、输入、归一化、控制器、奖励、场景访问次数。SL 全角度采样；RL 在线按当前策略采样，允许重复。
- 监督学习使用更多离线梯度更新，因此比较的是相同物理交互预算，不是相同优化步数或总计算量。
- 第三组用同一份 RL 记录训练监督模型，不增加物理试验，用于部分区分数据覆盖与学习目标的影响。
- 三组均使用最终权重，验证和测试数据不参与训练、调参、早停或权重选择。验证集仅作冻结后的独立诊断。

## 测试结果

24 个测试位置中有 10 个在候选网格内存在成功方案。
不能把网格内全部失败解释成连续角度空间或所有抓持方案都不可能成功。

| 方法 / 种子 | 全开成功 | 平均奖励 | 与网格最优的平均奖励差 | 碰撞次数 | 滑移次数 |
| --- | --- | --- | --- | --- | --- |
| 监督学习（完整角度采样） / 42 | 9/24 (37.5%) | 3.508 | 1.089 | 6 | 0 |
| 监督学习（完整角度采样） / 43 | 9/24 (37.5%) | 3.395 | 1.201 | 7 | 0 |
| 监督学习（完整角度采样） / 44 | 9/24 (37.5%) | 3.484 | 1.113 | 6 | 0 |
| REINFORCE（在线探索） / 42 | 10/24 (41.7%) | 3.856 | 0.740 | 7 | 0 |
| REINFORCE（在线探索） / 43 | 9/24 (37.5%) | 3.406 | 1.191 | 7 | 0 |
| REINFORCE（在线探索） / 44 | 9/24 (37.5%) | 3.341 | 1.255 | 7 | 0 |
| 监督学习（复用 RL 记录） / 42 | 9/24 (37.5%) | 3.061 | 1.536 | 10 | 0 |
| 监督学习（复用 RL 记录） / 43 | 8/24 (33.3%) | 2.243 | 2.354 | 12 | 0 |
| 监督学习（复用 RL 记录） / 44 | 8/24 (33.3%) | 2.537 | 2.059 | 10 | 0 |
| 固定 0° | 7/24 (29.2%) | 1.519 | 3.077 | 14 | 0 |
| 训练集选出的最佳固定角度 | 10/24 (41.7%) | 4.168 | 0.429 | 4 | 0 |

均匀随机角度的网格期望成功率：32.3%；穷举参考上限：41.7%。参考表不用于测试时选动作。

## SL 与 RL 的配对比较

- 种子 42：仅 SL 成功 0 个，仅 RL 成功 1 个；SL 覆盖 756 个训练位置–角度组合，RL 覆盖 438 个。
- 种子 43：仅 SL 成功 0 个，仅 RL 成功 0 个；SL 覆盖 756 个训练位置–角度组合，RL 覆盖 471 个。
- 种子 44：仅 SL 成功 0 个，仅 RL 成功 0 个；SL 覆盖 756 个训练位置–角度组合，RL 覆盖 446 个。

重复使用相同测试位置的多个种子不是更多独立场景样本。这里报告全部种子与配对计数，不宣称统计显著或某学习范式普遍优越。

## 可重放的例子

- rl_only_success：test_021，SL +10° / unexpected_collision；RL +18° / full_open_and_hold。
- both_success：test_000，SL +18° / full_open_and_hold；RL +6° / full_open_and_hold。

## 复现

```bash
python collect_grasp_data.py --output reproduced/grasp_learning_data_new
python compare_grasp_learning.py train --data reproduced/grasp_learning_data_new --output reproduced/grasp_learning_comparison_new
python compare_grasp_learning.py evaluate --data reproduced/grasp_learning_data_new --output reproduced/grasp_learning_comparison_new
```

用默认已生成模型观看某个场景：

```bash
mjpython compare_grasp_learning.py demo --scene test_000 --method supervised --seed 42
mjpython compare_grasp_learning.py demo --scene test_000 --method reinforce --seed 42
```

冻结后重新执行了 179 个去重后的测试位置–角度组合，全部与原始物理记录一致。

原始数据位于数据目录的 train/validation/test.csv 和 .jsonl；完整方法选择见 selections.json；指标见 evaluation.json；模型、训练历史与探索记录位于 seed_*。

局限：固定门族、固定抓点高度/力/速度、已知感知结果、预抓取初始化、离散角度，没有在线重抓或拒绝动作。SL 的输出是奖励预测值，不是校准成功概率。
实验设计参考：https://www.jmlr.org/papers/volume25/23-0183/23-0183.pdf
