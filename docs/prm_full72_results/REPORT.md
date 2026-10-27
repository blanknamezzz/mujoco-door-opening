# PRM 完整 72 站位实验（2026-10-27）

本次重新采集 36 / 12 / 24 个训练 / 验证 / 测试站位，使用与早期完整实验完全一致的坐标、朝向、划分和 21 个候选角度。旧的小规模 24 站位数据保留。所有成功与失败记录均保留，不按结果筛选站位。

## 数据规模

| 划分 | 站位 | 候选记录 | 准备失败 | 成功开门记录 |
|---|---:|---:|---:|---:|
| train | 36 | 756 | 0 | 251 |
| validation | 12 | 252 | 0 | 110 |
| test | 24 | 504 | 0 | 163 |

完整坐标、朝向和每个站位的候选结果见 [positions.csv](positions.csv)。一条记录代表一个站位和一个角度的完整流程。候选成功次数不是策略测试成功率。

## 三个训练种子的独立测试

| 方法 | 各种子成功数（每次 /24） | 平均成功率 |
|---|---|---:|
| supervised | 9, 9, 9 | 37.50% |
| reinforce | 10, 9, 9 | 38.89% |
| supervised_on_rl | 9, 8, 8 | 34.72% |
| 训练集选择的固定 +18° | 10/24 | 41.67% |

训练种子为 [42, 43, 44]，每种方法、每个种子 756 次训练决策。SL 对固定训练记录拟合；RL 按策略探索，可重复访问同一站位–角度组合。归一化、固定角度选择和模型拟合均只使用训练数据，测试前冻结权重。

测试集中有 10/24 个站位在本次角度网格和控制器下存在全开方案。这不是全局物理可行性上界。三个种子重复使用同一组 24 个测试站位，不能视为 72 个独立测试站位。

冻结后的动作按站位–角度去重，共重新物理执行 179 组，全部与参考记录一致。

## 路径准备与完整开门

候选 PRM 状态统计：`{'direct_path': 1512}`。PRM 检查的是到预抓取位置的路径，完整开门仍受碰撞、工作空间和接触稳定限制。

## 复现

```bash
python collect_grasp_data.py --output reproduced/prm_full72_data_20261027 --train-scenes 36 --validation-scenes 12 --test-scenes 24 --seed 20261026 --workers 4
python compare_grasp_learning.py train --data reproduced/prm_full72_data_20261027 --output reproduced/prm_full72_comparison_20261027 --seeds 42 43 44 --workers 4
python compare_grasp_learning.py evaluate --data reproduced/prm_full72_data_20261027 --output reproduced/prm_full72_comparison_20261027 --workers 4
```

已有数据目录使用采集器的 `--resume`，训练则选择新的空输出目录。完整原始记录、权重、训练历史和逐动作验证保存在上述 reproduced 目录。

## 结论边界

本次与早期实验使用相同站位和名义物理参数，但 simulate.py、训练和采集程序的旧版本哈希不同，且初始化方式发生变化，不能将结果差异全部归因于 PRM。PRM 本身的净增益需要同一当前代码、相同站位和预算下的有无 PRM 配对实验。当前仍只有一个门族、固定初始关节姿态，没有证明真实机器人或新门型泛化。

核对记录见 [verification.json](verification.json)。
