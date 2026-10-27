# 29角度扩展：小规模功能验证

本目录是已实际运行的 1/1/1 站位、87 条候选记录实验。用于验证角度扩展后的采集、SL/RL训练、模型加载和评估流程；不能代表完整72站位泛化表现。

| case | 用途 | 原21角度成功数 | 新增8角度成功数 | 准备失败 |
|---|---|---:|---:|---:|
| [train_000](train_000/README.md) | train | 8 | 3 | 1 |
| [validation_000](validation_000/README.md) | validation | 13 | 4 | 0 |
| [test_000](test_000/README.md) | test | 0 | 0 | 2 |

[评估指标](metrics.json) · [来源与源码哈希](provenance.json)

本次所有方法在唯一测试站位均未成功；4个去重后的测试动作重新执行，全部与记录匹配。训练与验证站位在原角度网格已有成功方案，因此新增角度成功不能解释为解救了新的站位。

数据采样和PRM seed=42，训练seed=42。原始数据和权重留在忽略目录，Git只保存本目录的小型结果摘要。中途判断/力调节尚未实现；完整72站位29角度实验尚未运行。

复现（使用新的输出目录）：

```bash
python collect_grasp_data.py --output reproduced/angle_diversity_recheck/data --train-scenes 1 --validation-scenes 1 --test-scenes 1 --workers 2 --angle-grid diverse29 --seed 42 --prm-seed 42
python compare_grasp_learning.py train --data reproduced/angle_diversity_recheck/data --output reproduced/angle_diversity_recheck/comparison --workers 2 --seeds 42
python compare_grasp_learning.py evaluate --data reproduced/angle_diversity_recheck/data --output reproduced/angle_diversity_recheck/comparison --workers 2
```

[角度实现与限制](../../docs/grasp_angle_diversity.md) · [所有案例](../README.md)
