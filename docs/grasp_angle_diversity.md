# 抓取角度多样性（2026-10-27）

新实验默认从 21 个离散抓角扩展到 29 个。角度仍表示绕竖直轴相对默认接近姿态的偏转，不是门的开度，也不是俯仰、侧倾或某一个关节的角度。

| 区域 | 候选角度 |
|---|---|
| 原细粒度区间 | −20° 到 +20°，间隔 2°，共 21 个 |
| 新增较大偏转 | −90°、−60°、−45°、−30°、+30°、+45°、+60°、+90° |

这是一组更广的探索候选，不代表这些角度一定可抓或可开门。先执行 PRM 预检和准备，再由实际 MuJoCo 接触、碰撞与开门结果产生标签；准备失败仍记录，不删除失败样本。

## 使用

```bash
# 默认29角度，72个站位，共2088条候选记录：1044训练、348验证、696测试。
python collect_grasp_data.py --angle-grid diverse29 --output reproduced/angles29_data
python compare_grasp_learning.py train --data reproduced/angles29_data --output reproduced/angles29_models --seeds 42
python compare_grasp_learning.py evaluate --data reproduced/angles29_data --output reproduced/angles29_models

# 自定义离散集合，采集器要求包含0度以保留固定0度基线。
python collect_grasp_data.py --angles-deg -60 -30 -10 0 10 30 60 --output reproduced/custom_angle_data

# 均匀范围，包含两个端点；步长必须整除范围。
python collect_grasp_data.py --angle-min -60 --angle-max 60 --angle-step 5 --output reproduced/uniform_angle_data

# 单独的在线REINFORCE训练入口也使用相同角度选项。
python train_grasp_rl.py train --angle-grid diverse29 --output reproduced/angles29_rl

# 历史采样设置；使用新目录，不能向旧数据追加不同角度。
python collect_grasp_data.py --angle-grid legacy21 --seed 20261026 --prm-seed 7 --output reproduced/legacy21_new_run
```

角度预设、显式列表、均匀范围三种方式互斥。候选必须有限且不重复。新默认输出目录带 `diverse29`；旧数据、模型和报告未改写。

SL 和 RL 的网络均按 manifest 动态建立 `4 → 16 → K`；SL 输出 K 个奖励预测，RL 输出 K 个动作概率。老 checkpoint 按其保存的角度列表加载，不能用29角度的动作索引解释21角度权重。扩大动作空间后应重新训练。恢复采集时以原 manifest 为准，显式给出不同角度集合会被拒绝；源码哈希检查仍有效。

GitHub可查看 [结果与三个站位case](../results/angle_diversity/README.md)。

## 实际验证

- 47 项回归测试通过，涵盖原候选保留、非法角度拒绝、两个入口一致、旧/新模型加载和续采动作索引保护。
- 小规模真实流程：seed=42，训练/验证/测试各1个站位，共87条候选记录；84次进入开门试验，3次准备失败。总耗时 25.10 秒。
- SL、REINFORCE、复用RL记录的SL均完成训练，配置正确记录29输出；冻结后 4 个去重测试动作重新执行，全部匹配。
- 训练站位新增 +30°、+45°、+60° 成功；验证站位新增 +30°、+45°、+60°、+90° 成功。它们说明较大偏转是可执行的候选，并不证明新增站位被解救，因为这些站位原网格也有成功方案。
- 该测试站位29个角度都未成功。这里仅验证数据、训练、评估的接口与物理流程，样本不足以比较泛化成功率，也未重跑完整72站位实验。
- 原始验证文件保留在 `reproduced/angle_diversity_smoke_20261027/`，不上传生成数据或模型。

## 下一步与限制

角度仍是一次性选择，抓点高度、肘分支、夹持力和轨迹固定；新增侧向接近不等于增加机械臂自由度，也不提供闭环纠错。大偏转会带来更多碰撞、IK或PRM失败，不能视为无条件提升。

29/21≈1.38，因此相同站位数下的全角度采集量与配对RL决策预算增加约38%。比较旧21角度与新29角度时，应匹配场景、物理参数与PRM设置，并区分动作空间扩大和预算增加的贡献；需要另做等预算对照。历史测试集已被查看，最终泛化结论应使用新冻结测试集。下一步应先看新角度能否使更多站位出现成功方案，再评价神经网络能否选中这些方案。
