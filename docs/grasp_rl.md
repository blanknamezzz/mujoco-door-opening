# 用强化学习选择抓取角度

入口是 `train_grasp_rl.py`。新增代码调用现有 `Experiment.trial()`，不修改机器人模型、接触判据或底层控制器。只需要现有 requirements.txt 的依赖；小型神经网络与 Adam 用 NumPy 实现。

仓库不分发生成的权重和大数据；先训练再演示。以下 `grasp_rl_new` 路径由训练命令生成。

## 快速开始

在项目根目录执行。以下示例使用仓库内的 `.venv` 环境：

```bash
# 在仓库根目录运行，并先激活已安装 requirements.txt 的环境
source .venv/bin/activate

# 从头学习，256 次实际物理试验，随后独立扫描所有角度作为参考。
python train_grasp_rl.py train --episodes 256 --sweep --output reproduced/grasp_rl_new

# 重新加载最终权重，执行所选角度与固定 0°，不更新权重。
python train_grasp_rl.py evaluate --checkpoint reproduced/grasp_rl_new/policy.npz --output reproduced/grasp_rl_new_eval

# macOS 图形演示：先执行学到的角度，再执行固定 0°。
mjpython train_grasp_rl.py demo --checkpoint reproduced/grasp_rl_new/policy.npz

# 只展示固定角度的对照。
mjpython train_grasp_rl.py demo --checkpoint reproduced/grasp_rl_new/policy.npz --fixed-angle 0
```

训练目录必须为空，防止覆盖历史实验。保存的权重用于推理；当前未实现断点续训。不同运行应使用不同输出目录。图形界面需要桌面环境；训练和评估不需要打开窗口。

## 它到底学习什么

这是单次高层动作的 episodic RL，也可以称为 contextual bandit。每个 episode：

1. 读取抓持前的底座与把手几何信息。
2. 神经网络输出各个接近角度的概率，按概率随机抽一个角度。
3. 重置仿真到该方案的预抓取姿态，执行接近、闭合、开门、保持；实际使用 `mj_step`，没有把手与夹爪焊接。
4. 根据真实执行结果计算奖励。
5. 每 8 个 episode 做一次 REINFORCE 策略梯度更新。

默认底座 (0.8, -1.0) m，角度范围 [-20°, 20°]、间隔 2°，共 21 个动作。所有角度初始概率相同，没有写死 +8° 或其他正确答案。角度表示绕竖直方向相对于默认接近姿态的偏角，暂不学习俯仰和侧倾。

网络为 4 → 16（tanh）→ 21（softmax）。输入为把手相对于底座的水平位移 dx、dy，以及底座朝向的 sin、cos；把手位置来自已有 RGB-D 检测结果。网络不接收执行后的角度或失败标签作为输入。高度、力上限、速度、摩擦和肘部解在一次训练中固定，不作为可变环境输入。默认只学习一个固定站位，不能声称泛化到任意位置。

策略优化目标为 `mean((R - b) / 10 * log π(a|s) + β H(π))`。`b` 为之前批次奖励的指数滑动平均，不使用本批次动作的结果预先设置 baseline；β 默认 0.02，用熵奖励鼓励探索。每一批都由当时的策略采样实际新轨迹，只有一次更新，没有缓存奖励查表，也没有监督“最佳角度”标签。

算法依据：[策略梯度与 baseline](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)。本实现是简化的 REINFORCE，不是 PPO，也不是世界模型或逐时刻关节控制 RL。

## 奖励

`R = 10S + G + 2P - 5C - 4L - 2W - 2U - 3V - 0.1T`

| 项 | 含义 |
| --- | --- |
| S | 安全全开并保持，由原仿真成功判据决定 |
| G | 建立稳定抓持 |
| P | 首次违规之前的最大安全开门角度 / 90°，截断到 [0,1] |
| C | 非预期接触 / 碰撞 |
| L | 检测到滑移或接触丢失 |
| W | 后续关节目标超出工作空间 |
| U | 初始抓持点或预抓取位置不可达 |
| V | 最终开门角度不足比例 |
| T | 实际执行时间 / 计划时间 |

上述奖励是当前版本的设计选择，不声称与 PPT 公式完全相同。滑移/工作空间失败有单独惩罚。即使最终接近 90°，碰撞也得不到成功奖励。奖励优化不是安全保证；执行阶段原有停止机制仍然生效。

## 调整训练条件

```bash
# 更细的角度离散化；增加训练次数。
python train_grasp_rl.py train --angle-min -20 --angle-max 20 --angle-step 1 --episodes 1024 --output reproduced/grasp_rl_fine

# 改变固定站位。
python train_grasp_rl.py train --base 0.6 -1.0 --output reproduced/grasp_rl_base06

# 在中心站位周围随机采样，每个坐标独立均匀扰动。
python train_grasp_rl.py train --base 0.8 -1.0 --base-jitter 0.05 0.03 --episodes 1024 --output reproduced/grasp_rl_jitter

# 用未参与训练的位置检查泛化，权重保持冻结。
python train_grasp_rl.py evaluate --checkpoint reproduced/grasp_rl_jitter/policy.npz --base 0.83 -0.99 --sweep --output reproduced/grasp_rl_unseen
```

也可以固定 `--height`、`--ik-branch`、`--grip-force`、`--open-seconds`、`--friction`、`--hinge-resistance` 再训练；它们暂时不是策略动作。夹持力参数是每指执行器驱动力上限，不等于实际接触法向力。改变这些参数需要重新训练，不能假设现有网络自动适应。

## 如何检查结果

- `config.json`：场景、随机种子、动作集合、训练设置和源文件哈希。
- `policy.npz`：最终网络权重与配置；`initial_policy.npz` 为初始权重。格式使用 NumPy 数组，不需要 pickle。
- `episodes.csv`：每次动作、奖励、是否成功、最终角度及失败原因。
- `episodes.jsonl`：详细奖励分解和执行结果。
- `updates.json`：每批次奖励、成功比例及中心站位的完整动作概率分布。
- `evaluation/evaluation.json`：冻结最终策略的实际执行，与固定 0° 对照。
- `evaluation/learned_plan.json`：可直接传给现有 `simulate.py --plan` 的方案。
- `evaluation/learned.csv`、`.npz`：最终策略的接触轨迹和关节状态。

`--sweep` 在训练完成后才评估所有候选，不参与更新或选择权重。它报告有限网格上最高奖励的角度、均匀随机选择的成功比例，可用于判断训练策略是否接近网格参考；不是连续角度空间的全局最优证明。

固定场景一次成功不是“测试成功率 100%”。应另外做多训练种子、独立场景测试，并比较相同物理交互预算下随机搜索、几何筛选与 RL。默认固定场景确定性很强，扫描可能比 RL 更简单；这份文件主要建立可运行、可观察的强化学习基线。

还没有实现任意初始关节姿态路径规划、在线重抓、连续角度策略、不可行拒绝动作，以及 world-model foresight。若所有候选均失败，策略仍会输出其中一个角度；必须查看评估结果，不能把推荐当成可行性承诺。

## 测试

```bash
python -m unittest test_grasp_rl -v
```

测试包括完整网络的有限差分梯度检查、风险奖励检查、模型保存加载一致性，以及在独立已知奖励问题上的策略学习方向。物理有效性仍以 MuJoCo 实际评估为准。
