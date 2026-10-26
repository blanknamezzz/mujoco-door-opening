# 在训练前增加 PRM：先检查能否到达预抓取位置

新的默认流程是：初始站位与关节姿态 → PRM 预检查 → 实际执行准备路径 → 接近/抓持 → 开门/保持 → 记录结果。

## 已接入的位置

- `collect_grasp_data.py`：每组场景先为每个站位的全部候选做 PRM 分析，保存 `*_prm_preflight.jsonl`，然后才执行候选试验。
- `train_grasp_rl.py`：每个被选择的角度必须先经过 PRM 和实际准备路径跟踪；默认开启，`--no-prm` 显式运行原来的预抓取初始化基线。
- `compare_grasp_learning.py`：读取数据 manifest 中的 PRM 设置，SL、RL、验证和测试都使用同一准备流程；旧 manifest 没有 PRM 设置时仍可运行旧基线。
- `prm_examples.py`：五个可复现例子、独立 MuJoCo 实时演示和离线交互重放。

新默认目录为 `reproduced/grasp_learning_data_prm`、`reproduced/grasp_learning_comparison_prm` 和 `reproduced/grasp_rl_prm`，避免覆盖先前无 PRM 的实验。

## PRM 实际分析什么

固定一个底座位置与朝向，夹爪张开、门关闭。节点是四维配置：

`q = [lift_m, shoulder_rad, elbow_rad, wrist_rad]`

默认初始关节姿态为 `[0 m, −90°, −120°, 0°]`。这是明确指定的初始状态，不能把它称为任意初始姿态；可用 `--initial-q` 修改。在每一轮学习内初始姿态固定，网络仍以站位信息为输入，尚未学习跨不同初始关节状态的泛化。

对一个抓持角度，先根据把手点和接近方向计算距离把手 0.16 m 的预抓取目标，再求所选肘分支的逆运动学。没有该解时返回 `goal_ik_unreachable`，不进入随机采样。

### CLEAR：配置有效性

检查关节限制，设置独立的探测 MjData，读取 MuJoCo 接触。为避免将深度穿过门板的初始姿态误认成自由空间，另用线段–有向盒距离检查机械臂胶囊与门板。地面和门板的零距离支撑接触不算穿透。容许穿透误差为 1e-5 m。

探测数据只用于几何检查，不修改执行中的机器人状态。

### LINK：边有效性

在关节空间插值，相邻检查点的升降变化不大于 0.008 m，各旋转关节变化不大于 0.06 rad。每个中间点调用 CLEAR。关节有物理限位，因此不会把 +165° 到 −165° 通过圆周最短角度直接绕过去。

这仍是离散碰撞检查，不是连续碰撞检测或严格安全保证。随后还要进行动力学执行监测。

### 采样、连边与搜索

先尝试起点到目标的直接 LINK。通过时直接返回，随机节点数为零是正常结果。

直连失败后，在四个关节范围内随机采样；默认保留 240 个有效节点、连接 12 个近邻。距离计算对升降和旋转坐标做尺度归一化。构造近邻候选边后，以 Dijkstra 寻路，只验证搜索路径上的边；无效边被删除，再搜索。这是 Lazy PRM 的实现方式，不是预先验证全部边的 BasicPRM。只有最终路径的每条边都通过检查，才返回 `path_found=true`。

相同静态场景和设置的路线图可在一个工作进程内复用；改变底座、模型或规划设置会使用不同缓存。有限预算没有找到路线记作 `no_path_within_budget`，不记作绝对不可行。

## 路径找到后如何执行

每个 episode 只在初始时将机器人重置到指定初始关节姿态。之后根据路线图的关节路标，用平滑插值和现有位置伺服执行 `mj_step`；仅对机器人关节做偏置力补偿，门没有驱动或外力。

执行中监测非预期接触和门是否提前移动，末端路标还检查关节跟踪误差与速度。成功到达后保留实际 qpos/qvel，将技能时钟归零，调用 `trial(reset_before_trial=False)` 接着抓持和开门。没有把机器人瞬移到预抓取点。

准备阶段耗时、轨迹和失败原因单独记录。奖励仍采用原开门阶段的奖励与时间项；准备失败的流程奖励为负值，准备耗时尚未纳入原奖励的时间惩罚。

## 三类结论必须分开

| 结果 | 含义 |
| --- | --- |
| `goal_ik_unreachable` | 当前预抓取目标和肘分支不存在合法逆运动学解 |
| `invalid_start` / `invalid_goal` | 当前指定端点碰撞或越界 |
| `no_path_within_budget` | 有限路线图未连通，其他预算或方法可能找到 |
| `direct_path` / `roadmap_path` | 找到了通过当前离散几何检查的预抓取路径 |
| 准备执行失败 | 几何路线存在，但当前控制器没有安全执行完成 |
| 开门失败 | 已到预抓取位置，抓持或之后开门过程失败 |

准备执行中碰撞会写入结果并计入流程碰撞次数；即使没有开门轨迹也可以导出 CSV。

数据中的 `opening_tested=false` 表示未执行开门，不应把这种记录解释成经过物理验证的“开门必然失败”。监督标签中的负奖励表示当前规划预算与控制流程的结果；如果要训练纯接触成功预测器，应排除未执行开门的样本。RL 对应的也是当前端到端流程奖励。

PRM 还没有对抓住之后的机器人–门闭环约束进行规划。底座保持固定，开门仍执行原铰链弧线控制器。因此，不能用 `path_found` 代替全开成功标签。

## 五个例子

```bash
python prm_examples.py
# 打开 reproduced/prm_examples/index.html

# 在 MuJoCo 窗口中完整观看准备路径与开门：
mjpython prm_examples.py --view detour
mjpython prm_examples.py --view partial
```

| 例子 | 站位 | 要观察的区别 |
| --- | --- | --- |
| A：direct | (0.6, −1.0) | 直连通过，无需随机建图，随后全开 |
| B：detour | (0.6, −1.0)，增加小障碍 | 直连被挡，采样 500 个有效节点后，通过降低现有升降轴绕行，随后全开 |
| C：limited_budget | 与 B 完全相同 | 零采样预算只检查直连，未找到；B 已证明这不等于绝对不可行 |
| D：partial | (0.3, −0.9) | 找到并执行到预抓取位置的路径，之后约 42.5° 处受工作空间限制 |
| E：unreachable | (2.0, −1.2) | 当前预抓取目标无 IK，跳过运动与开门 |

图上的灰边尚未全部验证，绿线是最终已检查的路径。肩–肘二维投影隐藏升降和腕角，线条交叉不代表四维节点相交。右侧是实际物理积分记录的关节姿态重放，不是仅按规划路标插值的成功动画。

## 训练与独立测试

完整默认规模：

```bash
python collect_grasp_data.py
python compare_grasp_learning.py train
python compare_grasp_learning.py evaluate
```

本次小规模验证使用新的场景划分：

```bash
python collect_grasp_data.py --train-scenes 12 --validation-scenes 4 --test-scenes 8 --output reproduced/prm_training_data_final
python compare_grasp_learning.py train --data reproduced/prm_training_data_final --output reproduced/prm_training_comparison_final --seeds 42
python compare_grasp_learning.py evaluate --data reproduced/prm_training_data_final --output reproduced/prm_training_comparison_final
```

共 24 个站位、504 次候选流程；RL 使用 252 次训练决策。这里只验证新准备步骤确实接入并可复现，不能与上一版 36/12/24、三个种子的结果直接比较优劣。

PRM 拒绝的候选会跳过实际开门，所以预算记录为候选决策次数；`opening_trials_executed` 和准备失败次数另外统计，不应把所有决策都称为实际开门试验。

修改起始关节姿态示例：

```bash
python collect_grasp_data.py --initial-q 0 -1.57 -2.09 0 --output reproduced/prm_custom_start
```

单位依次是米、弧度、弧度、弧度。训练后需要在同样的初始状态约定下评估。

数值与物理回归：`python -m unittest test_prm test_grasp_plan test_grasp_learning test_grasp_rl test_git_size -v`。
