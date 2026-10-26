# 完整抓握与开门方案

本次实现将一次尝试表示为 `(抓握高度, 接近角度, 肘关节分支, 运动时序, 夹爪驱动力上限)`。
`GraspPlan` 是可保存、加载和执行的方案；它不代表方案已经可行或最优。
目前只支持现有的平滑直线接近、夹持、绕已知门铰链圆弧开门和保持。

## Mac 上运行

在项目目录激活已有环境：

```bash
source /Users/andy/Documents/Codex/2026-10-15/zh/work/mujoco-venv/bin/activate
```

显式指定一个方案，使用包内 RGB-D 观测中的把手中心：

```bash
mjpython simulate.py --case nominal --view \
  --base 0.6 -1.0 --base-yaw-deg 90 \
  --grasp-height-offset 0.03 --grasp-angle-deg 0 \
  --ik-branch -1 --open-duration 14 --grip-force 25 \
  --output reproduced/plan_slow
```

把 `--grasp-angle-deg 0` 改成 `8` 就是在相同场景下尝试偏转 8° 的方案；
这仅指定候选，不保证能建立稳定抓握。换成 `--ik-branch 1` 可尝试另一肘关节分支，
执行过程中不会自动切换分支。不需要窗口时用 `python`，并去掉 `--view`。

未显式指定高度或角度时，继续使用原有几何基线选择的抓法；
只要指定了其中一个，另一个未指定的值取 0，直接执行该候选。
原有 `nominal`、`workspace_limit`、`low_friction` 等命令仍可使用。

## 方案与场景参数

| 参数 | 单位/范围 | 含义 |
| --- | --- | --- |
| `--grasp-height-offset` | 米 | 相对初始把手中心，正数向上 |
| `--grasp-angle-deg` | 度 | 相对当前操作侧的正常接近方向 |
| `--ik-branch` | -1 或 1 | 整段动作固定采用的肘关节逆解分支 |
| `--approach-distance` | 正数，米 | 抓取点到预抓取点的距离，默认 0.16 |
| `--grip-force` | (0, 35]，N | 每根手指的驱动器力上限，默认 35 |
| `--opening-angle` | (2, 90]，度 | 目标门开度，默认 90 |
| `--settle-duration` | 正数，秒 | 预抓取状态稳定时间，默认 0.5 |
| `--approach-duration` | 正数，秒 | 直线接近时间，默认 2 |
| `--close-duration` | 秒 | 闭合阶段总时长，默认 1 |
| `--close-motion-duration` | 正数，秒 | 闭合动作时间，默认 0.6 |
| `--open-duration` | 正数，秒 | 开门时间，默认 10 |
| `--hold-duration` | 至少 1 秒 | 完成开门后的保持时间，默认 2 |
| `--base X Y` | 米 | 覆盖案例的底座位置 |
| `--base-yaw-deg` | 度 | 底座朝向，默认 90 |
| `--friction` | 非负数 | 所有几何体的滑动摩擦系数 |
| `--hinge-resistance` | 非负数，Nm | 铰链摩擦阻力，默认 1.2 |

闭合阶段总时长至少为闭合动作时间加 0.2 秒，给接触建立留出时间。
力上限不是目标接触力，也不是已经测得的实际夹持力。实际左右接触力记录在 CSV 中。
总运行时长会随各阶段参数改变；成功条件仍要求最后一秒门角度稳定、双侧接触保持、
没有滑移告警、接触丢失或碰撞。夹爪开口目标和接触判据沿用现有控制器。

## 保存与重放

每次通过 `simulate.py` 运行时保存：

- `nominal_execution_plan.json`：可直接加载的完整动作方案。
- `nominal.json`：结果、方案、底座位置、摩擦和铰链参数，以及观测来源。
- `nominal.csv`、`nominal.npz`：接触日志、关节轨迹及时间；初始不可达时不生成轨迹。

重放同一个方案：

```bash
mjpython simulate.py --plan reproduced/plan_slow/nominal_execution_plan.json \
  --view --output reproduced/plan_replay
```

也可直接加载仓库中的示例：

```bash
mjpython simulate.py --plan examples/grasp_plan_slow.json \
  --view --output reproduced/plan_example
```

示例中的中心来自已知标称几何，不是新的视觉检测结果。
方案文件保存的是初始世界坐标和动作参数，**不包含场景配置**；
若原来修改了底座、摩擦或铰链阻力，重放时须传入同样的场景参数。
加载方案不会重新识别把手；移动把手后必须重新观测、生成新的方案。
`--plan` 不能与抓握/时序/力参数覆盖选项或 `--oracle` 混用，以避免执行的方案与文件不同。

当前单案例默认使用包内的固定 RGB-D 观测，并非每次启动都实时重新检测。
当前仍重置机械臂到预抓取姿态，不包含从任意初始关节状态出发的路径规划。

## 程序接口与学习模块

```python
from grasp_plan import GraspPlan, SkillTiming
from simulate import Experiment

plan = GraspPlan(
    handle_center_world_m=(0.8, -0.11, 0.85),
    height_offset_m=0.03,
    approach_angle_deg=0,
    ik_branch=-1,
    grip_force_limit_N=25,
    timing=SkillTiming(open_s=14),
)
result, rows, states = Experiment().trial(plan=plan)
plan.save('reproduced/my_plan.json')
```

`skill_target` 和 `hjepa_scene.action_at` 可使用同一 `SkillTiming` 与预抓取距离。
原有学习流程保持默认时序和候选集合，其动作特征与旧模型兼容；
这次没有将新参数加入网络输入，也没有重新训练模型。
后续可围绕 `GraspPlan` 采集不同角度、分支、时序和力上限的物理结果，再扩展选择器。
现在尚未实现连续候选搜索、完整轨迹可行性证明、最优方案搜索或在线纠错。

## 验证

```bash
python -m unittest test_grasp_plan -v
python benchmark.py --output reproduced/plan_regression
```

新增测试覆盖方案保存/加载、无效参数、默认轨迹兼容、自定义时间、抓握位置/角度、
肘关节分支、实际驱动力限幅和命令行重放。批量回归使用原有六个物理案例及八项验证。
