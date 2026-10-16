# 结构性门约束实验（2026-10-17）

这些场景新增实体柱子、双向/单向铰链，以及正反面独立把手。柱子与门板通过 MuJoCo 接触产生阻挡；没有按场景名硬编码失败。正面拉开对应正铰链角，反面拉开对应负铰链角，底座在开门时固定。

**换站位不能绕过同一条受阻的门轨迹。** 双向门能改用反向转动；单向门若唯一允许的转动路径也被柱子挡住，则当前候选池没有完整开门解。

![实体场景终止状态](hjepa_structural_results.png)

图中统一为顶视图，蓝色机械臂、棕色门板、灰色实体柱子。每格展示对应侧的中性抓取终止状态；无可见把手时展示到达状态。

## 冻结 checkpoint 的实测

沿用官方完整模型的更新前 `best.pt` 与更新后 `selected.pt`。共享世界模型、价值头及归一化参数逐张量相同；此次未训练或按测试结果调参。候选池为两侧各两个底座、每个底座四个抓取，共四个底座。**反面底座、反向动作和新结构都在旧训练分布之外**，这些是能力检查，不是已完成适配的模型。

| 方法 | 成功/9 | Position → Align → Grasp → Open/Hold |
|---|---:|---|
| P_pre/G_pre | 4/9 | 7 → 7 → 7 → 4 |
| P_pre/G_post | 4/9 | 7 → 7 → 7 → 4 |
| P_post/G_pre | 4/9 | 8 → 8 → 8 → 4 |
| P_post/G_post | 4/9 | 8 → 8 → 8 → 4 |
| fixed-front/G_post | 4/9 | 8 → 8 → 8 → 4 |
| fixed-back/G_post | 4/9 | 7 → 7 → 7 → 4 |
| P_post/world/h0.25 | 4/9 | 8 → 8 → 8 → 4 |
| P_post/world/h0.5 | 4/9 | 8 → 8 → 8 → 4 |
| P_post/world/h1.0 | 4/9 | 8 → 8 → 8 → 4 |
| geometry-side | 7/9 | 9 → 9 → 9 → 7 |

## 哪一侧实际可行

| 场景 | 候选执行验证的可行侧 | 旧 P/G 结果 | 更新后 P/G 结果 | 已知几何选侧结果 |
|---|---|---|---|---|
| free-two-way | back, front | back: 成功 | front: 成功 | back: 成功 |
| pillar-front | back | back: 成功 | front: door_obstructed | back: 成功 |
| pillar-back | front | back: door_obstructed | front: 成功 | front: 成功 |
| pillar-both | 无完整开门解 | back: door_obstructed | front: door_obstructed | back: door_obstructed |
| one-way-positive | front | back: slip_or_contact_loss | front: 成功 | front: 成功 |
| one-way-negative | back | back: 成功 | front: slip_or_contact_loss | back: 成功 |
| front-access-only | front | back: perception_failure | front: 成功 | front: 成功 |
| back-access-only | back | back: 成功 | front: perception_failure | back: 成功 |
| front-only-blocked | 无完整开门解 | back: perception_failure | front: door_obstructed | front: door_obstructed |

已知几何的完整路径筛选：执行 7 个场景、成功 7 个；拒绝 2 个，其中正确拒绝 2 个，误拒绝 0 个。拒绝不计为开门成功。该参考读取已知几何/铰链范围，属于显式特权基线；这些信息未送入 learned policy 或 latent rollout。

## 物理含义和限制

- `pillar-front/back`：门板运动到约 40° 时接触柱子；只有另一转向可完成 90°。
- `pillar-both`：两个转向都受阻，单纯重新选底座或抓取不能完成。
- `one-way-positive/negative`：两侧都有把手，铰链限位只允许一个转向。错误侧的固定“拉门”技能可以夹住，但拉不动并出现滑移。这不代表从该侧使用另一种推门技能也绝对不可能；当前动作库只有拉门。
- `front/back-access-only`：只在指定面安装把手；另一面看不到把手。额外的几何诊断确认从反面穿过门板去抓真实把手会碰撞，此诊断没有作为感知失败的回退。
- `front-only-blocked`：单面把手、单向铰链与允许方向上的柱子共同构成无完整开门解的场景。
- 现有 learned selector 始终选一个动作，没有训练过拒绝/换侧策略，也没有校准可行性阈值。真实碰撞停止属于执行器保护，不应当作模型提前预测了阻挡。
- 到达后传感器切换到对应操作面，并刷新标定 RGB-D。移动到对侧仍是模拟器重置；没有实现绕门导航或自动重抓。

## 记录与复现

144 条候选评估，120 次执行器调用，120 次实际动力学积分。失败分类使用到达观察、当前 IK、接近、稳定夹持和最终开门/保持状态；同时保存了中性抓取在 0/2.5/3.5 s 和终止时的关节状态。

```powershell
.\.venv\Scripts\python.exe evaluate_hjepa_structure.py --device cuda
.\.venv\Scripts\python.exe run_hjepa.py --scenario pillar-front --base-mode fixed --fixed-base-id 2 --method geometry --output runs/hjepa/structural_demo
.\.venv\Scripts\python.exe -m unittest test_structural_scene test_hjepa test_hjepa_stages test_git_size -v
```

JSON、逐候选 CSV 与 source/checkpoint SHA256 位于忽略目录 `runs/hjepa/structural_evaluation/`。旧参数/角度报告是其当时源码版本的历史结果；重跑旧套件时如源码校验变化，应指定新的 `--output`，避免混用缓存。
