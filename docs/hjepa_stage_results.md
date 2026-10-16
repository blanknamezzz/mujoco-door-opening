# H-JEPA 分阶段与角度实验（2026-10-17）

官方完整模型保持冻结，共 **504 条评估记录**：360 条场景候选、84 条执行角度/目标角度、60 条相机视角记录。其中 482 次调用执行器，389 次实际积分动力学；其余在视觉或初始 IK 阶段被拒绝，仍计为失败，不伪装成完成过的动力学执行。

新随机场景 8 个，seed=20261017；另有 9 个物理参数压力场景及 1 个无夹持对照。相机实验复用其中两个新场景，角度扫描复用四个场景。记录之间共享场景，不能作为独立场景数。

定位 P 与抓取 G 各有更新前/后两个版本：`best.pt` 的监督预热策略，以及 `selected.pt` 的 100 次交替离线更新策略。逐张量检查确认世界模型、价值头及归一化参数完全一致。候选执行结果仅用于事后评估，未用于测试时排序、训练或调参。

![分阶段与角度实验](hjepa_stage_results.png)

## 两个 policy 的独立作用

| 组合 | Position | Align | Grasp | Open + hold | 成功率 | 安全开度比例 |
|---|---:|---:|---:|---:|---:|---:|
| P_pre/G_pre | 18 | 18 | 17 | 0 | 0.0% | 0.421 |
| P_post/G_pre | 18 | 18 | 17 | 14 | 77.8% | 0.831 |
| P_pre/G_post | 18 | 18 | 17 | 0 | 0.0% | 0.421 |
| P_post/G_post | 18 | 18 | 17 | 14 | 77.8% | 0.831 |
| fixed/G_post | 18 | 18 | 17 | 14 | 77.8% | 0.830 |
| P_post/world/h0.25 | 18 | 18 | 17 | 14 | 77.8% | 0.831 |
| P_post/world/h0.5 | 18 | 18 | 17 | 14 | 77.8% | 0.831 |
| P_post/world/h1.0 | 18 | 18 | 17 | 14 | 77.8% | 0.831 |
| P_post/geometry | 18 | 18 | 17 | 14 | 77.8% | 0.831 |

上表分母均为 18 个场景；Position 指当前抓点和预抓点 IK 可达，Align 指 2.5 s 接近完成且此前没有碰撞，Grasp 指稳定双侧夹持，Open + hold 指达标并稳定保持一秒。成功率不能用“成功夹住的次数”替代。

| 场景组 | P_pre/G_pre | P_post/G_pre | P_pre/G_post | P_post/G_post |
|---|---:|---:|---:|---:|
| negative_control | 0/1 | 0/1 | 0/1 | 0/1 |
| new_in_range | 0/8 | 8/8 | 0/8 | 8/8 |
| stress | 0/9 | 6/9 | 0/9 | 6/9 |

固定每个到达状态、画面和底座后，抓取 policy 的对照如下（18×5=90 个到达状态，包含上游失败）：

| 抓取版本 | Position | Align | Grasp | Open + hold |
|---|---:|---:|---:|---:|
| pre | 69 | 69 | 65 | 28 |
| post | 69 | 69 | 65 | 28 |

配对效应（一次只替换一个 policy）：

| 替换 | 成功率变化（百分点） | 改善/退化场景 | 决策改变场景 |
|---|---:|---:|---:|
| position_update_with_G_pre | +77.8 | 14/0 | 18 |
| grasp_update_with_P_pre | +0.0 | 0/0 | 18 |
| position_update_with_G_post | +77.8 | 14/0 | 18 |
| grasp_update_with_P_post | +0.0 | 0/0 | 18 |

更新后的定位 policy 在这 18 个场景中均选择底座 [3]。固定名义底座 0 同样成功 14/18；因此本次改善说明修正了坏站位偏好，尚未证明优于固定好站位，或具备广泛的自适应定位能力。

仅用物理结果事后检查候选覆盖率：14/18 个场景至少有一个候选能成功。这不是可部署的 oracle policy，也不参与选动作。

## 更新后仍然失败的场景

| 场景 | 停止原因 | 最终角度 | 失败阶段 |
|---|---|---:|---|
| low_friction_005 | slip_or_contact_loss | 0.001° | opening |
| hinge_4 | slip_or_contact_loss | 63.114° | opening |
| hinge_8 | slip_or_contact_loss | 27.638° | opening |
| no_grip_control | grasp_not_acquired | 0.000° | grasp |

## 角度与视觉

底座 yaw 偏移 -45°～45°，抓取 yaw 偏移 -30°～30°；目标开度 30°、60°、90°。部分开度测试修改实体控制轨迹，要求最后一秒角度误差不超过 2°、持续双侧接触及低角速度。它们只评估控制器；当前 H-JEPA 没有训练为目标角度条件模型。

| 扫描 | 角度 | 成功/记录 | 平均最终角度 |
|---|---:|---:|---:|
| base_yaw | -45° | 0/4 | 44.51° |
| base_yaw | -30° | 3/4 | 67.24° |
| base_yaw | -15° | 3/4 | 67.24° |
| base_yaw | 0° | 3/4 | 67.24° |
| base_yaw | 15° | 3/4 | 67.24° |
| base_yaw | 30° | 3/4 | 67.24° |
| base_yaw | 45° | 3/4 | 67.24° |
| grasp_yaw | -30° | 3/4 | 67.34° |
| grasp_yaw | -15° | 3/4 | 67.27° |
| grasp_yaw | 0° | 3/4 | 67.24° |
| grasp_yaw | 15° | 3/4 | 67.25° |
| grasp_yaw | 30° | 3/4 | 67.33° |
| target_base0 | 30° | 3/4 | 22.30° |
| target_base0 | 60° | 3/4 | 44.75° |
| target_base0 | 90° | 3/4 | 67.25° |
| target_base1 | 30° | 3/4 | 22.31° |
| target_base1 | 60° | 0/4 | 32.15° |
| target_base1 | 90° | 0/4 | 31.87° |
| target_base3 | 30° | 3/4 | 22.30° |
| target_base3 | 60° | 3/4 | 44.77° |
| target_base3 | 90° | 3/4 | 67.24° |

![相机角度成功率](hjepa_stage_results_views.png)

| 相机测试方法 | 成功/记录 | 失败阶段 |
|---|---:|---|
| direct | 25/30 | {'arrival_perception': 5} |
| world-model | 25/30 | {'arrival_perception': 5} |

## 结论范围与复现

这些结果支持定位更新改善下游可行性；本次新随机场景中，单独更新抓取 policy 没有提高完整开门率。固定底座、direct 与 geometry 也是必要参照；当前结果不能据此宣称世界模型优于这些方法。摩擦与铰链阻力在初始 RGB 中不可见，当前单帧观察没有接触后的材料辨识信息；相机测试失败也应按视觉边界失败计入，不能记为抓取或开门能力。

没有用这批结果调整权重或重新训练。只有一个训练种子、八个新随机场景，压力场景有意构造，尚不足以给出通用泛化或统计显著性结论。下一轮改模型后应使用新的测试种子。

```powershell
.\.venv\Scripts\python.exe evaluate_hjepa_stages.py --device cuda
.\.venv\Scripts\python.exe evaluate_hjepa_views.py --device cuda
.\.venv\Scripts\python.exe report_hjepa_stages.py
.\.venv\Scripts\python.exe check_git_size.py
```

详细 JSON、逐候选物理结果、checkpoint/source SHA256 均位于忽略目录 `runs/hjepa/{stage_evaluation,view_evaluation}/`。已完成 14 项代码测试和原始基线 8 项回归检查。Git 检查默认拒绝超过 10 MiB 的可见文件及暂存 blob，模型/数据/视频输出不进入版本库。
