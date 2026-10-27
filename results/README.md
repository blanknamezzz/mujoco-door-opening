# 参考结果：按 case 整理

前六组是原有参考实验，目录整理未修改结果。新增角度扩展实验以独立 case 保存小型摘要；完整训练数据和模型仍在忽略的 `reproduced/`，相关报告在 `docs/`。

| 文件夹 | 案例含义 | 结果 |
|---|---|---|
| [nominal/](nominal/) | 完整打开并保持 | [nominal.json](nominal/nominal.json) |
| [reach_only/](reach_only/) | 到达并抓住；未测试开门 | [reach_only.json](reach_only/reach_only.json) |
| [low_friction/](low_friction/) | 低摩擦导致滑移、脱手 | [low_friction.json](low_friction/low_friction.json) |
| [workspace_limit/](workspace_limit/) | 开到约 42.5°，下一参考目标超出关节范围 | [workspace_limit.json](workspace_limit/workspace_limit.json) |
| [unreachable/](unreachable/) | 初始位置够不到把手 | [unreachable.json](unreachable/unreachable.json) |
| [no_grip/](no_grip/) | 未形成抓握，无法带动门 | [no_grip.json](no_grip/no_grip.json) |

新增实验：[angle_diversity/](angle_diversity/README.md)，包含 `train_000`、`validation_000`、`test_000` 三个站位 case，以及29角度的结果摘要与来源哈希。

共享文件保留在本目录：

- [summary.json](summary.json)：六组结果、验证检查、判据与步长敏感性。
- [grasp_plan.json](grasp_plan.json)：六组实验共用的抓点选择。
- [observation.npz](observation.npz) / [observation.png](observation.png)：初始 RGB-D 观测。
- [handle_estimate.json](handle_estimate.json) / [handle_detection.png](handle_detection.png)：把手识别结果。

文件类型：`.json` 是结果或参数，`.csv` 是接触与运动日志，`.npz` 是保存的轨迹，`.gif` 是动画，`.png` 是静态画面。

脚本仍可接受 `--output results`（诊断脚本使用 `--results results`），会定位对应 case 子目录；也可直接传入某个 case 目录。旧的平铺 `reproduced/` 输出仍兼容，`run_demo.py` 默认输出方式不变。
