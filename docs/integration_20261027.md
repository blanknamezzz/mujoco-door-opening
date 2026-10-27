# 2026-10-27 合并记录

合并远端 `6eefbb0`（H-JEPA 训练与评估更新）和本地 `57b3045`（站位诊断、完整 PRM 结果报告、案例目录整理）。README 与采集器自动三方合并成功。

## 合并行为

- 保留 `station_feasibility.py`、`station_workspace_check` 及其源码哈希记录；不自动筛除站位，不修改奖励。
- 采用远端新实验默认数据采样种子 42、PRM 种子 42、SL/RL 训练种子列表 `[42]`。
- `results/` 的 30 个原文件按六个 case 移动，内容与远端参考文件一致；共享观测保留原路径。渲染、诊断和复现入口兼容案例目录与旧平铺目录。
- 不重写历史实验报告和核对记录。历史数据采样种子为 20261026，PRM 种子为 7，三种子比较为 42/43/44；当前代码下沿用这些设置的命令见主 README。
- H-JEPA 与独立 PRM/抓角学习仍是各自的流程；本次代码合并不是端到端策略集成实验。

## 本次实际验证

```bash
python -m unittest test_station_feasibility test_grasp_learning test_prm test_grasp_plan test_grasp_rl test_git_size
python check_git_size.py
```

- 42 项测试通过，约 6.48 秒，包括新默认 seed=42、历史三种子显式选择、站位诊断与采集预检。
- 所有根目录 Python 源文件均通过语法解析。原有 Windows 命令字符串存在无效转义警告，本次未改写这些无关字符串。
- 合并前后 392 个历史报告、数据及模型文件的 SHA256 完全一致。
- Git 文件大小检查通过；未加入 `reproduced/` 或 `runs/` 中的数据、模型、日志。已有参考轨迹仅移动路径。
- CSV 保留原有 CRLF；差异检查使用 `git -c core.whitespace=cr-at-eol diff --check`。

本机 MuJoCo 虚拟环境及检查过的 Conda 环境均无 PyTorch，未重新运行完整 H-JEPA 测试、训练或 GPU 评估。远端文档中的 110 项检查及模型结果是队友原有记录，不冒充本次本机验证。
