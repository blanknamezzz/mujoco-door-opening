# 区分“这个站位不支持全开”和“选错了抓法”

入口：`station_feasibility.py`。原来 21 个角度都失败，只能表明该角度网格和当前控制器没有成功方案，不能作为站位全局不可行标签。

## 返回的三类结论

| 状态 | 证据 | 能否拒绝站位 |
|---|---|---|
| `feasible_baseline` / `feasible_with_alternative_grasp` | PRM 路径实际执行后，某个方案在 MuJoCo 中全开并保持 | 有正面证据，保留成功方案 |
| `excluded_by_workspace_bound` | 在达到最小成功角度前，把手轴必须越过所有手臂姿态的保守接触范围 | 可在证书明确的模型与任务条件下排除 |
| `unconfirmed_after_search` | 有限候选、IK 检查或 PRM/物理试验没有找到成功方案 | 不应当作无解，可增加搜索或重新定位 |

底座、场景、摩擦等条件相同，原方案失败而替代方案实际成功，是“该站位有解，原方案限制了结果”的直接证据。报告的 `changed_parameters` 明确列出改变的是角度、肘分支、抓点高度、力还是时序。

## 什么时候能判断与抓法无关

本模块先计算一个不依赖抓角的必要条件。现有模型的 TCP 最大水平半径不超过三段长度之和：

`R_tcp = 0.6 + 0.55 + 0.18 = 1.33 m`。

为了避免把把手中心不必严格对准 TCP 的情况误报，程序额外加入：夹爪在全手指行程内的偏移上界、指垫三维角点半径、把手胶囊半径，以及默认 5 mm 的保守余量。模型尺寸与关节范围直接读取 MuJoCo，得到更宽松的把手轴接触包络。

门成功容许 90±2 度，因此只检查连续开门必经的 0 至 88 度区间。对这段已知铰链圆弧，计算把手轴到肩部位置距离的精确极值，既检查端点，也检查区间内驻点。若某个必经角度的距离大于保守接触包络，则不论抓角、肘分支、升降高度、力与时间怎样选择，连续指垫抓持都不能通过该姿态。

这是一个单向必要条件：违反即可排除，通过只代表未被这个条件排除。它会漏掉许多因为关节角约束、障碍或接触导致的无解情况，但不会把“有限搜索没找到”转化为几何证书。

证书条件为：已知的固定底座平面三转动关节加升降模型、指定关节范围内的动作、竖直把手，以及持续的指垫–把手接触。它不覆盖移动底座、推门板、放手让门惯性滑过、重抓、真实机器人误差或改变模型。模型轴向或结构不满足假设时返回 `unsupported_geometry`，不作排除。

## 未被排除后，怎样搜索抓法

1. 先实际执行指定基线，记录真实停止原因。
2. 检查其他角度，默认包含原来的 -20 至 +20 度网格，以及 ±30、±45、±60、±90 度。
3. 扩展抓点高度 0、±0.07 m 和两个肘分支。
4. 继续尝试每指驱动力上限 20 / 35 N，以及开门时间 10 / 14 s。
5. 对每个方案先做完整参考弧的有限 IK 检查，再实际执行 PRM 准备和抓持开门。只有真实 `D_full_open` 才返回成功证据。

默认最多执行 32 次物理回合、检查 800 个候选。预算耗尽或整个有限候选池都失败，均返回 `unconfirmed_after_search`。有限 IK 检查通过也不代表没有碰撞或不会脱手。

## 运行和复现

```bash
# 常规站位诊断。默认输出到带时间戳的新目录。
python station_feasibility.py diagnose --base 0.8 -1.0 --baseline-angle 0

# 几何必要条件阶段，不执行物理搜索；结果只能是排除或未知。
python station_feasibility.py diagnose --base 1.4 -0.4 --max-rollouts 0

# 扩大原“开到一半失败”站位的搜索。失败仍保留为未知。
python station_feasibility.py diagnose --base 0.3 -0.9 --max-rollouts 32

# 固定的三个解释例子。
python station_feasibility.py examples

# macOS 上观看同站位的失败基线和成功替代方案。
mjpython station_feasibility.py replay --report reproduced/station_feasibility_examples/angle_rescue/diagnosis.json --which baseline
mjpython station_feasibility.py replay --report reproduced/station_feasibility_examples/angle_rescue/diagnosis.json --which witness
```

可用 `--baseline-plan` 传入完整 GraspPlan JSON、`--yaw-deg` 指定底座朝向。当前诊断针对 90 度全开目标；其他目标会明确拒绝。JSON 保存所有尝试方案、准备检查、执行结果、几何拒绝、预算、成功方案和排除证书。基线与成功方案还保存 CSV/NPZ 轨迹。

Python 中可直接调用 `diagnose_station(base, ...)`。`SearchSettings` 可修改角度、高度、肘分支、力、时间与预算。

## 与训练和 PRM 的关系

采集器 `collect_grasp_data.py` 的每个 `*_prm_preflight.jsonl` 记录现已增加 `station_workspace_check`。这是在开门标签产生之前、仅根据已知几何计算的必要条件。原有 `status`、`angles_with_path` 与 `queries` 仍专门描述预抓取路径。

新字段不自动删站位、不改变奖励，也不改变 SL/RL 输入和已有结果。独立诊断的扩大物理搜索需要显式运行上述入口，不会偷偷读取测试结果来改变训练策略。新采集 manifest 会包含新模块哈希；已有数据保持历史快照，源码变化后继续采集应使用新目录。

若之后训练站位可行性分类器：成功物理见证可作为正标签，明确模型条件下的几何证书可作为有范围限定的负标签；未知应单独保留。不能将“21 个角度都失败”或“PRM 有限搜索失败”直接标成不存在方案。学习模型的预测也不等于严格证明。
