# MuJoCo Door Opening

## Official H-JEPA setup

The staged world-model pipeline now uses the complete official `pusht_hjepa_l2`
configuration (224x224 ViT-Tiny, 23.3M parameters), two timescales,
separate positioning/grasping policies, fresh arrival RGB-D, physical trajectory
collection, downstream rewards, and scene-disjoint evaluation.
See [setup, training and limitations](docs/hjepa_setup.md).

```powershell
.\.venv\Scripts\python.exe setup_hjepa.py
.\.venv\Scripts\python.exe collect_hjepa.py
.\.venv\Scripts\python.exe train_hjepa.py --policy-updates 100
.\.venv\Scripts\python.exe run_hjepa.py
```

The default full run uses batch 128, BF16, 9 epochs and ViT activation
recomputation to fit the local 6 GB RTX 3060. The earlier 0.55M compact model is
retained as an explicit legacy profile. Base placement uses simulator resets. The original contact baseline
below remains available independently through `run_demo.py`.

Expanded frozen-policy evaluation (new scene seed, matched stage interventions,
base/grasp yaw sweeps and 30/60/90-degree controller targets):

```powershell
.\.venv\Scripts\python.exe evaluate_hjepa_stages.py --device cuda
.\.venv\Scripts\python.exe evaluate_hjepa_views.py --device cuda
.\.venv\Scripts\python.exe report_hjepa_stages.py
.\.venv\Scripts\python.exe -m unittest test_hjepa test_hjepa_stages test_git_size -v
.\.venv\Scripts\python.exe check_git_size.py
```

The stage evaluator crosses the pre/post-update positioning and grasping heads
while asserting identical world/value weights. Physical candidate tables are
evaluation labels, never selector inputs. Raw results stay in ignored `runs/`;
weights, arrays and video/archive files are also ignored. `check_git_size.py`
rejects Git-visible or staged files above 10 MiB, including tracked files that
ignore patterns cannot protect.
See [measured stage/angle results](docs/hjepa_stage_results.md) for the policy
interventions, stress failures and camera-view sensitivity.

Structural feasibility tests add collidable pillars, bidirectional/one-way
hinges and one/two-sided handles. Both operating sides are in the same four-base
candidate pool. See [structural results and physical states](docs/hjepa_structural_results.md).

```powershell
.\.venv\Scripts\python.exe evaluate_hjepa_structure.py --device cuda
# Observe and pull from the back: reverse swing avoids the front-side pillar.
.\.venv\Scripts\python.exe run_hjepa.py --scenario pillar-front --base-mode fixed --fixed-base-id 2 --method geometry --output runs/hjepa/structural_demo
```

These are historical structural tests of the original frozen checkpoints,
whose training data contained only front-side operation.
Geometry-based side selection/rejection in the report is a privileged reference.

Train the complete model on fresh structural scene splits, with two-sided
operation and an explicit refusal candidate:

```powershell
.\.venv\Scripts\python.exe collect_hjepa.py --structural --output runs/hjepa/structural_data --seed 20261017
.\.venv\Scripts\python.exe train_hjepa.py --data runs/hjepa/structural_data --output runs/hjepa/structural_model --init-checkpoint runs/hjepa/official_model/selected.pt --epochs 9 --policy-updates 1000 --device cuda
.\.venv\Scripts\python.exe refit_structural_position.py
.\.venv\Scripts\python.exe evaluate_structural_training.py --device cuda
.\.venv\Scripts\python.exe report_structural_training.py
.\.venv\Scripts\python.exe run_hjepa.py --checkpoint runs/hjepa/structural_model/selected.pt --scenario pillar-front --output runs/hjepa/structural_trained_demo
```

Structural fine-tuning keeps the official architecture/batch/precision. The
world checkpoint uses validation loss; frozen stage-policy updates use validation
return for selection. The six new test scenes are evaluated after selection.
The optional positioning refit probes the same full ViT before its JEPA
projector, fits the original task head and accepts it only when validation
return improves. Dynamics, value and grasp representations stay projected.

The completed full-model structural run used 9 epochs / 1,053 steps on the
RTX 3060 Laptop, with 2,848 MiB peak allocation. On six held-out scenes, opening
improved from 3/5 to 4/5 feasible doors; the blocked two-pillar case was not
rejected. See [training results and remaining failures](docs/hjepa_structural_training.md).

基于 MuJoCo 的机械臂开门仿真：识别固定拉杆把手、选择抓取姿态、两指夹持、通过接触力开门，并检测滑移与脱手。

场景使用固定底座、三个平面转动关节、一个竖直升降轴与两指平行夹爪。门铰链无驱动器，开门运动由夹爪与把手的实体接触产生。当前采用已知几何、受控 RGB-D 识别、解析逆运动学和关节伺服控制。

## 示例动画

### 完整打开

![完整打开：约 89.68°](results/nominal.gif)

### 开到一半后失败

![轨迹越过肘关节角度限制](results/workspace_limit.gif)

实际门角约 42.5° 时，下一参考目标需要肘关节超过 -165° 下限，控制器提前终止。此时夹爪仍保持接触；42.5° 是仿真终止时的实际角度，不是精确的物理最大开度。

### 滑动与脱手

![低摩擦下从夹持到滑移及脱手的近景](results/low_friction_closeup.gif)

低摩擦故障案例使用 μ=0.005；约 3.77 s 触发持续滑移告警，约 4.73 s 确认接触丢失。该极低摩擦参数用于复现故障，未对应实测材料。

## 安装与复现

验证环境：macOS、Python 3.12.5、MuJoCo 3.16.0、NumPy 2.5.4、Pillow 12.3.0。依赖版本固定在 `requirements.txt`。

在 VS Code 中打开仓库文件夹，并在终端执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_demo.py
```

完整流程会重新渲染初始 RGB-D、识别把手、选择抓点、运行六组动力学实验和验证、导出动画与诊断。新结果位于 `reproduced/`，包内 `results/` 是已保存的参考结果。

Windows 激活环境使用 `.venv\Scripts\activate`；可用 `py -3.12 -m venv .venv` 创建环境。图形渲染需要可用的图形环境。当前仅在上述 macOS 环境完成验证。

固定初始观测、重新计算运动与动画：

```bash
python run_demo.py --reuse-observation
```

macOS 实时观看：

```bash
mjpython simulate.py --case nominal --view
mjpython simulate.py --case workspace_limit --view
```

单独复现完整脱手过程：

```bash
python simulate.py --case low_friction --continue-after-slip
python render_slip_detail.py
```

默认单案例运行在滑移告警时结束；`--continue-after-slip` 是为观察后续接触丢失而启用的诊断模式。

## 实验与结果

| 案例 | 含义 | 已保存参考结果 |
|---|---|---|
| `unreachable` | 初始抓点或预抓取位姿不可达 | 无法建立抓握 |
| `reach_only` | 接近并夹住后结束，不测试后续开门 | 双侧抓握建立 |
| `workspace_limit` | 中途关节参考目标越界 | 实际门角约 42.505° |
| `low_friction` | 摩擦不足导致滑动和脱手 | 门基本未打开 |
| `nominal` | 正常夹持开门与保持 | 约 89.679° |
| `no_grip` | 手指始终张开的因果对照 | 门保持关闭 |

六个案例保持正常场景选出的抓点和朝向不变，分别改变站位、摩擦或闭合状态。8 项参考验证结果及步长敏感性结果见 [summary.json](results/summary.json)。

## 文件说明

| 文件 | 作用 |
|---|---|
| `scene.xml` | 门、把手、机械臂、夹爪、接触和驱动参数 |
| `perception.py` | RGB-D 采集、颜色分割和拉杆截面拟合 |
| `planning.py` | 9 个抓取候选与几何轨迹筛选 |
| `simulate.py` | 动力学控制、接触监测与结果分类 |
| `benchmark.py` | 六组对照实验和验证 |
| `diagnose_workspace.py` | 实际停止状态与首个越界目标的诊断 |
| `render_results.py` | 从保存的物理轨迹渲染动画 |
| `render_slip_detail.py` | 脱手近景动画和参数导出 |
| `run_demo.py` | 完整复现入口 |
| `results/` | 参考动画、输入观测、轨迹、接触数据与诊断 |

[详细实现、坐标与判据](docs/implementation.md) · [脱手案例完整参数](results/low_friction_parameters.json) · [半开失败诊断](results/workspace_limit_diagnosis.json)

## 当前范围

视觉方法依赖金色竖直拉杆、已知半径、标定相机及端部可见的先验；铰链与机器人几何来自已知场景。接触力与相对运动直接读取仿真状态。新增的官方 H-JEPA 训练与分阶段策略管线见上方链接。当前仍没有通用视觉模型、在线遮挡跟踪、真实六轴机器人、移动底座导航或自动重抓，亦未完成真实机器人验证。
