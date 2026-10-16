# Official H-JEPA in the door scene

> Historical record of the earlier compact adapter. The current default is the
> full official model described in [the current setup guide](hjepa_setup.md).
> The commands below reflect earlier defaults. To train the compact profile
> with the current CLI, pass `--profile compact --data runs/hjepa/data
> --output runs/hjepa/compact_model --steps 1000 --batch-size 16` explicitly.

This integration follows the staged positioning/grasping design in
`CS5478_Team42_Full_Opening.pptx` and the accompanying presentation scripts.
Those documents supersede the older joint-pair proposal in `Project.docx`.
The original contact benchmark remains available through `run_demo.py`.

## Model and source

The adapter imports the actual official [H-JEPA release](https://github.com/kevinghst/H-JEPA),
pinned to commit `97054ecaff83e13839195ec4b92d74c08a0127c5`.
`setup_hjepa.py` fetches the source into the ignored `external/H-JEPA` directory.
No upstream files are modified. Its MIT license is retained in that checkout.
The [paper](https://arxiv.org/abs/2610.06805) describes hierarchical latent
prediction and planning at different timescales.

The scene uses upstream `HJEPA`, `JEPA`, `FusionEncoder`, `ARPredictor`,
`SequenceMLPEncoder`, `SIGReg`, and `hjepa_forward`. Local additions are a compact
CNN, joint-state features, downstream value heads, two stage policy heads, and
a finite candidate planner. It does not load the paper's environment-specific
checkpoints: those have different observation/action spaces. Door weights are
trained from this scene's trajectories and saved with their upstream revision.

| Component | Door adaptation |
| --- | --- |
| Level 1 | 96-dimensional fused image/state latent; one step = 0.5 s |
| Level 2 | Separate 64-dimensional learned latent; stride 4 = 2 s |
| Dynamics | Official action-conditioned AdaLN autoregressive predictors |
| Macro action | Official MLP compresses four level-1 action embeddings |
| Objective | Official joint per-level prediction + SIGReg, weight 0.09 |
| Position policy | Initial image/state + candidate base pose -> downstream return |
| Grasp policy | Fresh arrival image/state + candidate grasp -> downstream return |
| Lookahead score | Mean of return heads on predicted level-1/level-2 latents |

H-JEPA jointly trains its encoders. This differs from the slides' proposed
frozen pretrained visual encoder. Policy losses detach the shared representation;
optional alternating policy updates freeze the entire world model. No pretrained
V-JEPA backbone or random frozen encoder is presented as a trained world model.

## Install on Windows

Use Python 3.11+ and a terminal in the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe setup_hjepa.py
.\.venv\Scripts\python.exe -m unittest test_hjepa -v
```

The adapter installs `requirements-hjepa.txt`, independent of the original
macOS baseline's exact NumPy/Pillow pins. It needs PyTorch, MuJoCo, a working
OpenGL context, Einops and OmegaConf; it does not require the upstream suite of
unrelated environments or its Lightning/SLURM setup. Training, execution and
evaluation CLIs default to `--device auto`: CUDA when available, CPU otherwise.
Use `--device cpu` or `--device cuda` to override. For GPU use, install a CUDA
PyTorch build compatible with the NVIDIA driver. On Linux choose a working MuJoCo rendering backend, such
as `MUJOCO_GL=egl` when EGL is available. Do not force EGL on Windows.

## Collect, train, execute

```powershell
.\.venv\Scripts\python.exe collect_hjepa.py --scenes 12
.\.venv\Scripts\python.exe train_hjepa.py --steps 1000 --policy-updates 100
.\.venv\Scripts\python.exe run_hjepa.py
.\.venv\Scripts\python.exe evaluate_hjepa.py
```

Defaults write data, models, a physical demonstration and held-out evaluation
under `runs/hjepa/`. Data collection refuses to overwrite a completed dataset;
use `--output` for a new one. Interrupted collections support `--resume`.
`best.pt` is selected using validation loss; `last.pt` contains the final
dynamics and any optional policy updates and is the execution/evaluation default.
Pass `--checkpoint runs/hjepa/model/best.pt` to use the validation-selected
checkpoint without those subsequent policy updates.

A quick pipeline check with smaller candidate budgets:

```powershell
.\.venv\Scripts\python.exe collect_hjepa.py --output runs/hjepa/smoke_data --scenes 3 --bases 2 --grasps 4
.\.venv\Scripts\python.exe train_hjepa.py --data runs/hjepa/smoke_data --output runs/hjepa/smoke_model --steps 100 --batch-size 8 --policy-updates 10
.\.venv\Scripts\python.exe run_hjepa.py --checkpoint runs/hjepa/smoke_model/last.pt --output runs/hjepa/smoke_demo --bases 2 --grasps 4
```

Use the same candidate budgets at collection, execution and evaluation. The
scripts expose `--bases` and `--grasps` for this purpose; a two-base smoke model
has not been trained to rank the remaining three bases.

## Task/data contract

The five base poses include a nominal full-opening position, the delayed
workspace-limit case, a difficult near-handle placement, a rotated placement,
and an unreachable position. Four grasps vary handle height and approach yaw.
All candidates use the same fixed Cartesian skill schedule and physical
controller. `skill_target()` is shared by execution and imagined action features.

The sequence is initial RGB-D -> position policy -> base reset placement ->
**new RGB-D** -> grasp policy -> physical alignment/grasp/open/hold. The base
remains fixed throughout manipulation. In this new pipeline, manipulation starts
from the observed arrival arm state; it does not teleport the arm to each
candidate's pregrasp pose. The original benchmark keeps its original reset
behavior to preserve its causal controls.

Images are 64x64 RGB. The 19 state features are `qpos[7]`, `qvel[7]`, base XY,
sine/cosine of base yaw, and elapsed time / 15.5. Joint and door state are
explicit simulator inputs, not an image-only model. Future IK feasibility,
collision checks, rewards and outcome labels are never input features.
Material parameters are not observations. Action features are the mean planned
Cartesian command over the next 0.5 s: XYZ, sine/cosine of hand yaw, and normalized
finger aperture. The controller still integrates at the scene's 1 ms timestep.
This temporal action compression is a scene-specific approximation.

`manifest.json` stores scene parameters, split assignments, candidate IDs,
termination reasons, outcomes and reward components. Each NPZ contains frames,
states, actions, timestamps, initial/arrival observations and candidate features.
The terminal action has no successor and receives no next-state target. Short
terminal intervals are omitted from dynamics clips rather than padded as fake
transitions. Initial-unreachable trials have zero trajectory frames and remain
negative policy examples. Nine-frame training clips never cross episodes.
Arrival detection failures are recorded as `B_perception_failed`, contribute
negative positioning targets, and are excluded from supervised grasp targets.
Execution reports the same failure without inventing a grasp from geometry.

Door scenes vary handle X/Z and contact/hinge properties. Splits are by scene,
before considering base/grasp variants. State normalization uses training data
only. Validation selects the checkpoint; test scenes are used only by evaluation.
These are variations of one door family, not independent robot/door assets.

## Reward and policy training

The reward implements slide 3:

`R = 10 S + sum(M_k) + 2 P - lambda_collision C - lambda_incomplete V - 0.1 T`.

`S` is safe opening to at least 88 degrees and one-second stable hold. The 2-degree
tolerance is the existing implementation criterion (the older proposal used 3).
`M_k` are ordered, once-only position, alignment and grasp milestones. Position
currently means reachable placement; alignment means reaching the close phase
without an earlier collision. `P` is maximum opening over the safe trajectory
prefix, frozen at the first collision/slip/contact-loss event. `V` is final-angle
shortfall / 90, clipped to [0,1] and assessed once. `T` is elapsed / 15.5, capped
at one. Collision and incompletion weights default to 5 and 3 and are separate
arguments to `task_reward`; physical joint limits remain hard execution constraints.

Warm starts regress measured downstream returns, scaled by 15. Position targets
use the best observed grasp return per base; grasp targets use each grasp's
measured return. `--policy-updates` alternates exact finite-pool policy gradients:
position optimizes expected downstream return under the fixed grasp policy, then
grasp optimizes return weighted by the fixed position policy. This uses the
collected counterfactual outcome table. It is **not online GRPO/PPO exploration**;
adaptive collection of new on-policy states remains future work.

## Evaluation and limits

`evaluate_hjepa.py` compares fixed/learned position policies with direct grasp
scores, model rollouts and full-path geometry on the **same four-grasp pool**.
Geometry is a grasp reference at the selected base, not a joint base/grasp oracle.
Positioning uses its downstream policy head; it does not imagine base navigation.
Horizon zero uses the direct grasp head; quarter/half horizons include preparation
then the requested opening-arc fraction; full horizon includes the hold.
The upper model predicts complete 2-second chunks within that horizon. The
planner refreshes its latent after placement. Opening then runs the fixed
controller; it does not replan from ongoing video.

Reports include safe opening fraction, full-opening success, collision rate,
joint-limit stops, slip/contact-loss rate and planning time. Direct and rollout
methods use the same trained model and measured returns, with no additional
simulator queries during ranking. The geometric reference performs explicit
known-scene path checks. A local-only objective ablation, mobile navigation,
base-travel metrics, online exploration and policy-driven recollection are not
implemented. The scene continues to use **real contact dynamics**; the slides'
ideal no-slip assumption is not silently imposed with a hand-door weld.

This adapter uses the official hierarchy for **finite candidate ranking**. It is
not a reproduction of the paper's continuous top-down subgoal optimizer, and
does not claim its benchmark performance. Longer rollout quality must be
measured on held-out scenes; decreasing training loss alone does not establish
better task decisions.

## Verified local run (2026-10-16)

The Windows CPU environment completed setup, seven contract tests, and all eight
original benchmark checks. The default dataset contains 240 candidate trials
across 12 scenes (8 train / 2 validation / 2 test), with 4,191 valid transitions.
Training ran 1,000 joint-model steps plus 100 alternating finite-pool policy
updates. The ready-to-run checkpoint is `runs/hjepa/model/last.pt`.

`python run_hjepa.py` selected base `(0.45, -1.05, 1.4208)` and the lower grasp;
physical execution reached 89.656 degrees and held for one second without slip
or collision flags. The detailed result and physical trajectory are in
`runs/hjepa/demo/hjepa.json` and `hjepa.npz`.

The 24 held-out evaluation runs (12 configurations on two test scenes) all
completed successfully, including the direct and geometric baselines. These
small, easy test scenes validate the pipeline but do not demonstrate a
world-model advantage. Results are in `runs/hjepa/evaluation/metrics.json` and
`trials.json`. Data, weights and downloaded source are local ignored artifacts;
the setup/collection/training commands reproduce them.

## RTX 3060 Laptop check

The local card has 6 GB VRAM and driver 561.09 (CUDA 12.6 support). The project
virtual environment now has PyTorch `2.8.0+cu126`, matching the
[official PyTorch 2.8 installation option](https://pytorch.org/get-started/previous-versions/#v280).
The global CPU installation was preserved. A fresh environment with the same
driver can install the CUDA build after setup:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade torch==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe benchmark_hjepa_device.py --device cuda --batches 16 64 128 --steps 100
```

Measured end-to-end training on the 240-trial dataset, including data loading,
validation and checkpoint saving:

| Device | Batch | 100 steps (seconds) | Peak PyTorch allocation | Peak reserved pool |
| --- | --- | --- | --- | --- |
| CPU, PyTorch 2.8 CPU build | 16 | 12.55 | — | — |
| RTX 3060, CUDA 12.6 | 16 | 7.51 | 71.6 MiB | 98 MiB |
| RTX 3060, CUDA 12.6 | 64 | 5.78 | 202.7 MiB | 268 MiB |
| RTX 3060, CUDA 12.6 | 128 | 6.74 | 377.5 MiB | 468 MiB |

GPU batch 16 was about 1.67 times faster than CPU in this short run. Larger
batches were tested later in the same process with warm CUDA/filesystem caches;
these are capacity checks, not a controlled throughput scaling study. Batch 16
remains the tested training default. Changing batch size also changes SIGReg's
batch-dependent objective and may require retuning, despite fitting in memory.
One `nvidia-smi` sample during the sweep showed 1,253 MiB total device usage,
including other applications and CUDA overhead; this was not a sampled peak.
The conclusion applies to the current 553,812-parameter, 64x64 adapter, not a
large pretrained video backbone.

The inherited environment contained `PYTORCH_CUDA_ALLOC_CONF=expandable_segments=True`,
which prevented CUDA initialization. `hjepa_runtime.py` removes that malformed,
unsupported Windows option within project processes while preserving unrelated
allocator settings. It does not change the system environment. On Linux it
corrects the option's separator to `:`. Raw results live in
`runs/hjepa/device_benchmark/{cpu,cuda}_benchmark.json`.

A complete CUDA run of 1,000 training steps plus 100 alternating policy updates
took **45.93 seconds** at batch 16. Its checkpoint is
`runs/hjepa/gpu_model/last.pt`; timing/memory details are in
`runs/hjepa/gpu_training_report.json`. GPU inference with that checkpoint opened
the nominal door to 89.656 degrees with a one-second hold and no slip/collision
flags, and completed both held-out scenes. Re-run that model with:

```powershell
.\.venv\Scripts\python.exe run_hjepa.py --checkpoint runs/hjepa/gpu_model/last.pt --output runs/hjepa/gpu_demo
```

## NUS SoC cluster fallback

The [official SoC GPU guide](https://www.comp.nus.edu.sg/~cs3210/student-guide/soc-gpus/)
documents the following access route:

1. Use a **SoC UNIX account**, separate from NUSNET. If needed, request one at
   [mySoC account registration](https://mysoc.nus.edu.sg/~newacct).
2. Enable **SoC Compute Cluster** in
   [mySoC services](https://mysoc.nus.edu.sg/~myacct/services.cgi).
3. Connect from the SoC network or via SoC VPN, then run
   `ssh YOUR_SOC_UNIX_ID@xlogin.comp.nus.edu.sg`.
4. Inspect resources with `sinfo` and request a GPU through Slurm, for example
   `srun --gpus=1 nvidia-smi`. The guide also documents typed A100/H100 requests.

Run training inside the allocated GPU job. The public guide lists A100 40/80 GB
and H100 resources, but the allocation and availability for an individual
student account must be checked after login. The public guide did not establish
a universal free quota or unlimited student entitlement; the detailed DocHub
policy pages were unavailable during this check. No remote account was accessed
or cluster job submitted. This cluster is unnecessary for the current model's
measured memory requirements.
