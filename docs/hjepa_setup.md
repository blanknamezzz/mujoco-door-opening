# Official full H-JEPA configuration for the door scene

The default now uses the complete **`pusht_hjepa_l2`** model configuration from
the [official release](https://github.com/kevinghst/H-JEPA), commit
`97054ecaff83e13839195ec4b92d74c08a0127c5`. `hjepa_model.default_config()` resolves
the release's `config/train/base/pusht.yaml` and `pusht_hjepa_l2.yaml`, then calls
its **`create_world_model()`** factory. The backbone is not a locally rebuilt
approximation or a resized CNN.

The older compact model and its measurements are retained in
[the compact experiment history](hjepa_compact_history.md). Those measurements
do not describe the full configuration. Old checkpoints remain loadable;
training one explicitly requires `--profile compact` and its original dataset.

## Model and training settings

| Setting | Full official configuration |
| --- | --- |
| Image input | Native 224x224 RGB; ImageNet normalization |
| Patch size | 14 |
| Visual encoder | ViT-Tiny, 12 layers, width 192, 3 attention heads |
| Level 1 predictor | 4 layers, 8 heads, dim_head 64, MLP width 1024 |
| Level 2 predictor | 6 layers, 16 heads, dim_head 64, MLP width 2048 |
| Latent dimensions | 192 / 192 |
| Projection heads | Official MLPs, hidden width 2048, BatchNorm |
| History / rollout training | 3 / 1, at each level |
| Upper temporal stride | 2 |
| Upper action encoder | Official sequence Transformer, output dimension 4 |
| Latent action queue | 4,000 |
| SIGReg | 1,024 random projections, weight 0.09 per level |
| Training batch | 128, shuffle, drop incomplete batches |
| Precision | BF16 mixed precision on CUDA |
| AdamW learning rates | Level 1: 5e-5; level 2: 1e-5 |
| Weight decay / gradient clip | 0.001 / 1.0 |
| Epoch budget | 9 |

The official world model has **23,222,914 parameters** with the door action
interface. Adding the two stage policy heads and two value heads brings the
total to **23,324,422**, approximately 89 MiB of FP32 parameter storage.

The only architecture change required by the scene is the primitive action
dimension: Push-T uses 2; the door controller uses 6. With frameskip 5, the
level-1 action embedder receives 30 values instead of 10. The native and adapted
world models have identical state-dict keys and tensor shapes except for that
input convolution weight, changing from `(10, 10, 1)` to `(10, 30, 1)`. A contract
test checks this against a second model constructed by the upstream factory.
No encoder, predictor, projection or action-encoder width/depth is reduced.

This configuration initializes its ViT from `ViTConfig`, as the upstream
factory does. It does not load a pretrained V-JEPA video backbone, nor is a
Push-T environment checkpoint presented as door-trained weights.

## Install and run on Windows

Use the existing project virtual environment, or create one with Python 3.11+:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe setup_hjepa.py
.\.venv\Scripts\python.exe -m pip install --upgrade torch==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m unittest test_hjepa -v
```

The CUDA command above matches this laptop's driver 561.09 / CUDA 12.6 support;
select a driver-compatible build on another machine. The tested Transformers
version is pinned in `requirements-hjepa.txt`. The local CUDA allocator fix in
`hjepa_runtime.py` handles the malformed inherited `expandable_segments=True`
option within project processes; it does not edit the system environment.

```powershell
.\.venv\Scripts\python.exe collect_hjepa.py
.\.venv\Scripts\python.exe train_hjepa.py --policy-updates 100
.\.venv\Scripts\python.exe run_hjepa.py
.\.venv\Scripts\python.exe evaluate_hjepa.py
```

Defaults use `runs/hjepa/official_data`, `official_model`, `official_demo` and
`official_evaluation`. Training/inference choose CUDA automatically when
available. `--device cpu` remains available; official BF16 training is tested
on the RTX 3060. `--steps N` overrides the epoch budget for an explicitly short
check. The default full run uses all 9 epochs. The collected dataset supplies
78 complete batches per epoch, hence **702 optimizer steps**.

`best.pt` selects the predictor by validation loss. The official CLI then freezes
that predictor and runs train-only stage-policy updates (100 by default), saving
**`selected.pt`**, the inference/evaluation default. It checks that every world
model weight and buffer remains unchanged during those policy updates.
`last.pt` preserves the final-epoch model and its policy-update branch for audit.
The raw best and final-epoch files are retained. Each run records a
`resolved_config.json` and `training.json` so its model size, data budget,
precision, optimizer rates and runtime are reviewable.

## RTX 3060 memory behavior

The unchanged model with batch 128 and BF16 ran out of memory without activation
recomputation. A real two-step training probe, including backward propagation,
AdamW state, validation and task heads, **passed** with recomputation:

| Runtime | Batch | Result | Peak PyTorch allocation | Peak reserved pool |
| --- | --- | --- | --- | --- |
| Standard autograd | 128 | CUDA OOM | See raw report; Windows used shared memory | — |
| ViT activation recomputation | 128 | Passed | 2,849.6 MiB | 3,516 MiB |

`enable_activation_checkpointing()` checkpoints the original ViT forward in
chunks of 32 images. It recomputes activations during backpropagation. Its
parameters and computations remain those of the official ViT. The projection
heads' BatchNorm and the two SIGReg objectives still operate over the **whole
128-example batch**. This is not small-batch gradient accumulation. Full model
dimensions, resolution, temporal context and training batch remain unchanged.
Checkpointing is enabled by default for the full profile. It trades runtime
for memory. Use `--no-activation-checkpointing` on a larger GPU to disable it.

Reproduce the capacity checks:

```powershell
.\.venv\Scripts\python.exe probe_hjepa_official.py
.\.venv\Scripts\python.exe probe_hjepa_official.py --activation-checkpointing
```

Results are in `runs/hjepa/official_capacity/*batch128.json`. The allocation
numbers refer to PyTorch tensors/pools, not all desktop and driver usage.
Windows WDDM may spill into shared host memory during an OOM attempt; its
reported logical allocation must not be read as physical VRAM capacity.

## Door data and interfaces

The 240-trial dataset was recollected at **native 224x224**, every **0.1 seconds**,
rather than upscaling the earlier 64x64 images. It contains 21,007 frames across
12 door scenes, split 8 train / 2 validation / 2 test. All base/grasp variants
of a scene stay in the same split.

The release's level-1 frameskip is 5. Each training sample spans 35 raw frames:
seven image states sampled at indices `0, 5, ..., 30`, with five consecutive
primitive actions flattened into each 30-dimensional action block. The
hierarchy derives its 4-frame level-1/level-2 training crops as upstream does.
Thus a level-1 transition covers 0.5 s and a level-2 transition covers 1 s.
Action z-score statistics come only from training scenes, use sample standard
deviation, and are saved as model buffers for planning. Constant columns are
handled with the upstream finite-value normalization convention.

RGB pixels live in memory-mapped `.pixels.npy` files (about 2.95 GiB total), so
the Windows trainer does not load the entire video dataset into RAM. Each NPZ
contains state/action/time arrays and the two stage observations; the manifest
links its pixel file. Initial/arrival camera frames are center-cropped to the
square renderer's field of view before resizing. Perception retains the
calibrated high-resolution RGB-D observation.

The primitive door command is XYZ, sine/cosine of hand yaw, and normalized
finger aperture, averaged over each raw 0.1-second interval. Candidate rollouts
use the same fixed command schedule and the same five-action packing and
normalization. Joint states remain available for logs and controller execution;
the official backbone keeps `use_proprio: false` and encodes images.

The task sequence remains initial observation -> position policy -> base reset
placement -> **fresh arrival observation** -> grasp policy -> physical
alignment/grasp/open/hold. The base is fixed during manipulation. No future IK
or collision oracle is an input to learned ranking. Arrival perception failures
and unreachable candidates remain negative positioning examples.

## Training/runtime scope

The world-model architecture and per-level predictive/SIGReg objective use the
official factory and forward function directly. The Windows trainer uses the
release's batch, precision, epoch count, per-level AdamW rates and gradient clip.
Its warmup/cosine formula and epoch scheduling interval follow
`main_hjepa.py` and stable-pretraining 0.1.7 defaults; these are recorded in the
training report. Pixel/action preprocessing matches the upstream conventions.
The local memory-mapped loader and `state_dict` checkpoints replace the release's
HDF5/Lightning/SLURM runtime. Those platform adapters are not paper benchmark
reproduction claims.

The two candidate-conditioned policies and downstream value heads are local
door-task components. They warm-start on measured returns. Optional
`--policy-updates` alternates finite-pool policy gradients with the world model
frozen; it is not online GRPO/PPO exploration. The reward remains
`10 S + sum(M_k) + 2 P - lambda_collision C - lambda_incomplete V - 0.1 T`.
Completion requires at least 88 degrees and a stable one-second hold; progress
uses the safe trajectory prefix and final-angle shortfall is assessed once.

Planning ranks finite base/grasp candidates using the shared representation,
position/grasp policy heads and imagined latent trajectories. It does not run
the paper's continuous top-down subgoal optimizer. Base navigation, on-policy
recollection, generic perception and real-robot validation remain outside this
scene implementation. Evaluation compares direct scores, rollout horizons and
full-path geometry at the chosen base; a small held-out door family alone
cannot establish general world-model benefits.

## NUS SoC alternative

For larger experiments, the [official SoC GPU guide](https://www.comp.nus.edu.sg/~cs3210/student-guide/soc-gpus/)
documents using a SoC UNIX account, enabling Compute Cluster in
[mySoC services](https://mysoc.nus.edu.sg/~myacct/services.cgi), connecting through
the SoC network/VPN, and SSH to `YOUR_SOC_UNIX_ID@xlogin.comp.nus.edu.sg`.
Use `sinfo` to inspect resources and request a GPU through Slurm, such as
`srun --gpus=1 nvidia-smi`. The guide lists A100/H100 resources. Account-specific
entitlements, quotas and availability need checking after login; no universal
free quota was verified from the public documentation. No remote job was
submitted from this workspace.

## Completed local full-model run (2026-10-16)

The full model completed **9 epochs / 702 optimizer steps**, batch 128 and BF16,
in **1,678 seconds (about 28 minutes)** on the 6 GB RTX 3060 Laptop. Peak PyTorch
allocation was **2,851.2 MiB**, with **3,518 MiB** reserved. All nine contract
tests passed, including native-vs-door architecture shapes and ViT recomputation
output/gradient equivalence.

The validation-selected predictor came from step 650. Selecting `best.pt` alone
omitted the subsequent stage-policy updates: its learned positioning initially
chose the delayed workspace-limit base and stopped the nominal door at 42.5
degrees. The finalization pipeline now freezes the selected predictor, updates
its two policies for 100 alternating updates using **training scenes only**, and
saves `selected.pt`. An explicit tensor equality check verified that every world
model parameter and buffer stayed unchanged. The raw failure results are retained
in `official_demo_before_policy_update` and
`official_evaluation_before_policy_update` under `runs/hjepa/`.

With `selected.pt`, the nominal demonstration reached **89.656 degrees** and
held for one second without slip/collision flags. The 24 integration evaluation
runs (12 configurations on two scene splits) all completed successfully,
including the direct and geometry references. Details are in
`runs/hjepa/official_model/training.json`, `official_demo/hjepa.json`, and
`official_evaluation/{metrics,trials}.json`.

These two test scenes were reused while debugging the stage-checkpoint pipeline.
They provide integration checks, not an untouched final research test set.
Reserve new scene seeds before making comparative generalization claims. The
current results also do not demonstrate a world-model advantage over the
references.

## Expanded stage evaluation (2026-10-17)

`evaluate_hjepa_stages.py` freezes the official full model and evaluates eight
fresh random scenes (seed 20261017), nine parameter stress scenes and one
no-grip negative control. The random scenes are checked against the physical
parameters of all collection splits, rather than only checking scene IDs.
Stress cases deliberately reuse the nominal geometry to isolate a changed
friction, hinge resistance, handle radius-from-hinge or height; they are not
additional independent random samples.

For every scene it executes all 5 bases × 4 grasps, starting each grasp from
the same actual constructor arrival state used to capture its RGB-D observation.
It compares the four `P_pre/G_pre`, `P_post/G_pre`, `P_pre/G_post`,
`P_post/G_post` combinations. `pre` is the supervised warm-start in `best.pt`;
`post` is the 100 alternating offline updates in `selected.pt`. Every non-policy
parameter/buffer, including value heads and input statistics, must match.
This isolates each policy update without retraining or changing the predictor.

The four diagnostic boundaries are reachable position, collision-free completed
alignment, acquired stable grasp, and opening with a one-second hold. Reports
include unconditional stage counts and conditional rates with their actual
denominators, failure stage and physical stop reason. An additional grasp-only
comparison holds *each* arrival base/image/state fixed, including failed
upstream placements, rather than comparing grasp policies on different states.

Fixed placement, direct policy, 1/4, 1/2 and full-arc latent lookahead, and
full-path geometry use the same candidate pools. The world-model selector uses
value heads instead of the direct grasp head for nonzero horizons; updating
that grasp head therefore cannot be credited for changes to world-model
selection. Cached counterfactual returns are used only after decisions have
been ranked, and never enter policy/world/geometric scores.

Four scenes also receive base yaw offsets -45 to +45 degrees, grasp offsets
-30 to +30 degrees, and target angles 30/60/90 degrees at three fixed bases.
Partial-angle tests change the physical reference and require an actual angle
within 2 degrees, bilateral contact and low door velocity throughout the last
second. They are controller diagnostics: the learned model was trained for
90 degrees and is not presented as goal-conditioned. Default execution remains
90 degrees. Suite/source/checkpoint hashes protect resume from mixing runs.

```powershell
.\.venv\Scripts\python.exe evaluate_hjepa_stages.py --device cuda
# Use another output directory when changing the seed or implementation.
.\.venv\Scripts\python.exe evaluate_hjepa_stages.py --seed 20261018 --output runs/hjepa/stages_seed2
.\.venv\Scripts\python.exe check_git_size.py --max-mib 10
```

Detailed generated tables live in `runs/hjepa/stage_evaluation/`: `suite.json`,
one candidate table per scene, `angles.json`, and `metrics.json`. Experiment
outputs, external repositories, weights, arrays and archives remain ignored.
The size checker also examines staged blob sizes, which catches a large staged
version even after the working file has been replaced with a smaller version.
Existing tracked reference assets are retained; the largest is the 5.45 MiB
nominal GIF, below the default 10 MiB guard.

`evaluate_hjepa_views.py` additionally orbits the calibrated camera around its
fixed nominal optical target: yaw -30/-15/0/15/30 degrees and elevation
-10/0/10 degrees, on the first two fresh scenes. The target does not follow the
true handle position. Intrinsics and extrinsics remain consistent with RGB-D;
lighting is unchanged. Both post-update direct and full-horizon world-model
selection run at each view, for 60 records total. Perception failures count
against end-to-end success instead of falling back to ground-truth handles.

```powershell
.\.venv\Scripts\python.exe evaluate_hjepa_views.py --device cuda
.\.venv\Scripts\python.exe report_hjepa_stages.py
```

The [measured results](hjepa_stage_results.md) include small versioned plots.
Raw output remains ignored. Counts distinguish candidate assessment records,
executor calls and actual dynamics integrations: a visual failure can reject
all four grasp candidates without running four physical trajectories.

## Structural constraints: pillars and operating sides

`structural_scene.py` builds nine physical MJCF variants: free bidirectional
swing, a pillar in either swing path, pillars in both paths, positive-only and
negative-only hinges, front/back-only handle access, and a single-sided,
single-direction door whose allowed arc is obstructed. Pillars are collidable
static cylinders; door-panel contact stops the trial as `door_obstructed`.
There are no scenario-name success/failure gates, door actuators or hand welds.

The structural candidate pool contains two front and two mirrored back bases,
each with the same four local grasps. Back operation mirrors the IK elbow branch,
refreshes RGB-D from that side, grasps its actual handle and commands negative
hinge rotation. Raw signed angles and progress in the commanded direction are
logged separately. `action_at`, physical execution and geometric screening use
the same signed reference. All existing checkpoint dimensions stay unchanged.

A pillar in the positive arc obstructs that arc for every base. A side switch
only helps if the door permits a reverse swing and has an accessible reverse
handle. A pillar in each arc, or one in the only permitted/accessed arc, leaves
no full-opening solution within this skill library. The geometry reference can
reject such cases; learned selectors currently always choose an action and
have no trained or calibrated abstention head.

One-way hinges and one-sided access test different constraints. A hinge stop
constrains rotation, not all possible actions from a face: a robot with an
appropriate push skill might operate a one-way door from the other side.
The current action library performs approach/grasp/pull/hold. Single-sided
handles additionally enforce grasp access through actual geometry; no handle
is invented when the arrival detector fails.

The two closed, two-handled one-way variants differ only in their hidden hinge
limits. Their initial RGB-D and robot state are indistinguishable. A single
frame cannot identify this difference: learning it requires an observable
doorstop/marking or a probing/contact history. Known-geometry screening reads
joint limits explicitly and must not be credited as learned visual foresight.
Pillars and missing handles, in contrast, provide geometric/visual differences.

`evaluate_hjepa_structure.py` crosses the two pre/post policy versions, compares
direct and three lookahead horizons, tests fixed front/back placement, and
compares full-path geometry across both sides. Every candidate starts from the
matched observed arrival state. Geometry-derived rankings and learned rankings
are fixed before their measured outcomes are used for evaluation. The report
separates successful opening, safe refusal, collision, grasp failure and slip.

```powershell
.\.venv\Scripts\python.exe evaluate_hjepa_structure.py --device cuda
.\.venv\Scripts\python.exe run_hjepa.py --scenario pillar-front --base-mode fixed --fixed-base-id 2 --method world-model --output runs/hjepa/structural_demo
.\.venv\Scripts\python.exe -m unittest test_structural_scene test_hjepa test_hjepa_stages test_git_size -v
```

[Structural results](hjepa_structural_results.md) contain the nine-scene tables
and actual terminal-state views. Detailed counterfactual outcomes, stage-state
snapshots, CSV and hashes remain in ignored `runs/hjepa/structural_evaluation/`.
These canonical scenes are deterministic diagnostic examples, not an independent
random generalization benchmark. No structural training or policy adaptation
has been claimed for the old front-only checkpoint.

## Structural training and refusal

`collect_hjepa.py --structural` creates 24 randomized door scenes, with 12 train,
6 validation and 6 held-out test scenes. Each of six observable families has
two training variants and one variant per other split: free two-way doors,
front/back pillars, two pillars and front/back-only handle access. Handle
position, friction, hinge resistance and pillar angle/radius vary independently.
One-way hinges with identical closed images remain ambiguity controls outside
the supervised family set; no hidden permission bit is supplied to the encoder.

Each scene has four physical base candidates, four grasps each and one virtual
positioning refusal record. Refusal uses the feature `[0,0,0,0]`, distinct from
physical yaw features with unit sin/cos norm. Its fixed return is zero regardless
of whether a different action could have succeeded. It never produces a physical
trajectory or a fictitious arrival observation for grasp training. Success,
collision and slip returns remain measured through the existing simulator.
The positioning warm start weights each base/refusal equally. The same policy
head dimensions and complete 23.3M parameter model are retained.

Fine-tuning loads the existing full model with a fresh optimizer, then trains
all world-model weights using the official 224x224 / batch 128 / BF16 / 9-epoch
profile and activation recomputation. State/action normalization uses training
scenes only. The selected world checkpoint is the minimum validation loss.
Up to 1,000 alternating offline policy updates then freeze that world model.
Policy snapshots are compared every 100 updates using validation returns;
the best pair is restored, with exact equality checks on frozen world weights.
The encoder latents are cached only for these frozen-head updates.

`evaluate_structural_training.py` selects actions from held-out current RGB-D
and candidate features before looking up any measured outcomes. It compares
the old checkpoint (using its own representation), the four pre/post stage-head
combinations at a fixed new world model, and latent lookahead. Correct refusals
are reported separately from opening successes. Final direct-policy choices
are also replayed through live perception and contact dynamics on all six test
scenes, with assertions against the matched candidate records.

```powershell
.\.venv\Scripts\python.exe collect_hjepa.py --structural --output runs/hjepa/structural_data --seed 20261017
.\.venv\Scripts\python.exe train_hjepa.py --data runs/hjepa/structural_data --output runs/hjepa/structural_model --init-checkpoint runs/hjepa/official_model/selected.pt --epochs 9 --policy-updates 1000 --device cuda
.\.venv\Scripts\python.exe evaluate_structural_training.py --device cuda
```

Use a structural `--scenario` when running this checkpoint: its four physical
bases include both faces. Outputs stay in ignored `runs/hjepa/` and previous
weights are preserved. Use a new output directory for a subsequent training run.

Validation checks can additionally compare `refit_structural_position.py`:
this freezes the complete world model and fits only the existing positioning
MLP on the same ViT's pre-projector CLS features. Training-only affine feature
standardization is folded into the MLP's existing first layer, so parameter
counts and shapes do not change. Train returns supervise soft-label ranking
(temperature 0.03 in return/15 units); validation return selects among 300
updates, checked every 10 updates. The replacement is accepted only if it
improves validation return. Original selected weights are archived before
replacement, and every non-position tensor must remain exactly equal.

Run this optional step **before** the first held-out evaluation. Policy
comparisons record the positioning representation interface for each version;
world-model/value/grasp latents continue using the official JEPA projection.
This is a task-policy adaptation, not a change to the official world model.
