# v5 frozen independent test and live RGB-D replay

Training/validation selected `depth_features_learned`. This phase evaluates frozen weights without changing models, rewards, maps, or thresholds. Preserve training records; see [development results and limitations](mobile_partial_development.md).

## Execution and controls

For each test scene, execute two branches (direct/physical inspection) x four stations x two grasps: 16 complete physical candidates. In separate episodes, each controller then makes fresh decisions from current camera images and depth. Cached candidates support outcome comparison without supplying live maps or actions. Initialize once per task, physically move the base for inspection, and open through gripper contact.

Retain v3 navigation, arm preparation, and opening interfaces, adding only initial RGB-D forwarding and fresh reached-depth reads. Interface checks confirm matching final q/v, trajectories, integration counts, and opening outcomes for old/new executors on the same scenes, with no inspected data in the initial callback. Camera reads must preserve q/v and simulated time.

Six learned policies cross image-only/image-plus-depth arms with never/always/learned inspection. Four references were declared before formal training:

| Policy | Opening map | Station and grasp | Inspection rule |
| --- | --- | --- | --- |
| `geometry_full_rear` | Complete known geometry | Rear first, fixed -15 degrees | Never |
| `geometry_depth_never` | Actual initial depth | Nearest allowed station, fixed -15 degrees | Never |
| `geometry_depth_rule` | Actual current depth | Nearest allowed station, fixed -15 degrees | Inspect once if sampled visibility at the initial selected station is <0.4 |
| `v4_always_depth` | Actual current depth | Frozen v4 P/G | Refuse if all initial actions are excluded; otherwise always inspect |

The full-map reference has extra obstacle information and is reported separately. Other opening constraints receive only door/robot CAD, current proprioception, and observed depth. Navigation and arm preparation retain known-map services rather than representing complete unknown-environment navigation.

## Data and verification

First check the full evaluation pipeline on the training smoke test's previously frozen 4-scene test manifest, using selected full-training weights. These scenes are disjoint from the formal 72-scene test and serve only execution verification. Afterward run the formal 18-group, 72-scene test without primary-policy reselection.

The formal test includes 1152 physical candidates and 10x72=720 independent controller replays. Compare cached/live inspection, refusal, station, grasp, opening, and collision outcomes, plus masks generated from live depth. Check one initialization, monotonic time, continuous opening-entry q/v, base speeds, and trajectory steps. Verify hashes for original inputs, 8 stage checkpoints, training/evaluation source, 422 original tensors, normalization, and non-I tensors during I training.

New runs set seed=42 and record Python, NumPy, Torch, CUDA, hash seeds, and deterministic limitations. Keep arrays, weights, logs, and trajectories in new ignored `runs/`. Use the training isolation workspace without mixing later root-level physics code.

Estimate runtime and queue time separately at launch. After confirming process/session and startup throughput, sleep continuously until the estimated deadline, then inspect local `origin/main` and the same authoritative session. Extend waiting for unfinished work rather than launching duplicates after an observation timeout.
