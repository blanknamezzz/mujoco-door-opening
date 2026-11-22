# PRM before learning: verify a path to pregrasp

New default: initial station/joints -> PRM preflight -> physical preparation -> approach/grasp -> opening/hold -> record outcomes.

## Integration points

- `collect_grasp_data.py`: analyze every station/candidate, save `*_prm_preflight.jsonl`, then execute trials.
- `train_grasp_rl.py`: every selected angle passes PRM and physical tracking; enabled by default, with `--no-prm` for old initialization.
- `compare_grasp_learning.py`: read PRM settings from the manifest and share preparation across SL/RL/validation/test. Old manifests without PRM retain baseline compatibility.
- `prm_examples.py`: five reproducible cases, live MuJoCo demos, offline replay.

Defaults are `reproduced/grasp_learning_data_prm`, `reproduced/grasp_learning_comparison_prm`, and `reproduced/grasp_rl_prm`, preserving earlier no-PRM runs.

## What PRM checks

Fix base pose with fingers open and door closed. Nodes are four-dimensional joint configurations:

`q = [lift_m, shoulder_rad, elbow_rad, wrist_rad]`

Default initial joints `[0 m, -90 degrees, -120 degrees, 0 degrees]` are explicit rather than arbitrary, configurable with `--initial-q`. They stay fixed within training; networks still receive station features rather than varying-initial-joint inputs.

For each grasp angle, compute pregrasp 0.16 m from the handle along approach, then selected-branch IK. If absent, return `goal_ik_unreachable` without random sampling.

### CLEAR: configuration validity

Check joint ranges and contacts in separate probe MjData. Also use segment-to-oriented-box distance for arm capsules/panel to reject deeply penetrating initial states. Zero-distance support on floor/panel is allowed; penetration tolerance is 1e-5 m.

Probe data do not modify live execution state.

### LINK: edge validity

Interpolate joints with lift increments <=0.008 m and rotations <=0.06 rad, calling CLEAR at each point. Physical ranges prevent wrapping +165 to -165 degrees through the shortest circular angle.

These are discrete collision checks rather than continuous proofs; dynamic monitoring still follows.

### Sampling, neighbors, and search

Try direct start-goal LINK first. Success with zero random nodes is expected behavior.

After failure, sample four joint ranges, retaining 240 valid nodes and 12 neighbors by default. Normalize lift/rotation scales in distance. Dijkstra searches candidate neighbor edges; validate only the current path, delete invalid edges, and search again. This is Lazy PRM rather than prevalidated BasicPRM. Return `path_found=true` only after every final edge passes.

Reuse roadmaps within a worker for identical static scenes/settings. Base/model/planning changes select different caches. Finite failure is `no_path_within_budget`, not absolute infeasibility.

## Executing a found path

Reset once to specified initial joints per episode, then smoothly interpolate waypoints with position servos and `mj_step`. Apply bias compensation only to robot joints, without door actuation/external forces.

Monitor unexpected contact/early door motion and final tracking error/speed. Preserve reached qpos/qvel, reset the skill clock, and call `trial(reset_before_trial=False)` for grasp/opening without teleportation.

Record preparation time/path/failure separately. Existing rewards/time penalties still describe opening; preparation failure gets negative pipeline reward, and preparation time is not yet included in that time penalty.

## Distinguish planning, execution, and opening outcomes

| Outcome | Meaning |
| --- | --- |
| `goal_ik_unreachable` | No legal IK for this pregrasp target/elbow branch |
| `invalid_start` / `invalid_goal` | Specified endpoint collides or exceeds ranges |
| `no_path_within_budget` | Finite roadmap not connected; other budgets/methods may succeed |
| `direct_path` / `roadmap_path` | A pregrasp route passes current discrete geometry checks |
| Preparation execution failure | A geometric route exists but current control cannot execute safely |
| Opening failure | Pregrasp reached, but grasp/opening fails |

Preparation collisions are recorded/count toward pipeline collisions; CSV export works without an opening trajectory.

`opening_tested=false` means opening was not executed, rather than physically proved to fail. Negative supervised reward describes current planning/control outcome. Exclude unexecuted openings when fitting pure contact-success predictors; RL likewise optimizes current end-to-end pipeline reward.

PRM does not plan post-grasp robot-door closed-chain constraints. Base stays fixed and the original hinge-arc controller opens; `path_found` is not complete-opening success.

## Five examples

```bash
python prm_examples.py
# Open reproduced/prm_examples/index.html

# Watch preparation and opening live in MuJoCo:
mjpython prm_examples.py --view detour
mjpython prm_examples.py --view partial
```

| Example | Station | Distinction |
| --- | --- | --- |
| A: direct | (0.6, -1.0) | Direct succeeds without sampling, followed by full opening |
| B: detour | (0.6, -1.0), added obstacle | Direct blocked; 500 valid nodes find a lift-lowering detour, then full opening |
| C: limited_budget | Same as B | Zero samples checks direct only; B establishes a path despite this failure |
| D: partial | (0.3, -0.9) | Reach pregrasp physically, then workspace limit near 42.5 degrees |
| E: unreachable | (2.0, -1.2) | No pregrasp IK; skip motion/opening |

Gray edges are not all verified; green is the checked final path. Shoulder/elbow projection hides lift/wrist, so crossing lines need not share four-dimensional nodes. The right panel replays integrated joints rather than interpolated success-only roadmap waypoints.

## Training and independent testing

Complete default scale:

```bash
python collect_grasp_data.py
python compare_grasp_learning.py train
python compare_grasp_learning.py evaluate
```

This pilot uses a fresh scene split:

```bash
python collect_grasp_data.py --train-scenes 12 --validation-scenes 4 --test-scenes 8 --output reproduced/prm_training_data_final
python compare_grasp_learning.py train --data reproduced/prm_training_data_final --output reproduced/prm_training_comparison_final --seeds 42
python compare_grasp_learning.py evaluate --data reproduced/prm_training_data_final --output reproduced/prm_training_comparison_final
```

Total 24 stations and 504 candidate procedures, with 252 RL training decisions. This verifies preparation integration/reproduction rather than comparison against the earlier 36/12/24-station, three-seed experiment.

PRM-rejected candidates skip opening, so budgets count candidate decisions. Report `opening_trials_executed` and preparation failures separately rather than calling every decision an opening trial.

Changing initial joints:

```bash
python collect_grasp_data.py --initial-q 0 -1.57 -2.09 0 --output reproduced/prm_custom_start
```

Units are meters, radians, radians, radians. Evaluate under the same initial-state convention after training.

Numerical/physical regressions: `python -m unittest test_prm test_grasp_plan test_grasp_learning test_grasp_rl test_git_size -v`.
