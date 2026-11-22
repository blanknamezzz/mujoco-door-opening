# Outcome factorization and measured door-sweep planning (v4, 2026-11-09)

Validation-selected **`factorized_always_sweep`** makes **71/72 correct decisions**, opens **52/53 feasible scenes**, and has **0 collisions** on 72 fresh scenes. The matched frozen-v3 baseline scores 61/72 with 50/53 openings and 1 collision. All 864 physical replays match; the original core remains frozen.

Clearer benefits come from known-geometry constraints and continued training. Equal-capacity outcome factorization does not outperform scalar-return supervision.

![Twelve frozen-policy comparisons](mobile_outcome_results.png)

## Training and data

Both models initialize from v3 validation-selected `view_free/selected.pt` with matching capacity, initialization, data, and update budgets:

| Stage | Completed training | Validation-selected checkpoint |
| --- | --- | --- |
| Scalar positioning/grasping | 320 updates to existing rank-8 encoder LoRA and P/G | Update 320 |
| Scalar inspection head | 800 updates; P/G, encoder, and LoRA frozen | Update 1 |
| Factorized positioning/grasping | 320 updates to existing rank-8 encoder LoRA and P/G | Update 200 |
| Factorized inspection head | 800 updates; P/G, encoder, and LoRA frozen | Update 280 |

Record completed budgets and selected checkpoints separately; last-step weights do not override validation selection. Stage 2 isolates inspection gradients. P/G still share LoRA, without gradient-angle measurement or projection in this run.

Reuse v3's **72 training and 36 validation scenes**, verifying physics/source/input hashes without changing arrays. Generate **18 fresh groups and 72 test scenes**: two parameter groups per familiar template, each with front/rear/both/no-obstacle layouts. Split whole groups and exclude historical geometry.

Execute direct/physical-inspection branches x 4 stations x 2 grasp angles from scratch: **1,152 new candidates**. Replay 12 policies on 72 scenes: **864 replays**. Freeze checkpoints, policy selection, and hashes before test collection.

New training, collection, and analysis fix **seed=42**, recording Python/NumPy/Torch/CUDA/startup hash seeds. Retain v3 physics, familiar handles, cameras, and ideal base, plus independent main-branch grasp experiments and historical tests.

## Reward differences

P/G output success, ordinary-failure, and collision probabilities. At risk coefficient 1 their values are `1, -1, -2`.

- `scalar`: combine probabilities into expected return and regress actual action return with MSE.
- `factorized`: classify the same physical outcome with cross-entropy, retaining supervision on failure composition.

P targets follow outcomes of current G's chosen actions. Supervise G only when its observation exists; mask downstream supervision after unsafe inspection. Refresh labels on training only without oracle action selection.

A separate I network predicts outcomes of frozen P/G, adding downstream refusal (value 0), and fits physical inspection cost with a positive output:

```text
cost = 0.2 * actual_inspection_distance(m)/6 + 0.1 * actual_inspection_time(s)/60
Inspection value = expected frozen-continuation return - predicted inspection cost
```

Both I models train on risk 1/3 crossed with ordinary/sweep planning. These are known execution inputs; scene IDs, layout labels, future images, and observed outcomes are excluded. Additional risk3 policies use the same factorized model with collision value -4; all primary test metrics use common risk 1 and actual motion cost.

## Planning differences

Plain retains known-geometry A*, RRT preparation, and contact opening. Sweep adds door-leaf/handle checks from closed to 90 degrees:

- Query door components against static obstacles and target base footprint in separate `MjData` every 1 degree, with 1 mm penetration tolerance.
- Exclude predicted colliding stations before selection; if all are excluded, refuse at the initial pose without inspection/navigation.
- Contact still drives the physical door; geometry queries preserve live door angle, robot state, and time.

The full known map adds geometric information. Queries do not cover every moving-arm trajectory or contact dynamic, and discrete sampling is not a continuous-collision proof.

The feasibility denominator remains the unfiltered candidate-success union over both branches: **53 feasible scenes and 19 library-infeasible scenes**. False refusals from conservative masks remain errors.

## Complete fresh test results

Correct decisions are complete openings or library-infeasible refusals. Inspection counts are actual physical inspections.

| Policy | Open /53 | Correct refusal /19 | False refusal | Correct /72 | Collisions | Inspections | Mean net utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen v3, always, plain | 50 | 11 | 3 | 61 | 1 | 72 | 0.3886 |
| Frozen v3, always, sweep | 50 | 18 | 3 | 68 | 0 | 54 | 0.5463 |
| Scalar, always, plain | 51 | 17 | 1 | 68 | 0 | 72 | 0.4858 |
| Scalar, always, sweep | 52 | 19 | 1 | 71 | 0 | 54 | 0.5880 |
| Scalar, learned, plain | 52 | 16 | 1 | 68 | 0 | 38 | 0.5839 |
| Scalar, learned, sweep | 52 | 19 | 1 | 71 | 0 | 21 | 0.6695 |
| Factorized, always, plain | 50 | 16 | 1 | 66 | 0 | 72 | 0.4442 |
| **Factorized, always, sweep (validation selected)** | **52** | **19** | **1** | **71** | **0** | **54** | **0.5880** |
| Factorized, learned, plain | 47 | 17 | 3 | 64 | 0 | 33 | 0.4985 |
| Factorized, learned, sweep | 51 | 19 | 2 | 70 | 0 | 24 | 0.6481 |
| Factorized risk3, learned, plain | 47 | 17 | 4 | 64 | 0 | 37 | 0.5028 |
| Factorized risk3, learned, sweep | 49 | 19 | 4 | 68 | 0 | 22 | 0.6252 |

Validation correctness in order is 31, 34, 32, 34, 32, 34, 30, 35, 30, 34, 30, 33. Select primary before testing. Scalar learned inspection has higher test utility but remains the preregistered secondary control.

Findings:

1. **Planning has a clear effect.** Adding sweeps to the same frozen old model raises correctness from 61 to 68 while openings remain 50, mainly reducing infeasible attempts. Exclude all 168 colliding candidates and retain all 192 successful candidates; these are correlated records rather than 360 independent scenes.
2. **Continuation helps without extra factorization benefit.** Plain scalar/factorized score 68/72 and 66/72; always-inspect sweep gives 71/72 for both. Retain scalar supervision as a strong baseline.
3. **Learned inspection reduces action cost.** With the same scalar/sweep weights, always/learned per-scene correctness matches. Inspections fall from 54 to 21, mean episode duration from 49.12 to 29.66 simulated seconds, and utility rises 0.0815.
4. **Higher collision penalty does not improve current outcomes.** Factorized sweep/learned risk1 versus risk3 both have zero collisions, but openings fall from 51 to 49 and false refusals rise from 2 to 4.

Never-inspect plus sweep was not a preregistered formal-test policy. A subsequent [read-only development diagnostic](mobile_no_observation_control.md) gives scalar validation 27/36 without inspection versus 34/36 learned, without test reads or primary reselection. Full-map geometric station rules and P0 calibration remain comparisons to make.

## Paired analysis and scope

Primary gains 10 correct decisions (+13.89 percentage points) over frozen v3. Exploratory exact sign-flip p=0.00195 clustered by 18 groups and p=0.03125 by 9 templates; the template-bootstrap 95% interval is +6.94 to +20.83 percentage points.

These analyses lack multiplicity correction and cover one seed, four familiar handles, and known maps. Plain factorized minus scalar correctness is -2, exploratory p=0.5. Results motivate further planning/staging study rather than conclusions about unfamiliar categories, partial maps, or real wheeled robots.

## Weight, physics, and freeze verification

- **864/864 replays match** actions, refusals, success, and collisions; every episode initializes once.
- **604** opening entries preserve qpos/qvel continuously; all timestamps increase monotonically.
- **553** physical inspections; successful inspections do not push the door early.
- **108** all-stations-excluded replays refuse at the initial pose without inspection.
- Maximum recorded translation speed **0.28124 m/s**; adjacent displacement **0.01653 m**.
- **8 stage checkpoints** across both models preserve **422 non-policy, non-LoRA tensors** and normalization bitwise.
- Both selected PG models change **148 LoRA tensors** from v3 initialization. I training preserves selected PG/all LoRA, while each selected I changes 6 tensors.
- **106 launch source/configuration** hashes match current isolation/archives; initialization, old frozen files, reused development data, manifest, and selected-model hashes are preserved.
- Frozen files precede every new-test observation; training reads 0 test arrays.

The smoke pipeline passes **41 tests** and 48 replays; formal evaluation performs full audits above. Preserve the original CPU/GPU normalization-transfer failure, receipt, and source, then rerun in a fresh directory after repair.

## Runtime and durations

Queue 0. Launch 14:31:17, PID **3352**, session **22687**; estimate 5.5 hours with a 20:01 check. Sleep continuously about **19,743 seconds**, then confirm the same session exits **0**. Completion **19:21:03**; actual **17,385.41 seconds, 4 hours 49 minutes 45 seconds**.

| Phase | Seconds | Minutes |
| --- | ---: | ---: |
| Reuse development data and query sweeps | 7.04 | 0.12 |
| Scalar PG training, refresh, validation | 1,262.57 | 21.04 |
| Scalar inspection stage | 71.96 | 1.20 |
| Factorized PG training, refresh, validation | 1,250.16 | 20.84 |
| Factorized inspection stage | 71.99 | 1.20 |
| Collect 72 scenes, 2 workers | 5,127.66 | 85.46 |
| 864 physical replays | 9,337.88 | 155.63 |
| Remaining selection, I/O, audits | 256.15 | 4.27 |

Collection and GPU training do not overlap. Mean PG updates take 2.45/2.44 seconds. Device telemetry is logged once per minute; training temperatures range 49-53 and 52-53 degrees C, insufficient alone to identify bottlenecks.

Mean replay time **10.81 seconds/case** exceeds smoke's 7.57, with more inspections/opening entries in formal scenes. Estimate future jobs from complete stages and action composition, allowing I/O and load variation rather than presenting smoke extrapolations as measurements.

## Files and reproduction

Convenience entry `runs/mobile_outcome_seed42_20261109/` points to the ignored isolation workspace. Arrays, models, logs, and telemetry remain in `runs/`, outside Git.

- [Main results and phase durations](../runs/mobile_outcome_seed42_20261109/report.json)
- [Pre-test selection and hashes](../runs/mobile_outcome_seed42_20261109/frozen_for_test.json)
- [Full tensor/physics audit](../runs/mobile_outcome_seed42_20261109/tensor_and_physics_audit.json)
- [Paired analysis and telemetry](../runs/mobile_outcome_seed42_20261109/post_analysis.json)
- [Protocol](../runs/mobile_outcome_seed42_20261109/protocol.json)
- [Full launch source](../runs/mobile_outcome_seed42_20261109/source/)

Preserve the main branch's original `simulate.py` and independent grasp updates. Development reuse requires matching physics source; formal receipts bind the isolated environment/launch snapshot. Reproduce there with a **new directory and stream**:

```powershell
$python = 'E:\Code\mujoco-door-opening\.venv\Scripts\python.exe'
Push-Location runs/mobile_outcome_workspace_seed42_20261109
& $python run_mobile_outcome_experiment.py `
  --output runs/mobile_outcome_new `
  --stream mobile_outcome_v4_new `
  --pg-steps 320 --gate-steps 800 --test-groups 18 `
  --estimate-seconds 21600
Pop-Location
```

`21600` is a 6-hour reproduction budget rounded up from complete stages/action composition; record queue time and refine for actual load. Integrated code adds this manifest to historical exclusions; launch snapshots retain original identity. Read archives with `audit_mobile_outcome.py --output runs/mobile_outcome_seed42_20261109 --archived-sources`.

Next compare never-inspect plus sweep against learned inspection on training/validation, then current-visible-map constraints to distinguish geometry/vision effects. Inspect gradients in shared P/G LoRA. Select new thresholds, objectives, and checkpoints on development data before a fresh frozen final test.
