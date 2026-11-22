# Overfitting review and output retention (v10, 2026-11-11)

**The selected primary policy shows no clear overfitting on fresh geometries within nine familiar templates; unseen templates and hardware remain untested.** An earlier H-JEPA last-step LoRA checkpoint did overfit (training 185/192, validation 9/24), but it is not this run's primary. Four-arm validation retains the old policy; retention selects different weights without validation improvement and is not promoted after testing.

## Training, validation, and frozen test

Train/validation/test have **72/36/72 scenes** in **18/9/18 complete groups**, using distinct parameters and four layouts within nine shared templates. Group/geometry identities do not overlap; new testing excludes **758** historical geometries. Freeze the manifest before training with `test_outcomes_available_at_creation=false`, and hash-freeze models, primary selection, aliases, and manifest before reading outcomes.

| Frozen policy | Training correct | Validation correct | New test correct | Test open / feasible | Test correct refusal / infeasible | Collisions | Test inspections | Test mean net utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Old primary `standard` (preselected)** | **71/72** | **35/36** | **70/72** | **50/52** | **20/20** | **0** | **43** | **0.5853** |
| `retention` (new weights, validation tie) | 71/72 | 35/36 | 71/72 | 51/52 | 20/20 | 0 | 41 | 0.6041 |

Old-policy correctness is **98.6%/97.2%/97.2%** across splits; new weights give **98.6%/97.2%/98.6%**, without the earlier large train/validation gap. Training is not independent generalization evidence; newly collected frozen-test geometries provide stronger evidence. Testing still shares nine templates, and validation was repeatedly used in development.

Four arms each complete **320 P/G plus LoRA and 1600 I updates**. All validation scores are 35/36, 25/26 openings, 10/10 correct refusals, zero collisions, and 21 inspections. Selected `standard`, `low_lr`, and `low_lr_retention` equal old v9/v6 tensor by tensor. Retention selects PG update 280 and I update 40 without validation improvement. Every last-step LoRA updates; selecting initialization is not failure to train. Preserve **422** core tensors/normalization bitwise and non-I tensors during I training.

## Matched results and limitations

New weights correct **1** additional test scene and remove **2** inspections. In `g006_front_blocked`, old primary falsely refuses after inspection, while retention selects station 2/grasp 0 and opens. Both retain the false refusal in `g004_rear_blocked`. In `g005_rear_blocked` and `g005_free`, retention opens directly while primary inspects first. No new failures occur. Mean utility difference **+0.01885**, path reduction **0.133 m**, and simulated duration reduction **0.764 s**.

One corrected decision does not establish a reliable accuracy gain. **Primary remains validation-selected `standard`**; deploying retention requires preregistered selection and another fresh frozen test. Old-policy v6/v8 score **68/72** and **72/72**; adding this **70/72** gives a descriptive **210/216** over three geometry cohorts. Correlated scenes, shared templates, and one seed do not establish cross-template/seed generalization. Exclude v5's different I weights.

Feasibility means success among frozen two-branch x four-station x two-grasp candidates rather than global continuous solvability. Navigation/preparation use known maps; opening uses current RGB-D, CAD, and physical contact with an ideal holonomic base. Zero collisions is an observed result rather than a guarantee. Preserve the earlier 185/192 versus 9/24 overfitting record in [H-JEPA history](hjepa_setup.md).

## Runtime, verification, and files

The original full run loses its process after arm four update 200; neither controller nor child remains, with no termination receipt or traceback, so **the cause is unknown**. Preserve all outputs. In a new ignored directory, hash-reuse the first three completed arms, teacher cache, and frozen manifest, then retrain arm four from update 0 with seed 42. Recovery takes **1492.54 seconds (24 minutes 53 seconds)**, exit 0; earlier arm times remain in their receipts rather than being replaced by recovery duration. Read-only development auditing then takes **134.58 seconds**, exit 0.

Formal estimate **3.5 hours, queue 0**, conservatively referencing previous 2-hour-15-minute and 2-hour-39-minute runs. Process **26884**, startup **17.33 seconds**, completion **10:13:39**, actual **7895.56 seconds (2 hours 11 minutes 36 seconds)**, exit 0; the agent sleeps continuously until estimated **11:32** before verification. Collect **1152** candidates. Pre-test model/normalization/inference/mode checks group seven names into four policies for **288** matching live replays, without counting aliases as repetitions. One reset per episode; **183** continuous opening-entry checks and **156** physical inspections pass. Source/weight/input hashes remain unchanged after testing. Repository size guard passes; arrays, weights, trajectories, and logs remain ignored in `runs/`.

[Training design](mobile_retention_plan.md) · [Pre-test protocol](mobile_retention_evaluation_plan.md) · [Four-arm selection hashes](../runs/mobile_partial_workspace_seed42_20261109/runs/p10_recovery_seed42_20261111/frozen_for_test.json) · [Development audit](../runs/mobile_partial_workspace_seed42_20261109/runs/p10_overfit_dev_seed42_20261111/report.json) · [New test report](../runs/mobile_partial_workspace_seed42_20261109/runs/p10t_seed42_20261111/report.json) · [Per-scene paired analysis](../runs/mobile_partial_workspace_seed42_20261109/runs/p10t_seed42_20261111/post_analysis.json) · [Replay/weight audit](../runs/mobile_partial_workspace_seed42_20261109/runs/p10t_seed42_20261111/tensor_and_physics_audit.json).
