# Inspection-regret training and development results (v6, 2026-11-10)

Three arms each complete **1600 I updates**. On the same 36 validation scenes, outcome supervision scores **35/36** with **23** inspections; joint-regret and regret-only objectives score **35/36** with **21** inspections, all with **0** collisions. The predefined order selects `joint_regret` at update **40**. Preserve these development records separately from the subsequent [new frozen independent test](mobile_inspection_regret_results.md).

## Controls and information scope

Initialize all arms from the same v5 depth-feature `selected_pg.pt`, matching capacity, inputs, sampling, 1600 updates, rate 3e-4, and weight decay 0.01. Set Python/NumPy/Torch/CUDA/hash seed=42 and warn-only determinism. Retain H-JEPA/rank-8 LoRA with core/LoRA/P/G/normalization frozen; update I only to isolate the objective, without full-model training.

Supervision compares returns of actual actions selected by frozen P/G on direct/physical-inspection branches, minus actual inspection cost, rather than oracle candidates. Weight decision BCE by the absolute net-return difference. Deployment retains v5's comparison of predicted inspection value against initial station/refusal value. Future views and outcomes supply labels only; I sees initial image, relative target, declared risk, and initial depth features. [Fixed design](mobile_inspection_regret_plan.md).

| Objective | Updates | Selected update | Open /26 | Correct refusal /10 | Correct /36 | Collisions | Inspections | Mean net utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Outcome CE + cost MSE | 1600 | 280 | 25 | 10 | 35 | 0 | 23 | 0.5774 |
| **Outcome + decision regret (validation selected)** | **1600** | **40** | **25** | **10** | **35** | **0** | **21** | **0.5876** |
| Decision regret + cost MSE | 1600 | 120 | 25 | 10 | 35 | 0 | 21 | 0.5876 |

Both new objectives retain opening/refusal counts, reducing mean path from 3.137 to 2.772 m and mean duration from 40.86 to 37.95 simulated seconds. Outcome supervision reproduces v5 depth validation metrics. Both regret arms tie, so preregistered arm order retains the joint objective.

Validation scenes were used in previous development and are not independent new evidence. Regret-only training may compensate for P0 value errors, so its logits are not calibrated probabilities. Scope remains one seed, frozen continuation, familiar categories, and known-map navigation, without new P/G or LoRA accuracy gains.

## Freezing and verification

Freeze **18 groups and 72 new test scenes**, excluding **526** historical geometries; retain old training/validation 72/36. Freeze primary at **2026-11-10 06:19:51 +08:00**. Manifest SHA256: `7cab1ea9650ba28f80eda83bad381350897d41aee93874e9d7015881cb971639`. Read **0** test outcomes during this phase.

Six selected/last checkpoints preserve all non-I tensors, 422 core tensors, and normalization; selected/last I each update 6 tensors. Preserve historical inputs, frozen models, and launch-source hashes. Seven contracts pass, covering selected-action labels, collision risk, positive/negative gradients, zero regret/forced refusal, and existing v5 capacity/checkpoints.

Subsequent fresh testing compares three arms with preregistered frozen-v5 learned/always/never controls using live RGB-D and full physical replays. Retain maps, executors, and frozen v3 physics in the isolation workspace without mixing root-level `simulate.py`.

## Runtime and waiting

| Run | Runtime estimate / queue | Actual duration | Authoritative session |
| --- | --- | --- | --- |
| Seven contracts + three-arm 8-update smoke test | 2 minutes / 0 | Controller 115.65 seconds; pipeline 89.05 | 57723, exit 0 |
| First formal launch | 10 minutes / 0 | 0.80 seconds; directory failure before training | Synchronous tool confirms exit 1 |
| Three full 1600-update arms | 10 minutes / 0 | Training 205.48 seconds; controller 207.77 | 20635, exit 0 |

The estimate uses smoke warmup of about 0.034-0.040 seconds per update and the previous approximately 72-second, 800-step I stages, allowing full caches, validation, and audits. Actual cache time is 64.62 seconds; arms take 28.92/37.02/34.15; startup/I/O/audits 40.77. Record measured update throughput and full-stage durations separately from estimates.

After a continuous 100.60-second smoke sleep, inspect the same session. Formal restart: 06:16:25, PID 22992, controller 31256. Confirm liveness and sleep toward 06:26:25; a user scheduling message interrupts after 7.56 seconds. After handling scheduling, the same session has completed at **06:19:52**, so no further wait or duplicate launch is needed.

The first launch failed because a convenience entry created an empty output directory before a trainer called mkdir without allowing existing directories. No training or test reads occurred. Preserve failed source/receipt, fix empty-directory handling, and restart in a new directory without objective or parameter changes.

## Original records

Keep models, arrays, labels, logs, telemetry, and source in new ignored `runs/`, outside Git.

- [Training report and validation curves](../runs/mobile_inspection_regret_training2_seed42_20261110/report.json)
- [Pre-test selection and frozen hashes](../runs/mobile_inspection_regret_training2_seed42_20261110/frozen_for_test.json)
- [Checkpoint audit](../runs/mobile_inspection_regret_training2_seed42_20261110/tensor_audit.json)
- [Process, sleep, and completion verification](../runs/mobile_inspection_regret_training2_seed42_20261110/verification.json)
- [Full training launch source](../runs/mobile_inspection_regret_training2_seed42_20261110/source/)

Report formal v5 testing separately from these development results: [v5 formal test](mobile_partial_results.md). Subsequent v6 formal testing is linked above; distinguish both test cohorts from validation in the final report.
