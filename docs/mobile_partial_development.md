# Partial-map training and development results (v5, 2026-11-09)

Two arms each complete 320 LoRA/P-G and 800 independent inspection-head updates. Validation selects `depth_features_learned`: **35/36** correct, **25/26** openings, **0** collisions, and **23** inspections. This document preserves training/validation records from before new-test collection and controller replays. The 2026-11-10 [independent formal test is complete](mobile_partial_results.md); development metrics remain separate.

Both P/G arms select **update 0**, so new LoRA updates do not beat initialization. Changes in the final selected model come from I rather than improved P/G or evidence that P learned depth-feature use.

![Development correctness, inspections, and PG validation curves](mobile_partial_development.png)

## Data and information scope

Keep v3's 72 training and 36 validation candidate sets frozen. Collect 108 independent RGB-D prefixes with physical base motion, one initialization each, and reached-view capture after inspection arrival. The 19-dimensional states, path lengths, and simulated durations match original prefixes; maximum RGB rounding difference is 1 gray level. Original arrays, source, models, and receipts are preserved.

Opening sweeps use only door/robot CAD and currently observed depth; obstacle geometry and layout labels do not enter mapping. Initial/inspected maps use depth available at their respective stages. Unknown space stays unknown. Each station receives observed obstruction, sampled visibility, and observed clearance. Navigation and preparation retain known-map services rather than complete unknown-environment navigation.

| Map diagnostic | Training, 72 scenes | Validation, 36 scenes |
| --- | ---: | ---: |
| Initially allowed stations blocked by the full map | 36 | 18 |
| Post-fusion allowed stations blocked by the full map | 0 | 0 |
| Post-fusion excluded stations allowed by the full map | 0 | 2 |

These count station records rather than independent successful/failed scenes. The two validation exclusions are layouts in one group, caused by silhouette endpoints and inflated point clouds. Retain this run's settings and records without interpreting the map as an exact safety certificate.

Earlier full-map development controls reach 35/36 with zero inspections using rear-first geometry and fixed -15-degree grasping. That reference has extra information and reveals information confounding in earlier inspection gains, rather than establishing that inspection is unnecessary everywhere. [Design and full-map controls](mobile_partial_map_plan.md).

## Matched training and validation

Both arms initialize from v4 `factorized/selected.pt`, adding equal P/I input slots with matching partial-map action constraints and random sequences. `images_only` zeros map slots; `depth_features` uses current physical depth features. Freeze the H-JEPA core; stage 1 continues rank-8 LoRA/P/G, and stage 2 freezes encoder/LoRA/P/G while training I only.

Use success/ordinary-failure/collision cross-entropy. Terminal utility is +1, -1, -1-c, and 0 for refusal, with c=1 by default, minus actual inspection distance/time cost. Train I on outcomes of actions selected by frozen continuation, using only initial image/map, relative inspection target, and declared risk. Future views/maps and oracle candidates are excluded from features.

| Model | Inspection | Open /26 | Correct /36 | Collisions | Additional inspections | Mean net utility |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Image features | Never | 15 | 24 | 0 | 0 | 0.3889 |
| Image features | Always | 25 | 35 | 0 | 36 | 0.5133 |
| Image features | Learned | 25 | 35 | 0 | 24 | 0.5725 |
| Image + depth features | Never | 15 | 24 | 0 | 0 | 0.3889 |
| Image + depth features | Always | 25 | 35 | 0 | 36 | 0.5133 |
| Image + depth features | Learned | 25 | 35 | 0 | 23 | 0.5774 |

The two I arms select updates 40/280. Later PG checkpoints return to initialization's validation metrics, and the predefined earliest-tie rule retains update 0. Selected P's new map columns remain zero; depth-arm I map columns update. One fewer validation inspection with depth features is a development difference rather than independent generalization evidence.

All 8 stage checkpoints preserve **422** frozen tensors and normalization bitwise. Both last-step PG checkpoints update **148** LoRA tensors; each selected I updates **6** tensors, preserving every non-I tensor during stage 2. Twenty map, information, capacity, checkpoint, and existing outcome-supervision tests pass.

## Runtime evidence and next stage

| Phase | Runtime estimate / queue | Actual duration | Authoritative session |
| --- | --- | --- | --- |
| Full-map geometric controls | 5 minutes / 0 | 159.42 seconds | 47352, exit 0 |
| Four-scene RGB-D pilot | 2 minutes / 0 | 42.19 seconds | 36086, exit 0 |
| 108-scene prefix collection | 20 minutes / 0 | 855.06 seconds | 22812, exit 0 |
| Twenty tests + two-model smoke run | 6 minutes / 0 | 117.36 seconds | 76762, exit 0 |
| Full two-model training and selection | 1 hour / 0 | 2724.39 seconds; controller 2726.65 | 62644, exit 0 |

Full stages take image PG 1288.06 seconds, I 72.44; depth PG 1237.81, I 72.07; loading/selection/audits 54.00. Mean PG updates are 2.54/2.41 seconds. Prefix collection runs without concurrent LoRA training. Each launch records separate runtime/queue estimates, verifies its live process, sleeps continuously to the deadline, and checks the same authoritative session without frequent polling.

New runs fix Python/NumPy/Torch/CUDA seed=42 and record CUDA/hash deterministic settings. Torch memory-efficient attention still warns about nondeterministic operators under `warn_only=True`; OpenGL can produce sparse pixel rounding, so cross-run bitwise determinism is not guaranteed.

Freeze **18 new groups and 72 test scenes**, excluding 450 historical geometries with group-disjoint splits. Freeze time is 2026-11-09 22:30:19 +08:00; primary policy `depth_features_learned`. No new-test outcomes exist when training completes. Subsequent collection and live replays retain models/policies/maps and report preregistered full-map, partial-map never-inspect, visibility-rule, and frozen-v4 always-inspect controls. Formal results are linked above.

[Training/validation report](../runs/mobile_partial_training_seed42_20261109/report.json) · [Frozen-test protocol](../runs/mobile_partial_training_seed42_20261109/frozen_for_test.json) · [Tensor audit](../runs/mobile_partial_training_seed42_20261109/tensor_audit.json) · [Process/sleep verification](../runs/mobile_partial_training_seed42_20261109/verification.json) · [RGB-D collection](../runs/mobile_partial_development_seed42_20261109/report.json).

Arrays, checkpoints, logs, and trajectories remain in ignored `runs/`. The actual workspace `runs/mobile_partial_workspace_seed42_20261109/` binds frozen v3 physics. The root contains a different later `simulate.py`; reuse rejects mixing it with old physical outcomes. Reproduce from the isolated workspace/archived source rather than assuming identity from root filenames.
