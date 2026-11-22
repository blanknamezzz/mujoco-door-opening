# Depth-fusion calibration retains the original map (v7, 2026-11-10)

Calibration covers 108 training/validation scenes and four map profiles. All **648 initial/fused mask comparisons** between the three new profiles and historical match exactly. Validation does not improve, so the predefined tie order retains **historical**. No models are updated, test outcomes read, or new maps deployed as primary.

The implementation and 4 contracts confirm that cross-frame deduplication prevents extra occupied support from repeated frames and free-ray rules preserve unknown space. However, exclusions in these physical development data remain. Duplicate counting alone does not explain the development errors; a code-level property improvement is not a measured task gain.

| Validation diagnostic (36 scenes) | Historical | Voxel union | Union + free rays | Union + 3 mm inflation |
| --- | ---: | ---: | ---: | ---: |
| Initially excluded stations allowed by full geometry | 2 | 2 | 2 | 2 |
| Initially allowed stations blocked by full geometry | 18 | 18 | 18 | 18 |
| Post-fusion excluded stations allowed by full geometry | 2 | 2 | 2 | 2 |
| Post-fusion allowed stations blocked by full geometry | 0 | 0 | 0 | 0 |
| Successful candidates excluded after fusion | 0 | 0 | 0 | 0 |
| Collision candidates allowed after fusion | 0 | 0 | 0 | 0 |

Station records and related candidates are not independent scenes. The two additional exclusions do not remove successful physical candidates in this validation set, so their full-geometry differences do not establish two lost task successes.

## PG selection issue found during calibration

Reuse validation candidate tables and frozen PG checkpoints only, without new physical replays:

| Checkpoint / mode | Correct /36 | Open /26 | Correct refusal /10 | Collisions | Mean net utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| Update 0, direct | 24 | 15 | 9 | 0 | 0.3889 |
| Update 320, direct | 25 | 22 | 3 | 2 | 0.3333 |
| Update 0, always inspect | 35 | 25 | 10 | 0 | 0.5133 |
| Update 320, always inspect | 35 | 25 | 10 | 0 | 0.5133 |

Previous PG validation considered only always inspection, missing direct-branch changes. Direct execution gains one correct decision but adds two collisions and lowers net utility. Future experiments check safety on both branches before comparing paired layouts and collision-weighted objectives, retaining the original map. [New experiment design](mobile_branch_balance_plan.md).

Additional perception failures are training 168/1152 and validation 88/576 `handle_not_detected` candidates. These are correlated execution records rather than scene-detection rates. Perception code is unchanged in this phase.

## Runtime, sources, and scope

Queue 0; estimated 8 minutes. Controller 12696, process 32240, authoritative session **32340**. Startup 6.50 seconds; three new profiles on the first scene take 1.62 seconds. The agent sleeps once for **457.39 seconds** until estimated 11:14:28, then confirms the same session completed at **11:09:45**, exit **0**. Actual pipeline **194.52 seconds**, controller 196.84: map loop 137.71, two PG validation forwards 49.06, startup 6.50, other 1.24. All 506 input and 123 source hashes remain unchanged.

The free-ray variant follows occupied/free/unknown distinctions in [Hornung et al., OctoMap, 2013](https://www.arminhornung.de/Research/pub/hornung13auro.pdf). It does not implement OctoMap's octree or probabilistic fusion, and the reference does not establish a benefit on this task.

[Pre-run profiles and selection rules](mobile_partial_fusion_plan.md) · [Full calibration report](../runs/mobile_partial_fusion_calibration_seed42_20261110/report.json) · [Validation PG diagnostic](../runs/mobile_partial_fusion_calibration_seed42_20261110/validation_pg_diagnostic.json) · [Runtime and mask verification](../runs/mobile_partial_fusion_calibration_seed42_20261110/verification.json).

Actual outputs are `runs/p7_cal_seed42_20261110/` inside the frozen v3 physics workspace. Original RGB-D, weights, and receipts are preserved. New arrays/logs remain ignored in `runs/`; new experiments use seed 42 and record Python/NumPy/Torch/CUDA settings and nondeterministic limitations.
