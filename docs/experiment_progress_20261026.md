# Overall experiment progress after local/remote integration

As of 2026-10-26. Fast-forward remote `51df28b` (grasp learning) and `72b3e60`
(PRM pregrasp checks) into local `main`, preserving all local H-JEPA workspace changes.

## Overall findings

Door-side selection, grasp-angle learning, and pregrasp planning have independent experiments but no unified-test evidence
for full-pipeline gains. Their success rates cannot be combined into one end-to-end opening rate.

| Module | Existing result | Current finding |
| --- | --- | --- |
| H-JEPA side/refusal selection | Latest validation-selected model scores 67/96, matching initialization | Stable generalization benefit remains unconfirmed |
| Independent grasp-angle learning | SL 37.50%, RL 38.89%, training-selected fixed +18 degrees 41.67% | Learned policies do not beat the strong fixed baseline |
| PRM pregrasp | Detour and complete opening; small integration SL/RL/fixed controls each 3/8 | Path preparation is demonstrated without a measured success-rate gain |

PRM is integrated into independent grasp learning. Main H-JEPA still uses pregrasp initialization and its own candidates,
without PRM or the remote SL/RL angle policies. PRM uses known MuJoCo geometry rather than showing that
the world model learns whole-path collision checking from vision alone.

## Local H-JEPA: independent multi-scene tests

| Independent comparison | Matched control | Validation-selected model | Outcome |
| --- | --- | --- | --- |
| A: 192 vs 576 training scenes | 49/96 | 51/96 | Collisions 15 to 9; false refusals 19 to 14; wrong sides 11 to 14 |
| B: CLS vs mean patches | 59/96 | 60/96 | Seven corrected, six regressed; only 4/30 infeasible scenes correctly refused |
| C: Refusal calibration | 67/96 | 67/96 | Openings 61 to 57, correct refusals 6 to 10; total unchanged |

Rows use different tests rather than one test improving from 51 to 67.
C's BCE candidate scores 69/96 but loses validation selection and is not promoted after testing.
The three rounds total 288 independent test scenes, frozen before each module's selection.

These stages use the official full architecture with LoRA/task-head adaptation, preserving backbone and historical models.
Recent work mainly studies endpoint supervision for positioning rather than retraining the entire temporal world model.
Independent benefit of H-JEPA lookahead over direct policies remains unestablished.

Sources: [coverage experiment](hjepa_setup.md#seed-42-coverage-expansion-2026-10-26),
[pooling experiment](hjepa_setup.md#native-vit-pooling-study-2026-10-26),
[refusal calibration](hjepa_setup.md#refusal-calibration-and-binary-side-supervision-2026-10-26),
and their `runs/hjepa/*/independent_test/heldout.json` records.

## Remote grasp-angle learning: compare strong fixed baselines too

The historical no-PRM experiment has 36/12/24 training/validation/test stations and 21 discrete angles.
Three training seeds share 24 test stations rather than 72 independent samples.

| Method | Matched test successes | Mean success rate |
| --- | --- | --- |
| SL, full-angle sampling | 9, 9, 9 /24 | 37.50% |
| REINFORCE, online exploration | 10, 9, 9 /24 | 38.89% |
| SL, reused RL records | 9, 8, 8 /24 | 34.72% |
| Fixed 0 degrees | 7/24 | 29.17% |
| Training-selected fixed +18 degrees | 10/24 | 41.67% |

Only 10/24 test stations have successful grid angles; fixed +18 degrees covers all ten.
The approximately 1.39-point mean RL-SL gap does not establish a stable algorithmic advantage.
These are new stations for one door family rather than new door types or real robots.

Values come from the published remote [report](grasp_learning_results/REPORT.md),
[analysis](grasp_learning_results/ANALYSIS.md), and
[aggregate metrics](grasp_learning_results/aggregate_metrics.json).
Remote records verify 179 distinct post-freeze action replays; this integration does not retrain the historical models.

## Remote PRM: reaching pregrasp is not complete opening

After integration, rerun all 5 examples locally:

| Scene | Path outcome | Physical result |
| --- | --- | --- |
| Clear | Valid direct connection | Full opening/hold near 89.7 degrees |
| Small obstacle blocks direct connection | Five-waypoint roadmap path with 500 valid nodes | Detour then full opening/hold near 89.7 degrees |
| Same obstacle, zero sampling budget | No path found within budget | Opening skipped; not proof of impossibility |
| Reachable handle, limited continuation | Reachable pregrasp path | Workspace limit near 42.5 degrees |
| No legal pregrasp IK | Exclude this target/elbow branch | Execution skipped |

Remote integration also covers 12/4/8 stations and 504 candidate procedures, with SL/RL/fixed controls each scoring 3/8,
and 32 matching distinct-action replays. This differs from the historical 24-station no-PRM test,
so 3/8 versus 10/24 does not measure a PRM gain. Five selected examples are not a generalization-rate sample.

Sources: [PRM measured report](prm_results/REPORT.md),
[integration verification](prm_results/verification.json).
Local reproduction record: `runs/git_sync20261026/prm_examples/summary.json`.

## Git integration and compatibility

- Direct conflict was limited to README, retaining both sides. Grasp candidate selection overlaps in research scope,
  but implementations are not yet one policy.
- Back up 57 workspace files and preserve a branch/stash before integration. Restore 56 non-README files identically,
  with unchanged SHA256 for 5 checkpoints.
- Fix UTF-8 template/scene-XML reads and writes in `prm_examples.py` to avoid Windows default-encoding differences.
- New collection, training, and PRM default to seed 42; published three-seed and five-example settings remain explicit.
- Initial integration passes 109 tests; seed-default regressions bring the total to 110.
  All 4,004 default skill commands and 404 world-model action features match pre-merge values.
- Four old/new executor cases match, comparing missing contact values with NaN semantics.
- Replaying the selected H-JEPA model on its original 96 scenes matches every action, outcome, and reward.
- Verification takes about 237 seconds; estimate 10 minutes, queue 0, with one continuous sleep to the deadline.

Local `HEAD` and fetched `origin/main` both point to `72b3e60`.
Experiment code and these additions remain local uncommitted changes; this integration makes no remote push.
Audits/backups, including `integration_verification.json`, are in `runs/git_sync20261026/`.

## Next integration step

Combine side selection, pregrasp paths, and grasp policies under a fresh unified protocol, retaining fixed-angle and original-initialization controls.
Score preparation failures, opening failures, and correct refusals separately; finite PRM failure is not physical infeasibility.
H-JEPA segmented observations and refusal calibration remain useful research directions, but contributions of world models and planning require
a common input/action interface and fresh independent testing.
