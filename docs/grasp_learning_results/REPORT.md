# Supervised versus reinforcement learning on new stations

See [ANALYSIS.md](ANALYSIS.md) for interpretation, failures, and next steps. SL does not clearly outperform RL; training-selected fixed +18 degrees also succeeds 10/24.

Findings apply to this implementation and cohort: unseen base positions/orientations around one door rather than new door families, unknown friction, or hardware.

## Data and fairness

- Train / validation / test: 36 / 12 / 24 scenes. Group all angles by scene and split before simulation.
- Twenty-one angles per scene, -20 to +20 degrees at step 2. Physics supplies labels without a prescribed best angle.
- Each main method/seed uses 756 training interactions. Seeds: [42, 43, 44].
- Match capacity, inputs, normalization, controller, rewards, and scene visits. SL samples every angle; RL samples online with repetition.
- SL has more offline gradient updates; match physical interactions rather than optimization steps or total compute.
- A third arm fits a supervised model to the same RL records without extra trials, partially separating coverage/objective effects.
- All use final weights. Validation/test do not tune, train, stop, or select; validation is a post-freeze diagnostic.

## Test results

Ten of 24 test stations have a successful grid candidate.
All-grid failure does not establish impossibility across continuous angles or every grasp plan.

| Method / seed | Full openings | Mean reward | Mean grid-oracle regret | Collisions | Slips |
| --- | --- | --- | --- | --- | --- |
| Supervised, full-angle coverage / 42 | 9/24 (37.5%) | 3.508 | 1.089 | 6 | 0 |
| Supervised, full-angle coverage / 43 | 9/24 (37.5%) | 3.395 | 1.201 | 7 | 0 |
| Supervised, full-angle coverage / 44 | 9/24 (37.5%) | 3.484 | 1.113 | 6 | 0 |
| REINFORCE, online exploration / 42 | 10/24 (41.7%) | 3.856 | 0.740 | 7 | 0 |
| REINFORCE, online exploration / 43 | 9/24 (37.5%) | 3.406 | 1.191 | 7 | 0 |
| REINFORCE, online exploration / 44 | 9/24 (37.5%) | 3.341 | 1.255 | 7 | 0 |
| Supervised on RL records / 42 | 9/24 (37.5%) | 3.061 | 1.536 | 10 | 0 |
| Supervised on RL records / 43 | 8/24 (33.3%) | 2.243 | 2.354 | 12 | 0 |
| Supervised on RL records / 44 | 8/24 (33.3%) | 2.537 | 2.059 | 10 | 0 |
| Fixed 0 degrees | 7/24 (29.2%) | 1.519 | 3.077 | 14 | 0 |
| Training-selected best fixed angle | 10/24 (41.7%) | 4.168 | 0.429 | 4 | 0 |

Uniform-random grid expectation: 32.3%; exhaustive reference ceiling: 41.7%. The reference table is not used for test-time action selection.

## Paired SL/RL comparison

- Seed 42: SL-only successes 0, RL-only 1; SL covers 756 training station-angle pairs, RL 438.
- Seed 43: SL-only successes 0, RL-only 0; SL covers 756 pairs, RL 471.
- Seed 44: SL-only successes 0, RL-only 0; SL covers 756 pairs, RL 446.

Repeated seeds on shared test stations do not add independent scenes. Report all seeds/pairs without significance or universal-paradigm claims.

## Replayable examples

- rl_only_success: test_021, SL +10 degrees / unexpected_collision; RL +18 degrees / full_open_and_hold.
- both_success: test_000, SL +18 degrees / full_open_and_hold; RL +6 degrees / full_open_and_hold.

## Reproduction

```bash
python collect_grasp_data.py --output reproduced/grasp_learning_data_new
python compare_grasp_learning.py train --data reproduced/grasp_learning_data_new --output reproduced/grasp_learning_comparison_new
python compare_grasp_learning.py evaluate --data reproduced/grasp_learning_data_new --output reproduced/grasp_learning_comparison_new
```

View a scene with generated default models:

```bash
mjpython compare_grasp_learning.py demo --scene test_000 --method supervised --seed 42
mjpython compare_grasp_learning.py demo --scene test_000 --method reinforce --seed 42
```

After freezing, all 179 distinct test station-angle combinations match the original physical records on replay.

Raw records are train/validation/test.csv and .jsonl; selections.json contains chosen actions, evaluation.json metrics, and seed_* directories models/training/exploration histories.

Scope: one door family; fixed height/force/speed; known perception; pregrasp initialization; discrete angles; no online regrasp/refusal. SL outputs predicted reward rather than calibrated success probability.
Design reference: https://www.jmlr.org/papers/volume25/23-0183/23-0183.pdf
