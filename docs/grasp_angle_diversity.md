# Grasp-angle diversity (2026-10-27)

New defaults expand 21 discrete angles to 29. Angles remain vertical-axis approach offsets, rather than door opening, pitch/roll, or individual joint angles.

| Region | Candidate angles |
|---|---|
| Original fine interval | -20 to +20 degrees, spacing 2 degrees, 21 values |
| Added larger offsets | -90, -60, -45, -30, +30, +45, +60, +90 degrees |

Broader candidates are not guaranteed grasps/openings. PRM preflight and physical preparation precede MuJoCo contact/collision/opening labels. Retain preparation failures rather than dropping negative samples.

## Usage

```bash
# Default: 29 angles, 72 stations, 2088 records: 1044 train, 348 validation, 696 test.
python collect_grasp_data.py --angle-grid diverse29 --output reproduced/angles29_data
python compare_grasp_learning.py train --data reproduced/angles29_data --output reproduced/angles29_models --seeds 42
python compare_grasp_learning.py evaluate --data reproduced/angles29_data --output reproduced/angles29_models

# Custom lists must include zero degrees to retain the fixed-zero baseline.
python collect_grasp_data.py --angles-deg -60 -30 -10 0 10 30 60 --output reproduced/custom_angle_data

# Uniform ranges include both endpoints; the step must divide the range.
python collect_grasp_data.py --angle-min -60 --angle-max 60 --angle-step 5 --output reproduced/uniform_angle_data

# The independent online REINFORCE entry supports the same angle options.
python train_grasp_rl.py train --angle-grid diverse29 --output reproduced/angles29_rl

# Historical sampling settings; use a fresh directory rather than appending changed angles to old data.
python collect_grasp_data.py --angle-grid legacy21 --seed 20261026 --prm-seed 7 --output reproduced/legacy21_new_run
```

Presets, explicit lists, and uniform ranges are mutually exclusive. Candidates must be finite and unique. New output defaults include `diverse29`; previous data/models/reports are preserved.

SL/RL build 4 -> 16 -> K from the manifest, outputting K rewards/probabilities respectively. Historical checkpoints retain saved angles/indices; 29-angle indices cannot interpret 21-angle weights. Retrain after expanding actions. Resume uses the original manifest and rejects changed angles, with source-hash checks retained.

See [results and three station cases](../results/angle_diversity/README.md) on GitHub.

## Measured verification

- Forty-seven regressions pass: old candidates, invalid-angle rejection, consistent entry points, model loading, and resume-index protection.
- Physical smoke run: seed=42, 1 station per split, 87 candidates; 84 opening trials and 3 preparation failures. Total 25.10 seconds.
- SL, REINFORCE, and SL on RL records complete training with 29-output configurations. All 4 distinct frozen-test action replays match.
- Added +30/+45/+60 degrees succeed on training; +30/+45/+60/+90 on validation. These establish executable large offsets, not newly feasible stations, since original-grid successes already exist there.
- All 29 test-station angles fail. This checks interfaces and physical execution, without enough samples for generalization comparisons or a full 72-station rerun.
- Original verification files remain in `reproduced/angle_diversity_smoke_20261027/`; generated data/models are not uploaded.

## Next steps and limitations

Angles are still one-time choices with fixed height, elbow branch, force, and trajectory. Lateral approaches add no degrees of freedom or closed-loop correction and can increase collision/IK/PRM failures.

29/21 is approximately 1.38, increasing full-grid collection and matched RL decision budgets about 38% for equal station counts. Match scenes/physics/PRM and separate action-space effects from added budget using equal-budget controls. Historical tests have been inspected; final generalization needs a fresh frozen test. First assess whether added angles make more stations feasible, then whether networks select them.
