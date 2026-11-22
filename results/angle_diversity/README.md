# 29-angle extension: small integration check

This directory summarizes a completed 1/1/1-station experiment with 87 candidate records. It verifies collection, SL/RL training, model loading, and evaluation with the extended angle grid, rather than full 72-station generalization.

| Case | Split | Successes among original 21 angles | Successes among 8 added angles | Preparation failures |
|---|---|---:|---:|---:|
| [train_000](train_000/README.md) | train | 8 | 3 | 1 |
| [validation_000](validation_000/README.md) | validation | 13 | 4 | 0 |
| [test_000](test_000/README.md) | test | 0 | 0 | 2 |

[Evaluation metrics](metrics.json) · [Provenance and source hashes](provenance.json)

All methods fail on the single test station. Four distinct test actions were replayed and matched the records. Training and validation stations already have successful original-grid angles, so successes at added angles do not establish newly feasible stations.

Data sampling, PRM, and training use seed=42. Raw data and weights remain ignored; Git stores only the small summaries here. Mid-grasp decisions and force adjustment are not implemented, and the full 72-station, 29-angle experiment has not been run.

Reproduction with new output directories:

```bash
python collect_grasp_data.py --output reproduced/angle_diversity_recheck/data --train-scenes 1 --validation-scenes 1 --test-scenes 1 --workers 2 --angle-grid diverse29 --seed 42 --prm-seed 42
python compare_grasp_learning.py train --data reproduced/angle_diversity_recheck/data --output reproduced/angle_diversity_recheck/comparison --workers 2 --seeds 42
python compare_grasp_learning.py evaluate --data reproduced/angle_diversity_recheck/data --output reproduced/angle_diversity_recheck/comparison --workers 2
```

[Angle implementation and limitations](../../docs/grasp_angle_diversity.md) · [All cases](../README.md)
