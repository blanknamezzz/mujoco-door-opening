# Full 72-station PRM experiment (2026-10-27)

Recollect 36/12/24 training/validation/test stations with the same coordinates, orientations, splits, and 21 angles as the earlier full experiment. Preserve the old 24-station pilot and all successful/failed records without outcome-based station filtering.

## Data scale

| Split | Stations | Candidate records | Preparation failures | Successful opening records |
|---|---:|---:|---:|---:|
| train | 36 | 756 | 0 | 251 |
| validation | 12 | 252 | 0 | 110 |
| test | 24 | 504 | 0 | 163 |

See [positions.csv](positions.csv) for every station/angle outcome. Each record is one complete station-angle procedure; candidate successes are not a policy success rate.

## Independent testing for three training seeds

| Method | Successes per seed, each /24 | Mean success rate |
|---|---|---:|
| supervised | 9, 9, 9 | 37.50% |
| reinforce | 10, 9, 9 | 38.89% |
| supervised_on_rl | 9, 8, 8 | 34.72% |
| Training-selected fixed +18 degrees | 10/24 | 41.67% |

Seeds [42, 43, 44], with 756 training decisions per method/seed. SL fits fixed records; RL explores online with repeated pairs. Normalization, fixed-angle choice, and fitting use training only; freeze weights before testing.

Ten of 24 test stations have complete-opening candidates in this grid/controller, without a global feasibility bound. Seeds share 24 stations rather than 72 independent stations.

Deduplicate frozen-test actions by station/angle and physically replay 179 combinations; all match reference records.

## Preparation paths and complete opening

Candidate PRM states: `{'direct_path': 1512}`. PRM checks paths to pregrasp; full opening remains limited by collision, workspace, and contact stability.

## Reproduction

```bash
python collect_grasp_data.py --output reproduced/prm_full72_data_20261027 --train-scenes 36 --validation-scenes 12 --test-scenes 24 --seed 20261026 --workers 4
python compare_grasp_learning.py train --data reproduced/prm_full72_data_20261027 --output reproduced/prm_full72_comparison_20261027 --seeds 42 43 44 --workers 4
python compare_grasp_learning.py evaluate --data reproduced/prm_full72_data_20261027 --output reproduced/prm_full72_comparison_20261027 --workers 4
```

Resume collection with `--resume` in the existing data directory and train in a fresh empty output. Raw records, weights, histories, and action checks remain in the reproduced directories above.

## Interpretation limits

Stations and nominal physics match earlier work, but source hashes for simulate.py/training/collection and initialization differ. A PRM effect needs matched current-code, station, and budget controls with/without PRM. Scope remains one door family and fixed initial joints, without hardware/new-door generalization evidence.

Verification: [verification.json](verification.json).
