# Generated data, supervised learning, and reinforcement-learning controls

New defaults add [PRM preparation](prm_pregrasp.md), analyzing and physically executing a path to pregrasp. New data/model directories use `_prm`. Linked historical 36/12/24-station results use original initialization; see [PRM integration results](prm_results/REPORT.md) for the new workflow.

## Why generated data support supervised learning

Supervised learning requires paired inputs/labels, not necessarily hardware data. Inputs are pre-execution scene/angle; full MuJoCo grasp/open/hold execution supplies success, collision, slip, angle, and reward labels. Networks neither invent labels nor receive a prescribed correct angle.

The model learns simulated `(state, action) -> outcome` for pre-execution candidate assessment. Automatic label collection remains supervised learning, distinct from self-supervised representation learning through future frames.

Validity requires reasonable physics/criteria, coverage of successes/failures/boundaries, and frozen evaluation on unseen scenes. Post-execution outcomes cannot enter inputs. Testing establishes new-station performance in the same simulator rather than hardware validity or simulator accuracy.

Networks may reduce candidate-execution costs on new scenes, while direct scans can suit small fixed problems. Retain fixed-angle/exhaustive references rather than treating completed training as proof that learning is necessary.

See [analysis and measured findings](grasp_learning_results/ANALYSIS.md). Regenerated evaluation files are in `reproduced/grasp_learning_comparison/`.

## Files and entry points

- `collect_grasp_data.py`: generate manifests, run MuJoCo, and record labeled data.
- `compare_grasp_learning.py`: train reward predictors/REINFORCE, evaluate, and generate reports.
- `test_grasp_learning.py`: gradients, split isolation, integrity, and model loading.
- Original `train_grasp_rl.py` and historical training remain independent; do not substitute old single-station models for multi-scene RL controls.

Run from the repository root:

```bash
# Run from the root after activating an environment with requirements.txt installed
source .venv/bin/activate

# Defaults: 36 training, 12 validation, 24 test stations; 29 angles each. Historical runs explicitly use --angle-grid legacy21.
python collect_grasp_data.py --output reproduced/grasp_learning_data_new

# New experiments default to seed 42; each method/seed has 756 training interactions.
python compare_grasp_learning.py train \
  --data reproduced/grasp_learning_data_new \
  --output reproduced/grasp_learning_comparison_new

# After freezing weights, read validation/test labels, replay selected actions, and generate REPORT.md.
python compare_grasp_learning.py evaluate \
  --data reproduced/grasp_learning_data_new \
  --output reproduced/grasp_learning_comparison_new
```

After integration on 2026-10-26, new data/model/PRM sampling defaults use seed 42.
Published historical no-PRM reports retain data seed 20261026 and training seeds 42/43/44.
Reproduce with explicit `collect_grasp_data.py --no-prm --seed 20261026`
and `compare_grasp_learning.py train --seeds 42 43 44`, using new data/output directories.
`prm_examples.py` preserves historical sampling seed 7 for the five published examples.

Resume interrupted collection with `--resume` in the same directory only when manifest physics-source hashes match; completed records are skipped. Training outputs must be empty, with no training checkpoint-resume implementation.

Use existing MuJoCo, NumPy, and Pillow without PyTorch. Workers execute independent physical episodes; learning updates remain sequential. Default `--workers 4` can be reduced for host load.

## Scene splits

Generate the manifest with a fixed scene seed before simulation:

- Base x: 0.35-0.95 m.
- Base y: -1.15 to -0.75 m.
- Base yaw: 60-120 degrees.
- Independent Latin hypercube sampling per split, without repeated initial states.
- All angles/attempts at one station stay in the same split.

This measures interpolation to new base positions/orientations within one door family, not new dimensions, reverse opening, unknown friction, or hardware transfer. Outcomes do not select an SL-favorable test set.

Training reads only `train.jsonl`; normalization uses training stations. Validation/test evaluate after all models are saved, without tuning, early stopping, or checkpoint selection in this experiment. Future validation tuning must retain a frozen test rather than repeatedly selecting on test outcomes.

## Data records

Data directory:

| File | Content |
| --- | --- |
| `manifest.json` | Scene IDs, splits, ranges, angles, fixed physics, library versions, source hashes |
| `train.csv` / `validation.csv` / `test.csv` | One attempt per row for inspection/plotting |
| Corresponding `.jsonl` | Reward components, plans, grasp results, first collision, slip/contact loss |
| `collection_summary.json` | Split counts, successful attempts, collection duration |

CSV records scene ID, base x/y/yaw, angle, reward, success/grasp flags, maximum safe/final angles, collision, slip, and stop reason. Labels come from physical execution rather than assumed angle quality.

Batch data store episode summaries rather than images/contact traces at every step. Temporal world-model training needs extra sequence collection; saved scenes/actions support replay. Selected report examples separately preserve full contact CSV and joint NPZ trajectories.

## What the three methods learn

All use 4 -> 16 tanh -> K, with K from manifest angles, default 29. Inputs are pre-execution handle-relative dx/dy and base-yaw sine/cosine. Retain 21 angles from -20 to +20 degrees at step 2, adding +/-30, +/-45, +/-60, +/-90. Preserve old checkpoint indices; see [angle extension](grasp_angle_diversity.md).

### A: Supervised reward prediction

Execute all K angles at every training station. Predict `R / 10`, fit observed actions by MSE, and choose maximum predicted reward at test. No separate success-probability head exists, so outputs are not calibrated probabilities.

Reuse the RL reward exactly: safe opening/hold, grasp, safe progress, and collision/slip/workspace/unreachable/incomplete/time penalties.

Use 1,200 full-batch updates at rate 0.01. Generate data once and reuse for training.

### B: REINFORCE

RL visits each training station K times, sampling from the current policy with possible repeated/missed angles. Update every 12 fresh episodes at rate 0.02, entropy 0.02, and a previous-batch mean-reward baseline. Every training reward comes from fresh physics rather than offline labels.

This is single-high-level-action RL rather than multi-step joint torques, PPO, DQN, or world-model RL.

### C: Supervised learning on RL records

Fit B's actual interactions with A's objective/training, without more physical trials. A versus C probes angle coverage; C versus B probes reuse for reward fitting. These finite controls do not isolate every factor completely.

## Fairness and metrics

Match training stations, visits, physical interactions, angles, controller/physics, input features, parameter count, and rewards.

Optimization updates, objectives, exploration, and total compute differ by workflow; results are not universal claims about abstract learning paradigms.

SL data support three initializations; each RL seed needs new interactions. Both use 756 per-seed training decisions, but reproduction collection costs differ. Report collection, optimization, and evaluation costs separately. Shared validation/test scans are excluded from training budgets.

Metrics include safe complete openings, mean reward, finite-grid regret, collisions, slip, and workspace failures. References include fixed zero, training-selected best fixed, uniform random, and exhaustive grid. The oracle requires all test candidates and is not a free deployment policy.

Neither method has refusal; both recommend an angle when all fail. Report how many test stations have at least one successful grid angle.

Three seeds share 24 test scenes rather than 72 independent samples. Report every seed without claiming significance.

## Viewing generated results

See the [published full report](grasp_learning_results/REPORT.md). Fresh clones lack weights; collect/train before launching these demos.

```bash
# View both models on the same test station.
mjpython compare_grasp_learning.py demo --scene test_000 --method supervised --seed 42
mjpython compare_grasp_learning.py demo --scene test_000 --method reinforce --seed 42

# Inspect the supervised model trained on RL records.
mjpython compare_grasp_learning.py demo --scene test_000 --method supervised_on_rl --seed 42

# Numerical and split-isolation checks.
python -m unittest test_grasp_learning test_grasp_rl -v
```

Available scene IDs are in the manifest. Report `examples.json` lists SL-only/RL-only/shared successes when present, with full trajectories in `examples/`. Recommendations do not guarantee success.

Design reference: [Empirical Design in Reinforcement Learning](https://www.jmlr.org/papers/volume25/23-0183/23-0183.pdf).
