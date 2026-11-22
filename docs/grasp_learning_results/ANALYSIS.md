# Did supervised learning outperform reinforcement learning?

**Not in this experiment.** At equal physical training budgets, REINFORCE has a slightly higher mean success rate, without evidence of general superiority. Training-selected fixed +18 degrees also matches or exceeds both networks.

See REPORT.md for all seed metrics and evaluation.json/selections.json for raw records. Test outcomes do not trigger retraining or checkpoint selection.

## Why generated training data are valid here

### Supervision needs input-label pairs, not necessarily human or hardware labels

A sample is `(initial state s, candidate action a, outcome y)`:

- s: pre-execution base pose/orientation and handle position.
- a: the grasp approach offset to attempt.
- y: physical success, collision, slip, opening angle, and reward.

Simulation generates y, and fitting `(s,a) -> y` is supervised learning. Automatically collected simulator labels neither make it unsupervised nor require an assigned best angle.

### Labels come from execution rather than desired answers

Collection calls existing `Experiment.trial()` and MuJoCo `mj_step`, then measures collision, bilateral contact, slip, and door angle. Only safe complete opening/hold receives success.

No rule declares +18 degrees successful, and the network does not generate its own success labels. Apply matching physics/criteria to every candidate and retain failures. The network approximates the simulator task's input-outcome relationship; the simulator, rather than learned Newtonian dynamics, solves the physics.

### A network can score candidates before complete simulation

Simulation generally needs a whole trajectory to establish the outcome. A trained model can quickly score several candidates before execution and transfer experience to new stations.

Potential value is fewer physical attempts/full simulations. For one fixed scene and 21 candidates, scanning may be simpler. Strong fixed +18 degrees means training completion alone does not establish necessity. Compare prediction errors, action quality, and compute on unseen conditions.

### Separation of learning and evaluation

Training reads only 36 training stations, with IDs/poses split before collection and training-only normalization. Freeze models, select angles on 24 different test stations, and replay physics. Final angles/collisions/rewards are labels rather than pre-execution features.

This checks adaptation to unseen stations in the same simulator rather than table memorization. Shared train/test physics and matching replays establish reproducibility, not independent real-world validity.

### When these data are insufficient

- Wrong contacts, geometry, or criteria teach wrong relationships; audit simulation with controls, numerical checks, and physical calibration when needed.
- Easy successful stations alone do not teach collision boundaries; include failures and borderline cases.
- Shared initial conditions or post-outcome features inflate scores; group by scene and check information timing.
- Unavailable deployment parameters must not become privileged inputs. Friction is fixed here and its truth is absent from policy features.
- Hardware transfer still needs execution validation and sensor/contact calibration; randomization expands training conditions without guaranteeing success.

Generated data support supervised learning and new-station validation in this simulation, without establishing hardware transfer.

A related precedent is Dex-Net 2.0's synthetic point clouds, candidate grasps, and analytical-quality labels. Its object-grasping task does not replace complete door-opening labels or validation. [Berkeley introduction](https://bair.berkeley.edu/blog/2017/06/27/dexnet-2.0/).

## Completed work and scope

- Seventy-two base poses: 36 training, 12 validation, 24 test.
- Execute 21 angles per station: 1,512 automatically labeled records.
- Each main method/seed uses 756 physical interactions; seeds 42, 43, 44.
- Networks are 4 -> 16 -> 21 with 437 parameters and matching inputs/controllers.
- Include SL on RL records, fixed zero, and training-selected best-fixed controls.
- Replay 179 distinct frozen-test station/action combinations, all matching collection.
- Ten numerical, split-isolation, save/load, and learning-direction tests pass.

New scenes are base poses/orientations around the same door, rather than new door types or hardware.

## Results

| Method | Successes on the same 24 stations | Mean success rate | Mean reward |
| --- | --- | --- | --- |
| Supervised, full-angle coverage | 9, 9, 9 | 37.50% | 3.462 |
| REINFORCE, online exploration | 10, 9, 9 | 38.89% | 3.534 |
| Supervised on RL records | 9, 8, 8 | 34.72% | 2.614 |
| Fixed 0 degrees | 7 | 29.17% | 1.519 |
| Training-selected fixed +18 degrees | 10 | 41.67% | 4.168 |

Seeds share test scenes rather than adding 72 independent samples. The approximately 1.39-point gap comes from one extra seed-42 success; the other seeds tie. Stable algorithmic superiority is not established.

Only 10/24 stations have successful grid angles. SL solves 9/10 for each seed, RL 10/10, 9/10, 9/10. The other 14 have no successes under current angles/height/elbow/speed/force, without proving all continuous plans impossible.

## RL's additional success: test_021

Seed 42, base about (0.83987, -0.91868) m, yaw 91.1557 degrees:

- SL selects +10 degrees, self-collides at 88.984 degrees, reward -2.141.
- RL selects +18 degrees, opens to 89.725 degrees and holds, reward 12.885.

SL ranks the candidates incorrectly:

| Candidate | Predicted SL reward | Actual reward | Outcome |
| --- | --- | --- | --- |
| +10 degrees | 9.792 | -2.141 | Collision |
| +18 degrees | 9.330 | 12.885 | Full opening/hold |

This is a reward-ranking error at an unseen station near a sharp collision boundary. Sparse boundary samples, MSE smoothing, and underfitting are possible contributors; no single cause is established.

RL's correct choice does not imply more accurate outcome prediction: it outputs action probabilities, not success probabilities. The other two RL seeds fail here.

```bash
# Run from the repository root after activating an environment with requirements.txt installed
source .venv/bin/activate
mjpython compare_grasp_learning.py demo --scene test_021 --method supervised --seed 42
mjpython compare_grasp_learning.py demo --scene test_021 --method reinforce --seed 42
```

Full contact CSV, joint NPZ, and plan JSON are in `examples/rl_only_success/`. Headless physics is checked; the graphical window was not manually reviewed.

## Practical value of supervision

1. Reuse one collected dataset across fits, label audits, and models; three SL initializations share data while each RL seed collects again.
2. Compare candidate predicted/actual rewards to locate ranking errors; RL probabilities alone estimate no outcome.
3. Full SL sampling covers 756 training pairs; RL covers 438, 471, 446.

Full-coverage SL averages 37.50% versus 34.72% on RL records, motivating coverage studies without proving SL must beat RL.

SL averages 6.33/24 collisions versus RL 7/24 but has more workspace failures, so collision counts alone do not establish overall safety. Neither refuses. No selected test action slips, without evidence of improved slip resistance.

## A strong fixed +18-degree baseline

Select +18 degrees by training mean reward only. Its 10/24 test successes cover every grid-feasible station, with higher mean reward than either network.

Many feasible stations share a good angle. Models learn and improve over fixed zero without establishing the need for adaptive grasp selection or world-model foresight.

## Next priorities

1. Design training/validation scene families requiring different grasps, using obstacles, door geometry/direction, and elbow configurations. Preserve fresh independent tests rather than tuning on these 24 stations.
2. Add training samples near collision/reachability boundaries, prioritizing uncertainty/disagreement.
3. Add refusal/repositioning for infeasible stations; angles alone cannot solve every fixed-skill failure.
4. Compare SL outcome/collision multitask or ranking losses and RL exploration/value learning, selecting on validation before fresh testing.
5. Retain best-fixed, random, and full-path geometry controls to measure what learning adds.

## Costs and limitations

Full collection takes about 153 s including shared evaluation references; each SL fit about 0.15-0.18 s; each RL seed about 86-104 s including interactions. Comparing 0.2 s to 100 s omits SL collection and differences in parallelism/early termination.

Physical trial counts match, rather than optimization steps, compute, or duration. Findings are not hardware deployment conclusions or universal learning-paradigm rankings.
