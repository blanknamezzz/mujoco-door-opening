# Reinforcement learning for grasp-angle selection

New defaults add PRM preparation from explicit initial joints to a physically reached pregrasp; see [PRM workflow](prm_pregrasp.md). Output/model defaults become `reproduced/grasp_rl_prm_diverse29`. Use `--no-prm` for old pregrasp initialization. The earlier 256-episode report describes that baseline.

Entry point `train_grasp_rl.py` calls existing `Experiment.trial()` without changing robot, contact criteria, or low-level control. Existing requirements suffice; NumPy implements the small network and Adam.

Generated weights/data are not distributed. Train before demonstrating; the commands below generate `grasp_rl_new`.

## Quick start

Run from the project root. Examples use its `.venv`:

```bash
# Run from the repository root after activating an environment with requirements.txt installed
source .venv/bin/activate

# Learn from scratch with 256 physical trials, then independently scan all angles as a reference.
python train_grasp_rl.py train --episodes 256 --sweep --output reproduced/grasp_rl_new

# Reload final weights and execute learned/fixed-zero angles without updates.
python train_grasp_rl.py evaluate --checkpoint reproduced/grasp_rl_new/policy.npz --output reproduced/grasp_rl_new_eval

# macOS graphical demo: learned angle first, then fixed zero degrees.
mjpython train_grasp_rl.py demo --checkpoint reproduced/grasp_rl_new/policy.npz

# Show only the fixed-angle control.
mjpython train_grasp_rl.py demo --checkpoint reproduced/grasp_rl_new/policy.npz --fixed-angle 0
```

Training requires an empty directory and does not support checkpoint resume. Use separate outputs for separate runs. Graphical viewing requires a desktop; training/evaluation can run headlessly.

## Learning task

This is episodic RL for one high-level action, also a contextual bandit. Each episode:

1. Read pregrasp base/handle geometry.
2. Predict angle probabilities and sample an approach angle.
3. Reset to pregrasp and execute approach, closure, opening, and hold through `mj_step`, without welding handle to fingers.
4. Compute reward from the physical outcome.
5. Apply REINFORCE every 8 episodes.

Default base (0.8, -1.0) m, with 29 angles: 21 values from -20 to 20 degrees at 2-degree spacing, plus +/-30, +/-45, +/-60, +/-90. Configure using `--angle-grid legacy21`, `--angles-deg`, or ranges; see [angle extension](grasp_angle_diversity.md). Initial probabilities are uniform, without a prescribed +8-degree answer. Angles are vertical-axis offsets from normal approach, not pitch/roll.

Network: 4 -> 16 tanh -> K softmax, default K=29; historical configuration restores K=21. Inputs are handle-relative dx/dy and base-yaw sine/cosine, using existing RGB-D detection. No post-execution angles or failure labels enter features. Height, force, speed, friction, and elbow branch stay fixed during a run. Default learning uses one station rather than arbitrary-position generalization.

Objective: `mean((R - b) / 10 * log pi(a|s) + beta H(pi))`. Baseline b is an exponential average of previous batches rather than current-action outcomes. Default beta=0.02 encourages exploration. Each batch uses fresh policy-sampled physical trajectories and one update, without cached reward tables or best-angle labels.

Reference: [policy gradients and baselines](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html). This is simplified REINFORCE rather than PPO, a world model, or per-timestep joint-control RL.

## Reward

`R = 10S + G + 2P - 5C - 4L - 2W - 2U - 3V - 0.1T`

| Term | Meaning |
| --- | --- |
| S | Safe full opening and hold under original success criteria |
| G | Stable grasp acquisition |
| P | Maximum safe angle before the first violation / 90 degrees, clipped to [0,1] |
| C | Unexpected contact / collision |
| L | Detected slip or contact loss |
| W | Downstream joint target outside workspace |
| U | Initial grasp/pregrasp target unreachable |
| V | Final opening shortfall fraction |
| T | Actual / planned execution time |

These are current reward-design choices rather than an exact copy of the slides. Slip/workspace failures receive separate penalties; collision prevents success reward even near 90 degrees. Reward optimization does not guarantee safety; existing execution stops remain active.

## Changing training conditions

```bash
# Finer angle discretization and more training episodes.
python train_grasp_rl.py train --angle-min -20 --angle-max 20 --angle-step 1 --episodes 1024 --output reproduced/grasp_rl_fine

# Change the fixed station.
python train_grasp_rl.py train --base 0.6 -1.0 --output reproduced/grasp_rl_base06

# Sample around the center station with independent uniform coordinate perturbations.
python train_grasp_rl.py train --base 0.8 -1.0 --base-jitter 0.05 0.03 --episodes 1024 --output reproduced/grasp_rl_jitter

# Evaluate an unseen position with frozen weights.
python train_grasp_rl.py evaluate --checkpoint reproduced/grasp_rl_jitter/policy.npz --base 0.83 -0.99 --sweep --output reproduced/grasp_rl_unseen
```

Fix `--height`, `--ik-branch`, `--grip-force`, `--open-seconds`, `--friction`, or `--hinge-resistance` before retraining; these are not policy actions. Force is an actuator cap per finger rather than measured normal contact force. Changed settings require retraining rather than assumed automatic adaptation.

## Inspecting results

- `config.json`: scene, seeds, actions, training settings, and source hashes.
- `policy.npz`: final weights/configuration; `initial_policy.npz`: initial weights. NumPy arrays require no pickle.
- `episodes.csv`: action, reward, success, final angle, and stop reason per episode.
- `episodes.jsonl`: detailed reward decomposition and execution outcomes.
- `updates.json`: batch rewards/success and complete center-station action probabilities.
- `evaluation/evaluation.json`: frozen final policy executed against fixed zero degrees.
- `evaluation/learned_plan.json`: plan loadable with existing `simulate.py --plan`.
- `evaluation/learned.csv`, `.npz`: final contact trajectory and joints.

`--sweep` evaluates every candidate only after training, without updates or checkpoint selection. It reports best grid reward and uniform-random success, serving as a finite-grid evaluation reference rather than a continuous global optimum.

One success on a fixed scene is not a 100% test rate. Use multiple training seeds, independent scenes, and equal-interaction random-search/geometry/RL controls. On small deterministic fixed scenes, scanning may be simpler; this establishes an executable, inspectable RL baseline.

Arbitrary-initial-joint planning, online regrasping, continuous angles, refusal actions, and world-model foresight are not implemented in this baseline. A policy still recommends an angle when all fail; inspect actual evaluation rather than treating recommendations as feasibility guarantees.

## Tests

```bash
python -m unittest test_grasp_rl -v
```

Checks cover finite-difference gradients, risk rewards, save/load identity, and learning direction on an independent known-reward problem. Physical validity still depends on actual MuJoCo evaluation.
