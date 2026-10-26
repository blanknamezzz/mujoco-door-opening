"""Learn grasp angles with on-policy REINFORCE and real MuJoCo rollouts.

One episode = one angle choice + the existing complete opening skill. This is
a contextual bandit / one-decision episodic RL problem, not joint-level RL.
NumPy implements a small neural policy and Adam; no extra packages are needed.
See docs/grasp_rl.md for commands, reward, and experimental limitations.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from grasp_plan import GraspPlan, SkillTiming
from perception import estimate
from simulate import Experiment, ROOT, save_trial

_PRM_CACHE = {}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def features(base, config):
    """Only information available BEFORE the chosen action; no outcome leakage.

    Geometry and contact parameters are known in this first controlled task.
    Only base position varies when training jitter is requested.
    """
    center = np.asarray(config["plan"]["handle_center_world_m"])
    delta = center[:2] - np.asarray(base)
    return np.array([delta[0], delta[1], np.sin(config["base_yaw_rad"]),
                     np.cos(config["base_yaw_rad"])], dtype=float)


class Policy:
    """4 -> 16 tanh -> K categorical angles. Explicit gradients for teaching."""

    def __init__(self, n_actions, rng):
        self.params = {
            "w1": rng.normal(0, .25, (4, 16)), "b1": np.zeros(16),
            # Zero output layer gives every angle equal initial probability.
            "w2": np.zeros((16, n_actions)), "b2": np.zeros(n_actions),
        }

    def forward(self, x):
        h = np.tanh(np.asarray(x) @ self.params["w1"] + self.params["b1"])
        logits = h @ self.params["w2"] + self.params["b2"]
        logits -= logits.max(axis=-1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=-1, keepdims=True)
        return h, p

    def gradient(self, x, actions, advantages, entropy_coef):
        """Gradient ASCENT on mean(A log pi + entropy_coef * H(pi))."""
        x = np.asarray(x)
        h, p = self.forward(x)
        logp = np.log(np.maximum(p, 1e-300))
        entropy = -(p * logp).sum(axis=1)
        one_hot = np.eye(p.shape[1])[actions]
        dl = np.asarray(advantages)[:, None] * (one_hot - p)
        dl -= entropy_coef * p * (logp + entropy[:, None])
        dl /= len(x)
        dh = (dl @ self.params["w2"].T) * (1 - h*h)
        return {"w1": x.T @ dh, "b1": dh.sum(axis=0),
                "w2": h.T @ dl, "b2": dl.sum(axis=0)}


class Adam:
    def __init__(self, params, lr):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        norm = np.sqrt(sum(float(np.sum(g*g)) for g in grads.values()))
        scale = min(1., 1. / max(norm, 1e-12))
        for k in params:
            g = grads[k] * scale
            self.m[k] = .9*self.m[k] + .1*g
            self.v[k] = .999*self.v[k] + .001*g*g
            params[k] += self.lr * (self.m[k] / (1-.9**self.t)) / (
                np.sqrt(self.v[k] / (1-.999**self.t)) + 1e-8)


def reward(result, rows, plan):
    """Dense progress helps exploration; only safe opening + hold earns +10."""
    collision = bool(result.get("unexpected_contact_samples", 0))
    slip = bool(result.get("slip_detected") or result.get("contact_lost"))
    # Exclude samples at/after the first violation from progress reward.
    safe_angles = [0.]
    for row in rows:
        if row["unexpected_contact"] or row["slip_detected"] or row["contact_lost"]:
            break
        safe_angles.append(row["door_deg"])
    progress = float(np.clip(max(safe_angles) / plan.opening_angle_deg, 0, 1))
    final = float(np.clip(result.get("final_angle_deg", 0) / plan.opening_angle_deg, 0, 1))
    terms = {
        "success": 10. * (result["outcome"] == "D_full_open"),
        "grasp": float(bool(result.get("grasp_acquired"))),
        "safe_progress": 2.*progress,
        "collision": -5.*collision,
        "slip": -4.*slip,
        "workspace": -2.*(result["stop_reason"] == "arm_workspace_limit"),
        "unreachable": -2.*(result["outcome"] == "A_initial_unreachable"),
        "shortfall": -3.*(1-final),
        "duration": -.1 * (rows[-1]["time_s"] / plan.timing.end if rows else 0.),
    }
    return float(sum(terms.values())), terms


def rollout(config, base, angle, viewer=False):
    plan = replace(GraspPlan.from_dict(config["plan"]), approach_angle_deg=float(angle))
    exp = Experiment(base=base, base_yaw=config["base_yaw_rad"],
                     friction=config["friction"], hinge_resistance=config["hinge_resistance"],
                     ik_branch=plan.ik_branch, model_xml=config.get('model_xml'))
    if viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(exp.m, exp.d) as window:
            return _execute_rollout(config, base, angle, exp, plan, window)
    return _execute_rollout(config, base, angle, exp, plan)


def _execute_rollout(config, base, angle, exp, plan, viewer=None):
    """One complete decision, including visible preparation when requested."""
    preparation = None
    if config.get('prm') is not None:
        from prm import PRMSettings, Roadmap, execute_path
        key = json.dumps(dict(config=config, base=list(map(float, base))), sort_keys=True)
        if key not in _PRM_CACHE:
            if len(_PRM_CACHE) >= 8:
                _PRM_CACHE.pop(next(iter(_PRM_CACHE)))
            _PRM_CACHE[key] = Roadmap(exp, PRMSettings(**config['prm']))
        query = _PRM_CACHE[key].plan(plan)
        preparation = dict(query=query, execution=None)
        if not query['path_found']:
            status = query['status']
            result = dict(outcome=('A_initial_unreachable' if status == 'goal_ik_unreachable'
                                  else 'P_pregrasp_unconfirmed'), stop_reason='prm_'+status,
                          max_angle_deg=0., final_angle_deg=0., execution_plan=plan.to_dict())
            value, terms = reward(result, [], plan)
            if status != 'goal_ik_unreachable':
                terms['pregrasp_failure'] = -2.
                value -= 2.
            return dict(angle_deg=float(angle), base_xy_m=list(map(float, base)), reward=value,
                reward_terms=terms, result=result, preparation=preparation, opening_tested=False), [], [], plan
        execution, _, _ = execute_path(exp, query['path_q'], viewer=viewer)
        preparation['execution'] = execution
        if not execution['success']:
            result = dict(outcome='P_pregrasp_execution_failed', stop_reason=execution['stop_reason'],
                max_angle_deg=0., final_angle_deg=0., execution_plan=plan.to_dict(),
                unexpected_contact_samples=int(bool(execution.get('collision'))),
                first_unexpected_contact=execution.get('collision'))
            return dict(angle_deg=float(angle), base_xy_m=list(map(float, base)), reward=-5.,
                reward_terms=dict(pregrasp_failure=-5., safe_progress=0.), result=result,
                preparation=preparation, opening_tested=False), [], [], plan
    result, rows, states = exp.trial(plan=plan, viewer=viewer, reset_before_trial=preparation is None)
    value, terms = reward(result, rows, plan)
    return dict(angle_deg=float(angle), base_xy_m=list(map(float, base)), reward=value,
                reward_terms=terms, result=result, preparation=preparation, opening_tested=True), rows, states, plan


def save_checkpoint(path, policy, config):
    np.savez(path, metadata=np.array(json.dumps(config)), **policy.params)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        config = json.loads(str(data["metadata"]))
        if config.get("schema_version") not in (1, 2):
            raise ValueError("Unsupported policy checkpoint version")
        policy = Policy(len(config["angles_deg"]), np.random.default_rng(0))
        for k, shape in ((k, v.shape) for k, v in policy.params.items()):
            value = data[k]
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid policy parameter: {k}")
            policy.params[k] = value.copy()
    return policy, config


def evaluate(policy, config, output, base=None, sweep=False, viewer=False, fixed_angle=None):
    output.mkdir(parents=True, exist_ok=True)
    base = config["base_xy_m"] if base is None else base
    _, probs = policy.forward(features(base, config))
    chosen = float(config["angles_deg"][int(np.argmax(probs))])
    distribution = [dict(angle_deg=a, probability=float(p))
                    for a, p in zip(config["angles_deg"], probs)]
    comparisons = []
    variants = [("learned", chosen), ("fixed_zero", 0.)] if fixed_angle is None else [("fixed", fixed_angle)]
    for name, angle in variants:
        trial, rows, states, plan = rollout(config, base, angle, viewer=viewer)
        trial["variant"] = name
        comparisons.append(trial)
        save_trial(name, trial["result"], rows, states, output)
        plan.save(output / f"{name}_plan.json")
    report = dict(scope="Single deterministic scene evaluation; not a generalization success rate.",
                  base_xy_m=list(base), selected_angle_deg=chosen, distribution=distribution,
                  comparisons=comparisons)
    if sweep:
        # Post-training reference only. Never used to update or select weights.
        trials = []
        for a in config["angles_deg"]:
            item, _, _, _ = rollout(config, base, a)
            trials.append(item)
        best = max(trials, key=lambda t: t["reward"])
        report["grid_reference"] = dict(trials=trials, best_angle_deg=best["angle_deg"],
            best_reward=best["reward"], uniform_policy_success_fraction=float(np.mean([
                t["result"]["outcome"] == "D_full_open" for t in trials])),
            note="Best on this finite grid in this deterministic scene; not a continuous global optimum.")
    write_json(output / "evaluation.json", report)
    for t in comparisons:
        print(f'{t["variant"]}: {t["angle_deg"]:+g} deg, reward={t["reward"]:.3f}, '
              f'{t["result"]["outcome"]}, {t["result"]["stop_reason"]}', flush=True)
    return report


def train(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Training output must be empty. Choose a new --output to preserve previous runs.")
    args.output.mkdir(parents=True, exist_ok=True)
    count = round((args.angle_max - args.angle_min) / args.angle_step)
    if count < 1 or not np.isclose(args.angle_min + count*args.angle_step, args.angle_max):
        raise ValueError("Angle range must be an integer multiple of --angle-step, with at least two angles")
    angles = np.linspace(args.angle_min, args.angle_max, count+1).tolist()
    center = tuple(estimate(args.observation)["center_world_m"])
    plan = GraspPlan(center, height_offset_m=args.height, ik_branch=args.ik_branch,
                     grip_force_limit_N=args.grip_force, timing=SkillTiming(open_s=args.open_seconds))
    config = dict(schema_version=2, algorithm="REINFORCE with past-batch EMA baseline and entropy",
        seed=args.seed, episodes=args.episodes, batch_size=args.batch_size,
        learning_rate=args.learning_rate, entropy_coef=args.entropy,
        base_xy_m=args.base, base_jitter_m=args.base_jitter,
        base_yaw_rad=float(np.deg2rad(args.base_yaw_deg)), friction=args.friction,
        hinge_resistance=args.hinge_resistance, plan=plan.to_dict(), angles_deg=angles,
        observation_path=str(args.observation.resolve()),
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                       for name in ("scene.xml", "simulate.py", "grasp_plan.py", "prm.py", "train_grasp_rl.py")})
    if not args.no_prm:
        from prm import PRMSettings
        config['prm'] = PRMSettings(samples=args.prm_samples, initial_q=args.initial_q).to_dict()
    write_json(args.output / "config.json", config)
    rng = np.random.default_rng(args.seed)
    policy = Policy(len(angles), rng)
    optimizer = Adam(policy.params, args.learning_rate)
    save_checkpoint(args.output / "initial_policy.npz", policy, config)
    baseline, done = 0., 0
    start = time.perf_counter()
    fields = ["episode", "base_x", "base_y", "angle_deg", "reward", "success", "final_angle_deg", "stop_reason"]
    updates = []
    with (args.output / "episodes.csv").open("w", newline="") as file, (
            args.output / "episodes.jsonl").open("w") as detail:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        while done < args.episodes:
            xs, actions, rewards = [], [], []
            successes = 0
            for _ in range(min(args.batch_size, args.episodes-done)):
                base = np.asarray(args.base) + rng.uniform(-1, 1, 2)*args.base_jitter
                x = features(base, config)
                _, probabilities = policy.forward(x)
                action = int(rng.choice(len(angles), p=probabilities))
                trial, _, _, _ = rollout(config, base, angles[action])
                result = trial["result"]
                success = result["outcome"] == "D_full_open"
                done += 1
                successes += success
                xs.append(x); actions.append(action); rewards.append(trial["reward"])
                writer.writerow(dict(episode=done, base_x=base[0], base_y=base[1],
                    angle_deg=angles[action], reward=trial["reward"], success=int(success),
                    final_angle_deg=result.get("final_angle_deg", 0), stop_reason=result["stop_reason"]))
                detail.write(json.dumps(dict(episode=done, **trial), allow_nan=False) + "\n")
            # Baseline is computed from PAST batches, independent of this batch's actions.
            # Fixed /10 scaling preserves the objective; no per-batch label fitting.
            advantages = (np.asarray(rewards)-baseline)/10.
            grads = policy.gradient(xs, actions, advantages, args.entropy)
            optimizer.step(policy.params, grads)
            baseline = .8*baseline + .2*float(np.mean(rewards))
            _, probs = policy.forward(features(args.base, config))
            update = dict(episodes=done, mean_reward=float(np.mean(rewards)),
                batch_success_fraction=successes/len(rewards), baseline=baseline,
                center_angle_deg=angles[int(np.argmax(probs))], probabilities=probs.tolist())
            updates.append(update)
            file.flush(); detail.flush()
            save_checkpoint(args.output / "policy.npz", policy, config)
            write_json(args.output / "updates.json", updates)
            print(f'{done}/{args.episodes}: reward={update["mean_reward"]:.2f}, '
                  f'success={successes}/{len(rewards)}, greedy angle={update["center_angle_deg"]:+g} deg', flush=True)
    report = evaluate(policy, config, args.output / "evaluation", sweep=args.sweep)
    write_json(args.output / "training_summary.json", dict(episodes=done,
        elapsed_seconds=time.perf_counter()-start, selected_angle_deg=report["selected_angle_deg"],
        note="Final policy; evaluation results are not used to choose a checkpoint. No global optimality guarantee."))


def positive(value):
    x = float(value)
    if not np.isfinite(x) or x <= 0:
        raise argparse.ArgumentTypeError("Expected a finite positive value")
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train", help="Train from fresh MuJoCo episodes")
    p.add_argument("--output", type=Path, default=ROOT/"reproduced"/"grasp_rl_prm")
    p.add_argument("--observation", type=Path, default=ROOT/"results"/"observation.npz")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=positive, default=.02)
    p.add_argument("--entropy", type=float, default=.02)
    p.add_argument("--base", type=float, nargs=2, default=[.8, -1.])
    p.add_argument("--base-jitter", type=float, nargs=2, default=[0., 0.], metavar=("DX", "DY"))
    p.add_argument("--base-yaw-deg", type=float, default=90.)
    p.add_argument("--angle-min", type=float, default=-20.)
    p.add_argument("--angle-max", type=float, default=20.)
    p.add_argument("--angle-step", type=positive, default=2.)
    p.add_argument("--height", type=float, default=0.)
    p.add_argument("--ik-branch", type=int, choices=[-1, 1], default=-1)
    p.add_argument("--grip-force", type=positive, default=35.)
    p.add_argument("--open-seconds", type=positive, default=10.)
    p.add_argument("--friction", type=positive, default=.8)
    p.add_argument("--hinge-resistance", type=float, default=1.2)
    p.add_argument("--sweep", action="store_true", help="After training, independently evaluate every angle")
    p.add_argument('--no-prm', action='store_true', help='Explicit legacy reset-to-pregrasp baseline')
    p.add_argument('--prm-samples', type=int, default=240)
    p.add_argument('--initial-q', type=float, nargs=4, default=[0., -np.pi/2, -2*np.pi/3, 0.],
                   metavar=('LIFT_M', 'SHOULDER_RAD', 'ELBOW_RAD', 'WRIST_RAD'))
    for command in ("evaluate", "demo"):
        p = commands.add_parser(command)
        p.add_argument("--checkpoint", type=Path, default=ROOT/"reproduced"/"grasp_rl_prm"/"policy.npz")
        p.add_argument("--output", type=Path, default=ROOT/"reproduced"/f"grasp_rl_{command}")
        p.add_argument("--base", type=float, nargs=2, help="Override base for evaluation only; no weight updates")
        p.add_argument("--sweep", action="store_true")
        if command == "demo":
            p.add_argument("--fixed-angle", type=float, help="Show a specified comparison angle instead of learned policy")
    args = parser.parse_args()
    numeric = [v for v in vars(args).values() if isinstance(v, (float, int))]
    numeric += [v for value in vars(args).values() if isinstance(value, list) for v in value]
    if not all(np.isfinite(v) for v in numeric):
        parser.error("All numeric arguments must be finite")
    if args.command == "train":
        if args.episodes < 1 or args.batch_size < 1 or args.entropy < 0 or args.hinge_resistance < 0:
            parser.error("Invalid episode count, batch size, entropy, or hinge resistance")
        if args.seed < 0 or min(args.base_jitter) < 0 or args.angle_max <= args.angle_min:
            parser.error("Invalid seed, jitter, or angle range")
        train(args)
    else:
        policy, config = load_checkpoint(args.checkpoint)
        evaluate(policy, config, args.output, args.base, args.sweep,
                 viewer=args.command == "demo", fixed_angle=getattr(args, "fixed_angle", None))


if __name__ == "__main__":
    main()
