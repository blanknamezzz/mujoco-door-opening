"""Supervised reward prediction vs REINFORCE on held-out robot base poses.

Same 4->16->K network, features, reward, controller, action grid and number of
training interactions per seed. SL sweeps candidates; RL explores on-policy.
A third arm fits SL to the exact RL records, isolating part of the data effect.
Only train.jsonl is read by training. Validation/test are loaded after weights
are frozen; no checkpoint/hyperparameter selection uses held-out outcomes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from collect_grasp_data import (DEFAULT_DATA, context, load_records, physics_job,
                               scene_config, validate_records)
from simulate import ROOT, save_trial
from train_grasp_rl import Adam, Policy, rollout, write_json

DEFAULT_OUTPUT = ROOT / "reproduced" / "grasp_learning_comparison_prm_diverse29"
METHODS = ("supervised", "reinforce", "supervised_on_rl")
LABELS = {"supervised": "Supervised (full grid)", "reinforce": "REINFORCE (online exploration)",
          "supervised_on_rl": "Supervised on RL records", "fixed_zero": "Fixed 0 degrees",
          "best_fixed_train": "Training-selected best fixed angle"}


class RewardModel:
    """Same parameterization as Policy, but linear outputs predict R/10."""
    def __init__(self, n_actions, rng):
        self.params = Policy(n_actions, rng).params

    def forward(self, x):
        h = np.tanh(np.asarray(x) @ self.params["w1"] + self.params["b1"])
        return h, h @ self.params["w2"] + self.params["b2"]

    def gradient(self, x, actions, targets):
        """ASCENT on negative MSE for observed (context, action) pairs only."""
        x = np.asarray(x)
        h, values = self.forward(x)
        error = values[np.arange(len(x)), actions] - targets
        dl = np.zeros_like(values)
        dl[np.arange(len(x)), actions] = -2*error/len(x)
        dh = (dl @ self.params["w2"].T) * (1-h*h)
        grads = dict(w1=x.T@dh, b1=dh.sum(axis=0), w2=h.T@dl, b2=dl.sum(axis=0))
        return grads, float(np.mean(error**2))


def normalizer(manifest):
    xs = np.array([context(manifest, s) for s in manifest["scenes"] if s["split"] == "train"])
    return dict(mean=xs.mean(axis=0).tolist(), std=np.maximum(xs.std(axis=0), 1e-6).tolist())


def encode(manifest, scene, norm):
    return (context(manifest, scene)-norm["mean"]) / norm["std"]


def training_arrays(manifest, records, norm):
    scenes = {s["scene_id"]: s for s in manifest["scenes"]}
    if any(t["split"] != "train" or scenes[t["scene_id"]]["split"] != "train" for t in records):
        raise ValueError("Refusing non-training records")
    return (np.array([encode(manifest, scenes[t["scene_id"]], norm) for t in records]),
            np.array([t["action_index"] for t in records]),
            np.array([t["reward"]/10. for t in records]))


def fit_supervised(manifest, records, norm, seed, steps, lr):
    model = RewardModel(len(manifest["angles_deg"]), np.random.default_rng(seed))
    adam = Adam(model.params, lr)
    x, a, y = training_arrays(manifest, records, norm)
    history = []
    for step in range(steps):
        grads, mse = model.gradient(x, a, y)
        adam.step(model.params, grads)
        if step == 0 or (step+1) % 100 == 0 or step+1 == steps:
            history.append(dict(update=step+1, train_reward_mse=mse*100))
    return model, history


def save_model(path, model, metadata):
    np.savez(path, metadata=np.array(json.dumps(metadata)), **model.params)


def load_model(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        cls = Policy if metadata["method"] == "reinforce" else RewardModel
        model = cls(len(metadata["angles_deg"]), np.random.default_rng(0))
        for k, v in model.params.items():
            if data[k].shape != v.shape or not np.all(np.isfinite(data[k])):
                raise ValueError(f"Invalid parameter {k}")
            model.params[k] = data[k].copy()
    return model, metadata


def select(model, manifest, scene, norm):
    _, scores = model.forward(encode(manifest, scene, norm))
    return int(np.argmax(scores))


def train(args):
    manifest = json.loads((args.data/"manifest.json").read_text())
    # No held-out outcome files are read anywhere in this function.
    records = load_records(args.data/"train.jsonl")
    validate_records(manifest, records, "train")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new empty --output for training")
    args.output.mkdir(parents=True, exist_ok=True)
    norm = normalizer(manifest)
    scenes = [s for s in manifest["scenes"] if s["split"] == "train"]
    n_actions = len(manifest["angles_deg"])
    budget = len(records)
    config = dict(schema_version=1, seeds=args.seeds, data=str(args.data.resolve()),
        manifest_sha256=hashlib.sha256((args.data/"manifest.json").read_bytes()).hexdigest(),
        train_data_sha256=hashlib.sha256((args.data/"train.jsonl").read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        architecture=f"4 -> 16 tanh -> {n_actions} outputs; softmax only for RL", normalizer=norm,
        train_interactions_per_method_per_seed=budget, supervised_updates=args.sl_updates,
        supervised_lr=args.sl_lr, rl_lr=args.rl_lr, rl_batch_size=args.batch_size,
        entropy_coef=.02, checkpoint_selection="Final weights only; no validation or test selection")
    config['preparation'] = manifest['physics'].get('prm')
    config['budget_unit'] = ('scheduled end-to-end decisions; PRM rejection may skip opening execution'
                             if config['preparation'] else 'physical opening attempts')
    write_json(args.output/"experiment_config.json", config)
    # A useful non-neural baseline, selected from training labels only.
    means = [float(np.mean([t["reward"] for t in records if t["action_index"] == a]))
             for a in range(n_actions)]
    config["best_fixed_train_index"] = int(np.argmax(means))
    write_json(args.output/"experiment_config.json", config)
    start = time.perf_counter()
    costs = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for seed in args.seeds:
            folder = args.output/f"seed_{seed}"
            folder.mkdir()
            metadata = dict(schema_version=1, seed=seed, normalizer=norm, angles_deg=manifest["angles_deg"],
                            manifest_sha256=config["manifest_sha256"])
            t0 = time.perf_counter()
            model, history = fit_supervised(manifest, records, norm, seed, args.sl_updates, args.sl_lr)
            save_model(folder/"supervised.npz", model, dict(metadata, method="supervised"))
            write_json(folder/"supervised_history.json", history)
            costs[f"supervised_{seed}"] = time.perf_counter()-t0
            print(f"seed {seed}: supervised fitting complete", flush=True)
            rng = np.random.default_rng(seed)
            policy = Policy(n_actions, rng)
            adam = Adam(policy.params, args.rl_lr)
            # Each training scene is visited exactly K times, matching SL budget
            # per scene. Only RL chooses which actions to revisit on-policy.
            schedule = np.concatenate([rng.permutation(len(scenes)) for _ in range(n_actions)])
            baseline = 0.
            rl_records, updates = [], []
            t0 = time.perf_counter()
            with (folder/"rl_episodes.jsonl").open("w") as file:
                for begin in range(0, budget, args.batch_size):
                    batch_scenes = [scenes[i] for i in schedule[begin:begin+args.batch_size]]
                    x = np.array([encode(manifest, s, norm) for s in batch_scenes])
                    _, probs = policy.forward(x)
                    actions = np.array([rng.choice(n_actions, p=p) for p in probs])
                    trials = list(pool.map(physics_job, [(manifest, s, int(a)) for s, a in zip(batch_scenes, actions)]))
                    rewards = np.array([t["reward"] for t in trials])
                    adam.step(policy.params, policy.gradient(x, actions, (rewards-baseline)/10., .02))
                    baseline = .8*baseline + .2*float(rewards.mean())
                    for trial in trials:
                        rl_records.append(trial)
                        file.write(json.dumps(trial, allow_nan=False)+"\n")
                    file.flush()
                    updates.append(dict(episodes=len(rl_records), mean_reward=float(rewards.mean()),
                        success_fraction=float(np.mean([t["result"]["outcome"] == "D_full_open" for t in trials]))))
                    if len(updates) % 8 == 0 or len(rl_records) == budget:
                        print(f"seed {seed}: RL {len(rl_records)}/{budget}, last batch reward {rewards.mean():.2f}", flush=True)
            save_model(folder/"reinforce.npz", policy, dict(metadata, method="reinforce"))
            write_json(folder/"reinforce_history.json", updates)
            costs[f"reinforce_with_collection_{seed}"] = time.perf_counter()-t0
            # No extra physical interactions: supervised fitting on SAME RL data.
            t0 = time.perf_counter()
            model, history = fit_supervised(manifest, rl_records, norm, seed, args.sl_updates, args.sl_lr)
            save_model(folder/"supervised_on_rl.npz", model, dict(metadata, method="supervised_on_rl"))
            write_json(folder/"supervised_on_rl_history.json", history)
            costs[f"supervised_on_rl_fitting_{seed}"] = time.perf_counter()-t0
            unique = len({(t["scene_id"], t["action_index"]) for t in rl_records})
            write_json(folder/"coverage.json", dict(supervised_unique_pairs=budget, rl_unique_pairs=unique,
                total_interactions_each=budget,
                rl_opening_trials_executed=sum(t.get('opening_tested', True) for t in rl_records),
                sl_opening_trials_executed=sum(t.get('opening_tested', True) for t in records)))
    write_json(args.output/"training_costs.json", dict(seconds=costs, total_seconds=time.perf_counter()-start,
        note="Supervised fitting time excludes its dataset generation; report both when comparing cost."))
    print("All model weights frozen. Run evaluate to open held-out outcomes.", flush=True)


def summarize(selected, grid):
    feasible = {sid for (sid, _), t in grid.items() if t["result"]["outcome"] == "D_full_open"}
    best = {}
    for (sid, _), t in grid.items():
        best[sid] = max(best.get(sid, -np.inf), t["reward"])
    successes = [t["result"]["outcome"] == "D_full_open" for t in selected]
    return dict(n_scenes=len(selected), successes=int(sum(successes)), success_rate=float(np.mean(successes)),
        feasible_scenes=len(feasible), feasible_success_rate=(sum(successes)/len(feasible) if feasible else None),
        mean_reward=float(np.mean([t["reward"] for t in selected])),
        mean_regret=float(np.mean([best[t["scene_id"]]-t["reward"] for t in selected])),
        collisions=sum(bool(t["result"].get("unexpected_contact_samples")) for t in selected),
        slips=sum(bool(t["result"].get("slip_detected") or t["result"].get("contact_lost")) for t in selected),
        workspace_failures=sum(t["result"]["stop_reason"] == "arm_workspace_limit" for t in selected),
        unreachable=sum(t["result"]["outcome"] == "A_initial_unreachable" for t in selected))


def evaluate(args):
    config = json.loads((args.output/"experiment_config.json").read_text())
    manifest = json.loads((args.data/"manifest.json").read_text())
    if hashlib.sha256((args.data/"manifest.json").read_bytes()).hexdigest() != config["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from training")
    norm, seeds = config["normalizer"], config["seeds"]
    models = {}
    for seed in seeds:
        for method in METHODS:
            model, metadata = load_model(args.output/f"seed_{seed}"/f"{method}.npz")
            if metadata["manifest_sha256"] != config["manifest_sha256"]:
                raise ValueError("Model belongs to another dataset")
            models[(method, seed)] = model
    all_metrics, selections, reference = {}, {}, {}
    for split in ("validation", "test"):
        records = load_records(args.data/f"{split}.jsonl")
        validate_records(manifest, records, split)
        grid = {(t["scene_id"], t["action_index"]): t for t in records}
        scenes = [s for s in manifest["scenes"] if s["split"] == split]
        all_metrics[split], selections[split] = {}, {}
        for (method, seed), model in models.items():
            key = f"{method}_{seed}"
            chosen = [grid[(s["scene_id"], select(model, manifest, s, norm))] for s in scenes]
            selections[split][key] = chosen
            metric = summarize(chosen, grid)
            if method != "reinforce":
                errors = []
                for s in scenes:
                    _, predictions = model.forward(encode(manifest, s, norm))
                    truth = np.array([grid[(s["scene_id"], a)]["reward"] for a in range(len(predictions))])
                    errors.extend(np.abs(predictions*10-truth).tolist())
                metric["all_candidate_reward_mae"] = float(np.mean(errors))
            all_metrics[split][key] = metric
        for method, action in (("fixed_zero", manifest["angles_deg"].index(0.)),
                               ("best_fixed_train", config["best_fixed_train_index"])):
            chosen = [grid[(s["scene_id"], action)] for s in scenes]
            selections[split][method] = chosen
            all_metrics[split][method] = summarize(chosen, grid)
        feasible = sum(any(grid[(s["scene_id"], a)]["result"]["outcome"] == "D_full_open"
                           for a in range(len(manifest["angles_deg"]))) for s in scenes)
        reference[split] = dict(scenes=len(scenes), grid_feasible_scenes=feasible,
            uniform_random_expected_success_rate=float(np.mean([t["result"]["outcome"] == "D_full_open" for t in records])),
            grid_oracle_success_rate=feasible/len(scenes))
    # Verify chosen test actions through NEW physical executions after freezing.
    scene_lookup = {s["scene_id"]: s for s in manifest["scenes"]}
    unique = {(t["scene_id"], t["action_index"]): t
              for trials in selections["test"].values() for t in trials}
    checks = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = [(manifest, scene_lookup[sid], a) for sid, a in unique]
        for actual in pool.map(physics_job, jobs):
            expected = unique[(actual["scene_id"], actual["action_index"])]
            matched = (actual["result"]["outcome"] == expected["result"]["outcome"] and
                       actual["result"]["stop_reason"] == expected["result"]["stop_reason"] and
                       abs(actual["reward"]-expected["reward"]) < 1e-6)
            checks.append(dict(scene_id=actual["scene_id"], angle_deg=actual["angle_deg"], matched=matched))
            if not matched:
                raise RuntimeError("Fresh simulation disagrees with reference; inspect physics/version changes")
    summary = dict(protocol=config, reference=reference, metrics=all_metrics,
        fresh_test_verification=dict(unique_attempts=len(checks), all_matched=all(c["matched"] for c in checks)),
        scope=manifest["scope"])
    write_json(args.output/"evaluation.json", summary)
    write_json(args.output/"selections.json", selections)
    write_json(args.output/"fresh_test_checks.json", checks)
    # Save full trajectories for the first discordant case in EACH direction,
    # not merely the case favoring the proposed method. Report every test case.
    seed = seeds[0]
    sl = selections["test"][f"supervised_{seed}"]
    rl = selections["test"][f"reinforce_{seed}"]
    examples = []
    for direction in ("sl_only_success", "rl_only_success", "both_success"):
        for a, b in zip(sl, rl):
            ok_a, ok_b = a["result"]["outcome"] == "D_full_open", b["result"]["outcome"] == "D_full_open"
            if (direction == "sl_only_success" and ok_a and not ok_b or
                direction == "rl_only_success" and ok_b and not ok_a or
                direction == "both_success" and ok_a and ok_b):
                scene = scene_lookup[a["scene_id"]]
                folder = args.output/"examples"/direction
                for method, trial in (("supervised", a), ("reinforce", b)):
                    actual, rows, states, plan = rollout(scene_config(manifest, scene), scene["base_xy_m"], trial["angle_deg"])
                    save_trial(method, actual["result"], rows, states, folder)
                    plan.save(folder/f"{method}_plan.json")
                examples.append(dict(kind=direction, scene=scene, seed=seed, supervised=a, reinforce=b))
                break
    write_json(args.output/"examples.json", examples)
    write_report(args.output, manifest, summary, selections, examples)
    print(json.dumps(summary["metrics"]["test"], indent=2), flush=True)


def write_report(output, manifest, summary, selections, examples):
    seeds = summary["protocol"]["seeds"]
    ref = summary["reference"]["test"]
    n = ref["scenes"]
    lines = ["# Supervised versus reinforcement learning on new stations", "",
        "Findings apply to this cohort/implementation: unseen base poses around one door, rather than new door families, unknown friction, or hardware.", "",
        "## Data and fairness", "",
        f"- Train / validation / test scenes: {sum(s['split']=='train' for s in manifest['scenes'])} / {sum(s['split']=='validation' for s in manifest['scenes'])} / {n}. Group angles by scene and split before simulation.",
        f"- {len(manifest['angles_deg'])} angles per scene: {manifest['angles_deg']}. Physics supplies labels without prescribed best angles.",
        f"- Each main method/seed receives {summary['protocol']['train_interactions_per_method_per_seed']} training decisions. Seeds: {seeds}. Budget unit: {summary['protocol'].get('budget_unit', 'physical opening attempts')}.",
        "- Match capacity, inputs, normalization, controller, reward, and visits. SL covers every angle; RL samples online with repetition.",
        "- SL uses more offline gradient updates; match physical interactions rather than optimization steps or total compute.",
        "- A third arm fits SL on the same RL records without new trials, partially separating coverage/objective effects.",
        "- Use final weights. Validation/test do not train, tune, stop, or select; validation is a post-freeze diagnostic.", "",
        "## Test results", "",
        f"24 test stations include {ref['grid_feasible_scenes']} with successful grid candidates.".replace("24 test", f"{n} test"),
        "All-grid failure does not establish impossibility across continuous angles or every grasp plan.", "",
        "| Method / seed | Full openings | Mean reward | Mean grid-oracle regret | Collisions | Slips |",
        "| --- | --- | --- | --- | --- | --- |"]
    for method in METHODS:
        for seed in seeds:
            m = summary["metrics"]["test"][f"{method}_{seed}"]
            lines.append(f"| {LABELS[method]} / {seed} | {m['successes']}/{n} ({m['success_rate']:.1%}) | {m['mean_reward']:.3f} | {m['mean_regret']:.3f} | {m['collisions']} | {m['slips']} |")
    for method in ("fixed_zero", "best_fixed_train"):
        m = summary["metrics"]["test"][method]
        lines.append(f"| {LABELS[method]} | {m['successes']}/{n} ({m['success_rate']:.1%}) | {m['mean_reward']:.3f} | {m['mean_regret']:.3f} | {m['collisions']} | {m['slips']} |")
    lines += ["", f"Uniform-random grid expectation: {ref['uniform_random_expected_success_rate']:.1%}; exhaustive ceiling: {ref['grid_oracle_success_rate']:.1%}. The reference is not used for test-time selection.", "",
              "## Paired SL/RL comparison", ""]
    for seed in seeds:
        a = selections["test"][f"supervised_{seed}"]
        b = selections["test"][f"reinforce_{seed}"]
        awin = sum(x["result"]["outcome"] == "D_full_open" and y["result"]["outcome"] != "D_full_open" for x, y in zip(a, b))
        bwin = sum(y["result"]["outcome"] == "D_full_open" and x["result"]["outcome"] != "D_full_open" for x, y in zip(a, b))
        coverage = json.loads((output/f"seed_{seed}"/"coverage.json").read_text())
        lines.append(f"- Seed {seed}: SL-only successes {awin}, RL-only {bwin}; SL covers {coverage['supervised_unique_pairs']} training station-angle pairs, RL {coverage['rl_unique_pairs']}.")
    lines += ["", "Seeds sharing test stations add no independent scenes. Report all seeds/pairs without significance or universal-paradigm claims.", "",
              "## Replayable examples", ""]
    for item in examples:
        a, b = item["supervised"], item["reinforce"]
        lines.append(f"- {item['kind']}: {item['scene']['scene_id']}, SL {a['angle_deg']:+g}° / {a['result']['stop_reason']}; RL {b['angle_deg']:+g}° / {b['result']['stop_reason']}.")
    lines += ["", "## Reproduction", "", "```bash", f"python collect_grasp_data.py --output reproduced/grasp_learning_data_new --seed {manifest['seed']} --prm-seed {(manifest['physics'].get('prm') or {}).get('seed', 42)} --angles-deg " + " ".join(map(str, manifest["angles_deg"])) + ("" if manifest["physics"].get("prm") else " --no-prm"),
        "python compare_grasp_learning.py train --data reproduced/grasp_learning_data_new --output reproduced/grasp_learning_comparison_new",
        "python compare_grasp_learning.py evaluate --data reproduced/grasp_learning_data_new --output reproduced/grasp_learning_comparison_new", "```", "",
        "View a scene with generated default models:", "", "```bash", "mjpython compare_grasp_learning.py demo --scene test_000 --method supervised --seed 42",
        "mjpython compare_grasp_learning.py demo --scene test_000 --method reinforce --seed 42", "```", "",
        f"After freezing, {summary['fresh_test_verification']['unique_attempts']} distinct test station-angle replays all match the original records.", "",
        "Raw data: train/validation/test.csv and .jsonl. Chosen actions: selections.json. Metrics: evaluation.json. Models, training, and exploration histories: seed_*.",
        "",
        "Scope: one door family, fixed height/force/speed, known perception, discrete angles, no online regrasp/refusal. " +
        ("PRM uses fixed initial joints; post-grasp motion follows a fixed hinge arc. " if summary['protocol'].get('preparation') else "Uses pregrasp initialization. ") +
        "SL outputs predicted reward rather than calibrated success probability.",
        "Design reference: https://www.jmlr.org/papers/volume25/23-0183/23-0183.pdf", ""]
    (output/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def demo(args):
    manifest = json.loads((args.data/"manifest.json").read_text())
    matches = [s for s in manifest["scenes"] if s["scene_id"] == args.scene]
    if not matches:
        raise ValueError("Unknown scene ID")
    scene = matches[0]
    model, metadata = load_model(args.output/f"seed_{args.seed}"/f"{args.method}.npz")
    if hashlib.sha256((args.data/"manifest.json").read_bytes()).hexdigest() != metadata["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from model")
    action = select(model, manifest, scene, metadata["normalizer"])
    angle = manifest["angles_deg"][action]
    print(f"{args.scene}, {args.method}: selected {angle:+g} deg", flush=True)
    trial, _, _, _ = rollout(scene_config(manifest, scene), scene["base_xy_m"], angle, viewer=True)
    print(json.dumps(trial, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "demo"):
        p = commands.add_parser(name)
        p.add_argument("--data", type=Path, default=DEFAULT_DATA)
        p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        if name != "demo":
            p.add_argument("--workers", type=int, default=4)
        if name == "train":
            p.add_argument("--seeds", type=int, nargs="+", default=[42])
            p.add_argument("--sl-updates", type=int, default=1200)
            p.add_argument("--sl-lr", type=float, default=.01)
            p.add_argument("--rl-lr", type=float, default=.02)
            p.add_argument("--batch-size", type=int, default=12)
        if name == "demo":
            p.add_argument("--scene", default="test_000")
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--method", choices=METHODS, default="supervised")
    args = parser.parse_args()
    if args.command != "demo" and args.workers < 1:
        parser.error("workers must be positive")
    if args.command == "train":
        if min(args.sl_updates, args.batch_size) < 1 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
            parser.error("Invalid update count, batch size, or seeds")
        if not all(np.isfinite(x) and x > 0 for x in (args.sl_lr, args.rl_lr)):
            parser.error("Learning rates must be finite and positive")
    {"train": train, "evaluate": evaluate, "demo": demo}[args.command](args)


if __name__ == "__main__":
    main()
