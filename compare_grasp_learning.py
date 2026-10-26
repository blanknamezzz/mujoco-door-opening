"""Supervised reward prediction vs REINFORCE on held-out robot base poses.

Same 4->16->21 network, features, reward, controller, action grid and number of
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

DEFAULT_OUTPUT = ROOT / "reproduced" / "grasp_learning_comparison"
METHODS = ("supervised", "reinforce", "supervised_on_rl")
LABELS = {"supervised": "监督学习（完整角度采样）", "reinforce": "REINFORCE（在线探索）",
          "supervised_on_rl": "监督学习（复用 RL 记录）", "fixed_zero": "固定 0°",
          "best_fixed_train": "训练集选出的最佳固定角度"}


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
        architecture="4 -> 16 tanh -> 21 outputs; softmax only for RL", normalizer=norm,
        train_interactions_per_method_per_seed=budget, supervised_updates=args.sl_updates,
        supervised_lr=args.sl_lr, rl_lr=args.rl_lr, rl_batch_size=args.batch_size,
        entropy_coef=.02, checkpoint_selection="Final weights only; no validation or test selection")
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
                                                    total_interactions_each=budget))
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
    lines = ["# 监督学习与强化学习：新站位实测对照", "",
        "结论仅适用于这批场景与实现。新场景指同一扇门周围未参与训练的底座位置和朝向，不代表新门型、未知摩擦或真实机器人。", "",
        "## 数据与公平性", "",
        f"- 训练 / 验证 / 测试场景数：{sum(s['split']=='train' for s in manifest['scenes'])} / {sum(s['split']=='validation' for s in manifest['scenes'])} / {n}。所有角度按场景分组，划分先于仿真。",
        f"- 每场景 {len(manifest['angles_deg'])} 个角度：−20° 至 +20°、间隔 2°。物理仿真产生标签，无人工指定最佳角度。",
        f"- 每种主方法每个种子使用 {summary['protocol']['train_interactions_per_method_per_seed']} 次训练交互。训练种子：{seeds}。",
        "- 同一网络规模、输入、归一化、控制器、奖励、场景访问次数。SL 全角度采样；RL 在线按当前策略采样，允许重复。",
        "- 监督学习使用更多离线梯度更新，因此比较的是相同物理交互预算，不是相同优化步数或总计算量。",
        "- 第三组用同一份 RL 记录训练监督模型，不增加物理试验，用于部分区分数据覆盖与学习目标的影响。",
        "- 三组均使用最终权重，验证和测试数据不参与训练、调参、早停或权重选择。验证集仅作冻结后的独立诊断。", "",
        "## 测试结果", "",
        f"24 个测试位置中有 {ref['grid_feasible_scenes']} 个在候选网格内存在成功方案。".replace("24 个", f"{n} 个"),
        "不能把网格内全部失败解释成连续角度空间或所有抓持方案都不可能成功。", "",
        "| 方法 / 种子 | 全开成功 | 平均奖励 | 与网格最优的平均奖励差 | 碰撞次数 | 滑移次数 |",
        "| --- | --- | --- | --- | --- | --- |"]
    for method in METHODS:
        for seed in seeds:
            m = summary["metrics"]["test"][f"{method}_{seed}"]
            lines.append(f"| {LABELS[method]} / {seed} | {m['successes']}/{n} ({m['success_rate']:.1%}) | {m['mean_reward']:.3f} | {m['mean_regret']:.3f} | {m['collisions']} | {m['slips']} |")
    for method in ("fixed_zero", "best_fixed_train"):
        m = summary["metrics"]["test"][method]
        lines.append(f"| {LABELS[method]} | {m['successes']}/{n} ({m['success_rate']:.1%}) | {m['mean_reward']:.3f} | {m['mean_regret']:.3f} | {m['collisions']} | {m['slips']} |")
    lines += ["", f"均匀随机角度的网格期望成功率：{ref['uniform_random_expected_success_rate']:.1%}；穷举参考上限：{ref['grid_oracle_success_rate']:.1%}。参考表不用于测试时选动作。", "",
              "## SL 与 RL 的配对比较", ""]
    for seed in seeds:
        a = selections["test"][f"supervised_{seed}"]
        b = selections["test"][f"reinforce_{seed}"]
        awin = sum(x["result"]["outcome"] == "D_full_open" and y["result"]["outcome"] != "D_full_open" for x, y in zip(a, b))
        bwin = sum(y["result"]["outcome"] == "D_full_open" and x["result"]["outcome"] != "D_full_open" for x, y in zip(a, b))
        coverage = json.loads((output/f"seed_{seed}"/"coverage.json").read_text())
        lines.append(f"- 种子 {seed}：仅 SL 成功 {awin} 个，仅 RL 成功 {bwin} 个；SL 覆盖 {coverage['supervised_unique_pairs']} 个训练位置–角度组合，RL 覆盖 {coverage['rl_unique_pairs']} 个。")
    lines += ["", "重复使用相同测试位置的多个种子不是更多独立场景样本。这里报告全部种子与配对计数，不宣称统计显著或某学习范式普遍优越。", "",
              "## 可重放的例子", ""]
    for item in examples:
        a, b = item["supervised"], item["reinforce"]
        lines.append(f"- {item['kind']}：{item['scene']['scene_id']}，SL {a['angle_deg']:+g}° / {a['result']['stop_reason']}；RL {b['angle_deg']:+g}° / {b['result']['stop_reason']}。")
    lines += ["", "## 复现", "", "```bash", "python collect_grasp_data.py --output reproduced/grasp_learning_data_new",
        "python compare_grasp_learning.py train --data reproduced/grasp_learning_data_new --output reproduced/grasp_learning_comparison_new",
        "python compare_grasp_learning.py evaluate --data reproduced/grasp_learning_data_new --output reproduced/grasp_learning_comparison_new", "```", "",
        "用默认已生成模型观看某个场景：", "", "```bash", "mjpython compare_grasp_learning.py demo --scene test_000 --method supervised --seed 42",
        "mjpython compare_grasp_learning.py demo --scene test_000 --method reinforce --seed 42", "```", "",
        f"冻结后重新执行了 {summary['fresh_test_verification']['unique_attempts']} 个去重后的测试位置–角度组合，全部与原始物理记录一致。", "",
        "原始数据位于数据目录的 train/validation/test.csv 和 .jsonl；完整方法选择见 selections.json；指标见 evaluation.json；模型、训练历史与探索记录位于 seed_*。",
        "",
        "局限：固定门族、固定抓点高度/力/速度、已知感知结果、预抓取初始化、离散角度，没有在线重抓或拒绝动作。SL 的输出是奖励预测值，不是校准成功概率。",
        "实验设计参考：https://www.jmlr.org/papers/volume25/23-0183/23-0183.pdf", ""]
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
            p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
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
