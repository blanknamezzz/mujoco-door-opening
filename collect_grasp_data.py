"""Generate grouped, reproducible MuJoCo grasp-angle outcome data.

Scenes here are NEW BASE POSES around the SAME door, not new door geometry.
Splits are committed before any simulation. One row = one full skill attempt.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import mujoco
import numpy as np

from grasp_plan import GraspPlan
from perception import estimate
from simulate import ROOT
from train_grasp_rl import rollout, write_json, features

DEFAULT_DATA = ROOT / "reproduced" / "grasp_learning_data_prm"


def make_scenes(counts, seed):
    """Independent Latin hypercube sample per split, with fixed bounds."""
    rng = np.random.default_rng(seed)
    low = np.array([.35, -1.15, 60.])
    high = np.array([.95, -.75, 120.])
    scenes = []
    for split, n in zip(("train", "validation", "test"), counts):
        unit = np.column_stack([(rng.permutation(n)+rng.random(n))/n for _ in range(3)])
        for i, pose in enumerate(low + unit*(high-low)):
            scenes.append(dict(scene_id=f"{split}_{i:03d}", split=split,
                               base_xy_m=pose[:2].tolist(), base_yaw_deg=float(pose[2])))
    if len({tuple(s["base_xy_m"]+[s["base_yaw_deg"]]) for s in scenes}) != len(scenes):
        raise ValueError("Duplicate scene across splits")
    return scenes


def scene_config(manifest, scene):
    return dict(manifest["physics"], base_yaw_rad=float(np.deg2rad(scene["base_yaw_deg"])))


def context(manifest, scene):
    return features(scene["base_xy_m"], scene_config(manifest, scene))


def physics_job(job):
    manifest, scene, action = job
    angle = manifest["angles_deg"][action]
    trial, _, _, _ = rollout(scene_config(manifest, scene), scene["base_xy_m"], angle)
    return dict(scene_id=scene["scene_id"], split=scene["split"],
                base_yaw_deg=scene["base_yaw_deg"], action_index=int(action), **trial)


def preflight_job(job):
    """Geometry only: no opening rewards or future-state labels are read."""
    from dataclasses import replace
    from prm import PRMSettings, Roadmap
    from simulate import Experiment
    manifest, scene = job
    config = scene_config(manifest, scene)
    exp = Experiment(base=scene['base_xy_m'], base_yaw=config['base_yaw_rad'],
                     friction=config['friction'], hinge_resistance=config['hinge_resistance'])
    roadmap = Roadmap(exp, PRMSettings(**config['prm']))
    plan = GraspPlan.from_dict(config['plan'])
    queries = [dict(angle_deg=a, **roadmap.plan(replace(plan, approach_angle_deg=a)))
               for a in manifest['angles_deg']]
    found = [q['angle_deg'] for q in queries if q['path_found']]
    return dict(scene_id=scene['scene_id'], split=scene['split'], base_xy_m=scene['base_xy_m'],
        base_yaw_deg=scene['base_yaw_deg'], angles_with_path=found, queries=queries,
        status='pregrasp_path_found' if found else 'no_pregrasp_path_confirmed',
        note='Pregrasp only. Search failure is not a proof that full opening is impossible.')


def load_records(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as file:
        return [json.loads(line) for line in file if line.strip()]


def export_csv(path, records):
    fields = ["scene_id", "split", "base_x", "base_y", "base_yaw_deg", "angle_deg",
              "action_index", "reward", "success", "grasp_acquired", "safe_max_angle_deg",
              "final_angle_deg", "collision", "slip", "stop_reason", "prm_status", "opening_tested"]
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for t in records:
            r = t["result"]
            writer.writerow(dict(scene_id=t["scene_id"], split=t["split"],
                base_x=t["base_xy_m"][0], base_y=t["base_xy_m"][1], base_yaw_deg=t["base_yaw_deg"],
                angle_deg=t["angle_deg"], action_index=t["action_index"], reward=t["reward"],
                success=int(r["outcome"] == "D_full_open"), grasp_acquired=int(r.get("grasp_acquired", False)),
                safe_max_angle_deg=t["reward_terms"].get("safe_progress", 0.) * 45.,
                final_angle_deg=r.get("final_angle_deg", 0.), collision=int(bool(r.get("unexpected_contact_samples"))),
                slip=int(bool(r.get("slip_detected") or r.get("contact_lost"))), stop_reason=r["stop_reason"],
                prm_status=((t.get('preparation') or {}).get('query') or {}).get('status', 'legacy_pregrasp_reset'),
                opening_tested=t.get('opening_tested', True)))


def validate_records(manifest, records, split):
    scenes = {s["scene_id"]: s for s in manifest["scenes"] if s["split"] == split}
    keys = [(t["scene_id"], t["action_index"]) for t in records]
    expected = {(sid, a) for sid in scenes for a in range(len(manifest["angles_deg"]))}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f"Incomplete/duplicate {split} data: expected {len(expected)}, got {len(keys)}")
    for t in records:
        s = scenes[t["scene_id"]]
        if t["split"] != split or t["base_xy_m"] != s["base_xy_m"] or t["base_yaw_deg"] != s["base_yaw_deg"]:
            raise ValueError("Record scene metadata does not match manifest")
        if t["angle_deg"] != manifest["angles_deg"][t["action_index"]]:
            raise ValueError("Record angle does not match manifest")


def collect(args):
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("Dataset exists. Use --resume or a new --output.")
        manifest = json.loads(manifest_path.read_text())
        for name, digest in manifest["source_sha256"].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source changed since collection: {name}. Use a new dataset directory.")
    else:
        if any(args.output.iterdir()):
            raise ValueError("New dataset directory must be empty")
        center = tuple(estimate(ROOT/"results"/"observation.npz")["center_world_m"])
        from prm import PRMSettings
        physics = dict(friction=.8, hinge_resistance=1.2, plan=GraspPlan(center).to_dict())
        if not args.no_prm:
            physics['prm'] = PRMSettings(samples=args.prm_samples, seed=args.prm_seed,
                                         initial_q=args.initial_q).to_dict()
        manifest = dict(schema_version=2, seed=args.seed,
            scope="Held-out base poses for the same door family; no new door geometry or real-robot claim.",
            angles_deg=np.arange(-20., 20.1, 2.).tolist(),
            scenes=make_scenes([args.train_scenes, args.validation_scenes, args.test_scenes], args.seed),
            bounds=dict(base_x=[.35, .95], base_y=[-1.15, -.75], base_yaw_deg=[60., 120.]),
            physics=physics,
            versions=dict(mujoco=mujoco.__version__, numpy=np.__version__),
            source_sha256={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                           for n in ("scene.xml", "simulate.py", "grasp_plan.py", "prm.py", "train_grasp_rl.py", "collect_grasp_data.py")})
        write_json(manifest_path, manifest)
    start = time.perf_counter()
    summary = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for split in ("train", "validation", "test"):
            if manifest['physics'].get('prm') is not None:
                preflight_path = args.output/f'{split}_prm_preflight.jsonl'
                existing = load_records(preflight_path)
                seen = {q['scene_id'] for q in existing}
                jobs = [(manifest, s) for s in manifest['scenes'] if s['split'] == split and s['scene_id'] not in seen]
                with preflight_path.open('a') as file:
                    for query in pool.map(preflight_job, jobs):
                        file.write(json.dumps(query, allow_nan=False)+'\n'); file.flush()
                        existing.append(query)
                print(f"{split}: PRM preflight {sum(bool(q['angles_with_path']) for q in existing)}/{len(existing)} positions with a pregrasp path", flush=True)
            path = args.output / f"{split}.jsonl"
            records = load_records(path)
            completed = {(t["scene_id"], t["action_index"]) for t in records}
            jobs = [(manifest, s, a) for s in manifest["scenes"] if s["split"] == split
                    for a in range(len(manifest["angles_deg"])) if (s["scene_id"], a) not in completed]
            with path.open("a") as file:
                for t in pool.map(physics_job, jobs, chunksize=1):
                    file.write(json.dumps(t, allow_nan=False)+"\n")
                    file.flush()
                    records.append(t)
                    if len(records) % 42 == 0:
                        print(f"{split}: {len(records)} attempts recorded", flush=True)
            validate_records(manifest, records, split)
            export_csv(args.output / f"{split}.csv", records)
            summary[split] = dict(scenes=len({t["scene_id"] for t in records}), attempts=len(records),
                successful_attempts=sum(t["result"]["outcome"] == "D_full_open" for t in records),
                opening_trials_executed=sum(t.get('opening_tested', True) for t in records),
                preparation_failures=sum(not t.get('opening_tested', True) for t in records))
    summary["collection_seconds_this_run"] = time.perf_counter()-start
    summary["note"] = "Collecting held-out reference labels does not authorize using them for training."
    write_json(args.output / "collection_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_DATA)
    p.add_argument("--train-scenes", type=int, default=36)
    p.add_argument("--validation-scenes", type=int, default=12)
    p.add_argument("--test-scenes", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument('--no-prm', action='store_true', help='Explicit legacy reset-to-pregrasp baseline')
    p.add_argument('--prm-samples', type=int, default=240)
    p.add_argument('--prm-seed', type=int, default=42)
    p.add_argument('--initial-q', type=float, nargs=4, default=[0., -np.pi/2, -2*np.pi/3, 0.],
                   metavar=('LIFT_M', 'SHOULDER_RAD', 'ELBOW_RAD', 'WRIST_RAD'))
    a = p.parse_args()
    if min(a.train_scenes, a.validation_scenes, a.test_scenes, a.workers) < 1 or min(a.seed, a.prm_seed, a.prm_samples) < 0:
        p.error("Counts must be positive and seed nonnegative")
    collect(a)


if __name__ == "__main__":
    main()
