"""Reproduce observation, physical trials, and GIFs with one command."""
from __future__ import annotations
import argparse
from result_paths import case_directory
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(title, script, *args):
    print(f"\n{title}", flush=True)
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced",
                        help="Directory for the new observation, logs and GIFs")
    parser.add_argument("--reuse-observation", action="store_true",
                        help="Reuse bundled RGB-D; still rerun physics and rendering")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    observation = output / "observation.npz"
    if args.reuse_observation:
        source = ROOT / "results" / "observation.npz"
        if observation != source:
            shutil.copy2(source, observation)
        print("[1/3] Reuse bundled RGB-D, then recompute dynamics.", flush=True)
    else:
        run("[1/3] Render fresh RGB-D and detect the handle.",
            "perception.py", "capture", "--observation", observation)
    run("[2/3] Select a grasp and run six physical cases with verification.",
        "benchmark.py", "--observation", observation, "--output", output)
    run("Check the partial-opening stopping state and first out-of-range target.",
        "diagnose_workspace.py", "--results", output)
    run("[3/3] Render saved trajectories, GIFs, and grasp close-ups.",
        "render_results.py", "--output", output)
    run("Export normal-speed slip/release close-up and full parameters.",
        "render_slip_detail.py", "--output", output)
    print(f"\nComplete. Opening animation: {case_directory(output, 'nominal') / 'nominal.gif'}", flush=True)
    print(f"Release animation: {case_directory(output, 'low_friction') / 'low_friction.gif'}", flush=True)
    print(f"Release close-up and parameters: {case_directory(output, 'low_friction') / 'low_friction_closeup.gif'}, {case_directory(output, 'low_friction') / 'low_friction_parameters.json'}", flush=True)
    print(f"Partial-opening failure: {case_directory(output, 'workspace_limit') / 'workspace_limit.gif'}", flush=True)
    print(f"Experiment and verification results: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"The step above stopped with exit code {error.returncode}. Check its output.") from error
