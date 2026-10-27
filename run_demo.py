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
        print("[1/3] 使用包内已有 RGB-D 观测；之后重新计算动力学。", flush=True)
    else:
        run("[1/3] 渲染新的 RGB-D 观测并识别把手。",
            "perception.py", "capture", "--observation", observation)
    run("[2/3] 选择抓取姿态，运行六组动力学实验与验证。",
        "benchmark.py", "--observation", observation, "--output", output)
    run("核对半开案例的实际停止状态与第一个越界参考目标。",
        "diagnose_workspace.py", "--results", output)
    run("[3/3] 渲染保存轨迹，导出 GIF 和夹持特写。",
        "render_results.py", "--output", output)
    run("导出正常速度的滑移到脱手近景及完整参数。",
        "render_slip_detail.py", "--output", output)
    print(f"\n运行完成。正常开门动画：{case_directory(output, 'nominal') / 'nominal.gif'}", flush=True)
    print(f"滑脱动画：{case_directory(output, 'low_friction') / 'low_friction.gif'}", flush=True)
    print(f"滑脱近景与参数：{case_directory(output, 'low_friction') / 'low_friction_closeup.gif'}，{case_directory(output, 'low_friction') / 'low_friction_parameters.json'}", flush=True)
    print(f"半开失败动画：{case_directory(output, 'workspace_limit') / 'workspace_limit.gif'}", flush=True)
    print(f"实验与验证结果：{output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"运行在上述步骤中止，退出码 {error.returncode}。请检查该步骤输出。") from error
