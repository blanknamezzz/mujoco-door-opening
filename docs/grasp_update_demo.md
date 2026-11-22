# Grasp-plan update comparison demo

This directory integrates `GraspPlan` and parameterized control. Previous `simulate.py` and `planning.py`
are preserved in `reproduced/before_grasp_update/`, with original reference results unchanged.

## Watching existing outputs

Open `reproduced/grasp_update_demo/index.html` to switch among angle, force, speed, and workspace-limit chapters,
with paired measured animations, parameters, and outcomes. `grasp_update_demo.webm` contains all chapters; each also has a GIF.

## Generation and reproduction

On Mac:

```bash
# Run from the repository root after activating an environment with requirements.txt installed
source .venv/bin/activate
python demo_grasp_update.py
```

Or separate physics computation from rendering:

```bash
python demo_grasp_update.py --physics-only
python demo_grasp_update.py --render-only
```

WebM export requires FFmpeg, using `ffmpeg` on PATH first and then an installed
Playwright FFmpeg. Specify `--ffmpeg /absolute/path/ffmpeg` if needed. GIF export works without FFmpeg.
Titles use system fonts. The verified Mac environment already provides fonts and an encoder.

## Viewing one plan live in MuJoCo

```bash
mjpython demo_grasp_update.py --view angle --variant before
mjpython demo_grasp_update.py --view angle --variant after
```

Replace `angle` with `force`, `speed`, or `limit`. Before/after are two manually configured plans,
**not old/new neural-network controls**. Live outputs go to `reproduced/grasp_update_demo/live/`.

## Demonstration details

1. **Angle**: same base `(0.8,-1.0)`, approach offset `0 to +8 degrees` only.
   Original forearm/right-finger self-collision occurs near full opening; the adjusted plan opens and holds.
   This illustrates whole-motion configuration rather than proving 8 degrees uniquely feasible or optimal.
2. **Grip force**: same friction `mu=0.2`, actuator cap per finger `2 N to 25 N` only.
   Original slips during pulling; adjusted completes. Caps differ from actual contact forces recorded in CSV.
3. **Speed**: same grasp/arc, opening duration `2 s to 14 s`.
   Fast motion slips; slow motion succeeds, without establishing monotonic benefit or optimality at 14 seconds.
4. **Workspace limit**: at `(0.3,-0.9)`, force caps `25 N to 35 N`.
   Both stop near 42.5 degrees at an out-of-range joint target while still grasping; force cannot expand workspace.

Each pair changes one plan field, checked automatically alongside expected success/failure and stop reasons.
Video plays at 2x simulated speed; the failed side freezes at termination rather than continuing to integrate/contact.
Aggregate results, plans, CSV, and NPZ share one output directory for independent inspection.

## Implemented and future capabilities

This demonstrates richer action parameters, traceable controls, and plan replay.
It supports studying narrow spaces, limited grip force, slippery contact, and speed tradeoffs,
without measured-material calibration or real-robot validation.

Current demos initialize at pregrasp with a fixed elbow branch, straight approach, and hinge-arc control.
This directory does not include H-JEPA training. Automatic candidate search, whole-trajectory screening,
learned success probabilities, and online regrasp/correction remain future work.
