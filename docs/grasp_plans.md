# Complete grasping and door-opening plans

Represent an attempt as `(grasp height, approach angle, elbow branch, stage timing, gripper actuator force cap)`.
`GraspPlan` supports saving, loading, and execution; representation alone does not establish feasibility or optimality.
Supported motion is smooth straight-line approach, closure, opening along the known hinge arc, and hold.

## Running on Mac

Activate the existing environment from the project directory:

```bash
source /Users/andy/Documents/Codex/2026-10-15/zh/work/mujoco-venv/bin/activate
```

Specify a plan using the handle center from bundled RGB-D:

```bash
mjpython simulate.py --case nominal --view \
  --base 0.6 -1.0 --base-yaw-deg 90 \
  --grasp-height-offset 0.03 --grasp-angle-deg 0 \
  --ik-branch -1 --open-duration 14 --grip-force 25 \
  --output reproduced/plan_slow
```

Changing `--grasp-angle-deg 0` to `8` tests an 8-degree approach offset in the same scene.
This specifies a candidate without guaranteeing stable grasp. Use `--ik-branch 1` for the other elbow branch;
execution does not switch branches automatically. For headless use, run `python` without `--view`.

If neither height nor angle is supplied, retain the original geometric baseline selection.
Supplying either sets the other unspecified value to 0 and executes that candidate directly.
Existing `nominal`, `workspace_limit`, and `low_friction` commands remain supported.

## Plan and scene parameters

| Option | Unit / range | Meaning |
| --- | --- | --- |
| `--grasp-height-offset` | m | Relative to initial handle center; positive upward |
| `--grasp-angle-deg` | degrees | Relative to the operating side's normal approach |
| `--ik-branch` | -1 or 1 | Elbow IK branch fixed for the complete action |
| `--approach-distance` | positive m | Grasp-to-pregrasp distance, default 0.16 |
| `--grip-force` | (0, 35] N | Actuator force cap per finger, default 35 |
| `--opening-angle` | (2, 90] degrees | Target opening, default 90 |
| `--settle-duration` | positive seconds | Pregrasp settling, default 0.5 |
| `--approach-duration` | positive seconds | Straight approach, default 2 |
| `--close-duration` | seconds | Total closure stage, default 1 |
| `--close-motion-duration` | positive seconds | Finger-closing motion, default 0.6 |
| `--open-duration` | positive seconds | Opening motion, default 10 |
| `--hold-duration` | at least 1 second | Post-opening hold, default 2 |
| `--base X Y` | m | Override case base position |
| `--base-yaw-deg` | degrees | Base orientation, default 90 |
| `--friction` | nonnegative | Sliding friction for all geometries |
| `--hinge-resistance` | nonnegative Nm | Hinge friction resistance, default 1.2 |

Total closure duration must exceed finger-closing motion by at least 0.2 seconds for contact acquisition.
The force cap is neither a target nor a measured contact force. Actual left/right contact forces are logged in CSV.
Total duration depends on stage settings. Success still requires final-second stable door angle and bilateral contact,
without slip alarm, contact loss, or collision. Finger opening targets and contact criteria retain the existing controller.

## Saving and replaying

Each `simulate.py` run saves:

- `nominal_execution_plan.json`: directly loadable complete action plan.
- `nominal.json`: outcome, plan, base pose, friction/hinge settings, and observation source.
- `nominal.csv`, `nominal.npz`: contact logs, joint trajectories, and time; no trajectory for initial unreachability.

Replay the same plan:

```bash
mjpython simulate.py --plan reproduced/plan_slow/nominal_execution_plan.json \
  --view --output reproduced/plan_replay
```

Or load a bundled example directly:

```bash
mjpython simulate.py --plan examples/grasp_plan_slow.json \
  --view --output reproduced/plan_example
```

Example centers come from known nominal geometry rather than a new visual detection.
Plan files store initial world coordinates and action settings, **not scene configuration**.
Replaying changes to base pose, friction, or hinge resistance requires the same scene options.
Loading does not redetect handles; after moving a handle, observe again and generate a fresh plan.
`--plan` cannot combine with grasp/timing/force overrides or `--oracle`, preserving the saved plan's identity.

A single case uses bundled fixed RGB-D by default rather than redetecting on every launch.
Direct single-case `simulate.py` still resets to pregrasp. New learning/collection workflows add
[PRM preparation](prm_pregrasp.md): plan and physically move from explicit initial joints,
then continue grasping with `trial(reset_before_trial=False)`. This is not post-grasp constrained planning.

## Programmatic and learning interfaces

```python
from grasp_plan import GraspPlan, SkillTiming
from simulate import Experiment

plan = GraspPlan(
    handle_center_world_m=(0.8, -0.11, 0.85),
    height_offset_m=0.03,
    approach_angle_deg=0,
    ik_branch=-1,
    grip_force_limit_N=25,
    timing=SkillTiming(open_s=14),
)
result, rows, states = Experiment().trial(plan=plan)
plan.save('reproduced/my_plan.json')
```

`skill_target` and `hjepa_scene.action_at` support the same `SkillTiming` and pregrasp distance.
Existing learning retains default timing/candidates, with action features compatible with previous models.
This update adds no network input parameters and retrains no models.
Future selectors can use physical outcomes collected across plan angles, branches, timing, and force caps.
Continuous candidate search, full-trajectory feasibility proofs, optimal planning, and online correction remain unimplemented.

## Verification

```bash
python -m unittest test_grasp_plan -v
python benchmark.py --output reproduced/plan_regression
```

New checks cover save/load, invalid parameters, default-trajectory compatibility, custom timing, grasp position/angle,
elbow branches, actuator force clipping, and CLI replay. Regression retains six physical cases and eight checks.
