# Implementation: handle detection, grasp selection, contact opening, and slip monitoring

Paths are relative to the repository root.

Sections 1-8 document the original contact baseline. See [scene integration and training](hjepa_setup.md) and Section 9 for the added official H-JEPA world model/staged policies.

This contact-dynamics implementation extends an earlier geometric reachability prototype for a **parallel two-finger gripper and fixed lever handle**, adding cameras, solid fingers, friction, force-driven opening, and contact monitoring.

**Baseline scope:** one known door, fixed base, simplified planar three-rotation arm plus vertical lift, and two fingers. Calibrated RGB-D and color/shape priors provide vision, without generic detection, hardware models, online tracking, or a learned world model in this baseline.

## 1. Handle and robot geometry

### Door and fixed lever

Panel dimensions: 0.9 m wide, 1.4 m high, 0.05 m thick; mass 8 kg. The hinge has one rotational degree about world -Z, opening toward the robot from closed to 90 degrees, with 1.2 Nm friction and 0.6 Nm s/rad damping.

The handle is a rigid child body without joints. Closed center `(0.8, -0.11, 0.85)` m. Its capsule has cylindrical endpoints at heights 0.70-1.00 m and radius 0.015 m, total height 0.33 m with caps. Axis-to-panel-center distance is 0.11 m, axis-to-front-surface 0.085 m, and rear-handle clearance about 0.07 m. Upper/lower supports connect it to the panel.

Handle, panel, supports, and finger pads collide physically. A handle site alone is not a success target, and no weld joins fingers to handle.

See `scene.xml`, `door -> handle -> handle_bar`. This fixed handle has no latch or downward-unlocking joint; lever-and-latch behavior requires extra modeling.

### Parallel two-finger gripper

The wrist supports a palm, two sliders, fingers, and pads. Hand X points along approach, Y along closure, Z upward. Front-side approach places pads on opposite sides of the vertical bar.

- Grasp center is 0.18 m ahead of the wrist.
- Each solid pad measures 64 x 20 x 50 mm.
- Sliders move along opposite Y directions with 0-45 mm travel each.
- Open joint targets are 40 mm per side, giving about 84 mm internal spacing.
- Closure target is 0; the 30 mm diameter solid bar stops further motion, and the dynamics solver determines contact force.
- Finger position stiffness is 1800 N/m, with +/-35 N actuator limits per finger. Nominal normal contact is about 23 N per side.

`condim=4` includes normal force, two tangential friction directions, and torsional friction. Nominal sliding coefficient is mu=0.8. Soft constraints allow small penetration/relative motion rather than ideal rigid attachment.

## 2. Detecting the handle

`perception.py capture` renders fixed-camera RGB/depth in meters and saves intrinsics K and camera-to-world T. `estimate()` reads only those four inputs, not handle-body truth, sites, or object-ID segmentation.

Pipeline:

1. Segment gold pixels using scene-specific thresholds rather than generic handle detection.
2. Backproject depth through intrinsics into camera coordinates, then transform into world coordinates.
3. Use vertical-axis priors and height quantiles to estimate endpoints. Both ends must be visible; limited central arm occlusion is allowed.
4. Exclude spherical caps and fit a circle to central XY cross-sections, checking against known radius 15 mm.
5. Output center, vertical axis, visible height, fit error, and segmentation count.

Estimated center is about `(0.800012, -0.110082, 0.852399)` m. Independent truth comparison gives about **2.4 mm** center error and **0.14 mm** cross-section RMS. These are controlled noiseless-rendering measurements rather than real-camera accuracy.

Backprojection follows MuJoCo's camera-facing -Z convention:

```text
Xc = (u + 0.5 - cx) · depth / fx
Yc = -(v + 0.5 - cy) · depth / fy
Zc = -depth
Pw = Rwc · Pc + twc
```

Information sources differ: RGB-D supplies the initial handle center; known geometry supplies hinge, robot dimensions, and collision models; simulation state supplies contact forces/points and relative motion. Hardware would require tactile/torque sensing or estimation for the latter.

## 3. Choosing the grasp

`planning.py` generates **3 heights x 3 approach orientations = 9 candidates** on the detected bar:

- Heights: estimated center minus 70 mm, center, plus 70 mm.
- Orientations: -15, 0, +15 degrees relative to the front normal.

Each candidate has 3D position and hand orientation, with clearance for pads and end supports; one center point is insufficient to specify a grasp.

First check approach from 160 mm in front, then opening at 2-degree intervals. Analytical IK checks lift, joint ranges, jumps, and unexpected collisions. Intended pad-handle contact is allowed.

Rank achievable opening angle first, then central height, smaller orientation offset, and greater joint margin. This run selects detected center/zero offset, geometrically reaching 90 degrees. Save every candidate and selection in `results/grasp_plan.json`.

Geometry screening does not prove force closure. Low friction can still cause release, requiring dynamics. Fixed IK branches/discrete samples do not establish global feasibility.

## 4. Contact-driven door motion

Control sequence:

```text
Camera -> handle detection -> 9 candidates -> approach -> closure
       -> stable bilateral contact -> pull -> hold -> evaluation
```

Settle at pregrasp for 0.5 s, approach for 2 s, and close for 1 s. Grasp acquisition requires both normal forces >1 N and tangential relative speeds <5 mm/s for at least 0.2 s.

Opening starts at simulated 3.5 s with a smooth 10 s angular reference. Rotate the vision-derived initial point about the known hinge while rotating hand orientation:

```text
p_des(θ) = Rz(-θ) · p_grasp
yaw_des(θ) = yaw_initial - θ
```

Convert desired hand pose to joint targets at every step; finite joint-servo torques drive motion. Compensate robot gravity/bias only. **The door hinge receives no actuation or compensation.**

`mj_step()` advances dynamics. No door actuator/weld/direct-angle updates exist; pad contact must generate hinge torque. Direct `qpos` assignment is restricted to initialization, separate geometry queries, and visualization replay.

Control uses known hinge trajectories and joint feedback, rather than Cartesian impedance or tracking through occlusion. Persistent slip/contact loss normally terminates simulation, without automatic regrasp or hardware braking control.

## 5. Contact motion and slip detection

**World-frame contact points normally move.** Hand and handle rotate along the door arc even under stable grasp.

Log contacts every 10 ms in world, handle-local, and finger-local frames. For multiple points per side, use a normal-force-weighted center and total normal force.

Transform world contact c into handle coordinates:

```text
c_handle = R_handleᵀ · (c_world - p_handle)
```

Local contact-center changes can reflect rolling, pressure redistribution, or solver point switching rather than slip. Also compute relative tangential contact velocity:

```text
v_rel = (J_finger(c) - J_handle(c)) · qvel
v_tangent = (I - n nᵀ) · v_rel
```

Log hand-handle translation/rotation drift and friction utilization `||Ft|| / (mu Fn)`. High utilization means little margin rather than proof of sliding, especially with torsional friction and soft constraints.

Explicit baseline criteria:

| State | Criterion |
|---|---|
| Grasp acquired | Both normal forces >1 N and tangential relative speeds <5 mm/s for >=0.2 s |
| Persistent-slip alarm | During bilateral contact, tangential speed >10 mm/s, translation drift >8 mm, or rotation drift >5 degrees for >=80 ms |
| Contact lost | After acquisition, either normal force no longer >1 N for >=100 ms |
| Complete opening | Angle >=88 degrees, final 1 s bilateral contact, absolute angular speed <0.05 rad/s, no slip/loss/unexpected collision |

These engineering thresholds need recalibration for pad size, speed, and noise. The 88-degree threshold gives 2-degree tolerance around 90; approximately 89.68 is not exactly 90.

Nominal maximum translation drift is about **4.66 mm**, peak tangential speed **0.85 mm/s**, without persistent-slip alarm. The force-weighted center can nevertheless move about **25 mm** as discrete contacts/weights change; this is not 25 mm finger sliding or a claim of zero microscopic slip.

## 6. Unreachable, grasp-only, incomplete, and complete opening

Reach is intermediate. `reach_only` stops after grasp acquisition without establishing subsequent opening feasibility.

Six cases retain nominally selected point/orientation while changing base, friction, or closure to isolate failures. Replanning after moving base is separate, through `choose_grasp(new Experiment, observation)`.

| Case | Setting / outcome | Classification |
|---|---|---|
| `unreachable` | Base `(2.0,-1.2)` m; initial grasp/pregrasp unreachable | A: unreachable |
| `reach_only` | Reach and stable bilateral grasp; door near zero degrees | B: acquired, opening untested |
| `workspace_limit` | Base `(0.3,-0.9)` m; grasp succeeds, then joint/workspace limit near **42.505 degrees** | C: reached but target opening incomplete |
| `low_friction` | Mu 0.8 to **0.005**; alarm **3.771 s**, loss **4.731 s**, maximum opening **0.006 degrees** | C: grasp lost through inadequate friction |
| `nominal` | Base `(0.6,-1.0)` m; final **89.679 degrees**, contact and hold checks pass | D: complete within tolerance |
| `no_grip` | Fingers remain open with identical targets; door near **0 degrees** | Control: hand motion alone cannot open without contact |

Low friction deliberately isolates failure rather than representing a measured material. With about 23 N per side, ideal tangential capacity is `2 * 0.8 * 23 = 36.8 N`, versus about 0.23 N at low friction. Overcoming 1.2 Nm hinge friction alone needs about 1.5 N tangentially, before inertia/damping, so low friction cannot sustain pulling.

### Exact partial-opening failure

`results/workspace_limit/workspace_limit.gif` records base `(0.3,-0.9)` m. At actual angle 42.505 degrees, elbow is -163.589 degrees; contacts are about 23.04/23.06 N with no slip/loss alarm.

The first rejected reference occurs at 8.404 s: reference door 43.704 degrees requires elbow -165.0005, beyond the roughly -165-degree lower bound. Maintaining grasp orientation moves the wrist toward the shoulder, requiring more folding of 0.60/0.55 m links. Minimum shoulder-wrist range at the elbow bound is about 0.158079 m, while this target requests 0.158074 m.

**The controller terminates on an infeasible next reference rather than actual elbow contact with the hard limit.** Tracking error leaves actual 42.505 below reference 43.704 degrees. The stopping angle is not the exact physical maximum. A final 1.8 s animation pause displays the terminal state without further integration.

The limit depends on base, point, orientation, and trajectory. Moving base to `(0.6,-1.0)` m with matching grasp/orientation opens to about 89.68 degrees. Initial grasp success does not guarantee a feasible full opening trajectory.

`python diagnose_workspace.py` writes `results/workspace_limit/workspace_limit_diagnosis.json`, preserving actual and first rejected targets. `python render_results.py --case workspace_limit` renders the case separately.

Batch low-friction diagnostics explicitly continue beyond slip to observe subsequent release. Normal `simulate.py --case low_friction` stops at persistent-slip alarm.

### Release case and parameters

`results/low_friction/low_friction_closeup.gif` shows 2.7-5.5 s at normal speed: closure, bilateral contact, slip, release. `results/low_friction/low_friction_parameters.json` preserves friction, actuators, handle/hinge, timing, thresholds, and events. `Experiment(friction=0.005)` overrides sliding friction for all geoms, retaining XML torsional/rolling coefficients. `condim=4` does not constrain rolling friction.

At 3.401 s normal forces are about 23.08/23.03 N. At slip alarm 3.771 s, tangential speeds are 15.0/15.6 mm/s with bilateral contact. At confirmed loss 4.731 s, both forces are 0 N and translation drift about 49 mm. Maximum positive door opening is about 0.006 degrees.

Reproduce the full release sequence:

```bash
python simulate.py --case low_friction --continue-after-slip
python render_slip_detail.py
```

Close-up BILATERAL CONTACT indicates current contact, while stable acquisition uses the separate 0.2 s dynamics criterion. After loss, tangential speed is n/a; recorded zeros without contact do not imply sliding has stopped.

## 7. Reproduction

Verified on macOS, Python 3.12.5, MuJoCo 3.16.0, NumPy 2.5.4, Pillow 12.3.0, with pinned dependencies. Use Python 3.12: python3 below must select that version, or use python3.12. Create an isolated environment in this directory:

The shortest full workflow installs dependencies then runs the unified entry, capturing fresh RGB-D, executing six cases, and exporting opening/slip animations. Defaults write `reproduced/` while preserving bundled `results/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_demo.py
```

Open `reproduced/nominal.gif` afterward. `reproduced/low_friction.gif` shows deliberately reduced-friction release; `reproduced/workspace_limit.gif` shows termination at an out-of-range elbow target. `reproduced/summary.json` records angles/checks, and `workspace_limit_diagnosis.json` provides detailed diagnosis.

`python run_demo.py --reuse-observation` reuses bundled RGB-D while recomputing physics/rendering. Both animation workflows need graphics. Change output with `--output new_directory`.

For stepwise inspection/debugging:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Capture fresh RGB-D using local graphics, then detect the handle.
python perception.py capture

# Headless batch dynamics using the saved RGB-D.
python benchmark.py

# A single case also defaults to RGB-D rather than handle-coordinate truth.
python simulate.py --case nominal
python simulate.py --case low_friction

# macOS live viewing uses mjpython.
mjpython simulate.py --case nominal --view

# Render animations/close-ups from saved physical trajectories.
python render_results.py
```

Other platforms normally use `python simulate.py --case nominal --view`. Live viewing/rendering needs graphics. Bundled `results/observation.npz` allows headless `python benchmark.py`, using saved rather than fresh observation.

Explicit `python simulate.py --oracle` is an ideal-perception control using known nominal center, labeled as bypassing vision. Its results are not visual-detection results.

## 8. Files and verification evidence

- `scene.xml`: door, solid handle, three-joint arm, lift, fingers, actuators.
- `perception.py`: RGB-D, segmentation, backprojection, cross-section fitting.
- `planning.py`: nine grasp candidates and geometry screening.
- `simulate.py`: physical control, forces, relative velocity, local contacts, states.
- `benchmark.py`: six cases, no-grip/no-door-actuator controls, timestep sensitivity.
- `render_results.py`: visualization replay of saved trajectories, outside result generation.
- `run_demo.py`: fresh observation, dynamics, and animation pipeline.
- `diagnose_workspace.py`: partial-opening actual state, first rejected target, limiting joint.
- `render_slip_detail.py`: normal-speed release close-up and complete parameter JSON.
- `results/summary.json`: classifications, thresholds, perception error, verification.
- `results/<case>/*.csv`: 100 Hz forces, tangential speeds, contacts, drift, alarms. Missing contacts use NaN rather than zero coordinates.
- `results/<case>/*.npz`: dynamic trajectories; `results/observation.npz`: shared initial RGB-D.
- `results/nominal/nominal.gif`, `results/low_friction/low_friction.gif`: opening/release near 2x speed.
- `results/nominal/nominal_grasp_closeup.png`: physical pad-handle close-up.

All 8 checks pass. Changing timestep from 1 ms to 0.5/2 ms gives nominal final 89.681/89.676 degrees, differing by <0.01. This is numerical consistency in one deterministic scene rather than a general success rate or hardware validation.

## 9. Course-project integration

[Official H-JEPA scene integration](hjepa_setup.md) adds scene-split collection, initial/reached RGB-D, two-level latent prediction, downstream returns, station/grasp policies, and fixed-controller execution. Sections 1-8 describe the contact baseline; new entries are `collect_hjepa.py`, `train_hjepa.py`, `run_hjepa.py`, and `evaluate_hjepa.py`.

The baseline data pipeline varies handle position, friction, and hinge resistance with scene-disjoint train/validation/test splits. Mobile navigation, hardware models, visual-noise/occlusion studies, online GRPO, and hardware generalization are not implemented in this stage. Stations use simulator resets, base stays fixed while opening, and contact dynamics retain no ideal no-slip weld.

Added 2026-10-17 [stage/angle measurements](hjepa_stage_results.md) freeze the official model, cross pre/post P/G, fix reached states for grasp comparisons, and scan base/grasp yaw, targets, and calibrated views. Natural-occlusion vision failures count; artificial noise/general occlusion tracking are not added.

Same-day [structural constraints](hjepa_structural_results.md) add colliding pillars, two-way doors, one-way limits, and one-sided handles. Two bases per face share a mirrored candidate library, observations, IK, directions, and criteria. Same-side repositioning cannot remove a blocked door arc; opposite-side success requires allowed reverse motion and handle access. Invisible hinge limits need cues or interaction rather than privileged geometry presented as visual prediction.

Official references:

- [MuJoCo contact modeling](https://mujoco.readthedocs.io/en/stable/modeling.html#contact)
- [MuJoCo contact/friction computation](https://mujoco.readthedocs.io/en/stable/computation.html#contact)
- [mj_contactForce and related APIs](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce)
- [Python bindings](https://mujoco.readthedocs.io/en/stable/python.html)
