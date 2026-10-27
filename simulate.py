"""Force-driven door opening: 3R planar arm, vertical lift, two sliding fingers.

Only reset/planning/replay assign qpos. Live trials integrate mj_step; the door
has no actuator, equality constraint, or externally applied force.
"""
from __future__ import annotations
import argparse
from result_paths import case_directory
import csv
import json
import math
from pathlib import Path
import time
import mujoco
import numpy as np
from grasp_plan import GraspPlan, SkillTiming

ROOT = Path(__file__).resolve().parent
ARM = ["shoulder", "elbow", "wrist_yaw"]


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.]])


def smooth(x):
    x = np.clip(x, 0, 1)
    return x*x*(3-2*x)


def skill_target(t, point, yaw_offset=0., mode="nominal", opening_angle_deg=90., opening_direction=1,
                 timing=None, approach_distance_m=.16):
    """Cartesian hinge-arc schedule; defaults preserve the trained action interface."""
    timing = DEFAULT_TIMING if timing is None else timing
    yaw0 = math.pi/2+yaw_offset
    pre = point-approach_distance_m*np.array([np.cos(yaw0), np.sin(yaw0), 0.])
    if t < timing.settle_s:
        target, yaw, aperture, phase = pre, yaw0, .04, "settle"
    elif t < timing.close_start:
        target, yaw, aperture, phase = pre+smooth((t-timing.settle_s)/timing.approach_s)*(point-pre), yaw0, .04, "approach"
    elif t < timing.open_start:
        target, yaw, aperture, phase = point, yaw0, .04*(1-smooth((t-timing.close_start)/timing.close_motion_s)), "close"
    else:
        theta = opening_direction*math.radians(opening_angle_deg)*smooth((t-timing.open_start)/timing.open_s)
        target, yaw, aperture, phase = rot(theta)@point, yaw0-theta, 0., "open" if t < timing.hold_start else "hold"
    if mode == "no_grip":
        aperture = .04
    return target, yaw, aperture, phase


DEFAULT_TIMING = SkillTiming()


class Experiment:
    def __init__(self, base=(0.6, -1.0), friction=0.8, hinge_resistance=1.2,
                 base_yaw=math.pi/2, handle_position=None, model_xml=None,
                 ik_branch=-1, opening_direction=1, handle_body="handle",
                 handle_geom="handle_bar", operating_side="front"):
        if opening_direction not in (-1, 1) or ik_branch not in (-1, 1):
            raise ValueError("Opening direction and IK branch must be +/-1.")
        if operating_side not in ("front", "back"):
            raise ValueError("Operating side must be front or back.")
        self.opening_direction = opening_direction
        self.ik_branch = ik_branch
        self.operating_side = operating_side
        self.grasp_yaw_offset = 0. if operating_side == "front" else math.pi
        self.grasp_yaw_sign = 1 if operating_side == "front" else -1
        self.m = (mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml")) if model_xml is None
                  else mujoco.MjModel.from_xml_string(model_xml))
        self.m.body("base").pos[:2] = base
        self.base = np.array(base)
        self.base_yaw = float(base_yaw)
        self.m.body("base").quat[:] = [math.cos(base_yaw/2), 0, 0, math.sin(base_yaw/2)]
        if handle_position is not None:
            self.m.body(handle_body).pos[:] = handle_position
        self.m.geom_friction[:, 0] = friction
        self.m.dof_frictionloss[self.m.joint("door_hinge").dofadr[0]] = hinge_resistance
        self.d = mujoco.MjData(self.m)
        self.arm_q = np.array([self.m.joint(n).qposadr[0] for n in ARM])
        self.arm_j = np.array([self.m.joint(n).id for n in ARM])
        self.lift_q = self.m.joint("lift_z").qposadr[0]
        self.door_q = self.m.joint("door_hinge").qposadr[0]
        self.finger_q = np.array([self.m.joint(n).qposadr[0] for n in ["left_slide", "right_slide"]])
        self.robot_dof = np.array([self.m.joint(n).dofadr[0] for n in ["lift_z", *ARM, "left_slide", "right_slide"]])
        self.bar = self.m.geom(handle_geom).id
        self.hbody = self.m.body(handle_body).id
        self.hand = self.m.body("hand").id
        self.pads = [self.m.geom(n).id for n in ["left_pad", "right_pad"]]
        self.site = self.m.site("grasp_center").id
        # Preserve constructor/XML defaults separately from per-trial plans.
        self._default_ik_branch = self.ik_branch
        self._default_grip_limits = {
            name: self.m.actuator(name).forcerange.copy()
            for name in ('left_grip', 'right_grip')
        }
        side = -1 if operating_side == "front" else 1
        self.reset(np.array([0.8, side*0.27, 0.85]), -side*math.pi/2)

    def ik(self, target, yaw, branch=None):
        branch = self.ik_branch if branch is None else branch
        wrist = np.asarray(target[:2]) - .18*np.array([np.cos(yaw), np.sin(yaw)])
        delta = wrist-self.base
        c = (delta@delta-.6**2-.55**2)/(2*.6*.55)
        if abs(c) > 1:
            return None
        q2 = branch*np.arccos(c)
        q1w = np.arctan2(delta[1], delta[0])-np.arctan2(.55*np.sin(q2), .6+.55*np.cos(q2))
        q = np.array([q1w-self.base_yaw, q2, yaw-q1w-q2])
        q = (q+math.pi) % (2*math.pi)-math.pi
        limits = self.m.jnt_range[self.arm_j]
        if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]) or abs(target[2]-.85) > .12:
            return None
        return q

    def reset(self, target, yaw):
        mujoco.mj_resetData(self.m, self.d)
        q = self.ik(target, yaw)
        if q is not None:
            self.d.qpos[self.arm_q] = q
            self.d.qpos[self.lift_q] = target[2]-.85
        self.d.qpos[self.finger_q] = .04
        self.d.ctrl[:] = [self.d.qpos[self.lift_q], *self.d.qpos[self.arm_q], .04, .04]
        mujoco.mj_forward(self.m, self.d)

    def command(self, point, yaw, aperture):
        q = self.ik(point, yaw)
        if q is None:
            return False
        self.d.ctrl[:] = [point[2]-.85, *q, aperture, aperture]
        # Model-based gravity/Coriolis compensation on ROBOT DOFs only.
        self.d.qfrc_applied[:] = 0
        self.d.qfrc_applied[self.robot_dof] = self.d.qfrc_bias[self.robot_dof]
        return True

    def contact_metrics(self):
        forces = np.zeros(2)
        speed = np.zeros(2)
        ratio = np.zeros(2)
        hlocal = np.zeros((2, 3))
        flocal = np.zeros((2, 3))
        unexpected = []
        R = self.d.xmat[self.hbody].reshape(3, 3)
        hp = self.d.xpos[self.hbody]
        for i, c in enumerate(self.d.contact):
            ids = {int(c.geom1), int(c.geom2)}
            sides = [s for s, pad in enumerate(self.pads) if ids == {self.bar, pad}]
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.m, self.d, i, wrench)
            if not sides:
                if wrench[0] > .5:
                    unexpected.append(":".join(self.m.geom(g).name for g in [c.geom1, c.geom2]))
                continue
            s = sides[0]
            fn = max(0., wrench[0])
            fb = self.m.geom_bodyid[self.pads[s]]
            jp1, jp2, jr = [np.zeros((3, self.m.nv)) for _ in range(3)]
            mujoco.mj_jac(self.m, self.d, jp1, jr, c.pos, self.hbody)
            mujoco.mj_jac(self.m, self.d, jp2, jr, c.pos, fb)
            vrel = (jp2-jp1)@self.d.qvel
            normal = c.frame[:3]
            vt = np.linalg.norm(vrel-normal*(normal@vrel))
            forces[s] += fn
            speed[s] += fn*vt
            ratio[s] = max(ratio[s], np.linalg.norm(wrench[1:3])/(c.friction[0]*fn+1e-9))
            hlocal[s] += fn*(R.T@(c.pos-hp))
            flocal[s] += fn*(self.d.xmat[fb].reshape(3, 3).T@(c.pos-self.d.xpos[fb]))
        for s in range(2):
            if forces[s] > 1e-8:
                speed[s] /= forces[s]
                hlocal[s] /= forces[s]
                flocal[s] /= forces[s]
            else:
                hlocal[s] = np.nan
                flocal[s] = np.nan
        rel = R.T@(self.d.site_xpos[self.site]-hp)
        return forces, speed, ratio, hlocal, flocal, rel, unexpected

    def trial(self, point=None, yaw_offset=0., mode="nominal", viewer=None, diagnostic_continue=False,
              observer=None, observation_interval=.5, reset_before_trial=True,
              opening_angle_deg=90., plan=None):
        """Execute contact dynamics; optional observer records actual reached states.

        The observer receives (experiment, terminal). It is called after reset,
        at the requested interval, and at termination, including early failures.
        """
        timing, approach_distance = DEFAULT_TIMING, .16
        plan_metadata = {}
        if plan is None:
            # mj_resetData resets dynamic state, not MjModel actuator limits.
            # A legacy/default trial must not inherit the previous plan.
            self.ik_branch = self._default_ik_branch
            for name, limits in self._default_grip_limits.items():
                self.m.actuator(name).forcerange[:] = limits
        if plan is not None:
            if point is not None or yaw_offset != 0. or opening_angle_deg != 90.:
                raise ValueError('Pass either point/yaw_offset or plan, not both.')
            point = np.asarray(plan.point_world_m)
            yaw_offset = plan.yaw_offset_rad(self)
            opening_angle_deg = plan.opening_angle_deg
            timing, approach_distance = plan.timing, plan.approach_distance_m
            self.ik_branch = plan.ik_branch
            for name in ('left_grip', 'right_grip'):
                self.m.actuator(name).forcerange[:] = [-plan.grip_force_limit_N, plan.grip_force_limit_N]
            plan_metadata = dict(execution_plan=plan.to_dict(), point_world_m=list(plan.point_world_m),
                                 yaw_offset_deg=math.degrees(yaw_offset), ik_branch=self.ik_branch)
        if point is None:
            raise ValueError('A grasp point or plan is required.')
        point = np.asarray(point, dtype=float)
        if observation_interval <= 0:
            raise ValueError("observation_interval must be positive")
        if not 2 < opening_angle_deg <= 90:
            raise ValueError("opening_angle_deg must be in (2, 90].")
        yaw0 = math.pi/2+yaw_offset
        approach_dir = np.array([np.cos(yaw0), np.sin(yaw0), 0.])
        pre = point-approach_distance*approach_dir
        if self.ik(point, yaw0) is None or self.ik(pre, yaw0) is None:
            return dict(outcome="A_initial_unreachable", max_angle_deg=0.,
                        stop_reason="initial_grasp_or_pregrasp_unreachable", **plan_metadata), [], []
        if reset_before_trial:
            self.reset(pre, yaw0)
        if observer is not None:
            observer(self, False)
        next_observation = observation_interval
        dt = self.m.opt.timestep
        sample_steps = max(1, round(.01/dt))
        sample_dt = sample_steps*dt
        rows, states = [], []
        baseline = None
        orientation_baseline = None
        local_baseline = None
        slip_duration = lost_duration = 0.
        slip_event = lost_event = False
        first_slip = first_loss = None
        acquired = False
        stable_time = 0.
        stop_reason = "time_limit"
        open_start, end = timing.open_start, timing.end
        if mode == "reach_only":
            end = open_start
        for step in range(round(end/dt)):
            t = self.d.time
            target, yaw, aperture, phase = skill_target(t, point, yaw_offset, mode, opening_angle_deg,
                                                       self.opening_direction, timing, approach_distance)
            if not self.command(target, yaw, aperture):
                stop_reason = "arm_workspace_limit"
                break
            mujoco.mj_step(self.m, self.d)
            if observer is not None and self.d.time >= next_observation-1e-9:
                observer(self, False)
                next_observation += observation_interval
            if step % sample_steps:
                continue
            mujoco.mj_forward(self.m, self.d)
            force, speed, ratio, hlocal, flocal, rel, unexpected = self.contact_metrics()
            both = bool(np.all(force > 1.))
            if phase == "close":
                stable_time = stable_time+sample_dt if both and max(speed) < .005 else 0.
            drift = migration = 0.
            rotation_drift = 0.
            relative_rotation = self.d.xmat[self.hbody].reshape(3, 3).T@self.d.xmat[self.hand].reshape(3, 3)
            if t >= open_start-.2 and baseline is None and stable_time >= .2:
                baseline, local_baseline = rel.copy(), hlocal.copy()
                orientation_baseline = relative_rotation.copy()
                acquired = True
            if baseline is not None:
                drift = float(np.linalg.norm(rel-baseline))
                rotation_drift = float(np.arccos(np.clip((np.trace(orientation_baseline.T@relative_rotation)-1)/2, -1, 1)))
                if both:
                    migration = float(np.max(np.linalg.norm(hlocal-local_baseline, axis=1)))
                else:
                    migration = float("nan")
                if t >= open_start:
                    slip_duration = slip_duration+sample_dt if (both and (max(speed) > .01 or drift > .008 or rotation_drift > math.radians(5))) else 0.
                    lost_duration = lost_duration+sample_dt if not both else 0.
                    if slip_duration >= .08 and not slip_event:
                        first_slip, slip_event = self.d.time, True
                    if lost_duration >= .1 and not lost_event:
                        first_loss, lost_event = self.d.time, True
            row = dict(time_s=float(self.d.time), phase=phase,
                       door_deg=float(self.opening_direction*np.degrees(self.d.qpos[self.door_q])),
                       door_signed_deg=float(np.degrees(self.d.qpos[self.door_q])),
                       door_speed_rad_s=float(self.d.qvel[self.m.joint("door_hinge").dofadr[0]]),
                       left_normal_N=float(force[0]), right_normal_N=float(force[1]),
                       left_tangent_speed_m_s=float(speed[0]), right_tangent_speed_m_s=float(speed[1]),
                       left_friction_ratio=float(ratio[0]), right_friction_ratio=float(ratio[1]),
                       relative_drift_m=drift, handle_contact_migration_m=migration,
                       relative_rotation_drift_deg=math.degrees(rotation_drift),
                       bilateral_contact=both, slip_detected=slip_event, contact_lost=lost_event,
                       unexpected_contact=";".join(sorted(set(unexpected))))
            for s, side in enumerate(["left", "right"]):
                world_point = self.d.xpos[self.hbody]+self.d.xmat[self.hbody].reshape(3, 3)@hlocal[s]
                for k, axis in enumerate("xyz"):
                    row[f"{side}_contact_world_{axis}_m"] = float(world_point[k])
                    row[f"{side}_contact_handle_{axis}_m"] = float(hlocal[s, k])
                    row[f"{side}_contact_finger_{axis}_m"] = float(flocal[s, k])
            rows.append(row)
            states.append(self.d.qpos.copy())
            if unexpected:
                stop_reason = ("door_obstructed" if any("door_panel" in pair and "pillar_" in pair
                                                       for pair in unexpected) else "unexpected_collision")
                break
            if (slip_event or lost_event) and not diagnostic_continue:
                stop_reason = "slip_or_contact_loss"
                break
            if viewer is not None:
                viewer.sync()
                time.sleep(sample_dt)
        max_angle = max((r["door_deg"] for r in rows), default=0.)
        final = rows[-1] if rows else {}
        hold_rows = [r for r in rows if r["time_s"] >= end-1]
        held = len(hold_rows) >= round(1/sample_dt)-1 and all(abs(r["door_deg"]-opening_angle_deg) <= 2 and r["bilateral_contact"] and abs(r["door_speed_rad_s"]) < .05 for r in hold_rows)
        success = (held and acquired and abs(final.get("door_deg", 0)-opening_angle_deg) <= 2 and
                   final.get("bilateral_contact", False) and not lost_event and not slip_event and
                   not any(r["unexpected_contact"] for r in rows))
        outcome = "D_full_open" if success else "C_grasped_but_failed" if acquired else "B_reachable_grasp_failed"
        if success and opening_angle_deg != 90.:
            outcome = "D_target_open"
        if mode == "reach_only" and acquired:
            outcome = "B_grasp_acquired_opening_untested"
        if success:
            stop_reason = "full_open_and_hold" if opening_angle_deg == 90. else "target_open_and_hold"
        elif slip_event or lost_event:
            stop_reason = "slip_or_contact_loss"
        elif not acquired and stop_reason == "time_limit":
            stop_reason = "grasp_not_acquired"
        result = dict(outcome=outcome, grasp_acquired=acquired, stop_reason=stop_reason,
                      target_angle_deg=opening_angle_deg,
                      opening_direction=self.opening_direction, operating_side=self.operating_side,
                      final_signed_angle_deg=round(final.get("door_signed_deg", 0.), 3),
                      first_unexpected_contact=next((r["unexpected_contact"] for r in rows if r["unexpected_contact"]), None),
                      diagnostic_continue_after_slip=diagnostic_continue, held_open_one_second=held,
                      max_angle_deg=round(max_angle, 3), final_angle_deg=round(final.get("door_deg", 0), 3),
                      slip_detected=slip_event, first_slip_s=first_slip, contact_lost=lost_event,
                      first_contact_loss_s=first_loss,
                      unexpected_contact_samples=sum(bool(r["unexpected_contact"]) for r in rows),
                      peak_relative_drift_mm=round(1000*max((r["relative_drift_m"] for r in rows if r["time_s"] >= open_start), default=0.), 3))
        result.update(plan_metadata)
        if observer is not None:
            observer(self, True)
        return result, rows, states


def save_trial(name, result, rows, states, output):
    output.mkdir(parents=True, exist_ok=True)
    (output/f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if rows:
        with (output/f"{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        np.savez_compressed(output/f"{name}.npz", qpos=np.asarray(states), time=np.array([r["time_s"] for r in rows]))
    else:
        # A newly unreachable attempt must not leave a previous run's trajectory.
        for suffix in ('.csv', '.npz'):
            (output/f'{name}{suffix}').unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=["nominal", "low_friction", "unreachable", "workspace_limit", "reach_only", "no_grip"], default="nominal")
    p.add_argument("--observation", type=Path, default=ROOT/"results"/"observation.npz")
    p.add_argument("--oracle", action="store_true", help="Explicitly bypass vision using known nominal geometry")
    p.add_argument("--output", type=Path, default=ROOT/"results")
    p.add_argument("--view", action="store_true")
    p.add_argument("--continue-after-slip", action="store_true", help="Diagnostic only: keep moving to observe eventual contact loss")
    p.add_argument("--plan", type=Path, help="Load a saved execution plan; coordinates are not re-detected")
    p.add_argument("--base", type=float, nargs=2, metavar=("X", "Y"), help="Override case base position in metres")
    p.add_argument("--base-yaw-deg", type=float, default=90.)
    p.add_argument("--friction", type=float, help="Override all geom sliding friction coefficients")
    p.add_argument("--hinge-resistance", type=float, default=1.2, help="Hinge friction loss in Nm")
    p.add_argument("--grasp-height-offset", type=float, help="Metres above/below the observed handle centre")
    p.add_argument("--grasp-angle-deg", type=float, help="Local approach angle offset in degrees")
    p.add_argument("--ik-branch", type=int, choices=[-1, 1], help="Keep this elbow branch throughout execution")
    p.add_argument("--approach-distance", type=float, help="Pregrasp distance in metres")
    p.add_argument("--grip-force", type=float, help="Per-finger actuator force cap in N, (0, 35]")
    p.add_argument("--opening-angle", type=float, help="Target door angle in degrees, (2, 90]")
    for flag, description in [('settle-duration', 'Initial settling'), ('approach-duration', 'Approach'),
                              ('close-duration', 'Closing stage including contact settling'),
                              ('close-motion-duration', 'Finger closing motion'), ('open-duration', 'Door opening'),
                              ('hold-duration', 'Final hold (at least 1 second)')]:
        p.add_argument('--'+flag, type=float, help=description+' duration in seconds')
    a = p.parse_args()
    plan_fields = ('grasp_height_offset', 'grasp_angle_deg', 'ik_branch', 'approach_distance',
                   'grip_force', 'opening_angle', 'settle_duration', 'approach_duration',
                   'close_duration', 'close_motion_duration', 'open_duration', 'hold_duration')
    if a.plan and (a.oracle or any(getattr(a, name) is not None for name in plan_fields)):
        p.error('--plan cannot be combined with --oracle or plan parameter overrides; edit the plan JSON instead.')
    base = (2., -1.2) if a.case == "unreachable" else (.3, -.9) if a.case == "workspace_limit" else (.6, -1.)
    base = tuple(a.base) if a.base is not None else base
    friction = a.friction if a.friction is not None else (.005 if a.case == "low_friction" else .8)
    if not all(math.isfinite(v) for v in (*base, a.base_yaw_deg, friction, a.hinge_resistance)) or friction < 0 or a.hinge_resistance < 0:
        p.error('Scene parameters must be finite; friction and hinge resistance must be nonnegative.')
    exp = Experiment(base, friction=friction, base_yaw=math.radians(a.base_yaw_deg),
                     hinge_resistance=a.hinge_resistance)
    a.output = case_directory(a.output, a.case)
    a.output.mkdir(parents=True, exist_ok=True)
    try:
        if a.plan:
            execution_plan = GraspPlan.load(a.plan)
            source = 'saved execution plan; fixed initial-world coordinates, no new perception'
        else:
            height, angle = 0., 0.
            if a.oracle:
                center = (.8, -.11, .85)
                source = 'known nominal geometry; NOT vision'
            else:
                from perception import estimate
                observation = estimate(a.observation)
                center = observation['center_world_m']
                source = 'saved RGB-D cylinder fit: '+str(a.observation)
                # Preserve the original fault-comparison grasp unless a local
                # grasp is explicitly requested. This is not a new search.
                if a.grasp_height_offset is None and a.grasp_angle_deg is None:
                    from planning import choose_grasp
                    geometric_plan = choose_grasp(Experiment(), observation)
                    selected = geometric_plan['selected']
                    height = selected['point_world_m'][2] - center[2]
                    angle = selected['local_yaw_offset_deg']
                    (a.output/f'{a.case}_grasp_plan.json').write_text(json.dumps(geometric_plan, indent=2), encoding='utf-8')
            def provided(name, default):
                value = getattr(a, name)
                return default if value is None else value
            timing = SkillTiming(**{key: provided(flag, getattr(DEFAULT_TIMING, key)) for key, flag in [
                ('settle_s', 'settle_duration'), ('approach_s', 'approach_duration'),
                ('close_s', 'close_duration'), ('close_motion_s', 'close_motion_duration'),
                ('open_s', 'open_duration'), ('hold_s', 'hold_duration')]})
            execution_plan = GraspPlan(tuple(center),
                height_offset_m=provided('grasp_height_offset', height),
                approach_angle_deg=provided('grasp_angle_deg', angle),
                ik_branch=provided('ik_branch', -1),
                approach_distance_m=provided('approach_distance', .16),
                grip_force_limit_N=provided('grip_force', 35.),
                opening_angle_deg=provided('opening_angle', 90.), timing=timing)
    except (ValueError, TypeError, OSError) as error:
        p.error(str(error))
    execution_plan.save(a.output/f'{a.case}_execution_plan.json')
    if a.view:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(exp.m, exp.d) as v:
            result, rows, states = exp.trial(plan=execution_plan, mode=a.case, viewer=v, diagnostic_continue=a.continue_after_slip)
    else:
        result, rows, states = exp.trial(plan=execution_plan, mode=a.case, diagnostic_continue=a.continue_after_slip)
    result.update(case=a.case, observation_source=source, mujoco_version=mujoco.__version__,
                  experiment_parameters=dict(base_xy_m=list(base), base_yaw_deg=a.base_yaw_deg,
                      sliding_friction=friction, hinge_resistance_Nm=a.hinge_resistance,
                      opening_direction=exp.opening_direction, operating_side=exp.operating_side),
                  initialization='reset to pregrasp; no motion planning from arbitrary initial joints')
    save_trial(a.case, result, rows, states, a.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
