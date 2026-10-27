"""Separate a feasible grasp witness, a bounded workspace exclusion, and UNKNOWN.

The certificate concerns a fixed base and continuous pad/handle contact in the
known planar MuJoCo model. Finite candidate or PRM failure never proves absence.
No learned policy or training split is changed by this independent diagnostic.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from itertools import product
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from grasp_plan import GraspPlan, SkillTiming
from prm import PRMSettings
from simulate import Experiment, ROOT, rot, save_trial


@dataclass(frozen=True)
class SearchSettings:
    angles_deg: tuple = tuple(range(-20, 21, 2)) + (-30, 30, -45, 45, -60, 60, -90, 90)
    heights_m: tuple = (0., -.07, .07)
    branches: tuple = (-1, 1)
    forces_N: tuple = (35., 20.)
    open_times_s: tuple = (10., 14.)
    max_rollouts: int = 32
    max_candidates: int = 800
    arc_step_deg: float = 2.

    def __post_init__(self):
        if self.max_rollouts < 0 or self.max_candidates < 1:
            raise ValueError('Invalid search budget')
        if not np.isfinite(self.arc_step_deg) or not 0 < self.arc_step_deg <= 90:
            raise ValueError('Invalid arc-check step')
        for values in (self.angles_deg, self.heights_m, self.forces_N, self.open_times_s):
            if not values or not np.all(np.isfinite(values)):
                raise ValueError('Search ranges must contain finite values')
        if not self.branches or any(b not in (-1, 1) for b in self.branches):
            raise ValueError('IK branches must be -1 or +1')
        if any(not 0 < f <= 35 for f in self.forces_N) or any(t <= 0 for t in self.open_times_s):
            raise ValueError('Invalid force or opening time')


def workspace_certificate(exp, target_deg=90., success_tolerance_deg=2., uncertainty_m=.005):
    """A one-sided outer-reach proof, allowing ALL arm and wrist orientations.

    TCP reach <= sum of link lengths. Any pad contact lies within a conservative
    pad-offset/corner radius of TCP; the vertical bar adds its capsule radius.
    If its axis must leave that envelope before the MINIMUM successful angle,
    continuous pad contact is impossible. No IK grid or collision sampler is
    used as a negative certificate. Passing this bound says only NOT EXCLUDED.
    """
    if not all(np.isfinite(x) for x in (target_deg, success_tolerance_deg, uncertainty_m)):
        raise ValueError('Certificate parameters must be finite')
    if not 0 <= success_tolerance_deg < target_deg <= 90 or uncertainty_m < 0:
        raise ValueError('Invalid target, tolerance or uncertainty')
    scope = ('Known fixed-base planar 3R + lift model within its joint limits, vertical handle and '
             'continuous finger-pad/handle contact. Moving the base, pushing '
             'the panel, letting the door coast, regrasping and model changes '
             'are outside this certificate. Passing does not prove feasibility.')
    common = dict(certified_exclusion=False, status='not_excluded', scope=scope)
    m = exp.m
    probe = mujoco.MjData(m)
    mujoco.mj_forward(m, probe)
    hinge = m.joint('door_hinge').id
    # Validate the assumptions needed by the analytical bound. Fail open.
    axes = [m.jnt_axis[j] for j in exp.arm_j]
    valid = (all(np.allclose(a, [0, 0, 1]) for a in axes)
             and all(np.allclose(probe.xaxis[j], [0, 0, 1]) for j in exp.arm_j)
             and all(m.jnt_type[m.joint(n).id] == mujoco.mjtJoint.mjJNT_SLIDE
                     and m.body(nm).parentid == exp.hand
                     for n, nm in (("left_slide", "left_finger"), ("right_slide", "right_finger")))
             and np.allclose(m.jnt_axis[m.joint('lift_z').id], [0, 0, 1])
             and np.allclose(probe.xaxis[m.joint('lift_z').id], [0, 0, 1])
             and all(np.allclose(m.jnt_pos[j], [0, 0, 0]) for j in exp.arm_j)
             and all(m.body(n).dofnum == 1 for n in ('lift', 'upper', 'forearm', 'hand', 'door'))
             and m.body('base').dofnum == 0
             and m.body('door').parentid == 0
             and m.body_dofnum[exp.hbody] == 0
             and m.body_parentid[exp.hbody] == m.body('door').id
             and m.geom_bodyid[exp.bar] == exp.hbody
             and np.allclose(probe.xaxis[hinge], [0, 0, -1])
             and np.allclose(probe.xanchor[hinge, :2], [0, 0])
             and np.allclose(m.body('lift').pos[:2], [0, 0])
             and np.allclose(m.body('upper').pos[:2], [0, 0])
             and m.body('base').parentid == 0
             and m.body('forearm').parentid == m.body('upper').id
             and m.body('hand').parentid == m.body('forearm').id
             and m.site_bodyid[exp.site] == exp.hand
             and m.geom_type[exp.bar] == mujoco.mjtGeom.mjGEOM_CAPSULE
             and np.allclose(np.abs(probe.geom_xmat[exp.bar].reshape(3, 3)[:, 2]), [0, 0, 1]))
    if not valid:
        return dict(common, status='unsupported_geometry', reason='Analytical model assumptions not met')
    links = [float(np.linalg.norm(m.body(n).pos[:2])) for n in ('forearm', 'hand')]
    links.append(float(np.linalg.norm(m.site_pos[exp.site, :2])))
    tcp_reach = sum(links)
    pad_offset = 0.
    for endpoint in (0, 1):
        for name in ('left_slide', 'right_slide'):
            j = m.joint(name).id
            probe.qpos[m.jnt_qposadr[j]] = m.jnt_range[j, endpoint]
        mujoco.mj_forward(m, probe)
        for gid in exp.pads:
            if m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
                return dict(common, status='unsupported_geometry', reason='Expected box finger pads')
            # Full 3D corner radius is deliberately conservative in projection.
            pad_offset = max(pad_offset, float(np.linalg.norm(
                probe.geom_xpos[gid, :2]-probe.site_xpos[exp.site, :2]) + np.linalg.norm(m.geom_size[gid])))
    bar_radius = float(m.geom_size[exp.bar, 0])
    bound = tcp_reach + pad_offset + bar_radius + uncertainty_m
    handle = probe.geom_xpos[exp.bar].copy()
    base = np.asarray(exp.base)
    needed = math.radians(target_deg-success_tolerance_deg)
    phi = math.atan2(handle[1], handle[0])
    beta = math.atan2(base[1], base[0])
    candidates = [0., needed]
    for k in range(-4, 5):
        theta = exp.opening_direction*(phi-beta-k*math.pi)
        if 0 <= theta <= needed:
            candidates.append(theta)
    # These are the exact distance extrema on the required hinge arc.
    distances = [float(np.linalg.norm((rot(exp.opening_direction*t)@handle)[:2]-base)) for t in candidates]
    i = int(np.argmax(distances))
    excluded = distances[i] > bound+1e-9
    return dict(common, certified_exclusion=excluded,
        status='workspace_exclusion' if excluded else 'not_excluded',
        link_lengths_m=links, tcp_max_radius_m=tcp_reach,
        pad_offset_bound_m=pad_offset, bar_radius_m=bar_radius, uncertainty_m=uncertainty_m,
        handle_axis_radius_bound_m=bound, minimum_success_angle_deg=target_deg-success_tolerance_deg,
        witness_door_angle_deg=math.degrees(candidates[i]),
        required_axis_distance_m=distances[i], excess_m=distances[i]-bound,
        reason=('Handle axis lies outside even an orientation-independent contact envelope '
                'at a mandatory door angle' if excluded else 'Outer-reach necessary condition passes only'))


def candidate_plans(baseline, settings):
    """Explore angles first, then heights/branches, then force and timing."""
    priority = [a for a in (8., -8., 18., -18.) if a in settings.angles_deg]
    angles = list(dict.fromkeys([baseline.approach_angle_deg, *priority, *settings.angles_deg]))
    seen = set()
    def unique(plan):
        key = json.dumps(plan.to_dict(), sort_keys=True)
        if key in seen:
            return False
        seen.add(key)
        return True
    yield baseline
    seen.add(json.dumps(baseline.to_dict(), sort_keys=True))
    for a in angles:
        p = replace(baseline, approach_angle_deg=a)
        if unique(p):
            yield p
    for h, b, a in product(settings.heights_m, settings.branches, angles):
        p = replace(baseline, height_offset_m=h, ik_branch=b, approach_angle_deg=a)
        if unique(p):
            yield p
    for h, b, a, f, t in product(settings.heights_m, settings.branches, angles,
                                settings.forces_N, settings.open_times_s):
        p = replace(baseline, height_offset_m=h, ik_branch=b, approach_angle_deg=a,
                    grip_force_limit_N=f, timing=replace(baseline.timing, open_s=t))
        if unique(p):
            yield p


def arc_ik_screen(exp, plan, step_deg=2.):
    """Reject this fixed-reference plan at a concrete failed IK waypoint only."""
    count = max(1, math.ceil(plan.opening_angle_deg/step_deg))
    point = np.asarray(plan.point_world_m)
    yaw0 = math.pi/2 + plan.yaw_offset_rad(exp)
    for deg in np.linspace(0., plan.opening_angle_deg, count+1):
        theta = exp.opening_direction*math.radians(float(deg))
        if exp.ik(rot(theta)@point, yaw0-theta, branch=plan.ik_branch) is None:
            return dict(passed=False, reason='reference_ik_failure', first_failed_reference_deg=float(deg))
    return dict(passed=True, reason='sampled_ik_only_not_a_collision_or_contact_guarantee')


def changed_parameters(a, b):
    av, bv = a.to_dict(), b.to_dict()
    return [k for k in av if av[k] != bv[k]]


def diagnose_station(base, baseline=None, base_yaw_deg=90., friction=.8,
                     hinge_resistance=1.2, settings=None, prm_settings=None,
                     output=None, progress=None):
    from train_grasp_rl import rollout
    if len(base) != 2 or not np.all(np.isfinite(base)) or not np.isfinite(base_yaw_deg):
        raise ValueError('Expected a finite base pose')
    if not np.isfinite(friction) or friction < 0 or not np.isfinite(hinge_resistance) or hinge_resistance < 0:
        raise ValueError('Invalid physical parameters')
    settings = settings or SearchSettings()
    prm_settings = prm_settings or PRMSettings()
    if baseline is None:
        from perception import estimate
        baseline = GraspPlan(tuple(estimate(ROOT/'results/observation.npz')['center_world_m']))
    if baseline.opening_angle_deg != 90.:
        raise ValueError('Station diagnosis currently requires a 90-degree full-opening plan')
    exp = Experiment(base=base, base_yaw=math.radians(base_yaw_deg), friction=friction,
                     hinge_resistance=hinge_resistance)
    cert = workspace_certificate(exp, target_deg=baseline.opening_angle_deg)
    report = dict(schema_version=1, status='unconfirmed_after_search',
        base_xy_m=list(map(float, base)), base_yaw_deg=float(base_yaw_deg),
        certificate=cert, search_settings=asdict(settings),
        physics=dict(friction=friction, hinge_resistance=hinge_resistance, prm=prm_settings.to_dict()),
        baseline_plan=baseline.to_dict(), baseline_result=None, witness_plan=None,
        attempts=[], geometry_rejections=[], proposed_candidates=0, physical_rollouts=0,
        scope='Finite fixed-base grasp-plan search. No failure count is a global infeasibility proof.',
        mujoco_version=mujoco.__version__)
    def finish():
        report['physical_rollouts'] = len(report['attempts'])
        if output is not None:
            folder = Path(output);folder.mkdir(parents=True, exist_ok=True)
            (folder/'diagnosis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        return report
    if cert['certified_exclusion']:
        report.update(status='excluded_by_workspace_bound', stop_reason='analytical_necessary_condition_violated')
        return finish()
    for index, plan in enumerate(candidate_plans(baseline, settings)):
        if index >= settings.max_candidates:
            report['stop_reason'] = 'candidate_budget_exhausted'
            break
        if len(report['attempts']) >= settings.max_rollouts:
            report['stop_reason'] = 'rollout_budget_exhausted'
            break
        report['proposed_candidates'] += 1
        # Always execute the requested baseline to preserve its actual failure.
        if index > 0:
            screen = arc_ik_screen(exp, plan, settings.arc_step_deg)
            if not screen['passed']:
                report['geometry_rejections'].append(dict(plan=plan.to_dict(), **screen))
                continue
        config = dict(base_yaw_rad=math.radians(base_yaw_deg), friction=friction,
            hinge_resistance=hinge_resistance, prm=prm_settings.to_dict(), plan=plan.to_dict())
        trial, rows, states, _ = rollout(config, base, plan.approach_angle_deg)
        report['attempts'].append(dict(plan=plan.to_dict(), **trial))
        if index == 0:
            report['baseline_result'] = trial
        if progress:
            progress(dict(rollout=len(report['attempts']), angle_deg=plan.approach_angle_deg,
                          outcome=trial['result']['outcome'], stop_reason=trial['result']['stop_reason']))
        success = trial['result']['outcome'] == 'D_full_open'
        if output is not None and (index == 0 or success):
            save_trial('baseline' if index == 0 else 'witness', trial['result'], rows, states, Path(output))
        if success:
            report.update(status='feasible_baseline' if index == 0 else 'feasible_with_alternative_grasp',
                stop_reason='physically_validated_full_open_and_hold', witness_plan=plan.to_dict(),
                changed_parameters=changed_parameters(baseline, plan))
            return finish()
    else:
        report['stop_reason'] = 'finite_candidate_set_exhausted'
    return finish()


def run_examples(output):
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    cases = [('angle_rescue', (.8, -1.), SearchSettings(max_rollouts=8)),
             ('workspace_excluded', (1.4, -.4), SearchSettings()),
             ('partial_unknown', (.3, -.9), SearchSettings(max_rollouts=1))]
    results = []
    for name, base, settings in cases:
        report = diagnose_station(base, settings=settings, output=output/name,
                                  progress=lambda r: print(json.dumps(r), flush=True))
        results.append(dict(name=name, base_xy_m=base, status=report['status'],
            baseline_stop=(report['baseline_result'] or {}).get('result', {}).get('stop_reason'),
            physical_rollouts=report['physical_rollouts'], witness_plan=report['witness_plan'],
            certificate=report['certificate']))
        print(name, report['status'], flush=True)
    (output/'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('diagnose')
    p.add_argument('--base', type=float, nargs=2, required=True)
    p.add_argument('--yaw-deg', type=float, default=90.)
    p.add_argument('--baseline-plan', type=Path)
    p.add_argument('--baseline-angle', type=float)
    p.add_argument('--max-rollouts', type=int, default=32)
    p.add_argument('--max-candidates', type=int, default=800)
    p.add_argument('--output', type=Path)
    p = commands.add_parser('examples')
    p.add_argument('--output', type=Path, default=ROOT/'reproduced/station_feasibility_examples')
    p = commands.add_parser('replay')
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--which', choices=['baseline', 'witness'], default='witness')
    args = parser.parse_args()
    if args.command == 'examples':
        run_examples(args.output)
        return
    if args.command == 'replay':
        from train_grasp_rl import rollout
        report = json.loads(args.report.read_text())
        plan = report['baseline_plan'] if args.which == 'baseline' else report['witness_plan']
        if plan is None:
            parser.error('This report has no successful witness. UNKNOWN is not success.')
        config = dict(report['physics'], base_yaw_rad=math.radians(report['base_yaw_deg']), plan=plan)
        trial, _, _, _ = rollout(config, report['base_xy_m'], plan['approach_angle_deg'], viewer=True)
        print(json.dumps(trial, indent=2))
        return
    baseline = GraspPlan.load(args.baseline_plan) if args.baseline_plan else None
    if args.baseline_angle is not None:
        if baseline is None:
            from perception import estimate
            baseline = GraspPlan(tuple(estimate(ROOT/'results/observation.npz')['center_world_m']))
        baseline = replace(baseline, approach_angle_deg=args.baseline_angle)
    output = args.output or ROOT/'reproduced/station_feasibility'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    report = diagnose_station(args.base, baseline, args.yaw_deg,
        settings=SearchSettings(max_rollouts=args.max_rollouts, max_candidates=args.max_candidates),
        output=output, progress=lambda r: print(json.dumps(r), flush=True))
    print(json.dumps(dict(status=report['status'], output=str(output),
        stop_reason=report['stop_reason'], physical_rollouts=report['physical_rollouts']), indent=2))


if __name__ == '__main__':
    main()
