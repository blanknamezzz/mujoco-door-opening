"""Nine local grasp candidates; sampled IK and known-scene collision screen.

This is a geometric ranking baseline, not a force-closure proof or learned
world model. Only a subsequent physical rollout can validate a candidate.
"""
import math
import mujoco
import numpy as np
from simulate import rot


def choose_grasp(exp, estimate):
    center = np.array(estimate["center_world_m"])
    low, high = estimate["visible_z_bounds_m"]
    probe = mujoco.MjData(exp.m)
    candidates = []
    for dz in [-.07, 0., .07]:
        for offset in [-15., 0., 15.]:
            point = center+np.array([0, 0, dz])
            yaw0 = math.pi/2+math.radians(offset)
            clearance = min(point[2]-low, high-point[2])-.025-.015
            last_angle = -1.
            reason = "end_clearance" if clearance < .02 else "full_geometric_path"
            previous = None
            min_margin = float("inf")
            if clearance >= .02:
                path = [(0., point-s*np.array([math.cos(yaw0), math.sin(yaw0), 0]), yaw0, .04)
                        for s in np.linspace(.16, 0, 9)]
                path += [(deg, rot(math.radians(deg))@point, yaw0-math.radians(deg), .013)
                         for deg in np.arange(0, 91, 2)]
                for angle, target, yaw, aperture in path:
                    q = exp.ik(target, yaw)
                    if q is None:
                        reason = "IK_or_joint_limit"
                        break
                    if previous is not None and np.max(np.abs(q-previous)) > math.radians(20):
                        reason = "joint_discontinuity"
                        break
                    previous = q
                    limits = exp.m.jnt_range[exp.arm_j]
                    min_margin = min(min_margin, float(np.min(np.minimum(q-limits[:, 0], limits[:, 1]-q))))
                    probe.qpos[exp.arm_q] = q
                    probe.qpos[exp.lift_q] = point[2]-.85
                    probe.qpos[exp.door_q] = math.radians(angle)
                    probe.qpos[exp.finger_q] = aperture
                    mujoco.mj_forward(exp.m, probe)
                    forbidden = []
                    for c in probe.contact:
                        if {int(c.geom1), int(c.geom2)} in [{exp.bar, pad} for pad in exp.pads]:
                            continue
                        if c.dist < -.0002:
                            forbidden.append(f"{exp.m.geom(c.geom1).name}:{exp.m.geom(c.geom2).name}")
                    if forbidden:
                        reason = "collision:"+forbidden[0]
                        break
                    last_angle = float(angle)
            # Prioritise complete arc, then central height and neutral approach.
            score = last_angle/90-.2*abs(dz)/.07-.2*abs(offset)/15
            if np.isfinite(min_margin):
                score += .02*min_margin
            candidates.append(dict(point_world_m=point.tolist(), yaw_offset_deg=offset,
                                   end_clearance_m=float(clearance), geometric_open_deg=last_angle,
                                   minimum_joint_margin_rad=float(min_margin) if np.isfinite(min_margin) else None,
                                   reason=reason, score=float(score)))
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return {"selected": candidates[0], "candidates": candidates,
            "method": "9 candidates; known scene; 2-degree opening samples; fixed elbow branch",
            "warning": "Geometric screening is not a guarantee of successful contact dynamics."}
