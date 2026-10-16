"""Explain the first rejected command in the fixed-base workspace trial.

The final logged physical pose and next infeasible reference are different:
the controller terminates before commanding a target outside the joint range.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from simulate import Experiment, ROOT, rot, smooth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"results")
    args = parser.parse_args()
    folder = args.results
    result = json.loads((folder/"workspace_limit.json").read_text())
    if "point_world_m" in result:
        point = np.array(result["point_world_m"])
        offset = math.radians(result["yaw_offset_deg"])
    else:
        plan = json.loads((folder/"workspace_limit_grasp_plan.json").read_text())
        point = np.array(plan["selected"]["point_world_m"])
        offset = math.radians(plan["selected"]["yaw_offset_deg"])
    e = Experiment((.3, -.9))
    states = np.load(folder/"workspace_limit.npz")
    rows = list(csv.DictReader((folder/"workspace_limit.csv").open()))
    last = rows[-1]
    limits = e.m.jnt_range[e.arm_j]
    report = dict(actual_stop_time_s=float(states["time"][-1]),
                  actual_door_angle_deg=float(last["door_deg"]),
                  actual_arm_joint_deg=dict(zip(["shoulder", "elbow", "wrist_yaw"], np.degrees(states["qpos"][-1, e.arm_q]).tolist())),
                  normal_force_N=[float(last["left_normal_N"]), float(last["right_normal_N"])],
                  relative_drift_mm=1000*float(last["relative_drift_m"]),
                  slip_detected=last["slip_detected"] == "True",
                  contact_lost=last["contact_lost"] == "True",
                  interpretation="Controller terminates at first infeasible reference; this is not a measured physical stall at the exact joint limit.")
    for step in range(round(3.5/e.m.opt.timestep), round(13.5/e.m.opt.timestep)+1):
        t = step*e.m.opt.timestep
        theta = math.pi/2*smooth((t-3.5)/10)
        target = rot(theta)@point
        yaw = math.pi/2+offset-theta
        if e.ik(target, yaw) is not None:
            continue
        wrist = target[:2]-.18*np.array([math.cos(yaw), math.sin(yaw)])
        delta = wrist-e.base
        cosine = (delta@delta-.6**2-.55**2)/(2*.6*.55)
        if abs(cosine) > 1:
            raise RuntimeError("Failure changed to link-length reachability; inspect the modified scene.")
        elbow = -math.acos(cosine)
        shoulder_world = math.atan2(delta[1], delta[0])-math.atan2(.55*math.sin(elbow), .6+.55*math.cos(elbow))
        q = np.array([shoulder_world-math.pi/2, elbow, yaw-shoulder_world-elbow])
        q = (q+math.pi)%(2*math.pi)-math.pi
        violations = [name for name, v, bounds in zip(["shoulder", "elbow", "wrist_yaw"], q, limits) if v < bounds[0] or v > bounds[1]]
        min_distance = math.sqrt(.6**2+.55**2+2*.6*.55*math.cos(abs(limits[1, 0])))
        report.update(first_rejected_reference_time_s=t,
                      reference_door_angle_deg=math.degrees(theta),
                      reference_arm_joint_deg=dict(zip(["shoulder", "elbow", "wrist_yaw"], np.degrees(q).tolist())),
                      joint_limits_deg=np.degrees(limits).tolist(), violating_joints=violations,
                      required_shoulder_wrist_distance_m=float(np.linalg.norm(delta)),
                      minimum_shoulder_wrist_distance_at_elbow_limit_m=min_distance)
        break
    else:
        raise RuntimeError("No infeasible command found; the saved scene/trial may have changed.")
    (folder/"workspace_limit_diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
