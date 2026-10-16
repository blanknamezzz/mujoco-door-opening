"""Reproduce the physical trials and independent causal/numerical checks."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from perception import estimate
from planning import choose_grasp
from simulate import Experiment, save_trial, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--observation", type=Path, default=ROOT/"results"/"observation.npz")
    p.add_argument("--output", type=Path, default=ROOT/"results")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    obs = estimate(a.observation, write_preview=True)
    plan = choose_grasp(Experiment(), obs)
    (a.output/"grasp_plan.json").write_text(json.dumps(plan, indent=2))
    point = np.array(plan["selected"]["point_world_m"])
    offset = math.radians(plan["selected"]["yaw_offset_deg"])
    all_results = []
    for name in ["nominal", "reach_only", "low_friction", "workspace_limit", "unreachable", "no_grip"]:
        base = (2., -1.2) if name == "unreachable" else (.3, -.9) if name == "workspace_limit" else (.6, -1.)
        e = Experiment(base, friction=.005 if name == "low_friction" else .8)
        result, rows, states = e.trial(point, offset, mode=name, diagnostic_continue=name == "low_friction")
        active = [r for r in rows if r["phase"] in ["open", "hold"]]
        result.update(case=name, base_xy_m=list(base), point_world_m=point.tolist(),
                      yaw_offset_deg=math.degrees(offset), observation_source="saved simulated RGB-D observation",
                      sliding_friction=.005 if name == "low_friction" else .8,
                      peak_tangent_speed_mm_s=1000*max((max(r["left_tangent_speed_m_s"],r["right_tangent_speed_m_s"]) for r in active), default=0),
                      peak_contact_centroid_migration_mm=1000*max((r["handle_contact_migration_m"] for r in active if np.isfinite(r["handle_contact_migration_m"])), default=0))
        save_trial(name, result, rows, states, a.output)
        all_results.append(result)
    checks = {}
    by_name = {r["case"]: r for r in all_results}
    checks["nominal_full_open_and_hold"] = by_name["nominal"]["outcome"] == "D_full_open"
    checks["reach_then_grasp"] = by_name["reach_only"]["grasp_acquired"]
    checks["slip_precedes_contact_loss"] = 0 < by_name["low_friction"]["first_slip_s"] < by_name["low_friction"]["first_contact_loss_s"]
    checks["fixed_base_can_limit_opening"] = by_name["workspace_limit"]["stop_reason"] == "arm_workspace_limit"
    checks["initial_unreachable"] = by_name["unreachable"]["outcome"] == "A_initial_unreachable"
    checks["without_grip_door_does_not_open"] = abs(by_name["no_grip"]["max_angle_deg"]) < .1
    e = Experiment()
    doorjid = e.m.joint("door_hinge").id
    checks["no_door_actuator_or_weld"] = e.m.neq == 0 and not np.any(e.m.actuator_trnid[:, 0] == doorjid)
    convergence = []
    for dt in [.0005, .002]:
        e = Experiment()
        e.m.opt.timestep = dt
        result, _, _ = e.trial(point, offset, diagnostic_continue=True)
        convergence.append({"timestep_s": dt, "final_angle_deg": result["final_angle_deg"]})
    checks["timestep_final_angle_within_0_5_deg"] = all(abs(x["final_angle_deg"]-by_name["nominal"]["final_angle_deg"]) < .5 for x in convergence)
    # Ground truth is used here only to evaluate the independent RGB-D estimate.
    truth = Experiment()
    error = float(np.linalg.norm(np.array(obs["center_world_m"])-truth.d.xpos[truth.hbody]))
    summary = dict(cases=all_results, validation=checks, timestep_sensitivity=convergence,
                   perception_center_error_mm=1000*error,
                   contact_thresholds={"normal_force_each_N": 1, "grasp_stable_s": .2,
                       "slip_speed_m_s": .01, "relative_drift_m": .008,
                       "relative_rotation_drift_deg": 5,
                       "slip_persistence_s": .08, "contact_loss_persistence_s": .1,
                       "full_open_min_deg": 88, "hold_s": 1},
                   scope="One deterministic known scene; simplified arm; no trained perception or real-robot validation")
    (a.output/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not all(checks.values()):
        raise SystemExit("Validation failed; inspect summary.json")


if __name__ == "__main__":
    main()
