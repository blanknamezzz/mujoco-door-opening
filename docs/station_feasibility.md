# Distinguishing an infeasible station from a poor grasp choice

Entry `station_feasibility.py`. Failure at all 21 angles means only that this grid/controller found no success, rather than global station infeasibility.

## Three returned outcomes

| Status | Evidence | Station decision |
|---|---|---|
| `feasible_baseline` / `feasible_with_alternative_grasp` | Physical PRM execution followed by complete MuJoCo opening/hold | Positive witness; retain successful plan |
| `excluded_by_workspace_bound` | Handle axis must leave every arm configuration's conservative contact region before minimum success angle | Exclude only under explicit certificate assumptions |
| `unconfirmed_after_search` | Finite candidates/IK/PRM/physical trials find no success | Remains unknown; expand search or reposition |

With matched base/scene/friction, baseline failure and physically successful alternative establish that the station is feasible and the original plan limiting. `changed_parameters` lists angle, elbow branch, height, force, or timing changes.

## A grasp-independent necessary condition

First compute a bound independent of angle. Maximum horizontal TCP radius is no greater than the sum of three link lengths:

`R_tcp = 0.6 + 0.55 + 0.18 = 1.33 m`.

Allow off-center contact by adding maximum gripper offset across finger travel, the 3D pad-corner radius, handle capsule radius, and default 5 mm margin. Read dimensions/ranges from MuJoCo to construct a conservative handle-axis contact envelope.

Success permits 90 +/-2 degrees, so check mandatory continuous opening from 0 to 88. Compute exact distance extrema between shoulder and handle axis along the known hinge arc, including endpoints/stationary points. If any required angle exceeds the conservative envelope, continuous pad contact cannot pass it regardless of angle, elbow branch, lift, force, or time.

This is a one-sided necessary condition: violation excludes, passing only avoids this exclusion. It misses other joint/obstacle/contact infeasibility and does not convert finite search failure into a certificate.

Assumptions: known fixed-base planar three-rotation-plus-lift model, specified joint ranges, vertical handle, and sustained pad contact. Exclude base motion, panel pushing, inertial coasting after release, regrasp, hardware errors, or model changes. Return `unsupported_geometry` without exclusion when axes/structure violate assumptions.

## Searching after the bound passes

1. Physically execute the baseline and record its stop reason.
2. Try other angles, defaulting to original -20 to +20 degrees plus +/-30, +/-45, +/-60, +/-90.
3. Expand height 0, +/-0.07 m and both elbow branches.
4. Try actuator caps 20/35 N per finger and opening times 10/14 s.
5. Check finite reference-arc IK, then physically execute PRM preparation/grasp/opening. Only actual `D_full_open` is a success witness.

Defaults allow 32 physical episodes and 800 candidate checks. Budget exhaustion or finite-pool failure returns `unconfirmed_after_search`; passing finite IK does not establish collision-free/contact-stable execution.

## Running and reproduction

```bash
# Standard station diagnostic; output defaults to a new timestamped directory.
python station_feasibility.py diagnose --base 0.8 -1.0 --baseline-angle 0

# Geometry bound only, without physical search: exclusion or unknown.
python station_feasibility.py diagnose --base 1.4 -0.4 --max-rollouts 0

# Expand search at the partial-opening failure station; failures remain unknown.
python station_feasibility.py diagnose --base 0.3 -0.9 --max-rollouts 32

# Three fixed explanatory examples.
python station_feasibility.py examples

# macOS live comparison of baseline failure and successful alternative at one station.
mjpython station_feasibility.py replay --report reproduced/station_feasibility_examples/angle_rescue/diagnosis.json --which baseline
mjpython station_feasibility.py replay --report reproduced/station_feasibility_examples/angle_rescue/diagnosis.json --which witness
```

Use `--baseline-plan` for complete GraspPlan JSON and `--yaw-deg` for base yaw. The diagnostic targets 90-degree complete opening and rejects other targets. JSON stores attempts, preparation, execution, exclusions, budgets, success witnesses, and certificates; baseline/success also retain CSV/NPZ.

Call `diagnose_station(base, ...)` from Python. `SearchSettings` changes angles, heights, branches, force, time, and budgets.

## Training and PRM integration

Every collector `*_prm_preflight.jsonl` adds `station_workspace_check`, computed from known geometry before opening labels. Existing `status`, `angles_with_path`, and `queries` still describe pregrasp paths only.

The field does not filter stations or change rewards, features, or existing SL/RL results. Expanded physical diagnosis requires explicit execution and does not secretly use test outcomes to alter training. New manifests include its source hash; preserve historical data and use fresh directories after source changes.

For a future feasibility classifier, physical witnesses are positive labels, scoped geometry certificates are conditional negatives, and unknown remains separate. All-21-angle or finite-PRM failure is not global impossibility, and learned predictions are not proofs.
