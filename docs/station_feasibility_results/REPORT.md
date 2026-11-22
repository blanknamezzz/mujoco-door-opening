# Measured station-feasibility diagnostics

[Implementation, certificate assumptions, reproduction](../station_feasibility.md)

| Station | Physical evidence | Finding |
|---|---|---|
| (0.8, -1.0) | Zero-degree grasp collides; changing only to +8 degrees opens and holds | Station feasible; original angle limits performance |
| (1.4, -0.4) | Required 88-degree handle-axis distance 1.535922 m exceeds conservative contact radius 1.448821 m | Excluded under certificate assumptions, without treating failed trials as proof |
| (0.3, -0.9), 1 episode | Original plan stops at 42.505 degrees on an out-of-range target | Unknown, rather than proven infeasible |
| (0.3, -0.9), expanded search | Check 364 candidates; exclude 332 with no reference-pose IK; 32 physical episodes find no complete opening | Budget exhausted; remains unknown |

The certificate assumes the known fixed base, current joint limits, vertical handle, and sustained fingertip-pad contact. It allows all arm/wrist orientations and includes finger travel, pad size, handle radius, and 5 mm margin. Release/regrasp, pushing the panel, moving the base, or changing the model invalidate these assumptions.

Expanded search includes angles outside the original grid, elbow branches, height, force, and opening time. Finite candidates/budgets do not prove every plan impossible.

Forty numerical/geometric/physical/split regressions pass. A 1/1/1-station, 63-candidate collection check adds `station_workspace_check` in every preflight. The field records geometry without removing stations or changing existing SL/RL labels.

Baseline/success trajectories and diagnostic JSON are in `reproduced/station_feasibility_examples/`. Use `station_feasibility.py replay` to view zero-degree failure versus +8-degree success. Preserve historical 72-station results without labeling all-21-angle failures globally infeasible.

Verification: [verification.json](verification.json).
