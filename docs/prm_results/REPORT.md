# PRM training integration: verification results

Workflow: initial station/joints -> PRM preflight -> physical preparation execution -> grasp -> opening/hold.

[Interactive roadmap and physical trajectory replay](index.html) · [Implementation, settings, training, reproduction](../prm_pregrasp.md)

## Five measured examples

| Example | Station | PRM outcome | Physical continuation |
| --- | --- | --- | --- |
| A | (0.6, -1.0) | Valid direct path, 2 waypoints, no sampling | Safe opening/hold, 89.676 degrees |
| B | (0.6, -1.0), small obstacle | Direct fails; 715 samples, 215 invalid, 500 valid nodes, 5-waypoint path | Existing lift lowers up to 9.6 cm to detour, restores height, opens to 89.676 degrees |
| C | Same as B, zero sampling budget | Direct fails; no path within budget | Opening skipped; B establishes a feasible route |
| D | (0.3, -0.9) | Pregrasp reached through physical preparation | Workspace limit after 42.505 degrees |
| E | (2.0, -1.2) | No legal pregrasp IK on this elbow branch | Preparation/opening skipped |

B uses Lazy PRM: connect neighboring candidates, find a shortest path, validate its edges, remove failures, and search again. Every final-path edge passes discrete checking; gray candidate edges need not all be verified. Shoulder/elbow projection hides lift/wrist, so crossing projected lines do not establish four-dimensional connectivity.

## Added capability

Previously the robot initialized at pregrasp. Now specify initial joints, plan reachability, and execute position servos with mj_step. The door stays passive during preparation; grasping continues from actual reached state without teleporting.

PRM adds no joints and cannot expand post-grasp reach. It adds motion-path assessment and separates unconfirmed planning, preparation-execution failure, and opening failure.

## Training integration verification

- New split: 12/4/8 training/validation/test stations, 504 candidate procedures.
- Every candidate runs PRM, physical preparation, and opening. All 24 stations have pregrasp paths, without establishing complete opening at all stations.
- Seed-42 RL completes 252 training decisions; SL uses candidates at the same training stations.
- After freezing, independently replay 32 distinct test station-angle combinations; all match collection.
- SL/RL each score 3/8, matching 3 grid-feasible test stations. This verifies integration rather than method ranking or a PRM success-rate gain.
- Independent train_grasp_rl.py completes 8 CLI episodes and save/load execution. Selected +6 degrees opens safely; fixed zero self-collides.
- Thirty-two numerical/geometric/physical/split tests pass before upload, including collision CSV export/counting. Mock DOM/canvas interactions pass without an available browser; no actual-browser visual review is performed.

## Reproduction

```bash
python prm_examples.py
mjpython prm_examples.py --view detour
python prm_examples.py --render detour
```

Geometry nodes, preparation contacts, actual joints, and opening outcomes are in reproduced/prm_examples/. Full data/models are in reproduced/prm_training_data_final/ and reproduced/prm_training_comparison_final/. See implementation notes for initial joints/check precision. Finite PRM failure means no path found in this run.
