# Reference results organized by case

The first six cases are the original reference experiments; reorganizing their directories did not change their results. New angle-diversity experiments store small summaries in separate cases. Full training data and models remain in ignored `reproduced/`, with reports in `docs/`.

| Folder | Case | Result |
|---|---|---|
| [nominal/](nominal/) | Complete opening and hold | [nominal.json](nominal/nominal.json) |
| [reach_only/](reach_only/) | Reach and grasp; opening not tested | [reach_only.json](reach_only/reach_only.json) |
| [low_friction/](low_friction/) | Low friction causes slip and contact loss | [low_friction.json](low_friction/low_friction.json) |
| [workspace_limit/](workspace_limit/) | Open to about 42.5 degrees before the next target exceeds joint range | [workspace_limit.json](workspace_limit/workspace_limit.json) |
| [unreachable/](unreachable/) | Initial station cannot reach the handle | [unreachable.json](unreachable/unreachable.json) |
| [no_grip/](no_grip/) | No grasp is acquired, so the door does not move | [no_grip.json](no_grip/no_grip.json) |

New experiment: [angle_diversity/](angle_diversity/README.md), containing `train_000`, `validation_000`, and `test_000` station cases with summaries of the 29-angle outcomes and source hashes.

Shared files remain in this directory:

- [summary.json](summary.json): six-case results, validation checks, criteria, and timestep sensitivity.
- [grasp_plan.json](grasp_plan.json): shared grasp-point selection for the six experiments.
- [observation.npz](observation.npz) / [observation.png](observation.png): initial RGB-D observation.
- [handle_estimate.json](handle_estimate.json) / [handle_detection.png](handle_detection.png): handle-detection results.

File types: `.json` stores results or parameters, `.csv` contact/motion logs, `.npz` trajectories, `.gif` animations, and `.png` still images.

Scripts accept `--output results` (`--results results` for diagnostics) and locate the relevant case subdirectory. A case directory may also be passed directly. Previous flat `reproduced/` outputs remain compatible, and `run_demo.py` retains its default output behavior.
