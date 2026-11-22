# Figures in the polished final report

| Figure | Asset | Purpose and source |
| --- | --- | --- |
| 1 | `failures/report_joint_limit.pdf` | Fixed-base diagnostic: the saved stopping configuration, measured door angle, and remaining elbow travel. Built from preserved qpos and telemetry. |
| 2 | `methods/report_system.pdf` | Shared H-JEPA encoder, I/P/G decisions, G-to-P-to-I targets, and the two training stages. |
| 3 | `methods/report_active_motion.pdf` | Selected-policy replay from test cohort C; viewpoint A/B markers link the path to its initial/reached camera observations. |
| 4 | `results/report_inspection_cost.pdf` | Cohort C correctness/time comparison of never, learned, and always inspection. |

The figures use vector text and plots. Camera observations are embedded unchanged. Figure 1 includes a newly rendered view of the recorded stopping configuration.

`../scripts/build_report_figures.py` regenerates the figures from saved run records. `../scripts/render_diagnostic_frame.py` renders the fixed-base pose with MuJoCo. Figure source paths and hashes are recorded in `SOURCES.json`; result definitions and implementation pointers are in `../EVIDENCE.md`.

Original PNG assets and optional figure blocks remain available for later editing. The five previously anchored objects retain their placement after the corresponding introductions.
