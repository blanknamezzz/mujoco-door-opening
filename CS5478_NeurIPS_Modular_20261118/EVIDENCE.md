# Result and implementation map

Reviewed against repository revision `0ae4c4ddc84fd50ab15e7ed73f12878a6584b4a0` and locally saved frozen run records on 2026-11-21.

| Report item | Repository record | Saved run record |
| --- | --- | --- |
| Cohort C primary and retention comparison (Table 3) | `docs/mobile_retention_results.md` | `runs/mobile_partial_workspace_seed42_20261109/runs/p10t_seed42_20261111/report.json` |
| Cohort A and inspection-objective comparison | `docs/mobile_inspection_regret_results.md` | `runs/mobile_inspection_regret_evaluation_seed42_20261110/report.json` |
| Cohort B and same-policy comparison | `docs/mobile_branch_balance_results.md` | `runs/mobile_branch_balance_evaluation_seed42_20261110/report.json` |
| Training/validation/test rates | `docs/mobile_retention_results.md` | `runs/mobile_partial_workspace_seed42_20261109/runs/p10t_seed42_20261111/post_analysis.json` |
| PCGrad | `docs/mobile_branch_pcgrad_results.md` | `runs/mobile_branch_pcgrad_training_seed42_20261111/report.json` |
| Branch balancing and checkpoint tradeoff | `docs/mobile_branch_balance_development.md` | `runs/mobile_branch_balance_training_seed42_20261110/report.json` |
| Independent 21-angle grasping study | `docs/prm_full72_results/REPORT.md` | Preserved `reproduced/prm_full72_*` experiment |
| Joint-limit diagnostic | `results/workspace_limit/workspace_limit_diagnosis.json` | Preserved CSV and image, figure manifest |
| Physical inspection example (cohort C primary policy) | `docs/mobile_retention_results.md` | `runs/mobile_partial_workspace_seed42_20261109/runs/p10t_seed42_20261111/policy_results/standard/replays/mobile_retention_v10_20261111_test_g000_front_blocked/` |

## Definitions used in the report

A mobile scene contributes 16 candidate trials: two inspection branches, four stations, and two grasps. Candidate feasibility is the union of successful recorded candidates. Correct decisions combine successful feasible openings and correct library-infeasible refusals. Episode time is measured in simulated seconds. The distance/time term in the inspection objective uses the inspection prefix only.

The A/B/C tests (archived as v6/v8/v10) contain 72 scenes each, 18 parameter groups each, and nine shared template families. Their primary policies share the same frozen model/inference identity. The descriptive total is 210/216 correct, 150/153 feasible openings, 60/63 correct refusals, zero observed collisions, and 125 learned inspections versus 216 always-inspect observations. Mean episode time is 37.6892 versus 55.8983 seconds.

## Cohort names and initialization

| Paper cohort | Archived identifier | Correct decisions | Saved run |
| --- | --- | --- | --- |
| A | v6 | 68/72 | `runs/mobile_inspection_regret_evaluation_seed42_20261110/` |
| B | v8 | 72/72 | `runs/mobile_branch_balance_evaluation_seed42_20261110/` |
| C | v10 | 70/72 | `runs/mobile_partial_workspace_seed42_20261109/runs/p10t_seed42_20261111/` |

`COHORTS.json` records each cohort's manifest/report hashes, selection identity, and checkpoint lineage. All three cohorts have 72 scenes in 18 groups, use seed 42, and have disjoint test geometries.

The continuation input is the earlier partial-map `depth_features/selected_pg.pt` checkpoint (SHA256 `fafec5f893f2963dbe24a82645de313382b3b24f59bb6fb2c971f1bb2bc17a70`). The partial-map stage selected P/G update 0 after adding the depth-feature input slots; its inherited adapters and outcome heads come from the preceding factorized mobile outcome-supervision checkpoint. The full input chain and original protocol hashes are recorded in `COHORTS.json`.

Figure 3 uses the first front-blocked test scene in cohort C's manifest and its preselected primary policy. Its recorded episode covers 4.6209688625 m in 61.992 simulated seconds. The camera arrays' poses match the replay's initial and inspection poses.

## Figure revision: 2026-11-22

- Figure 1 reads `results/workspace_limit/workspace_limit.npz`, the paired CSV, and `workspace_limit_diagnosis.json`. Its configuration image is rendered from the saved final qpos. Door angle is 42.5051 degrees and remaining elbow travel is 1.4110 degrees.
- Figure 2 shows the shared visual encoder, detached target construction, and the two training stages described in Section 4.
- Figure 3 uses the selected `standard` policy replay `mobile_retention_v10_20261111_test_g000_front_blocked` from cohort C: `trace.json`, `result.json`, and branch-1 `observations.npz`. The initial/reached RGB arrays are embedded unchanged.
- Figure 4 uses cohort C (archive v10) `standard`, `frozen_v9_always`, and `frozen_v9_never` rows. Their correctness counts are 70, 70, and 50 out of 72; mean episode durations are 38.3103, 55.6824, and 12.3815 simulated seconds.
- Table 3 preserves its numbers and separates the inspection-mode comparison from the retention training ablation.

The figure generator records input hashes and checks them again after rendering in ignored `runs/final_report_consistency_20261122/figure_data_audit.json`.

## Implementation pointers

- Execution and reached observations: `mobile_door.py`, `mobile_partial_execution_v5.py`.
- Current-depth features and masking: `mobile_partial_map_v5.py`, `mobile_partial_policy_v5.py`.
- Outcome utility and continuation classes: `mobile_outcome_v4.py`.
- Current-G-conditioned P targets: `run_mobile_outcome_experiment.py`.
- Fixed-policy inspection advantage and objective: `mobile_inspection_regret_v6.py`.
- Latest training settings, sampling, and selection: `run_mobile_retention.py`, `mobile_retention_v10.py`.
- Mobile evaluation physics: archived v3 workspace `runs/mobile_partial_workspace_seed42_20261109/`.

## Revision verification

The revision checks each cohort manifest hash, the selected checkpoint hashes, geometry separation within and across splits, saved frozen-core audits, and per-scene task outcomes for learned/always inspection. The three runs retain their seed 42 settings and source identities. Initial verification is kept in ignored `runs/final_report_revision_20261121/`; this alignment revision's source backup, build logs, and verification are in `runs/final_report_consistency_20261122/`.
