# No-inspection control: development diagnostic (2026-11-09)

With fixed models and known-geometry sweeps, validation still benefits from inspection: scalar correctness rises from 27/36 never-inspect to 34/36 learned. This diagnostic updates no models and reads no test outcomes.

## Validation results

Validation contains 36 scenes, including 26 library-feasible scenes. Every policy uses the same known-map sweep constraint and refuses initially if all candidates are excluded.

| Frozen model | Inspection | Open /26 | Correct /36 | Collisions | Physical inspections | Mean net utility |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v3 baseline | Never | 20 | 30 | 0 | 0 | 0.5556 |
| v3 baseline | Always | 24 | 34 | 0 | 27 | 0.5321 |
| v3 baseline | Learned | 24 | 34 | 0 | 27 | 0.5321 |
| v4 scalar | Never | 17 | 27 | 0 | 0 | 0.4722 |
| v4 scalar | Always | 24 | 34 | 0 | 27 | 0.5321 |
| v4 scalar | Learned | 24 | 34 | 0 | 13 | 0.6013 |
| v4 factorized | Never | 15 | 25 | 0 | 0 | 0.4167 |
| v4 factorized | Always | 25 | 35 | 0 | 27 | 0.5599 |
| v4 factorized | Learned | 24 | 34 | 0 | 13 | 0.6011 |

Corresponding training correctness (never/always/learned) is baseline 61/71/70, scalar 51/71/69, factorized 53/71/69, all out of 72.

## Interpretation

With current P/G/refusal thresholds, initial-image-only policies falsely refuse more scenes; inspection recovers some execution. Scalar learned inspection matches always-inspect correctness on validation with 14 fewer inspections.

This does not establish that every no-inspection policy is inadequate. Known geometry may support simpler station rules, and P0 calibration may explain differences. Compare geometry-selected stations followed by G and audit P0 calibration on development data; partial maps should isolate information available only through physical RGB-D. Use fresh frozen tests for final comparisons.

## Records

Read only 72 training and 36 validation scenes. Freeze three checkpoints with 0 updates. Compute decisions from existing physical candidate tables without new replays; checkpoint hashes match before/after.

Queue 0, estimate 6 minutes; session **38633**, PID **5628**. Sleep continuously to the deadline; final exit 0, actual **234.46 seconds**. Baseline/scalar/factorized inference takes 104.87/56.08/56.27 seconds.

[Development metrics](../runs/mobile_control_audit_seed42_20261109/development_controls.json) · [Run verification](../runs/mobile_control_audit_seed42_20261109/verification.json) · [v4 formal test](mobile_outcome_results.md). This informs the next design without revising or reselecting v4 primary results.
