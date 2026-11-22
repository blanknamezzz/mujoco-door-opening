# Integration notes: 2026-10-27

Merge remote `6eefbb0` (H-JEPA training/evaluation updates) with local `57b3045` (station diagnostics, full PRM reports, case directories). README and collector merge automatically by three-way merge.

## Integrated behavior

- Preserve `station_feasibility.py`, `station_workspace_check`, and source-hash records, without automatic station filtering or reward changes.
- Adopt remote defaults: data seed 42, PRM seed 42, SL/RL training seeds `[42]`.
- Move 30 reference files into six `results/` cases, retaining content and shared-observation paths. Rendering, diagnostics, and reproduction support case directories and older flat outputs.
- Preserve historical reports/checks: data seed 20261026, PRM seed 7, training seeds 42/43/44. Explicit commands for these settings appear in the main README.
- H-JEPA and independent PRM/angle learning remain separate workflows; code integration is not an end-to-end policy experiment.

## Verification performed here

```bash
python -m unittest test_station_feasibility test_grasp_learning test_prm test_grasp_plan test_grasp_rl test_git_size
python check_git_size.py
```

- Forty-two tests pass in about 6.48 seconds, covering seed=42 defaults, explicit historical seed selection, station diagnostics, and collection preflight.
- All root Python files parse. Existing Windows command strings raise invalid-escape warnings and are unchanged in this integration.
- All 392 historical report/data/model SHA256 hashes match before/after.
- Git size checks pass; no reproduced/ or runs/ data, models, or logs are added. Existing trajectories only change paths.
- Preserve CSV CRLF and check with `git -c core.whitespace=cr-at-eol diff --check`.

Neither local MuJoCo venv nor inspected Conda environments has PyTorch, so this integration does not rerun complete H-JEPA tests, training, or GPU evaluation. Remote 110-test/model outcomes are the collaborator's existing records rather than new local verification.
