# Factorized outcome supervision and door-sweep controls (v4)

Status: the formal experiment is complete, with 864/864 physical replays matching; see [full results and verification](mobile_outcome_results.md). The preceding smoke test passed 41 contracts and 48 replays. The v3 run remains independent historical evidence, with weights, frozen tests, and receipts preserved. The preregistered design and estimates follow.

## Questions

Among v3's 1,728 training/validation candidates, 152 collide and 304 complete opening. On validation, learned inspection with the cost model makes 4 fewer correct decisions than always inspecting. These development observations motivate comparing outcome representation, staged training, and opening-geometry constraints.

1. Scalar mean return merges ordinary failure and collision risk. Compare expected-return MSE against success/ordinary-failure/collision cross-entropy with equal-capacity networks.
2. Train and select positioning/grasping P/G and encoder LoRA first, freeze them, then fit inspection separately. Inspection labels use actions actually selected by frozen P/G rather than the best counterfactual action.
3. Include door-leaf and handle sweeps in known-map planning. Query overlap between door components, static obstacles, and the target base footprint only in a separate `MjData`; physical opening still comes from contact.

## Matched settings

- Both models initialize from v3's validation-selected `view_free/selected.pt`, with matching P/G hidden layers, three-class outputs, independent inspection networks, and seed 42.
- Stage 1 performs 320 updates per model to existing rank-8 LoRA and P/G only. Keep other tensors frozen and select PG on validation with no sweep, always inspection, and risk coefficient 1.
- Stage 2 freezes selected P/G and every encoder parameter, training only inspection for 800 updates. Both models use four conditions: risk 1/3 crossed with ordinary/sweep planning. Risk and planning mode are known at execution.
- Inspection predicts success, ordinary failure, collision, and downstream refusal, with a separate positive output for actual inspection cost. Inputs are initial-image features, actual relative target, risk, and planning mode.
- The three outcome values are `1, -1, -1-c`; `c=1` is primary, and `c=3` is a preregistered factorized-model control. Main metrics retain the common utility `2*success-1-collision-observation_cost`.
- Select I checkpoints on no-sweep, learned-inspection, risk-1 validation outcomes, without selecting across other planning conditions.

## Data and policies

Reuse v3's frozen 72 training and 36 validation scenes, checking physical-source and input-file hashes individually without resampling or rewriting arrays. Generate a fresh test of **18 groups and 72 scenes**, with front/rear/both/no-obstacle layouts per group and no historical geometry. Collect outcomes only after freezing models, policies, and hashes.

Twelve policies:

- The original v3 validation-selected model with always inspection, under ordinary/sweep planning.
- Scalar and factorized models with always/learned inspection, crossed with ordinary/sweep planning: eight policies.
- Factorized risk-3 learned inspection under ordinary/sweep planning: two policies.

Sweep planning excludes predicted colliding stations before selection; if all are excluded, refuse at the initial pose. Feasibility still uses the union of unfiltered physical candidates from both branches, so conservative refusal does not inflate success.

## Planning scope

By default query the sweep to 90 degrees at 1-degree intervals, with 1 mm penetration tolerance. In development, all 152 collision candidates are excluded and all 304 successful candidates retained. These describe the same training/validation records rather than fresh test performance.

The planner has a complete known map and extra geometric constraint information; report its effect separately from neural supervision. Queries cover door components, static obstacles, and base footprint. Moving-arm collisions, friction, grasp dynamics, and collisions between query angles still require physical execution.

## Verification and duration estimates

Passed checks cover equal scalar means with different collision mixtures, risk values, selected-action rather than oracle labels, frozen P/G during I training, checkpoint round trips, refusal sentinels, door/base collisions, unchanged live state during geometry queries, and existing mobile contracts.

First run 2 updates on each of one training group, one validation group, and one fresh test group, including all 12 policies and 48 replays. Estimate 25 minutes and zero queue time from v3's roughly 358-second 4-scene collection and 208-second 32-replay run, plus model loading, 40 tests, and margin.

The first smoke run failed because normalization stayed on CPU after loading a GPU model; session 74309 exited. After fixing loading and adding a CUDA regression check, rerun in a new directory. Session 79932 exited 0: 41 tests passed; pipeline 757.43 seconds, collection 295.34 seconds, 48 replays 363.38 seconds. The agent slept continuously for the revised 20-minute estimate. Two updates per model are an integration check rather than performance evidence.

Tensor checks of selected/last, PG, and I-stage checkpoints preserve all 422 original frozen tensors and normalization statistics. P/G and all LoRA tensors remain bitwise unchanged during I training. Failed receipts, source, and weights are preserved.

Revise the full estimate to **5.5 hours with zero queue time**: extrapolate 295.34 seconds for 4-scene collection to about 5,316 seconds for 72 scenes, and 363.38 seconds for 48 replays to about 6,541 seconds for 864. Reserve about 4,200 seconds for two-model training, label refresh, and validation using v3 as reference, plus I/O, tensor audits, and load variation. The 19,800-second budget is conservative rather than measured. Record the live process, throughput, and expected completion after launch.
