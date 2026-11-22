# Two-branch selection and scene-group training (v8, 2026-11-10)

Four arms each complete **320 LoRA/P-G and 1600 I updates**, keeping core and normalization frozen. Validation selects **paired_ce**, whose model state tensors equal frozen v6 exactly: **no new primary-policy improvement**. Paired layouts and collision weighting do not beat initialization. Random ordinary CE selects different weights but uses 22 validation inspections, one more than primary.

A new 18-group, 72-scene test is frozen at training launch. Preserve these development results; the subsequent [independent test is complete](mobile_branch_balance_results.md), with separate generalization metrics.

![Validation curves for direct-branch correctness and collisions](mobile_branch_balance_development.png)

## Four matched arms and selection

Match initialization, capacity, two-scene batches, 320 LoRA/P-G updates, and 1600 joint-regret I updates. Random sampling draws two training scenes; paired sampling draws different layouts within a random group. Collision-weighted CE changes only training weights. Formal utility remains success +1, ordinary failure -1, collision -2, refusal 0, minus actual inspection cost. [Preregistered design](mobile_branch_balance_plan.md).

PG selection compares total direct/always collisions, then always correctness, direct correctness, total openings, net utility, and path length; earliest ties win. Freeze encoder/LoRA/P/G for I training. Select the final arm by validation correctness, collisions, openings, net utility, path length, and inspections.

| Arm | Selected PG update | Selected I update | Open /26 | Correct /36 | Collisions | Inspections | Mean net utility | All tensors equal v6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| random_ce | 160 | 280 | 25 | 35 | 0 | 22 | 0.5825 | No |
| **paired_ce (validation selected)** | **0** | **40** | **25** | **35** | **0** | **21** | **0.5876** | **Yes** |
| random_collision4 | 0 | 40 | 25 | 35 | 0 | 21 | 0.5876 | Yes |
| paired_collision4 | 0 | 40 | 25 | 35 | 0 | 21 | 0.5876 | Yes |

The last three arms tie, selecting paired_ce by preregistered order. They retain the same old model rather than establishing gains from pairing or weighting. Random CE selects updated LoRA, changing direct validation from 15 openings + 9 correct refusals to 16 + 8: still 24/36 correct with zero collisions and a longer path, rather than better accuracy.

Last-step direct correctness is 25/36, 24/36, 24/36, 24/36, with 2/1/2/0 collisions. Every always-inspect branch scores 35/36 with zero collisions. Safety-first selection rejects some last checkpoints rather than treating more openings and fewer refusals as overall improvement.

At paired_collision4 update 240, direct correctness rises from 24/36 to **26/36**, while always inspection falls from 35/36 to **30/36**, both with zero collisions. The predefined rule retains the earlier checkpoint. This shows a branch tradeoff without identifying gradient conflict or its mechanism.

## Data, weights, and limitations

Use only the existing 72 training and 36 validation scenes/candidates, refreshing labels for current G's choices every 40 updates without oracle selection. Retain maps, perception, and continuous execution. New-test geometry excludes **602** historical identities; manifest SHA256 is `3e506223e2fc6bf8e59da03b132398e5f91de4dddaa4af624017986e4fe62c7e`.

All 16 checkpoints preserve **422** core tensors and normalization bitwise. Last-step LoRA updates in every arm; I stages retain all non-I tensors relative to selected PG. Three final models equal v6 tensor by tensor. Eight contracts and smoke tests cover splits, paired layouts, collision weighting, two-branch safety selection, and frozen-continuation action labels.

Fix seed 42 in Python, NumPy, Torch, CUDA, and hash settings. Attention remains warn-only nondeterministic. Scope is one seed, a reused small validation set, familiar classes, and fixed physics. P/G still share LoRA, without evidence that gradient conflict is eliminated.

Since random_ce selects different weights, evaluate a newly frozen test under the pretraining protocol, without reusing historical v6 outcomes or reselecting primary. Pre-test tensor/normalization/inference/mode checks group seven names into four deployed policies, yielding 288 live replays. List shared-record aliases explicitly without treating them as independent experiments. Formal results are linked above.

## Runtime and durations

The 8-contract, four-arm 8/8 smoke test is estimated at 5 minutes with queue 0. Actual pipeline **189.09 seconds**, controller 197.18; session **13937** exits 0 after one continuous 261.28-second agent sleep.

Full estimate: **2 hours, queue 0**, from historical PG stages of about 1238-1288 seconds per arm plus I, caches, both-branch validation, and audits. Launch 11:33:12; process **10532**, controller **14432**, session **8346**. Startup 9.28 seconds. The agent sleeps once for **7161.88 seconds** until estimated 13:33:12, then confirms the same session completed at **13:04:26**, exit **0**. Actual pipeline **5471.93 seconds (91 minutes 12 seconds)**; controller 5474.25.

| Phase | Seconds |
| --- | ---: |
| random_ce PG / I | 1267.32 / 99.76 |
| paired_ce PG / I | 1282.53 / 97.65 |
| random_collision4 PG / I | 1265.00 / 96.31 |
| paired_collision4 PG / I | 1244.59 / 97.29 |
| Startup, I/O, selection, audits | 21.46 |

No concurrent physical collection runs in this phase. Keep estimates separate from measurements. The repository stores implementation, reports, and figures; weights, arrays, logs, and telemetry remain in ignored `runs/`.

[Training report and curves](../runs/mobile_branch_balance_training_seed42_20261110/report.json) · [Pre-test selection/hashes](../runs/mobile_branch_balance_training_seed42_20261110/frozen_for_test.json) · [16-checkpoint audit](../runs/mobile_branch_balance_training_seed42_20261110/tensor_audit.json) · [Launch/sleep verification](../runs/mobile_branch_balance_training_seed42_20261110/verification.json).

Run in the frozen v3 physics workspace `runs/mobile_partial_workspace_seed42_20261109/`, with outputs under `runs/p8_train_seed42_20261110/`; do not mix the root's alternative physics implementation.
