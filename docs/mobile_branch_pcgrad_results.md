# Gradient-projection results across two branches

On November 11, 2026, both arms complete **320 LoRA/P-G and 1600 I updates**. PCGrad projects **12** negative-inner-product gradients in 320 updates. Both select PG update 0 and I update 40, matching every tensor of the old primary. **No new deployed policy or validation gain results.** Core/normalization remain frozen; the fresh 72-scene test is neither collected nor read.

![Two-branch validation curves and gradient statistics](mobile_branch_pcgrad_results.png)

## Matched training and selection

Tasks are direct and physically inspected execution, each with P cross-entropy plus 0.5 times visible-G cross-entropy. Match initialization, rank-8 LoRA, two-scene batches, scene sequence, optimizer, and budget, comparing gradient sum against two-task PCGrad. Both use two backward passes. Current G's selected-candidate labels refresh every 40 training-only updates rather than selecting oracle grasps.

Implement [Yu et al., PCGrad Algorithm 1](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) using the other task's original gradient for each projection, then summing. Apply to all trainable LoRA/P/G heads; block logs are analytical, without separate blockwise projection.

| Policy | Selected PG update | Selected I update | Open / feasible | Correct refusal | Correct | Collisions | Inspections | Mean net utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gradient sum | 0 | 40 | 25/26 | 10/10 | 35/36 | 0 | 21 | 0.5876 |
| PCGrad | 0 | 40 | 25/26 | 10/10 | 35/36 | 0 | 21 | 0.5876 |

These are the original **36 validation scenes**, not new testing. Exact ties retain gradient sum by pretraining order. Checkpoint file hashes differ through metadata, while tensors, normalization, and effective inference match. Both last-step LoRA checkpoints update 148 tensors; selecting initialization does not mean training was skipped.

## Gradient conflict and branch tradeoffs

| Negative original-gradient inner products | Gradient sum | PCGrad |
| --- | ---: | ---: |
| All trainable parameters | 11/320 | 12/320 |
| LoRA block | 10/320 | 11/320 |
| Positioning P head | 20/320 | 20/320 |
| Grasping G head | 0/318 | 0/318 |

Two G updates lack nonzero branch gradients, so angles are undefined and excluded from 318 valid observations. Other blocks have defined angles at every update. PCGrad's 12 projections follow negative all-parameter products; sum does not project. Mean full-parameter cosines are 0.6166/0.6168; LoRA 0.7084/0.7091. Trajectories diverge after the first projection, so counts at different parameter states are not paired effects.

At PCGrad update 200, direct validation reaches **26/36 with zero collisions**, above initial 24/36, but always inspection drops to **34/36** from 35/36, failing the predefined selection order. Both last steps restore always-inspect 35/36/zero collisions; direct scores 24/36 and 25/36 with one collision each. A single branch's best score omits other-branch degradation or safety failures.

Training contains a few negative inner products, but this projection intervention does not improve selected validation performance. It does not establish causation from negative products or measure curvature; AdamW, grouped rates, and decay alter actual updates. Tasks remain direct/inspected branches rather than isolated P/G objectives. One seed, small reused development data, and current map constraints limit conclusions.

## Test and weight protection

Freeze **18 groups and 72 test scenes**, excluding **682** historical geometries. Manifest SHA256: `98d4cfbb519d7fb125ae0222ebb9042ea00ad01ecfb42880b2e4f0818bbaaad3`. Both selected policies equal the old primary, so preserve the unread test under preregistered rules rather than recollecting the same policy or adding historical test scores to this run.

Eight gradient/masking/sampling/selection contracts and both 8/8 smoke arms pass. Eight formal checkpoints preserve **422** core tensors/normalization bitwise and non-I tensors during I training. Training batches/features contain no test identities; old model/array/source hashes match. Python/NumPy/Torch/CUDA/hash seed is 42, with warn-only/OpenGL determinism limitations recorded.

## Actual durations

The first smoke run fails before updates from duplicate configuration fields: session 46290, exit 1, controller 66.37 seconds. Fix in a new directory while preserving old snapshots/logs. Second estimate: 8 minutes, queue 0; actual pipeline **114.26 seconds**, controller 121.51; session **44693**, exit 0, checked after a continuous 437.20-second sleep.

Formal estimate **90 minutes, queue 0**. Two-backward smoke throughput 2.6-3.2 seconds/update plus full label refresh, validation, I, and audits extrapolates to about 55 minutes; reserve 5400. Launch 02:49:09: process **26788**, controller **6180**, session **63222**, startup 17.26 seconds. Sleep once **5359.49 seconds** until 04:19:09, then confirm completion **03:43:15**, exit **0**. Actual pipeline **3241.97 seconds (54 minutes 2 seconds)**; controller 3245.98.

| Phase | Actual seconds |
| --- | ---: |
| Gradient-sum PG / I | 1528.16 / 111.56 |
| PCGrad PG / I | 1464.68 / 98.53 |
| Startup, I/O, selection, audits | 39.04 |

Mean PG updates take 3.138/3.033 seconds with no concurrent physical collection. Keep estimates/measurements separate; wall-clock differences alone do not establish algorithmic speedup.

[Training design](mobile_branch_pcgrad_plan.md) · [Full report](../runs/mobile_branch_pcgrad_training_seed42_20261111/report.json) · [Selection freeze](../runs/mobile_branch_pcgrad_training_seed42_20261111/frozen_for_test.json) · [Checkpoint audit](../runs/mobile_branch_pcgrad_training_seed42_20261111/tensor_audit.json) · [Gradient/inference audit](../runs/mobile_branch_pcgrad_training_seed42_20261111/post_analysis.json) · [Runtime/sleep verification](../runs/mobile_branch_pcgrad_training_seed42_20261111/verification.json).

Raw weights, arrays, gradients, and logs remain ignored in `runs/`, with actual output `runs/p9_train_seed42_20261111/` in the frozen v3 workspace. Opening uses current depth/CAD; navigation/preparation use known maps. Existing formal results remain in the [two-cohort frozen-policy test](mobile_branch_balance_results.md).
