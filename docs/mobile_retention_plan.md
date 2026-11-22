# Continuation learning rates and output retention

November 11, 2026. In the previous full run, negative inner products between direct/inspected branch gradients occur in only a few updates, and PCGrad does not improve final validation selection. Ordinary continuation nevertheless drops always-inspect validation correctness from 35/36 to 12/36 at update 40 before recovery. Test whether smaller updates and retention of previous outputs preserve existing capability while learning actual task outcomes.

| Arm | PG learning-rate multiplier | Previous-output KL weight |
| --- | ---: | ---: |
| standard | 1 | 0 |
| low_lr | 0.1 | 0 |
| retention | 1 | 1 |
| low_lr_retention | 0.1 | 1 |

Original rates are 3e-4 for PG heads and 2e-5 for LoRA; low-rate arms use 3e-5 and 2e-6. Match architecture, initialization, rank-8 LoRA, alpha 16, dropout 0.05, AdamW weight decay 0.01, and clipping at 1. Each arm runs 320 PG updates with the same paired-layout scene sequence and one backward pass. Retrain the standard arm as a matched baseline rather than assuming numerical identity with the previous two-backward-pass trajectory.

Supervision remains branch-specific P cross-entropy plus 0.5 times visible-G cross-entropy. Refresh outcomes of current G's selected actions every 40 updates using training data only. Retention uses KL(frozen initial outputs || current outputs) on the same supervised rows, temperature 1, with G weighted by 0.5. Unsafe inspected-branch P and unobserved G rows are masked from both objectives. All arms compute KL to match forward cost; zero-weight arms do not include it in optimization. Terminal utility, refusal value, and maps stay fixed.

Use the output-retention idea from [knowledge distillation](https://arxiv.org/abs/1503.02531) and [Learning without Forgetting](https://arxiv.org/abs/1606.09282). This continues the same task on the same development set, without the papers' new-task setting, added capacity, or teacher test outcomes.

The teacher is the same frozen initial PG, encoding only **72 training scenes** once in eval mode with bf16 autocast and the student's candidate batching. Cache identity includes initial-checkpoint, scene, and file hashes. The teacher receives no gradients; validation/test scenes never enter its cache. Student dropout remains active, so KL also constrains stochastic variation. Validation and final live inference use standard encoding without sharing features between different student models.

Select PG with the existing two-branch safety-first validation order, retaining earliest exact ties. Freeze all non-I parameters, then train **1600 joint-regret I updates** per arm at the original rate. Select the primary arm using existing validation task/cost criteria, without test-based selection or choosing the minimum-KL step.

Reuse 72 training and 36 validation scenes. Before training, freeze 18 new groups and 72 test scenes, excluding every historical geometry, including earlier unread manifests. If all selected models equal the old primary policy, preserve the unread test; otherwise evaluate under a separately frozen physical-collection/live-replay protocol. Historical tests do not select models, thresholds, loss weights, or stopping budgets.

Fix seed 42 in Python, NumPy, Torch, CUDA, and hash settings; retain deterministic limitations such as warn-only operators. Preserve all 422 core tensors and normalization bitwise; audit 16 stage checkpoints and non-I tensors during I training. Preserve old models, frozen data, and source. Store teacher outputs, arrays, weights, and logs only in new ignored `runs/`, using the frozen v3 physics workspace.

Run 6 new objective/teacher-isolation and 4 existing sampling/safe-selection contracts, then 8 PG / 8 I updates per arm. Reference the earlier four-arm smoke run of about 197 seconds and two-arm two-backward-pass run of 121.51 seconds: initial estimate **8 minutes with zero queue time**, allowing teacher generation, loading, and 16-checkpoint audits. Reference the previous full 91-minute training for a conservative **2-hour, zero-queue** estimate, refined after smoke/warmup. Confirm process/session, evidence, and deadline, sleep continuously, then inspect the same session.

The smoke run completes: 10 contracts pass, pipeline time 293.92 seconds, controller time 302.60 seconds, session 66702 exits 0. Scene-sequence and teacher-cache hashes match across all four arms. All 16 checkpoints preserve core, normalization, and non-I tensors during I training. Mean updates take 3.18-3.30 seconds, slower than the previous single-backward reference; first teacher generation takes 2.78 seconds and later loads about 0.14 seconds.

Retain four arms of 320 PG / 1600 I updates without changing objectives from smoke results. Revise the estimate to **150 minutes with zero queue time**: 1280 updates at 3.3 seconds yield about 4224 seconds, plus about 500 seconds per arm for full-development label refresh/validation and about 500 seconds for I/startup audits. Baseline about 112 minutes; reserve 9000 seconds conservatively. Smoke uses only 4 training / 4 validation scenes versus 72/36 in the full run; do not extrapolate validation cost solely by update count. No physical collection runs concurrently.
