# Repository work rules

## Training and evaluation

- Continue H-JEPA with LoRA unless the user explicitly authorizes full-model
  training. Preserve the original model, existing checkpoints, and frozen test
  datasets; write each experiment into a new ignored `runs/` directory.
- Choose losses, hyperparameters, stopping rules and checkpoints using training
  and validation data. Previously inspected tests are historical comparisons;
  use a separately frozen test split for final evaluation.
- Before starting a long training, collection or evaluation job, estimate duration from comparable local
  runs. After launch, confirm its actual process/tool session is live and refine
  the estimate with measured startup/warmup throughput. Record the estimate,
  evidence, process/session handle and expected completion time in the run.
- **After reporting that estimate, sleep this agent until the estimated
  completion deadline.** Use the agent sleep tool, rather than only making a
  shell or training subprocess sleep. Calculate the remaining duration once
  and issue one uninterrupted sleep until that deadline. **Do not wake every
  few seconds or minutes to check the clock, poll the job, or send repetitive
  waiting updates.** Only split a sleep when a hard tool limit requires it;
  use the largest supported duration. User input may interrupt the sleep;
  handle it, then sleep once for the remaining duration. Do not replace the
  requested wait with unrelated work or an early final answer.
- At the deadline, inspect the same authoritative process/session. An estimate
  is not proof of completion. If it is still live, revise the remaining-time
  estimate and sleep again. If it failed, inspect and fix the concrete failure;
  never start a duplicate job because a status read or tool observation timed
  out. Collect evaluation results and verify frozen weights before reporting
  completion.
- Record actual phase durations after completion. Distinguish GPU-contended
  collection during LoRA training from collection-only/calibration jobs when
  estimating the next run; document a conservative estimate rather than
  silently presenting it as measured execution time.
- Keep generated arrays, weights, logs and videos out of Git. Run the repository
  size guard before delivery.

- Before submitting each experiment, estimate runtime and queue time separately and state the evidence in the launch note. After submission, sleep for their sum and prioritize listening for result commits on local `origin/main`. If the job is still queued or unfinished on waking, do not mark it failed; sleep another 30 minutes. Without Git results and before walltime, continue treating it as unfinished rather than polling SSH every few minutes.
- User requirement dated 2026-10-23: new experiments default to `seed=42`, set Python, NumPy, Torch, and CUDA seeds, and record the actual seed in run/result records. Preserve the original seeds and source identities of existing frozen experiments and historical receipts without retroactive edits. If a model exposes no explicit seed, record its deterministic settings and limitations.
