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

- 每次提交实验前先分别估计运行时长和排队时间，并在启动说明中写出估计依据；提交后先按两者之和`sleep`，优先监听本地`origin/main`是否出现结果提交。若醒来时仍在排队或未完成，不判定失败，再`sleep`30分钟；没有Git回传且未到walltime时仍按未完成处理，避免几分钟一次SSH轮询。
- 2026-10-23用户要求：以后新实验默认固定`seed=42`，同时设置Python、NumPy、Torch和CUDA，并把实际seed写入运行／结果记录。既有冻结实验和历史回执保留原seed及源码身份，不追溯回写；模型若不暴露显式seed，须记录确定性设置和限制。
