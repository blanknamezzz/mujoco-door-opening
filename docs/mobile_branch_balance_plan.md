# Selection across both branches and paired scene sampling (v8, 2026-11-10)

Development results show that the previous PG checkpoints at updates 0/320 both score 35/36 with zero collisions on the always-inspect branch, while the direct branch changes from 24/36 with zero collisions to 25/36 with two collisions. Selection must therefore check safety on both branches. In the v7 map calibration, all 648 mask comparisons across three fusion variants match the historical map, with no development benefit; the original map is retained.

This run uses a matched 2x2 design with the original 72 training and 36 validation scenes:

| Arm | Organization of the two scenes per update | Success / ordinary failure / collision CE weights |
| --- | --- | --- |
| random_ce | Two scenes sampled from all training scenes | 1/1/1 |
| paired_ce | Two different layouts from a randomly sampled training group | 1/1/1 |
| random_collision4 | Two scenes sampled from all training scenes | 1/1/4 |
| paired_collision4 | Two different layouts from a randomly sampled training group | 1/1/4 |

Each arm performs 320 rank-8 LoRA/P-G updates and 1600 joint-regret I updates. The original H-JEPA core and normalization remain frozen, and P/G share LoRA. Arms match capacity, initialization, two-scene batches, image counts, optimizer budgets, and seed 42. Loss arms with the same sampling method share their scene sequence. Pairing preserves marginal scene probabilities while changing batch co-occurrence, so images are not identical at each update across sampling methods.

Labels remain the terminal outcomes of actions selected by current G on the physical branch, refreshed every 40 training updates rather than choosing the best recorded action afterward. Only the collision class receives extra training weight; formal utility remains success +1, ordinary failure -1, collision -2, refusal 0, minus actual inspection cost. Weighted CE outputs are not calibrated posteriors.

PG validation prioritizes fewer collisions across both branches, then always-inspect correctness, direct correctness, total openings, net utility, and path length. Exact ties retain the earliest checkpoint. An additional direct-branch correct decision cannot offset a new collision. After selecting PG, freeze encoder/LoRA/P/G and train the same joint-regret I. Select the primary arm by the existing validation order of correctness, collisions, openings, net utility, and subsequent criteria.

Before training, freeze a fresh 18-group, 72-scene test excluding historical geometry. Historical tests do not select temperature, weights, budgets, or stopping rules. Preregister comparisons of four arms and frozen v6 with learned/always/never inspection. If all selected state tensors equal frozen v6, retain the unread test and report no new deployed policy; otherwise collect new independent physical candidates and live replays.

Keep the frozen map, perception, navigation, preparation, and opening implementations. Write new arrays, weights, and logs only to new ignored `runs/` directories. Verify the 422 original core tensors, normalization, non-I tensors during I training, and historical model/data hashes. Fix Python, NumPy, Torch, CUDA, and hash seeds to 42 and record nondeterministic limitations.

First run contracts and an 8/8-update smoke test per arm, estimated at **5 minutes with zero queue time**. For the four full arms, reference the previous two 320-step PG runs of 1288/1238 seconds and v6's three 1600-step I runs of about 29/37/34 seconds. Allow for both-branch validation and label refresh: initial estimate **2 hours with zero queue time**, refined after smoke tests and measured warmup. Confirm the actual process/session, record the deadline, sleep continuously until it, and inspect the same session without duplicate launches.
