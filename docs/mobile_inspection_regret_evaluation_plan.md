# v6 fresh frozen test and live replays

Training/validation selected `joint_regret` at update 40. The formal test uses 18 groups and 72 scenes separately frozen at training launch; historical v5 results do not select weights, temperature, loss weights, or maps.

| Policy | Weights | Inspection mode |
| --- | --- | --- |
| outcome | Matched outcome-supervised arm | Learned |
| joint_regret (primary) | Outcome plus regret arm | Learned |
| decision_regret | Regret-only arm | Learned |
| frozen_v5_learned | Frozen v5 depth-feature arm | Learned |
| frozen_v5_always | Same frozen v5 | Always; refuse if every initial action is excluded |
| frozen_v5_never | Same frozen v5 | Never |

The last three controls were declared in the training freeze record. All policies share P/G, existing LoRA, core, and normalization, verified tensor by tensor before sharing fixed-forward caches. Each policy still makes independent live RGB-D decisions and physically executes the task without cached maps or actions. Retain the unmodified v5 collector and replay engine, changing only preregistered model paths and inspection modes.

For every fresh scene, execute two branches x four stations x two grasps from scratch, totaling **1152** physical candidates. Six independent policy replays total **432**. Report openings, correct/false refusals, collisions, inspections, net utility, actual path length, and simulated time. Correct decisions include correct refusals, and feasibility refers to the candidate library. Retain the preselected primary policy rather than choosing a new winner on the test.

First use the smoke test's separately frozen 4-scene manifest, disjoint from formal scenes, for 64 candidates and 24 live replays to check routing and audits. It verifies execution without retuning training. Collect formal results separately. Check one initialization, continuous opening-entry q/v, current depth masks, clocks, speeds, and timesteps. Preserve all non-I tensors across six new stage checkpoints and frozen-v5 references, plus the original 422 core tensors and historical inputs.

Estimate the smoke test at **15 minutes with zero queue time**, referencing v5's 4-scene collection of 368.51 seconds and 40 replays of 551.98 seconds; this run uses 24 replays plus loading/audits. Initially reserve **4 hours with zero queue time** for the formal test, from v5's 72-scene collection of 6560.84 seconds and about 4184.63 seconds for six comparable neural-policy replay sets, plus caching/audits and load variation. Refine by phase after the new smoke test, distinguishing collection from concurrent training. No training or concurrent collection runs in this phase.

After launch, confirm the actual process/authoritative session and warmup, record the deadline, and sleep continuously. Inspect the same session at the deadline; if incomplete, extend by 30 minutes under repository rules without duplicate launches. Fix seed=42 and keep raw data, models, videos, and logs only in new ignored `runs/` directories.
