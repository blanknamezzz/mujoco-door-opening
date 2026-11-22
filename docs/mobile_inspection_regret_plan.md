# Inspection-decision regret with a frozen continuation policy (v6, 2026-11-10)

The v5 primary policy retains full correctness on its historical independent test and reduces inspections to 47/72, but its new P/G does not beat initialization and a full-map geometric reference has higher net utility. Change only I's objective to test whether direct supervision of inspection value reduces extra action cost. Historical tests do not enter training, validation, threshold selection, or model selection.

## Three matched objectives

Initialize from the same v5 depth-feature `selected_pg.pt`. Freeze the original core, existing LoRA, P/G, and normalization; update only the same I architecture, which outputs four outcome logits and an inspection cost. All arms use seed=42, matching initialization, inputs, 32-row sampling sequences, learning rate 3e-4, weight decay 0.01, 1600 updates, and validation every 40 updates including steps 0/1. No full-model training is used.

- `outcome`: retain v5's actual post-inspection outcome cross-entropy plus 0.5 times cost MSE.
- `joint_regret`: the preceding objective plus inspection-decision regret.
- `decision_regret`: inspection-decision regret plus 0.5 times cost MSE.

Let U0 and U1 be returns of actions selected by frozen P/G on direct and physically inspected branches, with actual inspection cost C. The new label is `Delta = U1-C-U0`, without selecting the best recorded training action afterward. Utility is success +1, ordinary failure -1, collision -1-risk, and refusal 0; risk=1/3 is a known decision-time input.

Deployment still compares I's predicted inspection return minus predicted cost against frozen P0's maximum predicted value, including refusal at 0. Decision loss is BCE on the difference divided by 0.2, with label `Delta>0` and normalized `|Delta|` weights. Mask the decision loss when every initial action is excluded, and force refusal at deployment; Delta=0 gives no decision gradient. Fix temperature, weights, and candidate groups before the run without test-based tuning.

The `decision_regret` outcome logits support decisions and are not calibrated outcome probabilities. The method can also change I by compensating for P0 value errors. Hold downstream P/G actions identical and use matched evaluation to assess actual outcomes.

## Information and validation

All arms use only the original 72 training and 36 validation scenes; labels come from their physical candidates and current fixed P/G. I receives only initial-image features, the current relative target, risk, and initial-depth features. Future images, fused maps, and execution outcomes supply training labels without entering policy features.

Select checkpoints by validation correctness, fewer collisions, openings, net utility, shorter paths, and fewer inspections; exact ties retain the earliest checkpoint and the arm order above. Repeated use of a small validation set does not establish generalization. At training launch, freeze another 18-group, 72-scene test excluding every historical geometry. Final evaluation separately collects candidates and live replays for the three arms and frozen v5 learned/always/never policies.

Verify preservation of all non-I tensors, 422 original core tensors, normalization, historical data, and source hashes. Formal replay checks current RGB-D, one initialization, stage continuity, and actual inspection cost. Use the frozen v3 isolated physics workspace without mixing the repository root's alternative `simulate.py`. Store new models, arrays, and logs only in new ignored `runs/` directories.

## Runtime estimate

First run 4 new label/gradient contracts and an 8-update smoke test per arm: estimated runtime 2 minutes and queue time 0. Reference v5's two 800-step I stages of 72.44/72.07 seconds and the earlier roughly 2-minute smoke pipeline; reserve 10 minutes with zero queue time for three full 1600-step arms. Shared frozen feature caches reduce encoding, but these are estimates rather than measurements. Confirm the live process and warmup, sleep continuously until the deadline, and inspect the same session. Record actual durations and phase statistics afterward.
