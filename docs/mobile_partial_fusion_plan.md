# Partial-depth fusion and subsequent training design (v7, 2026-11-10)

Calibrate only on the existing 72 training and 36 validation scenes without historical-test tuning. Development has 168/88 handle-detection failures and two validation station records additionally excluded by fusion despite passing full geometry. Study map fusion first, keeping handle perception fixed to isolate changes.

The old map concatenates clouds after per-frame voxelization. Two frames may count the same spatial voxel twice toward the two-hit blocking threshold. The new implementation selects one deterministic measured endpoint per spatial voxel, rather than replacing measurements with voxel centers; duplicate frames and frame order should preserve the occupied voxel set.

Four preregistered profiles:

| Profile | Cross-frame deduplication | Free-ray consistency | Endpoint inflation |
| --- | --- | --- | --- |
| historical | No; original v5 | None | 15 mm |
| voxel_union | Yes | None | 15 mm |
| ray_consistent_union | Yes | Remove endpoints when any reached view sees behind them with a 15 mm margin | 15 mm |
| tight_voxel_union | Yes | None | 3 mm |

Free-space consistency follows the sensor-ray distinction between occupied, free, and unknown space in [Hornung et al., OctoMap, 2013](https://www.arminhornung.de/Research/pub/hornung13auro.pdf). This implementation uses deterministic endpoint fusion/filtering rather than an octree or Bayesian log-odds. Out-of-view, invalid-depth, and occluded regions stay unknown. Ray filtering and discrete sweeps are not safety certificates.

Runtime inputs remain current physical RGB-D, door/robot CAD, and proprioception. Complete obstacle geometry supplies calibration labels only, not map/model inputs. Preserve the four-dimensional map features and information timing, as well as navigation, preparation, and contact-opening physics. Preserve v5 source and historical records.

Validation selection first requires no additional allowed collision candidates on the initial branch relative to historical. Then minimize allowed collisions after fusion, excluded successful candidates, extra exclusions relative to full geometry, remaining full-geometry obstructions, and initial excluded successes. Exact ties retain table order. Never fill initially unknown obstacles using ground-truth labels.

Additional validation diagnostic: v5 PG selection checked only always inspection and may miss direct-branch changes. Compare direct/always outcomes of updates 0/320 on the same 36 validation scenes using existing physical candidates, without weight updates or test reads. This diagnostic does not select map profiles.

Two historical-map queries across 108 scenes take 71.34 seconds total, mean 0.661 per scene, maximum 1.368. Three new profiles extrapolate to about 214 seconds. Including RGB-D reads, mapping, labels, hashes, and validation of two frozen PG checkpoints gives a conservative **8-minute, zero-queue** estimate: about 6 minutes for mapping and 2 for model loading/forward passes. No physical collection or LoRA training runs concurrently. Record process, first-scene duration, and deadline; sleep continuously and inspect the same session afterward.

If validation improves, compare old/new maps with equal capacity and budgets, continuing rank-8 LoRA/P-G before freezing continuation to train I, with the original core preserved. Freeze fresh test geometry before final collection. Do not select weights or claim new generalization from historical tests. Retain negative calibration results without selecting more aggressive maps on test performance.
