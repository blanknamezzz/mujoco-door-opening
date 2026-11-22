# Inspection and door opening with partial maps (v5)

Status: RGB-D prefixes for 108 scenes, two 320-update PG arms, and 800-update I training are complete. Validation selects `depth_features_learned`: 35/36 correct, zero collisions, 23 inspections. Both P/G arms select update 0; independent testing remains pending. [Development results and run verification](mobile_partial_development.md).

## Motivation

The previous no-inspection diagnostic fixed neural P's refusal threshold but did not isolate answers supplied by the map. New geometric station-selection controls use only the known-map sweep mask and current base pose, followed by a fixed grasp angle or frozen G at arrival. Validation contains 36 scenes, of which 26 are feasible under the candidate library:

| Station / grasp rule | Open /26 | Correct /36 | Collisions | Inspections | Mean net utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nearest geometric station, fixed -15 degrees | 25 | 34 | 0 | 0 | 0.6389 |
| Rear-first geometric station, fixed -15 degrees | 26 | 35 | 0 | 0 | 0.6944 |
| Nearest geometric station, frozen scalar G | 25 | 34 | 0 | 0 | 0.6389 |
| Rear-first geometric station, frozen scalar G | 26 | 35 | 0 | 0 | 0.6944 |
| Rear-first geometric station, fixed +22 degrees | 6 | 15 | 0 | 0 | -0.4167 |

Frozen factorized G gives the same corresponding results as scalar G. These controls reuse physical candidate tables without updating models, adding live replays, or reading test outcomes. Fixed -15 degrees nearly reaches the development ceiling, showing a strong preferred library angle rather than establishing that learned G is essential.

[Full control metrics](../runs/mobile_geometry_control_seed42_20261109/report.json) and [process verification](../runs/mobile_geometry_control_seed42_20261109/verification.json). Seed=42; queue 0; estimated 300 seconds; actual 159.42 seconds. Authoritative session 47352 exited 0, and both checkpoint hashes remain unchanged.

## Information and physical scope

- Opening-sweep planning receives obstacle-free door/handle CAD, robot proprioception, and calibrated depth at physically reached states. Obstacle geometry, layout labels, segmentation ground truth, and future images do not enter map construction.
- The single-view map uses initial RGB-D; fusion updates only after real base motion and successful inspection capture. Depth endpoints form obstacle voxels after removing known door/robot surfaces and filtering depth discontinuities.
- Discrete sweep tests between obstacle voxels and door components provide blocking evidence. Occluded and out-of-view space remains unknown; sampled visibility is a feature rather than a complete safety certificate.
- Fixed settings: 1-degree sweeps, 2.5 cm voxels, 1.5 cm endpoint inflation, a 3x3 depth-edge filter (3 cm + 1% depth), and at least 2 hit voxels. Exclude the lowest 4 cm above the floor. Door and robot CAD are known priors.
- Only obstacle information for opening planning changes. Navigation and arm preparation still use known-map services; this is not complete navigation in an unknown environment.
- Retain continuous physics: one initialization per episode, integrated base motion for inspection, and contact-driven opening.

## Pilot collection and subsequent training

The first training group's 4-layout pilot takes 42.19 seconds (session 36086, exit 0). Initial/inspected states match frozen candidate prefixes. Maximum RGB difference is 1 gray level in at most 3 color channels per frame. Two mixed depth pixels create false obstacles; independent calibration confirms that edge filtering leaves both sides available in free scenes and recognizes a previously unknown rear obstacle only after inspection. Preserve the earlier pilot's source, arrays, and results.

Collect physical RGB-D prefixes for the existing 72 training and 36 validation scenes, verify states and inspection costs against the original candidates, and retain historical outcomes. Train equal-capacity image-only/image-plus-map P/I models: freeze the original H-JEPA core, continue existing LoRA/P/G, then freeze encoder/P/G to train I. Select parameters and policies on training/validation only. Freeze 18 new test groups and separately report partial-map geometric rules and frozen previous-model controls.

The next question is whether inspection resolves unknown obstruction enough to improve decisions and repay actual motion cost. Report full-map geometric rules as an information-rich reference, outside partial-map primary-policy selection.
