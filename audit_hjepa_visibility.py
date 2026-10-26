"""Training-only observation counterfactuals; never used to rank policy actions."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from collect_hjepa import small_rgb
from perception import observe
from structural_scene import structural_scene_from_spec
from position_observation import observe_rear_camera


def audit(data, output):
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    scenes = [s for s in manifest['scenes'] if s['split'] == 'train' and
              s['family'] in ('pillar-both', 'low-back')]
    rows = []
    for spec in scenes:
        scene = structural_scene_from_spec(spec)
        # Keep geometry, view, light and robot fixed. Only remove the posts on
        # the far side of the closed door. No model or test scenes are loaded.
        other = replace(scene, pillars=tuple(p for p in scene.pillars if p.y <= 0))
        if len(other.pillars) == len(scene.pillars):
            continue
        original = small_rgb(observe(scene.experiment()), 224, True)
        counterfactual = small_rgb(observe(other.experiment()), 224, True)
        delta = np.abs(original.astype(np.int16)-counterfactual.astype(np.int16))
        changed = np.any(delta != 0, axis=-1)
        rear=small_rgb(observe_rear_camera(scene.experiment()),224,True)
        rear_other=small_rgb(observe_rear_camera(other.experiment()),224,True)
        rear_changed=np.any(rear!=rear_other,axis=-1)
        rows.append(dict(scene_id=spec['scene_id'], family=spec['family'],
            changed_pixels=int(changed.sum()), changed_fraction=float(changed.mean()),
            mean_absolute_channel_difference=float(delta.mean()),
            max_channel_difference=int(delta.max()), identical_policy_rgb=bool(not changed.any()),
            rear_changed_pixels=int(rear_changed.sum()),rear_changed_fraction=float(rear_changed.mean()),
            rear_identical=bool(not rear_changed.any()),front_ambiguity_resolved=bool(not changed.any() and rear_changed.any())))
    result = dict(split='train', test_scenes_loaded=0, model_updates=0,
        intervention='remove far-side pillars while holding all other scene parameters fixed',
        pairs=len(rows), identical_rgb_pairs=sum(r['identical_policy_rgb'] for r in rows),
        front_identical_rear_distinct_pairs=sum(r['front_ambiguity_resolved'] for r in rows),
        nearly_identical_pairs_under_0_1_percent_changed=sum(r['changed_fraction'] < .001 for r in rows),
        rows=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('runs/hjepa/diverse_lora_trainval'))
    p.add_argument('--output', type=Path, default=Path('runs/hjepa/lora_continued_model/visibility_audit.json'))
    args = p.parse_args()
    audit(args.data, args.output)
