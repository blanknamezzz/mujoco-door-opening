"""Compose disjoint policy corpora by reference, without copying trajectories."""
from copy import deepcopy
import json
import os
from pathlib import Path
from prepare_hjepa_splits import scene_fingerprint
from run_hjepa_lora_experiment import hash_file,write_json


def compose(sources,output,seed=42):
    if (output/'manifest.json').exists():raise FileExistsError('Preserve existing policy corpus.')
    output.mkdir(parents=True,exist_ok=True)
    result=None;seen_ids=set();seen_geometry=set();provenance=[]
    invariant_keys=('image_size','base_budget','grasp_budget','observation_interval_s','candidate_base_poses',
        'positioning_abstention','abstention_reward')
    copy_keys=('schema_version','image_size','base_budget','grasp_budget','observation_interval_s','structural','upstream',
        'candidate_base_poses','positioning_abstention','abstention_reward','positioning','dynamics')
    for source,splits in sources:
        source=Path(source);path=source/'manifest.json';m=json.loads(path.read_text(encoding='utf-8'))
        if m.get('evaluation_only') or set(splits)-{'train','val'}:raise ValueError('Only development splits may be composed.')
        if result is None:
            result={k:deepcopy(m[k]) for k in copy_keys if k in m}
            result.update(scenes=[],episodes=[],position_rear_files={})
        elif any(m.get(k)!=result.get(k) for k in invariant_keys):raise ValueError('Incompatible physical/action contracts.')
        selected=[s for s in m['scenes'] if s['split'] in splits];ids={s['scene_id'] for s in selected}
        if seen_ids&ids or seen_geometry&{scene_fingerprint(s) for s in selected}:
            raise ValueError('Repeated scene ID or geometry across policy sources.')
        if not ids<=set(m.get('position_rear_files',{})):raise ValueError('Every selected scene needs actual rear RGB.')
        seen_ids.update(ids);seen_geometry.update(scene_fingerprint(s) for s in selected)
        result['scenes'].extend(deepcopy(selected))
        for sid in ids:
            actual=(source/m['position_rear_files'][sid]).resolve()
            if not actual.is_file():raise FileNotFoundError(actual)
            result['position_rear_files'][sid]=Path(os.path.relpath(actual,output.resolve())).as_posix()
        for ep in m['episodes']:
            if ep['scene_id'] not in ids:continue
            if ep['split'] not in splits:raise ValueError('Episode/scene split mismatch.')
            item=dict(ep)
            for field in ('file','pixels_file'):
                if item.get(field):
                    actual=(source/item[field]).resolve()
                    if not actual.is_file():raise FileNotFoundError(actual)
                    item[field]=Path(os.path.relpath(actual,output.resolve())).as_posix()
            result['episodes'].append(item)
        provenance.append(dict(manifest=str(path),sha256=hash_file(path),source_seed=m.get('seed'),
            selected_splits=list(splits),scenes=len(ids)))
    result.update(seed=seed,seed_role='New experiment seed; historical source collection seeds remain as recorded',
        evaluation_only=False,policy_only=True,temporal_sequences_available=False,
        position_views_available=['front','rear'],source_manifests=provenance,
        split_counts={s:sum(x['split']==s for x in result['scenes']) for s in ('train','val')})
    write_json(output/'manifest.json',result)
    return result
