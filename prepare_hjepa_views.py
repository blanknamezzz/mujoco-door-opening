"""Render an extra current sensor for existing physical action tables."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import numpy as np
from collect_hjepa import small_rgb
from hjepa_scene import proprio
from perception import observe
from position_observation import observe_rear_camera
from structural_scene import structural_scene_from_spec


def front_alignment(recorded,rendered):
    delta=np.abs(rendered.astype(np.int16)-recorded.astype(np.int16))
    changed=np.any(delta,axis=-1)
    return dict(exact_front_match=bool(not delta.any()),changed_pixels=int(changed.sum()),
        max_channel_error=int(delta.max()),changed_fraction=float(changed.mean()),
        acceptable=bool(delta.max()<=1 and changed.mean()<=.001))


def prepare(source,output):
    started=time.monotonic()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Fresh sensor-data output required.')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('position_rear_files'):raise ValueError('Dataset already has rear sensors.')
    output.mkdir(parents=True,exist_ok=True)
    result=deepcopy(manifest);result['position_rear_files']={}
    first={};alignment=[]
    for e in manifest['episodes']:first.setdefault(e['scene_id'],e)
    for spec in manifest['scenes']:
        exp=structural_scene_from_spec(spec).experiment()
        with np.load(source/first[spec['scene_id']]['file'],allow_pickle=False) as arrays:
            mismatches=[]
            for attempt in range(3):
                front=small_rgb(observe(exp),manifest['image_size'],True)
                check=front_alignment(arrays['initial_rgb'],front)
                if check['acceptable']:break
                mismatches.append(check)
            if not check['acceptable']:
                failure=dict(scene_id=spec['scene_id'],render_mismatches=mismatches)
                (output/'alignment_failure.json').write_text(json.dumps(failure,indent=2))
                raise AssertionError('Re-rendered front differs from recorded observation: '+json.dumps(failure))
            np.testing.assert_array_equal(proprio(exp),arrays['initial_proprio'])
            alignment.append(dict(scene_id=spec['scene_id'],**check,retried_render_mismatches=mismatches))
        rear=small_rgb(observe_rear_camera(exp),manifest['image_size'],True)
        file=spec['scene_id']+'.rear.npy';np.save(output/file,rear)
        result['position_rear_files'][spec['scene_id']]=file
    for e in result['episodes']:
        for field in ('file','pixels_file'):
            if e.get(field):e[field]=Path(os.path.relpath((source/e[field]).resolve(),output.resolve())).as_posix()
    result.update(position_views_available=['front','rear'],position_sensor_contract='two fixed calibrated views before placement; robot state unchanged',
        source_manifest=str(source/'manifest.json'),view_preparation_seconds=time.monotonic()-started,
        front_alignment_checks=alignment,front_alignment_tolerance='max channel error <=1/255 and <=0.1% pixels changed; proprioception exact')
    (output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(scenes=len(result['scenes']),elapsed_seconds=result['view_preparation_seconds'],source=str(source)),indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('runs/hjepa/diverse_lora_trainval'))
    p.add_argument('--output',type=Path,default=Path('runs/hjepa/paired_view_trainval20261026'))
    a=p.parse_args();prepare(a.source,a.output)
