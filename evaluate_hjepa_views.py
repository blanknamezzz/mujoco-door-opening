"""Camera viewpoint interventions with frozen official H-JEPA policies."""
import argparse
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import time
import numpy as np
import mujoco
import hjepa_runtime
import torch

from evaluate_hjepa_stages import sha256, summarize, write_json
from hjepa_model import load_checkpoint, resolve_device
from hjepa_scene import DoorScene
from run_hjepa import select_and_execute


@dataclass(frozen=True)
class CameraScene(DoorScene):
    view_yaw_deg: float = 0.
    view_elevation_deg: float = 0.

    def experiment(self, base=(.6, -1., math.pi/2)):
        exp = super().experiment(base)
        cid = exp.m.camera('perception').id
        # Orbit the existing camera about its existing optical-axis target.
        # Target/radius stay fixed across scenes, so the camera does not receive
        # privileged handle localization. Only extrinsics change, with valid K/T.
        target = np.array([.8, -.11, .85])
        yaw, elevation = np.radians([self.view_yaw_deg, self.view_elevation_deg])
        offset = 2.09*np.array([math.sin(yaw)*math.cos(elevation),
                               -math.cos(yaw)*math.cos(elevation), math.sin(elevation)])
        exp.m.cam_pos[cid] = target+offset
        backward = offset/np.linalg.norm(offset)
        right = np.cross([0., 0., 1.], backward)
        right /= np.linalg.norm(right)
        up = np.cross(backward, right)
        mujoco.mju_mat2Quat(exp.m.cam_quat[cid], np.column_stack([right, up, backward]).ravel())
        mujoco.mj_forward(exp.m, exp.d)
        return exp


def evaluate(stage_output, checkpoint_path, output, device='cuda'):
    suite = json.loads((stage_output/'suite.json').read_text())
    scenes = [DoorScene(**c['scene']) for c in suite['cases'] if c['family'] == 'new_in_range'][:2]
    if len(scenes) != 2:
        raise ValueError('View test requires two fresh stage-suite scenes.')
    spec = dict(schema=1, scenes=[asdict(s) for s in scenes], yaw=[-30, -15, 0, 15, 30],
                elevation=[-10, 0, 10], checkpoint_sha256=sha256(checkpoint_path),
                source_sha256={name: sha256(Path(name)) for name in
                    ['evaluate_hjepa_views.py', 'run_hjepa.py', 'evaluate_hjepa_stages.py',
                     'simulate.py', 'perception.py', 'collect_hjepa.py', 'hjepa_model.py']})
    output.mkdir(parents=True, exist_ok=True)
    path = output/'suite.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError('Changed view suite; choose another output directory.')
    write_json(path, spec)
    trials = json.loads((output/'trials.json').read_text()) if (output/'trials.json').exists() else []
    done = {t['key'] for t in trials}
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    start = time.perf_counter()
    from evaluate_hjepa_stages import stage_metrics
    for scene in scenes:
        for elevation in spec['elevation']:
            for yaw in spec['yaw']:
                viewed = CameraScene(**asdict(scene), view_yaw_deg=yaw, view_elevation_deg=elevation)
                for method in ['direct', 'world-model']:
                    key = f'{scene.scene_id}/{elevation}/{yaw}/{method}'
                    if key in done:
                        continue
                    plan, rows, _ = select_and_execute(model, checkpoint, viewed, method=method)
                    plan['metrics'] = stage_metrics(plan['result'], rows)
                    trials.append(dict(key=key, yaw=yaw, elevation=elevation, **plan))
                    write_json(output/'trials.json', trials)
                print(f'Camera {scene.scene_id}: yaw={yaw}, elevation={elevation}', flush=True)
    metrics = dict(by_method={method: summarize([t for t in trials if t['method'] == method])
                              for method in ['direct', 'world-model']},
                   by_view={f'yaw{yaw}/elev{elevation}': {
                       method: summarize([t for t in trials if t['yaw'] == yaw and t['elevation'] == elevation
                                          and t['method'] == method]) for method in ['direct', 'world-model']}
                            for yaw in spec['yaw'] for elevation in spec['elevation']},
                   metadata=dict(physical_trials=len(trials), scene_count=2, camera_views=15,
                                 invocation_seconds=time.perf_counter()-start, device=str(device),
                                 note='Same two new door scenes repeated under views, not 60 independent scenes. '
                                      'Frozen post-update policies; camera pose changed, lighting unchanged.'))
    write_json(output/'metrics.json', metrics)
    print(json.dumps(metrics['by_method'], indent=2))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-output', type=Path, default=Path('runs/hjepa/stage_evaluation'))
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/hjepa/official_model/selected.pt'))
    parser.add_argument('--output', type=Path, default=Path('runs/hjepa/view_evaluation'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.stage_output, args.checkpoint, args.output, resolve_device(args.device))
