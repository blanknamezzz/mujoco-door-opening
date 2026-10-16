"""Matched held-out-scene evaluation of fixed/learned placement and lookahead."""
import argparse
import json
from pathlib import Path
import numpy as np
import hjepa_runtime  # process-local CUDA environment fix
import torch
from hjepa_model import load_checkpoint, resolve_device
from hjepa_scene import DoorScene
from run_hjepa import select_and_execute


def evaluate(data, checkpoint_path, output, methods, horizons, base_modes, bases=5, grasps=4, device='cpu'):
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    train_report = checkpoint_path.parent/'training.json'
    if not train_report.is_file():
        raise ValueError('Missing training.json; cannot verify held-out scenes.')
    training_scenes = json.loads(train_report.read_text(encoding='utf-8'))['scenes']
    seen = {s['scene_id'] for s in training_scenes if s['split'] != 'test'}
    tests = [s for s in manifest['scenes'] if s['split'] == 'test']
    if not tests or any(s['scene_id'] in seen for s in tests):
        raise ValueError('Test scenes missing or overlap training/validation scenes.')
    output.mkdir(parents=True, exist_ok=True)
    trials = []
    for spec in tests:
        scene = DoorScene(**{k: v for k, v in spec.items() if k != 'split'})
        for base_mode in base_modes:
            for method in methods:
                for horizon in horizons if method == 'world-model' else [0.]:
                    plan, _, _ = select_and_execute(model, checkpoint, scene, method, horizon,
                                                    base_mode, bases, grasps)
                    trials.append(plan)
                    print(json.dumps(dict(scene=scene.scene_id, method=method, horizon=horizon,
                                          base_mode=base_mode, **plan['metrics'])), flush=True)
                    (output/'trials.json').write_text(json.dumps(trials, indent=2), encoding='utf-8')
    groups = {}
    for trial in trials:
        key = f'{trial["base_mode"]}/{trial["method"]}/h{trial["horizon"]}'
        groups.setdefault(key, []).append(trial)
    metrics = {}
    for key, group in groups.items():
        metrics[key] = dict(episodes=len(group),
            safe_opening_fraction=float(np.mean([x['metrics']['safe_opening_fraction'] for x in group])),
            full_open_success=float(np.mean([x['metrics']['success'] for x in group])),
            collision_rate=float(np.mean([x['metrics']['collision'] for x in group])),
            joint_limit_stops=sum(x['result']['stop_reason'] == 'arm_workspace_limit' for x in group),
            slip_or_loss_rate=float(np.mean([x['result'].get('slip_detected', False) or
                                            x['result'].get('contact_lost', False) for x in group])),
            mean_planning_time_s=float(np.mean([x['planning_time_s'] for x in group])))
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('runs/hjepa/official_data'))
    p.add_argument('--checkpoint', type=Path, default=Path('runs/hjepa/official_model/selected.pt'))
    p.add_argument('--output', type=Path, default=Path('runs/hjepa/official_evaluation'))
    p.add_argument('--methods', nargs='+', choices=['direct', 'world-model', 'geometry'], default=['direct', 'world-model', 'geometry'])
    p.add_argument('--horizons', nargs='+', type=float, choices=[0., .25, .5, 1.], default=[0., .25, .5, 1.])
    p.add_argument('--base-modes', nargs='+', choices=['fixed', 'learned'], default=['fixed', 'learned'])
    p.add_argument('--bases', type=int, default=5)
    p.add_argument('--grasps', type=int, default=4)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a = p.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(evaluate(a.data, a.checkpoint, a.output, a.methods, a.horizons, a.base_modes, a.bases, a.grasps, resolve_device(a.device)), indent=2))
