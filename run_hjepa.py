"""Two-stage door preparation with official H-JEPA and fixed skill execution."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
import hjepa_runtime  # process-local CUDA environment fix
import torch
from collect_hjepa import small_rgb
from hjepa_model import load_checkpoint, resolve_device
from hjepa_scene import (BASE_CANDIDATES, GRASP_OFFSETS, DoorScene, action_at,
                         base_features, grasp_features, grasps, lookahead_steps, proprio, task_reward)
from perception import observe, estimate_frame
from simulate import save_trial
from train_hjepa import rgb_tensor, state_tensor


def select_and_execute(model, checkpoint, scene, method='world-model', horizon=1.,
                       base_mode='learned', bases=5, grasp_count=4, execute=True, fixed_base_id=0):
    if horizon not in (0., .25, .5, 1.):
        raise ValueError('Lookahead must be 0, .25, .5 or 1.')
    if not 1 <= bases <= 5 or not 1 <= grasp_count <= 4:
        raise ValueError('Invalid candidate budget.')
    if bases > checkpoint.get('base_budget', 5) or grasp_count > checkpoint.get('grasp_budget', 4):
        raise ValueError('Candidate budget exceeds the training collection; reduce --bases/--grasps.')
    base_pool = getattr(scene, 'base_candidates', BASE_CANDIDATES)[:bases]
    count = len(base_pool)
    if not 0 <= fixed_base_id < count:
        raise ValueError('Fixed base ID is outside the scene candidate pool.')
    device = next(model.parameters()).device
    mean, std = checkpoint['state_mean'].to(device), checkpoint['state_std'].to(device)
    size = model.config['image_size']

    def encode(exp, frame):
        return model.encode(rgb_tensor(small_rgb(frame, size, model.is_official)[None], device),
                            state_tensor(proprio(exp)[None], mean, std, device))

    start = time.perf_counter()
    initial = scene.experiment()
    initial_frame = observe(initial)
    with torch.no_grad():
        latent = encode(initial, initial_frame)
        features = torch.tensor(np.stack([base_features(b) for b in base_pool]), device=device)
        scores = model.position_policy(torch.cat([latent.expand(count, -1), features], -1)).flatten()
        base_id = int(scores.argmax()) if base_mode == 'learned' else fixed_base_id
    base = base_pool[base_id]
    reached = scene.experiment(base)
    # Predictor is refreshed with a real post-placement frame, not an imagined image.
    arrival = observe(reached)
    try:
        handle = estimate_frame(arrival)
    except ValueError as error:
        result = dict(outcome='B_perception_failed', grasp_acquired=False,
                      stop_reason='perception_failure', final_angle_deg=0.,
                      max_angle_deg=0., perception_error=str(error))
        plan = dict(scene=asdict(scene), method=method, horizon=horizon, base_mode=base_mode,
                    base_id=base_id, base_pose=list(base), grasp_id=None,
                    base_scores=scores.cpu().tolist(), fresh_arrival_observation=True,
                    planning_time_s=time.perf_counter()-start,
                    result=result, metrics=task_reward(result, []),
                    upstream_commit=checkpoint['upstream_commit'])
        return plan, [], []
    candidates = grasps(handle, reached.grasp_yaw_offset, reached.grasp_yaw_sign)[:grasp_count]
    with torch.no_grad():
        latent = encode(reached, arrival)
        features = torch.tensor(np.stack([grasp_features(p, y) for p, y in candidates]), device=device)
        direct = model.grasp_policy(torch.cat([latent.expand(grasp_count, -1), features], -1)).flatten()
        candidate_scores = direct.clone()
        steps = lookahead_steps(horizon)
        if method == 'world-model' and steps:
            skip = model.config.get('frame_skip', 1)
            raw_interval = model.config.get('raw_observation_interval_s', .5)
            actions = torch.tensor(np.stack([
                np.stack([np.concatenate([action_at(i*.5+j*raw_interval, point, yaw, interval=raw_interval,
                                                   opening_direction=reached.opening_direction)
                                          for j in range(skip)]) for i in range(steps)])
                for point, yaw in candidates]), device=device)
            expanded = latent.expand(grasp_count, -1)
            low = model.imagine(expanded, actions, level=1)
            high = model.imagine(expanded, actions, level=2)
            candidate_scores = (model.value_low(low).flatten()+model.value_high(high).flatten())/2
        elif method == 'geometry':
            from planning import choose_grasp
            offsets = [(dz, float(round(math.degrees(yaw), 6))) for dz, yaw in GRASP_OFFSETS[:grasp_count]]
            geometric = choose_grasp(reached, handle, candidate_offsets=offsets)
            candidate_scores = torch.tensor([
                next(c['score'] for c in geometric['candidates']
                     if np.allclose(c['point_world_m'], point) and abs(c['yaw_offset_deg']-math.degrees(yaw)) < 1e-6)
                for point, yaw in candidates], device=device)
    # Only CURRENT reachability is a hard execution screen. Future limits are
    # left to the predictor except in the explicit full-path geometry baseline.
    feasible = []
    for point, yaw in candidates:
        yaw0 = math.pi/2+yaw
        pre = point-.16*np.array([math.cos(yaw0), math.sin(yaw0), 0.])
        feasible.append(reached.ik(point, yaw0) is not None and reached.ik(pre, yaw0) is not None)
    eligible = np.flatnonzero(feasible)
    grasp_id = int(max(eligible, key=lambda i: float(candidate_scores[i]))) if len(eligible) else int(candidate_scores.argmax())
    planning_s = time.perf_counter()-start
    point, yaw = candidates[grasp_id]
    plan = dict(scene=asdict(scene), method=method, horizon=horizon, base_mode=base_mode,
                base_id=base_id, base_pose=list(base), grasp_id=grasp_id,
                point_world_m=point.tolist(), yaw_offset_rad=yaw, planning_time_s=planning_s,
                base_scores=scores.cpu().tolist(), grasp_scores=candidate_scores.cpu().tolist(),
                current_reachability=feasible, fresh_arrival_observation=True,
                placement='simulator reset; navigation not implemented',
                operating_side=reached.operating_side, opening_direction=reached.opening_direction,
                candidate_base_poses=[list(b) for b in base_pool],
                candidate_pool_out_of_training_distribution=hasattr(scene, 'base_candidates'),
                upstream_commit=checkpoint['upstream_commit'], inference_device=str(device))
    if not execute:
        return plan, [], []
    result, rows, states = reached.trial(point, yaw, reset_before_trial=False)
    plan.update(result=result, metrics=task_reward(result, rows))
    return plan, rows, states


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('runs/hjepa/official_model/selected.pt'))
    p.add_argument('--output', type=Path, default=Path('runs/hjepa/official_demo'))
    p.add_argument('--method', choices=['direct', 'world-model', 'geometry'], default='world-model')
    p.add_argument('--horizon', type=float, choices=[0., .25, .5, 1.], default=1.)
    p.add_argument('--base-mode', choices=['fixed', 'learned'], default='learned')
    p.add_argument('--fixed-base-id', type=int, default=0)
    from structural_scene import structural_scenes
    p.add_argument('--scenario', choices=['nominal', *structural_scenes()], default='nominal')
    p.add_argument('--bases', type=int, default=5)
    p.add_argument('--grasps', type=int, default=4)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a = p.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(a.checkpoint, resolve_device(a.device))
    scene = DoorScene('demo') if a.scenario == 'nominal' else structural_scenes()[a.scenario]
    plan, rows, states = select_and_execute(model, checkpoint, scene,
        a.method, a.horizon, a.base_mode, a.bases, a.grasps, fixed_base_id=a.fixed_base_id)
    output = a.output if a.scenario == 'nominal' else a.output/a.scenario
    save_trial('hjepa', plan, rows, states, output)
    print(json.dumps(plan, indent=2))
