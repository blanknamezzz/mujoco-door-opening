"""Collect physical RGB/state/action trajectories and staged counterfactuals."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from hjepa_scene import (BASE_CANDIDATES, DoorScene, action_at, base_features,
                         grasp_features, grasps, proprio, scene_set, task_reward)
from perception import observe, estimate_frame


def small_rgb(frame, size, square_camera=False):
    image = Image.fromarray(frame['rgb'])
    if square_camera:
        width, height = image.size
        side = min(width, height)
        left, top = (width-side)//2, (height-side)//2
        image = image.crop((left, top, left+side, top+side))
    return np.asarray(image.resize((size, size)))


def collect(output, scenes=12, seed=42, bases=5, grasp_count=4, image_size=224, resume=False,
            observation_interval=.1, mmap_pixels=True, structural=False, diverse=False,
            train_per_family=16, val_per_family=2, test_per_family=2, shard_index=0, shard_count=1):
    if not 1 <= bases <= 5 or not 1 <= grasp_count <= 4:
        raise ValueError('Use 1..5 bases and 1..4 grasps.')
    output = Path(output)
    if (output/'manifest.json').exists():
        raise FileExistsError('Use a new output directory to preserve the previous dataset.')
    output.mkdir(parents=True, exist_ok=True)
    if diverse:
        structural = True
    if structural:
        from structural_scene import structural_training_scenes, diverse_structural_scenes, STRUCTURAL_BASES
        specs = (diverse_structural_scenes(seed, train_per_family, val_per_family, test_per_family)
                 if diverse else structural_training_scenes(seed))
        base_pool = STRUCTURAL_BASES
        if bases != 5 or grasp_count != 4:
            raise ValueError('Structural training uses four physical bases plus refusal, and four grasps.')
    else:
        specs, base_pool = scene_set(scenes, seed), BASE_CANDIDATES[:bases]
    specs = json.loads(json.dumps(specs))  # normalize tuples for exact resume matching
    if not 0 <= shard_index < shard_count:
        raise ValueError('Require 0 <= shard index < shard count.')
    specs = specs[shard_index::shard_count]
    manifest = dict(schema_version=2 if mmap_pixels else 1, seed=seed, image_size=image_size,
                    base_budget=bases, grasp_budget=grasp_count,
                    observation_interval_s=observation_interval, upstream='https://github.com/kevinghst/H-JEPA',
                    scenes=specs, episodes=[], structural=structural,
                    candidate_base_poses=[list(b) for b in base_pool],
                    positioning_abstention=structural, abstention_reward=0. if structural else None,
                    positioning='reset placement; no simulated navigation',
                    dynamics='physical contact; no weld or no-slip constraint')
    partial = output/'manifest.partial.json'
    if partial.exists():
        if not resume:
            raise FileExistsError('Interrupted dataset exists; use --resume or a new output directory.')
        previous = json.loads(partial.read_text(encoding='utf-8'))
        for key in ['seed', 'image_size', 'scenes', 'observation_interval_s', 'structural']:
            if previous.get(key, False) != manifest[key]:
                raise ValueError(f'Cannot resume with changed {key}.')
        for key, budget in [('base_budget', bases), ('grasp_budget', grasp_count)]:
            if key in previous and previous[key] != budget:
                raise ValueError(f'Cannot resume with changed {key}.')
        manifest['episodes'] = previous['episodes']
    completed = {ep['file'] for ep in manifest['episodes']}
    for spec in manifest['scenes']:
        if structural:
            from structural_scene import structural_scene_from_spec
            scene = structural_scene_from_spec(spec)
        else:
            scene = DoorScene(**{k: v for k, v in spec.items() if k != 'split'})
        initial = scene.experiment()
        initial_frame = observe(initial)
        for bi, base in enumerate(base_pool):
            reached = scene.experiment(base)
            # A new render after base placement is mandatory. Never reuse the
            # initial handle estimate for the grasp policy's observation.
            arrival_frame = observe(reached)
            try:
                estimate = estimate_frame(arrival_frame)
                candidates = grasps(estimate, reached.grasp_yaw_offset, reached.grasp_yaw_sign)[:grasp_count]
                perception_error = None
            except ValueError as error:
                candidates = [(None, None)]*grasp_count
                perception_error = str(error)
            for gi, (point, yaw) in enumerate(candidates):
                name = f'{scene.scene_id}_b{bi}_g{gi}.npz'
                if name in completed:
                    continue
                exp = scene.experiment(base)
                pixels, states, actions, times = [], [], [], []
                with mujoco.Renderer(exp.m, height=image_size, width=image_size) as renderer:
                    def record(e, terminal):
                        # Terminal states closer than one full interval are kept
                        # in metrics, not treated as .5-second model transitions.
                        if terminal and (not times or e.d.time-times[-1] < observation_interval-1e-6):
                            return
                        if times and e.d.time <= times[-1]+1e-6:
                            return
                        renderer.update_scene(e.d, camera='perception')
                        pixels.append(renderer.render().copy())
                        states.append(proprio(e))
                        times.append(float(e.d.time))
                        actions.append(action_at(e.d.time, point, yaw, interval=observation_interval,
                                                 opening_direction=e.opening_direction))
                    if point is None:
                        result, rows = dict(outcome='B_perception_failed', grasp_acquired=False,
                                            stop_reason='perception_failure', final_angle_deg=0.,
                                            max_angle_deg=0., perception_error=perception_error), []
                    else:
                        result, rows, _ = exp.trial(point, yaw, observer=record, reset_before_trial=False,
                                                  observation_interval=observation_interval)
                metrics = task_reward(result, rows)
                pixel_array = np.asarray(pixels, dtype=np.uint8).reshape(-1, image_size, image_size, 3)
                pixels_file = None
                if mmap_pixels and len(pixel_array):
                    pixels_file = name.replace('.npz', '.pixels.npy')
                    np.save(output/pixels_file, pixel_array)
                # Unreachable trials still contribute negative policy examples.
                np.savez_compressed(output/name,
                    pixels=pixel_array[:0] if mmap_pixels else pixel_array,
                    proprio=np.asarray(states, dtype=np.float32).reshape(-1, len(proprio(exp))),
                    action=np.asarray(actions, dtype=np.float32).reshape(-1, 6),
                    time_s=np.asarray(times),
                    initial_rgb=small_rgb(initial_frame, image_size, image_size == 224), initial_proprio=proprio(initial),
                    arrival_rgb=small_rgb(arrival_frame, image_size, image_size == 224), arrival_proprio=proprio(reached),
                    base=base_features(base), grasp=grasp_features(point, yaw) if point is not None else np.zeros(5, np.float32))
                episode = dict(file=name, scene_id=scene.scene_id, split=spec['split'],
                               base_id=bi, grasp_id=gi, base_pose=list(base),
                               point_world_m=point.tolist() if point is not None else None,
                               yaw_offset_rad=yaw, grasp_observed=point is not None,
                               frames=len(times), pixels_file=pixels_file, result=result, **metrics)
                episode.update(positioning_only=False, operating_side=exp.operating_side,
                               opening_direction=exp.opening_direction,
                               current_reachable=(point is not None and
                                   exp.ik(point, np.pi/2+yaw) is not None and
                                   exp.ik(point-.16*np.array([np.cos(np.pi/2+yaw), np.sin(np.pi/2+yaw), 0.]), np.pi/2+yaw) is not None))
                manifest['episodes'].append(episode)
                print(f'{name}: {result["outcome"]}, reward={metrics["reward"]:.3f}', flush=True)
        if structural:
            # An explicit no-execution action. Its fixed zero return does not
            # consult an oracle about whether some other action could succeed.
            name = f'{scene.scene_id}_reject.npz'
            if name not in completed:
                np.savez_compressed(output/name, pixels=np.empty((0, image_size, image_size, 3), np.uint8),
                    proprio=np.empty((0, len(proprio(initial))), np.float32), action=np.empty((0, 6), np.float32),
                    time_s=np.empty(0), initial_rgb=small_rgb(initial_frame, image_size, image_size == 224),
                    initial_proprio=proprio(initial), arrival_rgb=small_rgb(initial_frame, image_size, image_size == 224),
                    arrival_proprio=proprio(initial), base=np.zeros(4, np.float32), grasp=np.zeros(5, np.float32))
                manifest['episodes'].append(dict(file=name, scene_id=scene.scene_id, split=spec['split'],
                    base_id=4, grasp_id=0, frames=0, pixels_file=None, positioning_only=True, grasp_observed=False,
                    result=dict(outcome='E_policy_abstained', stop_reason='policy_abstention'),
                    reward=0., safe_opening_fraction=0., success=False, collision=False,
                    shortfall=1., milestones=[False, False, False]))
        (output/'manifest.partial.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('runs/hjepa/official_data'))
    p.add_argument('--scenes', type=int, default=12)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--bases', type=int, default=5)
    p.add_argument('--grasps', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--observation-interval', type=float, default=.1)
    p.add_argument('--compressed-pixels', action='store_true', help='Legacy compact NPZ pixel storage')
    p.add_argument('--structural', action='store_true', help='Collect scene-disjoint two-sided structural training data')
    p.add_argument('--diverse', action='store_true', help='Expanded geometry, camera and lighting suite (implies structural)')
    p.add_argument('--train-per-family', type=int, default=16)
    p.add_argument('--val-per-family', type=int, default=2)
    p.add_argument('--test-per-family', type=int, default=2)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    args = p.parse_args()
    collect(args.output, args.scenes, args.seed, args.bases, args.grasps, image_size=args.image_size,
            resume=args.resume, observation_interval=args.observation_interval,
            mmap_pixels=not args.compressed_pixels, structural=args.structural, diverse=args.diverse,
            train_per_family=args.train_per_family, val_per_family=args.val_per_family, test_per_family=args.test_per_family,
            shard_index=args.shard_index, shard_count=args.shard_count)
