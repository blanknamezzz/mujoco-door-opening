"""Door task contract: scene splits, staged candidates and downstream reward.

Base placement is a simulator reset, not a mobile navigation controller. Opening
always uses the existing physical contact controller with a locked base.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class DoorScene:
    scene_id: str
    handle_x: float = .8
    handle_z: float = .85
    friction: float = .8
    hinge_resistance: float = 1.2

    def experiment(self, base=(.6, -1., math.pi/2)):
        from simulate import Experiment
        return Experiment(base=base[:2], base_yaw=base[2], friction=self.friction,
                          hinge_resistance=self.hinge_resistance,
                          handle_position=(self.handle_x, -.11, self.handle_z))


BASE_CANDIDATES = ((.6, -1., math.pi/2), (.3, -.9, math.pi/2),
                   (.8, -1., math.pi/2), (.45, -1.05, math.pi/2-.15),
                   (2., -1.2, math.pi/2))
GRASP_OFFSETS = ((0., 0.), (-.07, 0.), (.07, 0.), (0., math.radians(15)))


def grasps(estimate, yaw_origin=0., yaw_sign=1):
    center = np.asarray(estimate['center_world_m'])
    return [(center+np.array([0., 0., dz]), yaw_origin+yaw_sign*yaw) for dz, yaw in GRASP_OFFSETS]


def base_features(base):
    return np.array([*base[:2], math.sin(base[2]), math.cos(base[2])], np.float32)


def grasp_features(point, yaw_offset):
    return np.array([*point, math.sin(yaw_offset), math.cos(yaw_offset)], np.float32)


def proprio(exp):
    # Simulator joint/door state is an explicit privileged state baseline.
    # No future state, collision checks, material constants or outcomes are inputs.
    return np.array([*exp.d.qpos, *exp.d.qvel, *exp.base,
                     math.sin(exp.base_yaw), math.cos(exp.base_yaw), exp.d.time/15.5], np.float32)


def action_at(t, point, yaw_offset, interval=.5, opening_angle_deg=90., opening_direction=1):
    """Mean Cartesian command over the next observation interval (6 values).

    Unlike an IK path check this feature does not reveal future feasibility.
    The schedule is deterministic; execution still integrates every 1 ms.
    """
    from simulate import skill_target
    commands = []
    for time in np.linspace(t, t+interval, 10, endpoint=False):
        target, yaw, aperture, _ = skill_target(time, point, yaw_offset, opening_angle_deg=opening_angle_deg,
                                               opening_direction=opening_direction)
        commands.append([*target, math.sin(yaw), math.cos(yaw), aperture/.04])
    return np.mean(commands, axis=0).astype(np.float32)


def lookahead_steps(fraction, interval=.5):
    """Include preparation, then reach the requested fraction of the door arc."""
    if fraction == 0:
        return 0
    if fraction == 1:
        return math.ceil(15.5/interval)  # full opening plus hold
    if not 0 < fraction < 1:
        raise ValueError('Lookahead fraction must be in [0, 1].')
    lo, hi = 0., 1.
    for _ in range(40):
        mid = (lo+hi)/2
        if mid*mid*(3-2*mid) < fraction:
            lo = mid
        else:
            hi = mid
    return math.ceil((3.5+10*(lo+hi)/2)/interval)


def task_reward(result, rows, collision_weight=5., incomplete_weight=3.,
                full_angle=90., duration=15.5):
    """Slide 3 reward; shortfall is paid once using the FINAL angle.

    Position = reachable reset. Alignment = completed approach without a prior
    collision. Grasp = acquired stable contact, gated by earlier milestones.
    Safe progress freezes at the first collision, slip or contact loss.
    """
    positioned = result['outcome'] != 'A_initial_unreachable'
    aligned = positioned and any(r['time_s'] >= 2.5 for r in rows)
    aligned = aligned and not any(r['unexpected_contact'] for r in rows if r['time_s'] <= 2.5)
    grasped = aligned and result.get('grasp_acquired', False)
    collision = any(bool(r['unexpected_contact']) for r in rows)
    safe_angle = 0.
    for row in rows:
        if row['unexpected_contact'] or row['slip_detected'] or row['contact_lost']:
            break
        safe_angle = max(safe_angle, row['door_deg'])
    progress = float(np.clip(safe_angle/full_angle, 0, 1))
    shortfall = float(np.clip((full_angle-result.get('final_angle_deg', 0.))/full_angle, 0, 1))
    elapsed = rows[-1]['time_s'] if rows else 0.
    success = result['outcome'] in ('D_full_open', 'D_target_open')
    reward = (10*success + positioned + aligned + grasped + 2*progress
              - collision_weight*collision - incomplete_weight*shortfall
              - .1*min(elapsed/duration, 1.))
    return dict(reward=float(reward), safe_opening_fraction=progress,
                success=bool(success), collision=bool(collision), shortfall=shortfall,
                milestones=[bool(positioned), bool(aligned), bool(grasped)])


def scene_set(count, seed):
    if count < 3:
        raise ValueError('Need at least 3 scenes for disjoint train/val/test splits.')
    rng = np.random.default_rng(seed)
    scenes = [DoorScene('scene_0000')]
    for i in range(1, count):
        scenes.append(DoorScene(f'scene_{i:04d}', handle_x=float(rng.uniform(.73, .84)),
                               handle_z=float(rng.uniform(.82, .88)),
                               friction=float(rng.uniform(.5, .9)),
                               hinge_resistance=float(rng.uniform(.8, 1.6))))
    # Split door scenes, keeping EVERY base/grasp variant in the same split.
    order = rng.permutation(count)
    nval = max(1, round(.15*count))
    ntest = max(1, round(.15*count))
    assignment = {int(j): 'val' for j in order[:nval]}
    assignment.update({int(j): 'test' for j in order[nval:nval+ntest]})
    return [dict(**asdict(scene), split=assignment.get(i, 'train')) for i, scene in enumerate(scenes)]
