"""Serializable grasp-and-opening plans for the existing fixed-base controller.

A plan describes an attempted motion, not a feasibility or stability guarantee.
Positions are metres in the initial world frame; the door initially is closed.
"""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class SkillTiming:
    settle_s: float = .5
    approach_s: float = 2.
    close_s: float = 1.
    close_motion_s: float = .6
    open_s: float = 10.
    hold_s: float = 2.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive.')
        if self.close_s < self.close_motion_s + .2 - 1e-9:
            raise ValueError('close_s must allow close_motion_s plus at least .2 s to establish contact.')
        if self.hold_s < 1:
            raise ValueError('hold_s must be at least 1 s for the success check.')

    @property
    def close_start(self):
        return self.settle_s + self.approach_s

    @property
    def open_start(self):
        return self.close_start + self.close_s

    @property
    def hold_start(self):
        return self.open_start + self.open_s

    @property
    def end(self):
        return self.hold_start + self.hold_s


@dataclass(frozen=True)
class GraspPlan:
    handle_center_world_m: tuple[float, float, float]
    height_offset_m: float = 0.
    approach_angle_deg: float = 0.
    ik_branch: int = -1
    approach_distance_m: float = .16
    grip_force_limit_N: float = 35.
    opening_angle_deg: float = 90.
    timing: SkillTiming = field(default_factory=SkillTiming)

    def __post_init__(self):
        if len(self.handle_center_world_m) != 3 or not all(math.isfinite(v) for v in self.handle_center_world_m):
            raise ValueError('handle_center_world_m must contain three finite coordinates.')
        object.__setattr__(self, 'handle_center_world_m', tuple(self.handle_center_world_m))
        if self.ik_branch not in (-1, 1):
            raise ValueError('ik_branch must be -1 or +1.')
        if not isinstance(self.timing, SkillTiming):
            raise ValueError('timing must be a SkillTiming.')
        for name in ('height_offset_m', 'approach_angle_deg', 'approach_distance_m',
                     'grip_force_limit_N', 'opening_angle_deg'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite.')
        if self.approach_distance_m <= 0:
            raise ValueError('approach_distance_m must be positive.')
        if not 0 < self.grip_force_limit_N <= 35:
            raise ValueError('grip_force_limit_N must be in (0, 35], the original per-finger actuator limit.')
        if not 2 < self.opening_angle_deg <= 90:
            raise ValueError('opening_angle_deg must be in (2, 90].')

    @property
    def point_world_m(self):
        x, y, z = self.handle_center_world_m
        return (x, y, z + self.height_offset_m)

    def yaw_offset_rad(self, experiment):
        return experiment.grasp_yaw_offset + experiment.grasp_yaw_sign * math.radians(self.approach_angle_deg)

    def to_dict(self):
        data = asdict(self)
        data['handle_center_world_m'] = list(self.handle_center_world_m)
        return dict(schema_version=1, trajectory='door_hinge_arc', **data)

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        if data.pop('schema_version', None) != 1 or data.pop('trajectory', None) != 'door_hinge_arc':
            raise ValueError('Expected grasp plan schema_version=1 and trajectory=door_hinge_arc.')
        data['timing'] = SkillTiming(**data.get('timing', {}))
        return cls(**data)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + '\n', encoding='utf-8')
