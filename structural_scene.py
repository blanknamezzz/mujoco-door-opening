"""Physical door obstructions, signed swings and one/two-sided handle access.

No policy is given a permitted-side label or future collision outcome. Scene
metadata builds geometry; learned selectors receive only observations/actions.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from hjepa_scene import DoorScene
from simulate import Experiment, ROOT


@dataclass(frozen=True)
class Pillar:
    x: float
    y: float
    radius: float = .045
    height: float = 1.5

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.x, self.y, self.radius, self.height)):
            raise ValueError('Pillar dimensions must be finite.')
        if self.radius <= 0 or self.height <= 0:
            raise ValueError('Pillar dimensions must be positive.')


# Same number of candidates for all structural experiments, with both sides
# available. This is an explicit OOD pool relative to the front-only training.
STRUCTURAL_BASES = ((.6, -1., math.pi/2), (.45, -1.05, math.pi/2-.15),
                    (.6, 1., -math.pi/2), (.45, 1.05, -math.pi/2+.15))


@dataclass(frozen=True)
class StructuralDoor(DoorScene):
    hinge_min_deg: float = -90.
    hinge_max_deg: float = 90.
    handle_faces: tuple = ('front', 'back')
    pillars: tuple = ()
    camera_azimuth_deg: float = 0.
    camera_height_offset: float = 0.
    light_scale: float = 1.

    def __post_init__(self):
        if not -90 <= self.hinge_min_deg <= 0 <= self.hinge_max_deg <= 90 or self.hinge_min_deg == self.hinge_max_deg:
            raise ValueError('Hinge range must include zero and lie within +/-90 degrees.')
        if not self.handle_faces or set(self.handle_faces)-{'front', 'back'} or len(set(self.handle_faces)) != len(self.handle_faces):
            raise ValueError('Handle faces must be a nonempty unique front/back subset.')
        if not all(math.isfinite(v) for v in (self.camera_azimuth_deg, self.camera_height_offset, self.light_scale)) or self.light_scale <= 0:
            raise ValueError('Camera and lighting parameters must be finite, with positive light scale.')

    @property
    def base_candidates(self):
        return STRUCTURAL_BASES

    def experiment(self, base=STRUCTURAL_BASES[0]):
        side = 'front' if base[1] < 0 else 'back'
        tree = ET.parse(ROOT/'scene.xml')
        root = tree.getroot()
        world = root.find('worldbody')
        world.find('light').set('diffuse', ' '.join([str(.7*self.light_scale)]*3))
        door = world.find("body[@name='door']")
        door.find("joint[@name='door_hinge']").set('range',
            f'{math.radians(self.hinge_min_deg)} {math.radians(self.hinge_max_deg)}')
        front = door.find("body[@name='handle']")
        front.set('pos', f'{self.handle_x} -.11 {self.handle_z}')
        if 'back' in self.handle_faces:
            back = deepcopy(front)
            back.set('name', 'back_handle')
            back.set('pos', f'{self.handle_x} .11 {self.handle_z}')
            for geom in back.findall('geom'):
                geom.set('name', 'back_'+geom.get('name'))
                coords = [float(x) for x in geom.get('fromto').split()]
                coords[1], coords[4] = -coords[1], -coords[4]
                geom.set('fromto', ' '.join(map(str, coords)))
            door.append(back)
        if 'front' not in self.handle_faces:
            door.remove(front)
        for i, pillar in enumerate(self.pillars):
            ET.SubElement(world, 'geom', name=f'pillar_{i}', type='cylinder',
                          pos=f'{pillar.x} {pillar.y} {pillar.height/2}',
                          size=f'{pillar.radius} {pillar.height/2}', rgba='.48 .50 .55 1')
        active = side if side in self.handle_faces else self.handle_faces[0]
        exp = Experiment(base=base[:2], base_yaw=base[2], friction=self.friction,
            hinge_resistance=self.hinge_resistance, model_xml=ET.tostring(root, encoding='unicode'),
            ik_branch=-1 if side == 'front' else 1, opening_direction=1 if side == 'front' else -1,
            handle_body='handle' if active == 'front' else 'back_handle',
            handle_geom='handle_bar' if active == 'front' else 'back_handle_bar', operating_side=side)
        # The arrival camera moves to the operated face, independently of handle
        # truth. Its calibration is read from MuJoCo by observe(). Navigation is
        # still a reset, not a learned mobile-base controller.
        if side == 'back':
            cid = exp.m.camera('perception').id
            exp.m.cam_pos[cid] = [.8, 2.2, .85]
            right, up, backward = [-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]
            mujoco.mju_mat2Quat(exp.m.cam_quat[cid], np.column_stack([right, up, backward]).ravel())
            mujoco.mj_forward(exp.m, exp.d)
        if self.camera_azimuth_deg or self.camera_height_offset:
            cid = exp.m.camera('perception').id
            target = np.array([.8, 0., .85])
            offset = exp.m.cam_pos[cid]-target
            angle = math.radians(self.camera_azimuth_deg)
            c, s = math.cos(angle), math.sin(angle)
            offset[:2] = np.array([[c, -s], [s, c]])@offset[:2]
            offset[2] += self.camera_height_offset
            exp.m.cam_pos[cid] = target+offset
            backward = offset/np.linalg.norm(offset)
            right = np.cross([0., 0., 1.], backward)
            right /= np.linalg.norm(right)
            up = np.cross(backward, right)
            mujoco.mju_mat2Quat(exp.m.cam_quat[cid], np.column_stack([right, up, backward]).ravel())
            mujoco.mj_forward(exp.m, exp.d)
        return exp


def pillar_at(angle_deg, radius=.4):
    theta = math.radians(angle_deg)
    return Pillar(radius*math.cos(theta), -radius*math.sin(theta))


def structural_scenes():
    return {
        'free-two-way': StructuralDoor('free-two-way'),
        'pillar-front': StructuralDoor('pillar-front', pillars=(pillar_at(50),)),
        'pillar-back': StructuralDoor('pillar-back', pillars=(pillar_at(-50),)),
        'pillar-both': StructuralDoor('pillar-both', pillars=(pillar_at(50), pillar_at(-50))),
        'one-way-positive': StructuralDoor('one-way-positive', hinge_min_deg=0.),
        'one-way-negative': StructuralDoor('one-way-negative', hinge_max_deg=0.),
        'front-access-only': StructuralDoor('front-access-only', hinge_min_deg=0., handle_faces=('front',)),
        'back-access-only': StructuralDoor('back-access-only', hinge_max_deg=0., handle_faces=('back',)),
        'front-only-blocked': StructuralDoor('front-only-blocked', hinge_min_deg=0., handle_faces=('front',),
                                            pillars=(pillar_at(50),)),
    }


def structural_scene_from_spec(spec):
    fields = {k: v for k, v in spec.items() if k not in ('split', 'family')}
    fields['handle_faces'] = tuple(fields.get('handle_faces', ('front', 'back')))
    fields['pillars'] = tuple(Pillar(**p) if isinstance(p, dict) else p for p in fields.get('pillars', ()))
    return StructuralDoor(**fields)


def structural_training_scenes(seed=20261017):
    """Two train, one validation and one untouched test variant per family.

    Hidden hinge-only direction changes are deliberately not supervised from
    an identical closed image. Single-sided access provides observable cues.
    """
    from dataclasses import asdict
    rng = np.random.default_rng(seed)
    specs = []
    for family in ['free-two-way', 'pillar-front', 'pillar-back', 'pillar-both',
                   'front-access-only', 'back-access-only']:
        for variant, split in enumerate(['train', 'train', 'val', 'test']):
            front = pillar_at(float(rng.uniform(40, 70)), float(rng.uniform(.36, .5)))
            back = pillar_at(float(rng.uniform(-70, -40)), float(rng.uniform(.36, .5)))
            pillars = ((front,) if family == 'pillar-front' else (back,) if family == 'pillar-back'
                       else (front, back) if family == 'pillar-both' else ())
            scene = StructuralDoor(f'struct_{seed}_{family}_{variant}',
                handle_x=float(rng.uniform(.76, .84)), handle_z=float(rng.uniform(.835, .865)),
                friction=float(rng.uniform(.65, .9)), hinge_resistance=float(rng.uniform(.9, 1.4)),
                hinge_min_deg=0. if family == 'front-access-only' else -90.,
                hinge_max_deg=0. if family == 'back-access-only' else 90.,
                handle_faces=('front',) if family == 'front-access-only' else
                             ('back',) if family == 'back-access-only' else ('front', 'back'), pillars=pillars)
            specs.append(dict(**asdict(scene), split=split, family=family))
    return specs


def diverse_structural_scenes(seed=20261018, train_per_family=16, val_per_family=2, test_per_family=2, stream=None):
    """Scene-disjoint geometry/view randomization; no outcome labels as inputs.

    The old 24 scenes are not reused. Outside-arc posts teach that a visible
    post need not obstruct a door. Low posts test occlusion, and single-handle
    blocked cases provide physically unsolvable examples. Hidden hinge-only
    controls remain excluded: identical RGB cannot reveal their constraint.
    """
    from dataclasses import asdict, replace
    if min(train_per_family, val_per_family, test_per_family) < 1:
        raise ValueError('Each split needs at least one scene per family.')
    families = ['free-two-way', 'pillar-front', 'pillar-back', 'pillar-both',
                'front-access-only', 'back-access-only', 'front-only-blocked',
                'back-only-blocked', 'outside-arc', 'mixed-posts', 'low-front', 'low-back']
    # Explicit named streams permit disjoint future cohorts while preserving
    # the user's global seed42. The default path is bit-for-bit historical.
    if stream is not None:
        import hashlib
        import re
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',stream):
            raise ValueError('Scene stream needs 1..64 letters, numbers, underscores or hyphens.')
        digest=hashlib.sha256(stream.encode('utf-8')).digest()
        spawn_key=tuple(int.from_bytes(digest[i:i+4],'little') for i in range(0,16,4))
        rng=np.random.default_rng(np.random.SeedSequence(seed,spawn_key=spawn_key))
    else:
        rng = np.random.default_rng(seed)
    identity=f'{seed}_{stream}' if stream is not None else str(seed)
    specs = []
    splits = ['train']*train_per_family+['val']*val_per_family+['test']*test_per_family
    for family in families:
        for variant, split in enumerate(splits):
            def post(sign=1, low=False):
                p = pillar_at(sign*float(rng.uniform(25, 78)), float(rng.uniform(.3, .68)))
                return replace(p, radius=float(rng.uniform(.028, .075)),
                               height=float(rng.uniform(.45, .95) if low else rng.uniform(1.35, 1.9)))
            front, back = post(), post(-1)
            outside = replace(pillar_at(float(rng.uniform(-75, 75)), float(rng.uniform(1.08, 1.4))),
                              radius=float(rng.uniform(.03, .07)), height=float(rng.uniform(.7, 1.9)))
            pillars = {'pillar-front': (front,), 'pillar-back': (back,), 'pillar-both': (front, back),
                'front-only-blocked': (front,), 'back-only-blocked': (back,), 'outside-arc': (outside,),
                'mixed-posts': ((front if variant % 2 == 0 else back), outside),
                'low-front': (post(low=True),), 'low-back': (post(-1, low=True),)}.get(family, ())
            front_only = family in ('front-access-only', 'front-only-blocked')
            back_only = family in ('back-access-only', 'back-only-blocked')
            scene = StructuralDoor(f'diverse_{identity}_{family}_{variant:03d}',
                handle_x=float(rng.uniform(.74, .85)), handle_z=float(rng.uniform(.82, .88)),
                friction=float(rng.uniform(.55, .95)), hinge_resistance=float(rng.uniform(.8, 1.6)),
                hinge_min_deg=0. if front_only else -90., hinge_max_deg=0. if back_only else 90.,
                handle_faces=('front',) if front_only else ('back',) if back_only else ('front', 'back'),
                pillars=pillars, camera_azimuth_deg=float(rng.uniform(-12, 12)),
                camera_height_offset=float(rng.uniform(-.08, .18)), light_scale=float(rng.uniform(.75, 1.2)))
            specs.append(dict(**asdict(scene), split=split, family=family))
    return specs


def independent_test_scenes(seed=20261019, per_family=4, stream=None):
    """New seed and geometry, test-only; never appended to a training manifest."""
    specs = diverse_structural_scenes(seed, 1, 1, per_family,stream=stream)
    identity=f'{seed}_{stream}' if stream is not None else str(seed)
    result = []
    counters = {}
    for spec in specs:
        if spec['split'] != 'test':
            continue
        family = spec['family']
        index = counters.get(family, 0)
        counters[family] = index+1
        result.append(dict(spec, scene_id=f'holdout_{identity}_{family}_{index:03d}'))
    return result
