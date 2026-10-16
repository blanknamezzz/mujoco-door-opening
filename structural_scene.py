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

    def __post_init__(self):
        if not -90 <= self.hinge_min_deg <= 0 <= self.hinge_max_deg <= 90 or self.hinge_min_deg == self.hinge_max_deg:
            raise ValueError('Hinge range must include zero and lie within +/-90 degrees.')
        if not self.handle_faces or set(self.handle_faces)-{'front', 'back'} or len(set(self.handle_faces)) != len(self.handle_faces):
            raise ValueError('Handle faces must be a nonempty unique front/back subset.')

    @property
    def base_candidates(self):
        return STRUCTURAL_BASES

    def experiment(self, base=STRUCTURAL_BASES[0]):
        side = 'front' if base[1] < 0 else 'back'
        tree = ET.parse(ROOT/'scene.xml')
        root = tree.getroot()
        world = root.find('worldbody')
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
