"""Procedural stress-test scenes for explore-diagnose-retry door opening.

These scenes are intentionally separate from the frozen H-JEPA training and
test splits. They add visibly different doors, handles, walls and clutter while
retaining the existing robot, contact controller and strict opening criterion.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from simulate import Experiment, ROOT


FRONT_BASE = (.6, -1., math.pi/2)
BACK_BASE = (.6, 1., -math.pi/2)
GOLD = '1 .72 .04 1'
METAL = '.38 .42 .47 1'


@dataclass(frozen=True)
class Obstacle:
    """Static world geometry. Prefixes feed the simulator's contact category."""

    name: str
    shape: str
    pos: tuple[float, float, float]
    size: tuple[float, ...]
    rgba: tuple[float, float, float, float]
    euler_deg: tuple[float, float, float] = (0., 0., 0.)

    def __post_init__(self):
        if self.shape not in ('box', 'cylinder'):
            raise ValueError('Obstacle shape must be box or cylinder.')
        if not self.name.startswith(('wall_', 'clutter_', 'barrel_', 'obstacle_')):
            raise ValueError('Obstacle names must use a collision-category prefix.')
        if len(self.pos) != 3 or not all(math.isfinite(v) for v in (*self.pos, *self.size)):
            raise ValueError('Obstacle position and dimensions must be finite.')
        if len(self.euler_deg) != 3 or not all(math.isfinite(v) for v in self.euler_deg):
            raise ValueError('Obstacle Euler angles must be finite.')
        if any(v <= 0 for v in self.size) or len(self.rgba) != 4:
            raise ValueError('Obstacle dimensions and RGBA must be valid.')
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.rgba):
            raise ValueError('Obstacle RGBA values must lie in [0, 1].')

    def append(self, world):
        attributes = dict(name=self.name, type=self.shape,
                          pos=' '.join(map(str, self.pos)),
                          rgba=' '.join(map(str, self.rgba)),
                          euler=' '.join(map(str, map(math.radians, self.euler_deg))),
                          contype='1', conaffinity='1')
        if self.shape == 'cylinder':
            attributes['size'] = f'{self.size[0]} {self.size[1]}'
        else:
            attributes['size'] = ' '.join(map(str, self.size))
        return ET.SubElement(world, 'geom', **attributes)


@dataclass(frozen=True)
class StressDoorScene:
    scene_id: str
    door_style: str
    handle_style: str
    obstacles: tuple[Obstacle, ...] = ()
    wall_layout: str = ''
    expected_open_side: str | None = None
    handle_faces: tuple[str, ...] = ('front', 'back')
    handle_x: float = .8
    handle_z: float = .85
    friction: float = .8
    handle_friction: float = .1
    hinge_resistance: float = 1.2
    description: str = ''

    def __post_init__(self):
        if self.door_style not in DOOR_STYLES:
            raise ValueError(f'Unknown door style {self.door_style!r}.')
        if self.handle_style not in HANDLE_STYLES:
            raise ValueError(f'Unknown handle style {self.handle_style!r}.')
        if not self.handle_faces or set(self.handle_faces) - {'front', 'back'}:
            raise ValueError('handle_faces must be a nonempty front/back subset.')
        if len(set(self.handle_faces)) != len(self.handle_faces):
            raise ValueError('handle_faces cannot contain duplicates.')
        if self.expected_open_side not in (None, 'front', 'back'):
            raise ValueError('expected_open_side must be front, back or None.')
        if self.expected_open_side not in self.handle_faces:
            raise ValueError('expected_open_side must expose a handle.')
        if not all(math.isfinite(v) for v in
                   (self.handle_x, self.handle_z, self.friction,
                    self.handle_friction, self.hinge_resistance)):
            raise ValueError('Scene parameters must be finite.')
        if min(self.friction, self.handle_friction, self.hinge_resistance) < 0:
            raise ValueError('Friction values must be nonnegative.')

    @property
    def accessible_side(self):
        """Geometry-label used only by tests/reporting, never by the agent."""
        if self.expected_open_side is not None:
            return self.expected_open_side
        return 'front' if self.handle_faces == ('front',) else 'back'

    def experiment(self, side='front'):
        if side not in ('front', 'back'):
            raise ValueError('side must be front or back.')
        if side not in self.handle_faces:
            raise ValueError(f'{side} handle is not installed in {self.scene_id}.')
        tree = ET.parse(ROOT/'scene.xml')
        root = tree.getroot()
        world = root.find('worldbody')
        door = world.find("body[@name='door']")
        door.find("joint[@name='door_hinge']").set(
            'range', f'{-math.pi/2} {math.pi/2}')
        _style_door(door, self.door_style)
        _style_handles(door, self.handle_style, self.handle_faces,
                       self.handle_x, self.handle_z)
        for obstacle in self.obstacles:
            obstacle.append(world)

        base = FRONT_BASE if side == 'front' else BACK_BASE
        physical_side = 'front' if base[1] < 0 else 'back'
        active = physical_side if physical_side in self.handle_faces else self.handle_faces[0]
        exp = Experiment(
            base=base[:2], base_yaw=base[2], friction=self.friction,
            hinge_resistance=self.hinge_resistance,
            model_xml=ET.tostring(root, encoding='unicode'),
            ik_branch=-1 if physical_side == 'front' else 1,
            opening_direction=1 if physical_side == 'front' else -1,
            handle_body='handle' if active == 'front' else 'back_handle',
            handle_geom='handle_bar' if active == 'front' else 'back_handle_bar',
            operating_side=physical_side)
        exp.m.geom_friction[exp.bar, 0] = self.handle_friction
        _set_perception_side(exp, physical_side)
        return exp


def _visual_geom(parent, name, geom_type, size, pos=None, fromto=None,
                 rgba=METAL, group='1'):
    attrs = dict(name=name, type=geom_type, size=size, rgba=rgba,
                 contype='0', conaffinity='0', group=group, mass='0.001')
    if pos is not None:
        attrs['pos'] = pos
    if fromto is not None:
        attrs['fromto'] = fromto
    return ET.SubElement(parent, 'geom', **attrs)


DOOR_STYLES = {
    'oak-panel': (.47, .27, .12, 1.),
    'brushed-steel': (.42, .47, .5, 1.),
    'frosted-glass': (.56, .78, .86, .42),
    'fire-door': (.64, .12, .1, 1.),
    'walnut-flush': (.28, .16, .1, 1.),
    'sage-painted': (.31, .43, .34, 1.),
}


def _style_door(door, style):
    panel = door.find("geom[@name='door_panel']")
    panel.set('rgba', ' '.join(map(str, DOOR_STYLES[style])))
    if style == 'oak-panel':
        for row, z in enumerate((.38, .77), start=0):
            for column, x in enumerate((.23, .67), start=0):
                _visual_geom(door, f'panel_{row}_{column}', 'box',
                             '.17 .003 .22', pos=f'{x} -.029 {z}',
                             rgba='.57 .34 .16 1')
    elif style == 'brushed-steel':
        for i, z in enumerate((.25, .5, .75, 1.)):
            _visual_geom(door, f'steel_rivet_{i}', 'sphere', '.009',
                         pos=f'.06 -.031 {z}', rgba='.72 .75 .78 1')
    elif style == 'frosted-glass':
        for name, x in (('left', .025), ('right', .875)):
            _visual_geom(door, f'glass_frame_{name}', 'box',
                         '.025 .032 .7', pos=f'{x} 0 .7', rgba=METAL)
        for name, z in (('top', 1.375), ('bottom', .025)):
            _visual_geom(door, f'glass_frame_{name}', 'box',
                         '.45 .032 .025', pos=f'.45 0 {z}', rgba=METAL)
    elif style == 'fire-door':
        _visual_geom(door, 'fire_kickplate', 'box', '.35 .004 .1',
                     pos='.45 -.03 .18', rgba='.62 .64 .66 1')
        _visual_geom(door, 'fire_push_rail', 'box', '.39 .025 .025',
                     pos='.45 -.055 .72', rgba='.77 .78 .8 1')
    elif style == 'sage-painted':
        _visual_geom(door, 'painted_border_top', 'box', '.39 .004 .018',
                     pos='.45 -.03 1.26', rgba='.48 .58 .5 1')
        _visual_geom(door, 'painted_border_bottom', 'box', '.39 .004 .018',
                     pos='.45 -.03 .14', rgba='.48 .58 .5 1')
        _visual_geom(door, 'painted_border_left', 'box', '.018 .004 .57',
                     pos='.07 -.03 .7', rgba='.48 .58 .5 1')
        _visual_geom(door, 'painted_border_right', 'box', '.018 .004 .57',
                     pos='.83 -.03 .7', rgba='.48 .58 .5 1')


HANDLE_STYLES = ('lever', 't-bar', 'd-pull', 'offset-bar',
                 'paddle', 'service-knob')


def _style_handles(door, style, faces, handle_x, handle_z):
    front = door.find("body[@name='handle']")
    front.set('pos', f'{handle_x} -.11 {handle_z}')
    bar = front.find("geom[@name='handle_bar']")
    bar.set('rgba', GOLD)
    # The robot has two straight, parallel jaw pads and no articulated digits.
    # Each profile therefore exposes one vertical rod, one flat blade or one
    # spherical pinch surface that can be clamped by translation alone.
    if style in ('lever', 't-bar', 'd-pull', 'offset-bar'):
        bar.set('type', 'capsule')
        bar.set('fromto', '0 0 -.15 0 0 .15')
        bar.set('size', '.014')
        bar.attrib.pop('pos', None)
    elif style == 'paddle':
        bar.set('type', 'box')
        bar.set('size', '.042 .006 .13')
        bar.set('pos', '0 0 0')
        bar.attrib.pop('fromto', None)
    elif style == 'service-knob':
        bar.set('type', 'sphere')
        bar.set('size', '.025')
        bar.set('pos', '0 0 0')
        bar.attrib.pop('fromto', None)

    if style == 'lever':
        _visual_geom(front, 'lever_arm', 'capsule', '.009',
                     fromto='-.16 .01 .09 0 .01 .09', rgba=METAL)
        _visual_geom(front, 'lever_knob', 'sphere', '.024',
                     pos='-.18 .01 .09', rgba=METAL)
    elif style == 't-bar':
        _visual_geom(front, 'tbar_cross', 'capsule', '.011',
                     fromto='-.14 0 .14 .14 0 .14', rgba=METAL)
    elif style == 'd-pull':
        _visual_geom(front, 'dpull_outer', 'capsule', '.009',
                     fromto='-.13 0 -.13 -.13 0 .13', rgba=METAL)
        _visual_geom(front, 'dpull_top', 'capsule', '.009',
                     fromto='-.13 0 .13 0 0 .13', rgba=METAL)
        _visual_geom(front, 'dpull_bottom', 'capsule', '.009',
                     fromto='-.13 0 -.13 0 0 -.13', rgba=METAL)
    elif style == 'offset-bar':
        _visual_geom(front, 'offset_bar_bridge_top', 'capsule', '.008',
                     fromto='-.10 0 .13 0 0 .13', rgba=METAL)
        _visual_geom(front, 'offset_bar_bridge_bottom', 'capsule', '.008',
                     fromto='-.10 0 -.13 0 0 -.13', rgba=METAL)
    elif style == 'paddle':
        _visual_geom(front, 'paddle_frame', 'box', '.065 .009 .15',
                     pos='-.09 0 -.15', rgba=METAL)
    elif style == 'service-knob':
        _visual_geom(front, 'knob_collar', 'cylinder', '.032 .008',
                     pos='0 -.105 0', rgba=METAL)
        _visual_geom(front, 'knob_body', 'sphere', '.045',
                     pos='0 -.15 0', rgba='.14 .16 .18 1')

    if 'back' in faces:
        back = deepcopy(front)
        back.set('name', 'back_handle')
        back.set('pos', f'{handle_x} .11 {handle_z}')
        for geom in back.findall('geom'):
            geom.set('name', 'back_'+geom.get('name'))
            if geom.get('fromto'):
                values = [float(v) for v in geom.get('fromto').split()]
                values[1], values[4] = -values[1], -values[4]
                geom.set('fromto', ' '.join(map(str, values)))
            if geom.get('pos'):
                values = [float(v) for v in geom.get('pos').split()]
                values[1] = -values[1]
                geom.set('pos', ' '.join(map(str, values)))
        door.append(back)
    if 'front' not in faces:
        door.remove(front)


def _set_perception_side(exp, side):
    if side == 'front':
        return
    cid = exp.m.camera('perception').id
    exp.m.cam_pos[cid] = [.8, 2.2, .85]
    right, up, backward = [-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]
    mujoco.mju_mat2Quat(
        exp.m.cam_quat[cid], np.column_stack(
            [right, up, backward]).ravel())
    mujoco.mj_forward(exp.m, exp.d)


def _front_cross_wall(height=.78, wing=False):
    obstacles = [
        Obstacle('wall_front_cross', 'box', (.48, -.59, height/2),
                 (.56, .07, height/2), (.70, .72, .74, 1.))
    ]
    if wing:
        obstacles.append(
            Obstacle('wall_front_wing', 'box', (1.02, -.30, .74),
                     (.07, .29, .74), (.64, .67, .70, 1.)))
    return tuple(obstacles)


def _rear_cross_wall(height=.78, wing=False):
    obstacles = [
        Obstacle('wall_rear_cross', 'box', (.48, .59, height/2),
                 (.56, .07, height/2), (.70, .72, .74, 1.))
    ]
    if wing:
        obstacles.append(
            Obstacle('wall_rear_wing', 'box', (1.02, .30, .74),
                     (.07, .29, .74), (.64, .67, .70, 1.)))
    return tuple(obstacles)


def _front_low_wall():
    return (
        Obstacle('wall_front_low', 'box', (.48, -.59, .3),
                 (.56, .07, .3), (.64, .66, .68, 1.)),
        Obstacle('clutter_front_low_box', 'box', (.91, -.67, .18),
                 (.14, .15, .18), (.61, .39, .18, 1.), euler_deg=(0., 0., 7.)),
    )


def _rear_low_wall():
    return (
        Obstacle('wall_rear_low', 'box', (.48, .59, .3),
                 (.56, .07, .3), (.64, .66, .68, 1.)),
        Obstacle('clutter_rear_low_box', 'box', (.91, .67, .18),
                 (.14, .15, .18), (.61, .39, .18, 1.), euler_deg=(0., 0., -7.)),
    )


def _front_l_alcove():
    return _front_cross_wall(wing=True) + (
        Obstacle('clutter_front_cart', 'box', (.58, -.73, .24),
                 (.27, .13, .24), (.18, .22, .26, 1.), euler_deg=(0., 0., 4.)),
    )


def _rear_l_alcove():
    return _rear_cross_wall(wing=True) + (
        Obstacle('clutter_rear_cart', 'box', (.58, .73, .24),
                 (.27, .13, .24), (.18, .22, .26, 1.), euler_deg=(0., 0., -4.)),
    )


def _front_corridor():
    return _front_cross_wall(wing=True) + (
        Obstacle('wall_front_short_wing', 'box', (-.18, -.30, .74),
                 (.07, .29, .74), (.64, .67, .70, 1.)),
    )


def _rear_corridor():
    return _rear_cross_wall(wing=True) + (
        Obstacle('wall_rear_short_wing', 'box', (-.18, .30, .74),
                 (.07, .29, .74), (.64, .67, .70, 1.)),
    )


def _front_crate_stack():
    return (
        Obstacle('clutter_crate_a', 'box', (.82, -.78, .22), (.22, .18, .22),
                 (.59, .35, .18, 1.), euler_deg=(0., 0., 8.)),
        Obstacle('clutter_crate_b', 'box', (.6, -.62, .3), (.27, .24, .3),
                 (.25, .48, .64, 1.), euler_deg=(0., 0., -6.)),
    )


def _rear_crate_stack():
    return (
        Obstacle('clutter_crate_a', 'box', (.82, .78, .22), (.22, .18, .22),
                 (.59, .35, .18, 1.), euler_deg=(0., 0., -8.)),
        Obstacle('clutter_crate_b', 'box', (.6, .62, .3), (.27, .24, .3),
                 (.25, .48, .64, 1.), euler_deg=(0., 0., 6.)),
    )


def scenario_suite():
    """Thirteen one-sided-access scenes crossing doors, handles and layouts."""
    return (
        StressDoorScene(
            'oak-lever-wall', 'oak-panel', 'lever',
            obstacles=(
                Obstacle('wall_front', 'box', (.48, -.62, .75), (.5, .07, .75),
                         (.72, .74, .76, 1.)),
                Obstacle('clutter_crate_a', 'box', (.82, -.78, .22), (.22, .18, .22),
                         (.59, .35, .18, 1.), euler_deg=(0., 0., 8.)),
            ),
            wall_layout='front-cross-wall',
            expected_open_side='back',
            handle_friction=.1,
            description='Oak panel door, lever, front wall and crate'),
        StressDoorScene(
            'steel-tbar-crates', 'brushed-steel', 't-bar',
            obstacles=_front_crate_stack(),
            wall_layout='front-crate-stack',
            expected_open_side='back',
            handle_z=.86,
            handle_friction=.18,
            description='Steel door, T-bar, stacked front crates'),
        StressDoorScene(
            'glass-dpull-alcove', 'frosted-glass', 'd-pull',
            obstacles=_front_l_alcove(),
            wall_layout='front-l-alcove',
            expected_open_side='back',
            handle_x=.78,
            handle_friction=.28,
            description='Frosted-glass door, D-pull, side-wall alcove'),
        StressDoorScene(
            'fire-knob-cart', 'fire-door', 'service-knob',
            obstacles=_front_low_wall(),
            wall_layout='front-low-wall',
            expected_open_side='back',
            handle_faces=('back',),
            handle_z=.83,
            handle_friction=.5,
            description='Fire door, one-sided service knob, barrel and low wall'),
        StressDoorScene(
            'walnut-paddle-clutter', 'walnut-flush', 'paddle',
            obstacles=(
                Obstacle('clutter_tall_box', 'box', (.47, -.6, .6), (.3, .2, .6),
                         (.45, .28, .16, 1.), euler_deg=(0., 0., -3.)),
                Obstacle('barrel_corner', 'cylinder', (.94, -.66, .3), (.16, .3),
                         (.23, .42, .32, 1.)),
            ),
            wall_layout='front-clutter',
            expected_open_side='back',
            handle_x=.82,
            handle_friction=.12,
            description='Walnut flush door, paddle, tall clutter'),
        StressDoorScene(
            'sage-offset-rear-wall', 'sage-painted', 'offset-bar',
            obstacles=_rear_cross_wall(),
            wall_layout='rear-cross-wall',
            expected_open_side='front',
            handle_x=.79,
            handle_friction=.14,
            description='Sage painted door, offset pull, rear cross wall'),
        StressDoorScene(
            'steel-lever-rear-crates', 'brushed-steel', 'lever',
            obstacles=_rear_crate_stack(),
            wall_layout='rear-crate-stack',
            expected_open_side='front',
            handle_z=.84,
            handle_friction=.09,
            description='Steel door, lever, rear crate stack'),
        StressDoorScene(
            'glass-tbar-rear-low', 'frosted-glass', 't-bar',
            obstacles=_rear_low_wall(),
            wall_layout='rear-low-wall',
            expected_open_side='front',
            handle_x=.77,
            handle_friction=.22,
            description='Glass door, T-bar, rear low wall and box'),
        StressDoorScene(
            'oak-dpull-rear-corridor', 'oak-panel', 'd-pull',
            obstacles=_rear_corridor(),
            wall_layout='rear-corridor',
            expected_open_side='front',
            handle_x=.78,
            handle_friction=.16,
            description='Oak door, D-pull, rear corridor layout'),
        StressDoorScene(
            'walnut-lever-front-lwall', 'walnut-flush', 'lever',
            obstacles=_front_l_alcove(),
            wall_layout='front-l-alcove',
            expected_open_side='back',
            handle_x=.81,
            handle_friction=.08,
            description='Walnut flush door, lever, front L-shaped alcove'),
        StressDoorScene(
            'fire-tbar-rear-alcove', 'fire-door', 't-bar',
            obstacles=_rear_l_alcove(),
            wall_layout='rear-l-alcove',
            expected_open_side='front',
            handle_x=.79,
            handle_friction=.24,
            description='Fire door, T-bar, rear L-shaped alcove'),
        StressDoorScene(
            'sage-knob-front-corridor', 'sage-painted', 'service-knob',
            obstacles=_front_corridor(),
            wall_layout='front-corridor',
            expected_open_side='back',
            handle_faces=('back',),
            handle_z=.84,
            handle_friction=.45,
            description='Sage painted door, one-sided knob, front corridor'),
        StressDoorScene(
            'steel-paddle-rear-wall', 'brushed-steel', 'paddle',
            obstacles=_rear_cross_wall(),
            wall_layout='rear-cross-wall',
            expected_open_side='front',
            handle_x=.78,
            handle_z=.83,
            handle_friction=.12,
            description='Steel door, paddle, rear cross wall'),
    )


def scenario_by_id(scene_id):
    scenes = {scene.scene_id: scene for scene in scenario_suite()}
    if scene_id not in scenes:
        raise KeyError(f'Unknown stress scene {scene_id!r}; choose from {sorted(scenes)}')
    return scenes[scene_id]


def write_scene_xml(scene, side, path):
    """Write the exact XML for a scene without changing any tracked asset."""
    if side not in scene.handle_faces:
        raise ValueError(f'{side} handle is not installed in {scene.scene_id}.')
    path = Path(path)
    tree = ET.parse(ROOT/'scene.xml')
    root = tree.getroot()
    world = root.find('worldbody')
    door = world.find("body[@name='door']")
    door.find("joint[@name='door_hinge']").set(
        'range', f'{-math.pi/2} {math.pi/2}')
    _style_door(door, scene.door_style)
    _style_handles(door, scene.handle_style, scene.handle_faces,
                   scene.handle_x, scene.handle_z)
    for obstacle in scene.obstacles:
        obstacle.append(world)
    base = world.find("body[@name='base']")
    camera = world.find("camera[@name='perception']")
    if side == 'back':
        base.set('pos', '0.6 1 0')
        base.set('euler', f'0 0 {-math.pi/2}')
        camera.set('pos', '0.8 2.2 0.85')
        camera.set('xyaxes', '-1 0 0 0 0 1')
    else:
        base.set('pos', '0.6 -1 0')
        base.set('euler', f'0 0 {math.pi/2}')
        camera.set('pos', '0.8 -2.2 0.85')
        camera.set('xyaxes', '1 0 0 0 0 1')
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding='unicode')

