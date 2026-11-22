"""Render a preserved joint-limit pose; no control or dynamics steps are run."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
REPORT=Path(__file__).resolve().parents[1]
WORK=ROOT/'runs/final_report_visual_polish_20261122'
WORK.mkdir(parents=True,exist_ok=True)
source=ROOT/'results/workspace_limit/workspace_limit.npz'
with np.load(source,allow_pickle=False) as data:
    qpos=data['qpos'][-1].copy()
    time=float(data['time'][-1])
tree=ET.parse(ROOT/'scene.xml')
asset=tree.getroot().find('asset')
if asset is None:asset=ET.SubElement(tree.getroot(),'asset')
ET.SubElement(asset,'texture',name='paper_background',type='skybox',builtin='flat',
              rgb1='1 1 1',width='64',height='64')
model=mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'))
model.body('base').pos[:2]=(.3,-.9)
model.body('base').quat[:]=[np.cos(np.pi/4),0,0,np.sin(np.pi/4)]
state=mujoco.MjData(model)
assert qpos.shape==(model.nq,)
state.qpos[:]=qpos
state.time=time
mujoco.mj_forward(model,state)
elbow=float(np.rad2deg(qpos[model.joint('elbow').qposadr[0]]))
door=float(np.rad2deg(qpos[model.joint('door_hinge').qposadr[0]]))
assert abs(elbow+163.58901410901402)<1e-6
assert abs(door-42.505084386970125)<1e-6
camera=mujoco.MjvCamera()
mujoco.mjv_defaultCamera(camera)
camera.lookat[:]=(.48,-.57,.68)
camera.distance=2.35
camera.azimuth=125
camera.elevation=-24
with mujoco.Renderer(model,height=720,width=960) as renderer:
    renderer.update_scene(state,camera=camera)
    rgb=renderer.render().copy()
output=REPORT/'figures/failures/report_joint_limit_scene.png'
Image.fromarray(rgb).save(output)
record=dict(source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    scene_xml_sha256=hashlib.sha256((ROOT/'scene.xml').read_bytes()).hexdigest(),
    time_s=time,door_deg=door,elbow_deg=elbow,mujoco_version=mujoco.__version__,
    rendering_only=True,dynamics_steps=0,output=str(output.relative_to(REPORT)))
(WORK/'diagnostic_render.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
