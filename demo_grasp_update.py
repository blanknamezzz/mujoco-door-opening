"""Generate measured, paired MuJoCo demos of configurable grasp plans.

These are hand-configured counterfactuals, not a learned selector or optimizer.
Each pair changes exactly one parameter and keeps the scene fixed.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, replace
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from grasp_plan import GraspPlan, SkillTiming
from perception import estimate
from simulate import Experiment, ROOT, save_trial


@dataclass(frozen=True)
class Comparison:
    key: str
    title: str
    condition: str
    before_label: str
    after_label: str
    explanation: str
    practical: str
    changes: str
    before: GraspPlan
    after: GraspPlan
    base: tuple = (.6, -1.)
    friction: float = .8
    after_succeeds: bool = True

    def experiment(self):
        return Experiment(base=self.base, friction=self.friction)


def comparisons():
    center = tuple(estimate(ROOT/'results'/'observation.npz')['center_world_m'])
    default = GraspPlan(center)
    return [
        Comparison('angle', 'Angle: reach and avoid collision throughout',
            'Same base (0.80, -1.00) m, grasp, elbow branch, speed, and force cap',
            'Approach offset 0 deg', 'Approach offset +8 deg',
            'At 0 degrees the forearm/right finger self-collide near full opening; +8 degrees opens and holds.',
            'Narrow spaces require grasp configurations that support the complete continuation.',
            'approach_angle_deg', default, replace(default, approach_angle_deg=8.), base=(.8, -1.)),
        Comparison('force', 'Grip force: same surface, different load capacity',
            'Same station and path; sliding friction mu = 0.20',
            'Force cap 2 N/finger', 'Force cap 25 N/finger',
            'The 2 N plan grasps but slips during pulling; a higher actuator cap completes opening and hold.',
            'This models limited grip budget on slippery handles; actuator caps are not prescribed contact forces.',
            'grip_force_limit_N', replace(default, grip_force_limit_N=2.),
            replace(default, grip_force_limit_N=25.), friction=.2),
        Comparison('speed', 'Timing: pulling speed matters after grasping',
            'Same station, grasp, angle, contact settings, and force cap',
            'Opening duration 2 s', 'Opening duration 14 s',
            'Fast pulling triggers slip; slowing the same arc completes opening.',
            'Execution time trades off against contact stability.',
            'timing.open_s', replace(default, timing=SkillTiming(open_s=2.)),
            replace(default, timing=SkillTiming(open_s=14.))),
        Comparison('limit', 'Workspace: more force cannot add reach',
            'Same limited station (0.30, -0.90) m, angle, grasp, and speed',
            'Force cap 25 N/finger', 'Force cap 35 N/finger',
            'Both stop near 42.5 degrees at an out-of-range joint target while maintaining grasp.',
            'Change posture, station, or motion; increased grip force cannot remove geometric limits.',
            'grip_force_limit_N', replace(default, grip_force_limit_N=25.), default,
            base=(.3, -.9), after_succeeds=False),
    ]


def changed_paths(a, b, prefix=''):
    paths = []
    for key in a:
        path = prefix+key
        if isinstance(a[key], dict):
            paths.extend(changed_paths(a[key], b[key], path+'.'))
        elif a[key] != b[key]:
            paths.append(path)
    return paths


def run_physics(output):
    measurements = []
    for case in comparisons():
        assert changed_paths(case.before.to_dict(), case.after.to_dict()) == [case.changes]
        pair = dict(key=case.key, title=case.title, condition=case.condition,
                    explanation=case.explanation, practical=case.practical,
                    changed_parameter=case.changes, scene=dict(base_xy_m=list(case.base),
                    base_yaw_deg=90., friction=case.friction, hinge_resistance_Nm=1.2), variants=[])
        for variant, plan, label in [('before', case.before, case.before_label), ('after', case.after, case.after_label)]:
            exp = case.experiment()
            result, rows, states = exp.trial(plan=plan)
            result.update(scene=pair['scene'],
                observation_source='saved calibrated RGB-D cylinder fit; same observation for each pair',
                plan_selection='manually configured; no neural network or automatic search',
                mujoco_version=mujoco.__version__)
            name = f'{case.key}_{variant}'
            save_trial(name, result, rows, states, output)
            plan.save(output/f'{name}_plan.json')
            successful = result['outcome'] == 'D_full_open'
            expected = variant == 'after' and case.after_succeeds
            if successful != expected:
                raise RuntimeError(f'{name}: measured outcome {result["outcome"]} changed; inspect raw logs before presenting.')
            if case.key == 'angle' and variant == 'before':
                assert result['first_unexpected_contact'] == 'forearm_link:right_finger_link'
            if case.key == 'limit':
                assert result['stop_reason'] == 'arm_workspace_limit' and rows[-1]['bilateral_contact']
            pair['variants'].append(dict(name=name, label=label, result=result,
                elapsed_s=rows[-1]['time_s'], final_bilateral_contact=rows[-1]['bilateral_contact']))
            print(f'{name}: {result["outcome"]}, {result["final_angle_deg"]:.3f} deg, {result["stop_reason"]}', flush=True)
        measurements.append(pair)
    summary = dict(schema_version=1, mujoco_version=mujoco.__version__,
        scope='Four paired deterministic simulations; manually configured plans, no automatic optimization or real-robot validation.',
        playback='2x simulated time; stopped trajectories freeze and do not continue physical integration.', comparisons=measurements)
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def find_font(size):
    paths = ['/System/Library/Fonts/STHeiti Medium.ttc',
             '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
             'C:/Windows/Fonts/msyh.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise RuntimeError('A CJK font is required to render the Chinese captions.')


def find_ffmpeg(explicit=None):
    if explicit:
        return str(explicit)
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    for path in sorted((Path.home()/'Library'/'Caches'/'ms-playwright').glob('ffmpeg-*/ffmpeg-mac')):
        return str(path)
    return None


BG, FG, MUTED = '#101b2a', '#eef4fa', '#a7b8ca'
BEFORE, AFTER = '#f7b267', '#59d4bc'
W, H, FPS = 1152, 832, 12


def reason_text(result):
    return {'full_open_and_hold': 'Success: complete opening and hold',
            'unexpected_collision': 'Stop: forearm/finger self-collision',
            'slip_or_contact_loss': 'Stop: slip or contact loss',
            'arm_workspace_limit': 'Stop: joint target outside workspace',
            'grasp_not_acquired': 'Failure: no stable grasp'}.get(result['stop_reason'], result['stop_reason'])


class VideoWriter:
    def __init__(self, executable, path):
        self.path = path
        self.log = path.with_suffix('.encoder.log').open('wb')
        self.process = subprocess.Popen([executable, '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'image2pipe', '-vcodec', 'mjpeg', '-r', str(FPS), '-i', 'pipe:0',
            '-an', '-c:v', 'libvpx', '-deadline', 'realtime', '-cpu-used', '5', '-b:v', '1600k',
            str(path)], stdin=subprocess.PIPE, stderr=self.log)

    def add(self, frame):
        data = BytesIO()
        frame.save(data, format='JPEG', quality=95)
        self.process.stdin.write(data.getvalue())

    def close(self):
        self.process.stdin.close()
        code = self.process.wait()
        self.log.close()
        if code:
            raise RuntimeError(f'Video encoding failed; inspect {self.path.with_suffix(".encoder.log")}')


def render_comparison(case, pair, output, ffmpeg, combined):
    import csv
    font, small, title_font = find_font(22), find_font(18), find_font(30)
    exps, trajectories, logs, renderers = [], [], [], []
    for variant in pair['variants']:
        e = case.experiment()
        exps.append(e)
        data = np.load(output/(variant['name']+'.npz'))
        trajectories.append(dict(qpos=data['qpos'].copy(), time=data['time'].copy()))
        data.close()
        with (output/(variant['name']+'.csv')).open() as stream:
            logs.append(list(csv.DictReader(stream)))
        renderers.append(mujoco.Renderer(e.m, height=312, width=552))
    max_time = max(t['time'][-1] for t in trajectories)
    times = np.arange(0., max_time+1e-9, 2./FPS).tolist()+[max_time]*18
    writer = VideoWriter(ffmpeg, output/(case.key+'.webm')) if ffmpeg else None
    gif_frames = []
    chapter = ['angle', 'force', 'speed', 'limit'].index(case.key)+1
    try:
        for frame_id, t in enumerate(times):
            frame = Image.new('RGB', (W, H), BG)
            d = ImageDraw.Draw(frame)
            d.text((24, 14), f'Grasp-plan update  /  {chapter:02d}', font=small, fill=MUTED)
            d.text((24, 43), case.title, font=title_font, fill=FG)
            d.text((24, 88), case.condition, font=small, fill=MUTED)
            for side, color in [(0, BEFORE), (1, AFTER)]:
                x = 16+side*568
                name = 'Original plan' if side == 0 else 'Adjusted plan'
                variant = pair['variants'][side]
                result = variant['result']
                data, e = trajectories[side], exps[side]
                idx = int(np.clip(np.searchsorted(data['time'], t, side='right')-1, 0, len(data['time'])-1))
                row = logs[side][idx]
                terminal = t >= data['time'][-1]
                if terminal and result.get('first_unexpected_contact'):
                    for geom_name in result['first_unexpected_contact'].split(':'):
                        e.m.geom(geom_name).rgba[:3] = [.92, .25, .20]
                d.text((x+8, 126), f'{name}  ·  {variant["label"]}', font=font, fill=color)
                e.d.qpos[:] = data['qpos'][idx]
                mujoco.mj_forward(e.m, e.d)
                renderers[side].update_scene(e.d, camera='overview')
                frame.paste(Image.fromarray(renderers[side].render().copy()), (x, 164))
                d.rectangle((x+8, 171, x+158, 201), fill=BG)
                d.text((x+12, 176), f't = {data["time"][idx]:.2f} s', font=small, fill=FG)
                # Inset follows the actual handle pose, including the forearm/finger contact region.
                cam = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(cam)
                cam.lookat[:] = e.d.xpos[e.hbody]
                cam.distance, cam.azimuth, cam.elevation = .70, 115, -20
                renderers[side].update_scene(e.d, camera=cam)
                inset = Image.fromarray(renderers[side].render().copy()).resize((212, 120))
                frame.paste(inset, (x+332, 348))
                d.rectangle((x+332, 326, x+544, 348), fill=BG)
                d.text((x+336, 327), 'Contact close-up', font=small, fill=FG)
                d.text((x+8, 488), f'Door {float(row["door_deg"]):.1f} deg   Drift {1000*float(row["relative_drift_m"]):.1f} mm', font=font, fill=FG)
                d.text((x+8, 520), f'Normal force L/R: {float(row["left_normal_N"]):.1f} / {float(row["right_normal_N"]):.1f} N', font=small, fill=MUTED)
                status = reason_text(result) if terminal else ('Bilateral contact' if row['bilateral_contact']=='True' else 'Approach / contact acquisition')
                d.text((x+8, 548), status, font=small, fill=color)
            # Angle traces use simulated time, including distinct termination times.
            gx, gy, gw, gh = 70, 594, 1048, 94
            for angle in (0, 45, 90):
                y = gy+gh*(1-angle/90)
                d.line((gx,y,gx+gw,y), fill='#334256', width=1)
                d.text((20,y-10), f'{angle}°', font=small, fill=MUTED)
            for side, color in [(0, BEFORE), (1, AFTER)]:
                data = trajectories[side]
                values = logs[side]
                indices = [i for i in range(0,len(values),4) if data['time'][i] <= t]
                if indices:
                    last = min(int(np.searchsorted(data['time'],t,side='right')-1),len(values)-1)
                    indices.append(max(last,0))
                    points = [(gx+gw*data['time'][i]/max_time, gy+gh*(1-float(values[i]['door_deg'])/90)) for i in indices]
                    if len(points)>1:
                        d.line(points, fill=color, width=3)
            for tick in (0., max_time/2, max_time):
                label = f'{tick:.1f} s'
                tx = gx+gw*tick/max_time
                d.text((tx-d.textlength(label,font=small)/2, gy+gh+5), label, font=small, fill=MUTED)
            d.text((24, 731), case.explanation, font=small, fill=FG)
            d.text((24, 762), 'MuJoCo physics · One parameter per pair · Manual plans, no network optimization', font=small, fill=MUTED)
            d.text((24, 793), '2x playback; failed sides freeze at termination. Close-ups use the same physical state.', font=small, fill=MUTED)
            if writer:
                writer.add(frame)
                combined.add(frame)
            if frame_id % 3 == 0 or frame_id == len(times)-1:
                gif_frames.append(frame.resize((864,624)).convert('P', palette=Image.Palette.ADAPTIVE, colors=128))
            if frame_id == len(times)-1:
                frame.save(output/(case.key+'_final.png'))
                if case.key == 'angle':
                    frame.save(output/'poster.png')
        durations = [250]*len(gif_frames)
        durations[-1] = 1500
        gif_frames[0].save(output/(case.key+'.gif'), save_all=True, append_images=gif_frames[1:], duration=durations, loop=0)
    finally:
        for renderer in renderers:
            renderer.close()
        if writer:
            writer.close()
    print('Rendered:',case.key,flush=True)


def write_presentation(summary, output, has_video):
    data = json.dumps(summary, ensure_ascii=False).replace('</','<\\/')
    html = PRESENTATION.replace('__DATA__', data).replace('__VIDEO__', 'true' if has_video else 'false')
    (output/'index.html').write_text(html, encoding='utf-8')
    lines = ['# Grasp-plan updates: measured comparison', '',
             'Every result comes from actual MuJoCo integration. Each pair fixes the scene and changes one plan parameter.',
             'Plans are manual, without networks or automatic optimization; comparisons are not old/new model win rates.', '',
             '| Control | Original plan | Adjusted plan |', '| --- | --- | --- |']
    for pair in summary['comparisons']:
        cells=[]
        for variant in pair['variants']:
            r=variant['result']
            cells.append(f'{variant["label"]}: {r["final_angle_deg"]:.3f}°, {reason_text(r)}')
        lines.append('| '+pair['title']+' | '+' | '.join(cells)+' |')
    lines += ['', '## Demonstrated capabilities', '',
              '- Save/replay complete plans including point, angle, elbow branch, timing, and force cap.',
              '- Isolate angle/force/speed effects on complete execution rather than grasp acquisition or final angle alone.',
              '- Workspace examples show that force cannot compensate for insufficient geometric reach.', '',
              '## Practical interpretation and scope', '',
              'These mechanisms illustrate narrow spaces, friction differences, grip budgets, and speed tradeoffs. ',
              'No material calibration, hardware, occlusion, mobile navigation, or online regrasp is validated.',
              'Current demos initialize at pregrasp. Automatic selection needs candidate search, trajectory screening, and closed-loop correction.', '',
              '## Reproduction', '', '```bash',
              'python demo_grasp_update.py --physics-only',
              'python demo_grasp_update.py --render-only',
              'mjpython demo_grasp_update.py --view angle --variant before',
              'mjpython demo_grasp_update.py --view angle --variant after',
              '```', '', 'Open index.html to switch chapters. summary.json, case CSV/NPZ, and plan JSON preserve the evidence.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


PRESENTATION = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grasp-plan updates · MuJoCo comparisons</title>
<style>
:root{color-scheme:dark;--bg:#0d1723;--text:#eef4fa;--muted:#acbbcd;--line:#2a394d;--a:#f7b267;--b:#59d4bc}[hidden]{display:none!important}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.7 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1200px;padding:44px 26px 64px;margin:auto}h1{font-size:clamp(27px,4vw,42px);line-height:1.25;letter-spacing:-.04em;margin:14px 0}h2{font-size:22px;margin:10px 0}p{margin:10px 0}small,.muted{color:var(--muted)}.eyebrow{color:var(--b);letter-spacing:.12em;font-size:12px}nav{display:flex;gap:9px;flex-wrap:wrap;margin:25px 0 20px}button{font:inherit;color:var(--text);background:transparent;border:1px solid var(--line);border-radius:9px;padding:9px 16px;cursor:pointer}button[aria-pressed=true]{background:var(--text);color:var(--bg)}button:focus-visible{outline:3px solid var(--b);outline-offset:3px}video,img{display:block;width:100%;border-radius:10px;background:#101b2a}.facts{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin:22px 0}.fact{border-top:3px solid var(--a);padding-top:14px}.fact.after{border-color:var(--b)}.value{font-size:30px;font-variant-numeric:tabular-nums}.section{border-top:1px solid var(--line);padding-top:22px;margin-top:30px}a{color:var(--b)}code{font-size:14px;white-space:pre-wrap;overflow-wrap:anywhere}.command{padding:14px;background:#172435;border-radius:9px}details{margin:18px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}td{overflow-wrap:anywhere}ul{padding-left:22px}li{margin:9px 0}@media(max-width:600px){main{padding:28px 16px}.facts{grid-template-columns:1fr}.value{font-size:25px}nav button{padding:8px 10px}}
</style></head><body><main>
<div class="eyebrow">MUJOCO / GRASP PLAN DEMO</div>
<h1>From a grasp point to a complete plan</h1>
<p class="muted">Same scene, one changed parameter. Compare angle, force, and speed over the full opening.</p>
<p><small>Manual plans · Physical simulation · No automatic network optimization or hardware validation</small></p>
<nav id="chapters" aria-label="Comparison chapters"></nav>
<h2 id="case-title"></h2><p id="condition" class="muted"></p>
<video id="player" controls playsinline preload="metadata" aria-label="Physical trajectories of two grasp plans"></video>
<img id="gif" alt="Measured grasp-plan comparison animation" hidden>
<div class="facts" id="facts" aria-live="polite"></div>
<p id="explanation"></p><p id="practical" class="muted"></p>
<details><summary>Reproducible plan parameters</summary><table><thead><tr><th>Parameter</th><th>Original</th><th>Adjusted</th></tr></thead><tbody id="parameters"></tbody></table><p id="raw"></p></details>
<div class="section"><h2>Implemented capabilities</h2>
<ul><li><strong>Richer actions:</strong> grasp point, approach angle, elbow branch, timing, and actuator caps.</li>
<li><strong>Complete-process comparison:</strong> self-collision or slip can follow successful reach/grasp; reach and angle alone are insufficient.</li>
<li><strong>Traceability:</strong> retain plans, actual forces, drift, trajectories, and stop reasons for replay/training.</li></ul>
<h2>Future capabilities</h2><p class="muted">Optimal search, reliable prediction from unknown poses, visual occlusion handling, and online regrasp remain unimplemented. Current demos begin at pregrasp.</p>
<p><a href="REPORT.md">Measured report</a> · <a href="summary.json">Raw summary</a> · <a href="grasp_update_demo.webm" id="full-video">Complete four-chapter demo</a></p>
<details><summary>Run this chapter live in MuJoCo</summary><pre class="command"><code id="commands"></code></pre><small>Activate a Python environment with MuJoCo installed; use mjpython for Mac live windows.</small></details>
</div></main><script>
const summary=__DATA__;const hasVideo=__VIDEO__;
const $=id=>document.getElementById(id);
const reasons={full_open_and_hold:'Complete opening and hold',unexpected_collision:'Forearm/right-finger self-collision',slip_or_contact_loss:'Stopped by slip alarm',arm_workspace_limit:'Joint target outside workspace'};
const fields=[['height_offset_m','Height offset / m'],['approach_angle_deg','Approach offset / deg'],['ik_branch','Elbow branch'],['grip_force_limit_N','Force cap per finger / N'],['timing.open_s','Opening duration / s']];
const get=(o,path)=>path.split('.').reduce((v,k)=>v[k],o);
function select(index){const c=summary.comparisons[index];$('case-title').textContent=c.title;$('condition').textContent=c.condition;
document.querySelectorAll('nav button').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===index)));
const player=$('player');player.pause();player.hidden=!hasVideo;$('gif').hidden=hasVideo;
player.poster=c.key+'_final.png';player.src=c.key+'.webm';if(hasVideo)player.load();else $('gif').src=c.key+'.gif';
$('facts').replaceChildren();c.variants.forEach((v,i)=>{const div=document.createElement('div');div.className='fact'+(i?' after':'');const label=document.createElement('div');label.textContent=v.label;const value=document.createElement('div');value.className='value';value.textContent=v.result.final_angle_deg.toFixed(2)+'°';const status=document.createElement('div');status.textContent=reasons[v.result.stop_reason]||v.result.stop_reason;div.append(label,value,status);$('facts').append(div)});
$('explanation').textContent=c.explanation;$('practical').textContent=c.practical;
$('parameters').replaceChildren();fields.forEach(([key,label])=>{const tr=document.createElement('tr');[label,...c.variants.map(v=>get(v.result.execution_plan,key))].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td)});$('parameters').append(tr)});
$('raw').replaceChildren();c.variants.forEach(v=>{const a=document.createElement('a');a.href=v.name+'_plan.json';a.textContent=v.label+' JSON';a.style.marginRight='20px';$('raw').append(a)});
$('commands').textContent='# Run from the project root after activating Python\n\nmjpython demo_grasp_update.py --view '+c.key+' --variant before\nmjpython demo_grasp_update.py --view '+c.key+' --variant after';}
summary.comparisons.forEach((c,i)=>{const b=document.createElement('button');b.type='button';b.textContent=['01 Angle','02 Grip force','03 Speed','04 Workspace'][i];b.addEventListener('click',()=>select(i));$('chapters').append(b)});
$('full-video').hidden=!hasVideo;select(0);
</script></body></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'reproduced'/'grasp_update_demo')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--physics-only',action='store_true')
    mode.add_argument('--render-only',action='store_true')
    mode.add_argument('--view',choices=['angle','force','speed','limit'])
    p.add_argument('--variant',choices=['before','after'],default='after')
    p.add_argument('--ffmpeg',type=Path,help='Optional FFmpeg executable for WebM video; GIF works without it')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.view:
        from mujoco import viewer
        case=next(c for c in comparisons() if c.key==a.view)
        exp=case.experiment()
        with viewer.launch_passive(exp.m,exp.d) as window:
            result,rows,states=exp.trial(plan=getattr(case,a.variant),viewer=window)
        save_trial(case.key+'_'+a.variant,result,rows,states,a.output/'live')
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return
    summary=json.loads((a.output/'summary.json').read_text()) if a.render_only else run_physics(a.output)
    if a.physics_only:
        write_presentation(summary,a.output,False)
        return
    ffmpeg=find_ffmpeg(a.ffmpeg)
    combined=VideoWriter(ffmpeg,a.output/'grasp_update_demo.webm') if ffmpeg else None
    try:
        for case,pair in zip(comparisons(),summary['comparisons']):
            render_comparison(case,pair,a.output,ffmpeg,combined)
    finally:
        if combined:
            combined.close()
    write_presentation(summary,a.output,ffmpeg is not None)
    print('Demo ready:',a.output/'index.html',flush=True)


if __name__=='__main__':
    main()
