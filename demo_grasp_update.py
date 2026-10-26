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
        Comparison('angle', '角度：抓得到，还要全程不碰撞',
            '相同站位 (0.80, -1.00) m；相同抓点、肘分支、速度和力上限',
            '接近偏角 0°', '接近偏角 +8°',
            '0° 方案接近全开时前臂与右手指发生自碰撞；+8° 方案通过并稳定保持。',
            '对应狭窄操作空间：抓取姿态会改变整段机械臂配置，必须考虑后续动作。',
            'approach_angle_deg', default, replace(default, approach_angle_deg=8.), base=(.8, -1.)),
        Comparison('force', '夹持力：相同表面，接触承载能力不同',
            '相同站位与轨迹；滑动摩擦系数 μ = 0.20',
            '每指驱动力上限 2 N', '每指驱动力上限 25 N',
            '2 N 方案已抓住但拉门时滑移；提高允许的驱动力后完成开门并保持。',
            '对应较滑的把手或夹持力预算不足；这是驱动力上限，不是直接指定接触力。',
            'grip_force_limit_N', replace(default, grip_force_limit_N=2.),
            replace(default, grip_force_limit_N=25.), friction=.2),
        Comparison('speed', '时序：同一个抓点，拉动速度也重要',
            '相同站位、抓点、角度、接触参数和驱动力上限',
            '开门动作 2 秒', '开门动作 14 秒',
            '快速拉动中途触发滑移；放慢相同圆弧轨迹后完成开门。',
            '对应时间与接触稳定性的取舍：夹住后不能只追求更快。',
            'timing.open_s', replace(default, timing=SkillTiming(open_s=2.)),
            replace(default, timing=SkillTiming(open_s=14.))),
        Comparison('limit', '能力边界：更用力不能扩大机械臂工作空间',
            '相同受限站位 (0.30, -0.90) m；角度、抓点和速度相同',
            '每指驱动力上限 25 N', '每指驱动力上限 35 N',
            '两个方案均在约 42.5° 因后续关节目标越界而停止；此时仍保持抓握。',
            '这类失败需要改姿态、站位或运动方案；加大夹持力无法解决几何限制。',
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
    return {'full_open_and_hold': '成功：完全打开并稳定保持',
            'unexpected_collision': '停止：前臂与右手指自碰撞',
            'slip_or_contact_loss': '停止：滑移或接触丢失',
            'arm_workspace_limit': '停止：后续关节目标越界',
            'grasp_not_acquired': '失败：未建立稳定抓握'}.get(result['stop_reason'], result['stop_reason'])


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
            d.text((24, 14), f'抓握方案更新  /  {chapter:02d}', font=small, fill=MUTED)
            d.text((24, 43), case.title, font=title_font, fill=FG)
            d.text((24, 88), case.condition, font=small, fill=MUTED)
            for side, color in [(0, BEFORE), (1, AFTER)]:
                x = 16+side*568
                name = '原方案' if side == 0 else '调整方案'
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
                d.text((x+336, 327), '实际接触近景', font=small, fill=FG)
                d.text((x+8, 488), f'门角度 {float(row["door_deg"]):.1f}°   相对漂移 {1000*float(row["relative_drift_m"]):.1f} mm', font=font, fill=FG)
                d.text((x+8, 520), f'接触法向力 L/R：{float(row["left_normal_N"]):.1f} / {float(row["right_normal_N"]):.1f} N', font=small, fill=MUTED)
                status = reason_text(result) if terminal else ('双侧接触已建立' if row['bilateral_contact']=='True' else '接近 / 建立接触')
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
            d.text((24, 762), 'MuJoCo 物理实测 · 两侧仅改变一个参数 · 手动配置，非网络自动优化', font=small, fill=MUTED)
            d.text((24, 793), '播放 2 倍速；失败侧到停止时刻后冻结，未继续积分。接触近景来自同一物理状态。', font=small, fill=MUTED)
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
    lines = ['# 抓握方案更新：实测对比演示', '',
             '全部结果来自当前 MuJoCo 模型的实际积分。每组固定场景，只改变一个方案参数。',
             '方案由人工指定，未运行神经网络或自动优化，不能把这些对照解释为新旧模型胜率。', '',
             '| 对照 | 原方案 | 调整方案 |', '| --- | --- | --- |']
    for pair in summary['comparisons']:
        cells=[]
        for variant in pair['variants']:
            r=variant['result']
            cells.append(f'{variant["label"]}：{r["final_angle_deg"]:.3f}°，{reason_text(r)}')
        lines.append('| '+pair['title']+' | '+' | '.join(cells)+' |')
    lines += ['', '## 已展示的优势', '',
              '- 可将抓点、接近角度、肘关节分支、时序和夹持力上限保存为完整方案并重放。',
              '- 可以隔离角度、力和速度对全程成功的影响，而不是只看能否夹住或最终角度。',
              '- 边界案例明确展示：几何工作空间不足无法靠增加夹持力解决。', '',
              '## 更现实的问题及当前边界', '',
              '这些机制对应狭窄空间、表面摩擦差异、有限夹持力和动作速度的取舍；',
              '但本演示没有对真实材料标定，也没有验证真实机器人、遮挡、移动底座导航或在线重抓。',
              '当前从预抓取姿态初始化。后续需加入候选搜索、完整轨迹筛选和闭环纠错，才能自动选择方案。', '',
              '## 复现', '', '```bash',
              'python demo_grasp_update.py --physics-only',
              'python demo_grasp_update.py --render-only',
              'mjpython demo_grasp_update.py --view angle --variant before',
              'mjpython demo_grasp_update.py --view angle --variant after',
              '```', '', '打开 index.html 可切换章节；summary.json、各案例 CSV/NPZ 和方案 JSON 为原始证据。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


PRESENTATION = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>抓握方案更新 · MuJoCo 实测对照</title>
<style>
:root{color-scheme:dark;--bg:#0d1723;--text:#eef4fa;--muted:#acbbcd;--line:#2a394d;--a:#f7b267;--b:#59d4bc}[hidden]{display:none!important}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.7 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1200px;padding:44px 26px 64px;margin:auto}h1{font-size:clamp(27px,4vw,42px);line-height:1.25;letter-spacing:-.04em;margin:14px 0}h2{font-size:22px;margin:10px 0}p{margin:10px 0}small,.muted{color:var(--muted)}.eyebrow{color:var(--b);letter-spacing:.12em;font-size:12px}nav{display:flex;gap:9px;flex-wrap:wrap;margin:25px 0 20px}button{font:inherit;color:var(--text);background:transparent;border:1px solid var(--line);border-radius:9px;padding:9px 16px;cursor:pointer}button[aria-pressed=true]{background:var(--text);color:var(--bg)}button:focus-visible{outline:3px solid var(--b);outline-offset:3px}video,img{display:block;width:100%;border-radius:10px;background:#101b2a}.facts{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin:22px 0}.fact{border-top:3px solid var(--a);padding-top:14px}.fact.after{border-color:var(--b)}.value{font-size:30px;font-variant-numeric:tabular-nums}.section{border-top:1px solid var(--line);padding-top:22px;margin-top:30px}a{color:var(--b)}code{font-size:14px;white-space:pre-wrap;overflow-wrap:anywhere}.command{padding:14px;background:#172435;border-radius:9px}details{margin:18px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}td{overflow-wrap:anywhere}ul{padding-left:22px}li{margin:9px 0}@media(max-width:600px){main{padding:28px 16px}.facts{grid-template-columns:1fr}.value{font-size:25px}nav button{padding:8px 10px}}
</style></head><body><main>
<div class="eyebrow">MUJOCO / GRASP PLAN DEMO</div>
<h1>从一个抓点，到完整抓握方案</h1>
<p class="muted">同一场景，只改一个参数。看角度、夹持力与速度如何改变整段开门结果。</p>
<p><small>人工配置方案 · 物理仿真实测 · 不代表网络自动优化或真实机器人验证</small></p>
<nav id="chapters" aria-label="对照章节"></nav>
<h2 id="case-title"></h2><p id="condition" class="muted"></p>
<video id="player" controls playsinline preload="metadata" aria-label="两套抓握方案的真实仿真轨迹"></video>
<img id="gif" alt="抓握方案物理实测对比动画" hidden>
<div class="facts" id="facts" aria-live="polite"></div>
<p id="explanation"></p><p id="practical" class="muted"></p>
<details><summary>查看可复现的两套方案参数</summary><table><thead><tr><th>参数</th><th>原方案</th><th>调整方案</th></tr></thead><tbody id="parameters"></tbody></table><p id="raw"></p></details>
<div class="section"><h2>这次更新已经带来了什么？</h2>
<ul><li><strong>扩大动作表达：</strong>抓握不再只是一个点，还包含接近角度、肘分支、速度安排和驱动力上限。</li>
<li><strong>能比较整段过程：</strong>接近全开仍可能自碰撞，能抓住仍可能滑脱；不能只凭“够到了”或门角度判断成功。</li>
<li><strong>可追溯：</strong>每次尝试保留方案、实际接触力、漂移、轨迹和终止原因，可重放和用于后续训练。</li></ul>
<h2>下一步才能做到的事</h2><p class="muted">自动搜索最优方案、未知初始位置的可靠预测、视觉遮挡处理和在线重抓均尚未实现。当前演示从预抓取姿态开始。</p>
<p><a href="REPORT.md">实测报告</a> · <a href="summary.json">原始结果汇总</a> · <a href="grasp_update_demo.webm" id="full-video">完整四段演示</a></p>
<details><summary>在 MuJoCo 中实时运行当前章节</summary><pre class="command"><code id="commands"></code></pre><small>先激活已安装 MuJoCo 的 Python 环境；Mac 实时窗口使用 mjpython。</small></details>
</div></main><script>
const summary=__DATA__;const hasVideo=__VIDEO__;
const $=id=>document.getElementById(id);
const reasons={full_open_and_hold:'完全打开并稳定保持',unexpected_collision:'前臂与右手指自碰撞',slip_or_contact_loss:'滑移告警触发停止',arm_workspace_limit:'后续关节目标越界'};
const fields=[['height_offset_m','抓握高度偏移 / m'],['approach_angle_deg','接近偏角 / °'],['ik_branch','肘分支'],['grip_force_limit_N','每指驱动力上限 / N'],['timing.open_s','开门时长 / s']];
const get=(o,path)=>path.split('.').reduce((v,k)=>v[k],o);
function select(index){const c=summary.comparisons[index];$('case-title').textContent=c.title;$('condition').textContent=c.condition;
document.querySelectorAll('nav button').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===index)));
const player=$('player');player.pause();player.hidden=!hasVideo;$('gif').hidden=hasVideo;
player.poster=c.key+'_final.png';player.src=c.key+'.webm';if(hasVideo)player.load();else $('gif').src=c.key+'.gif';
$('facts').replaceChildren();c.variants.forEach((v,i)=>{const div=document.createElement('div');div.className='fact'+(i?' after':'');const label=document.createElement('div');label.textContent=v.label;const value=document.createElement('div');value.className='value';value.textContent=v.result.final_angle_deg.toFixed(2)+'°';const status=document.createElement('div');status.textContent=reasons[v.result.stop_reason]||v.result.stop_reason;div.append(label,value,status);$('facts').append(div)});
$('explanation').textContent=c.explanation;$('practical').textContent=c.practical;
$('parameters').replaceChildren();fields.forEach(([key,label])=>{const tr=document.createElement('tr');[label,...c.variants.map(v=>get(v.result.execution_plan,key))].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td)});$('parameters').append(tr)});
$('raw').replaceChildren();c.variants.forEach(v=>{const a=document.createElement('a');a.href=v.name+'_plan.json';a.textContent=v.label+' JSON';a.style.marginRight='20px';$('raw').append(a)});
$('commands').textContent='# 在项目根目录运行，先激活 Python 环境\n\nmjpython demo_grasp_update.py --view '+c.key+' --variant before\nmjpython demo_grasp_update.py --view '+c.key+' --variant after';}
summary.comparisons.forEach((c,i)=>{const b=document.createElement('button');b.type='button';b.textContent=['01 角度','02 夹持力','03 速度','04 能力边界'][i];b.addEventListener('click',()=>select(i));$('chapters').append(b)});
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
