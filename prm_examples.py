"""Reproducible PRM examples and an offline interactive roadmap/robot replay."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from grasp_plan import GraspPlan
from prm import PRMSettings, Roadmap, execute_path
from simulate import Experiment, ROOT, save_trial
from train_grasp_rl import write_json


def render_recorded(case, output):
    """Render the saved physical trajectory, never a synthetic success path."""
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    folder = output/case['key']
    if not (folder/'preparation.npz').exists():
        raise ValueError('No executed preparation trajectory for this case')
    with np.load(folder/'preparation.npz') as a:
        poses, times = a['qpos'].copy(), a['time'].copy()
    preparation_end = float(times[-1])
    if (folder/'opening.npz').exists():
        with np.load(folder/'opening.npz') as a:
            poses = np.vstack([poses, a['qpos']])
            times = np.r_[times, preparation_end+a['time']]
    exp = Experiment(base=case['base'], model_xml=case.get('model_xml'))
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 18) if Path('/System/Library/Fonts/Supplemental/Arial.ttf').exists() else ImageFont.load_default()
    images = []
    with mujoco.Renderer(exp.m, height=480, width=640) as renderer:
        for t in np.arange(0, times[-1]+1e-8, 2/12):
            index = min(int(np.searchsorted(times, t)), len(times)-1)
            exp.d.qpos[:] = poses[index]
            mujoco.mj_forward(exp.m, exp.d)
            renderer.update_scene(exp.d, camera='overview')
            frame = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0,0,640,58), fill='#13283a')
            stage = 'PRM path execution' if t <= preparation_end else 'Grasp + open + hold'
            draw.text((14,5), stage+'  |  actual MuJoCo trajectory  |  2x', font=font, fill='white')
            draw.text((14,31), f't={t:.1f}s   lift={poses[index, exp.lift_q]*100:+.1f}cm   door={math.degrees(poses[index,exp.door_q]):.1f}deg', font=font, fill='white')
            images.append(frame)
    path = output/f"{case['key']}.gif"
    images[0].save(path, save_all=True, append_images=images[1:], duration=83, loop=0)
    print('Rendered', path, flush=True)


def cases():
    # Put a small obstacle on the DIRECT joint interpolation, with endpoints
    # clear. The lift has room to pass above/below it. No new robot joints.
    plan = GraspPlan((.8, -.11, .85), approach_angle_deg=8.)
    exp = Experiment()
    probe = Roadmap(exp)
    direct = probe.plan(plan)
    a, b = np.array(direct['initial_q']), np.array(direct['goal_q'])
    probe.clear(a+.55*(b-a))
    obstacle = probe.probe.xpos[exp.m.body('hand').id].copy()
    tree = ET.parse(ROOT/'scene.xml')
    ET.SubElement(tree.getroot().find('worldbody'), 'geom', name='prm_obstacle', type='box',
                  pos=' '.join(map(str, obstacle)), size='.035 .035 .035', rgba='.8 .2 .2 1')
    xml = ET.tostring(tree.getroot(), encoding='unicode')
    return [
        dict(key='direct', title='A：直接到达并全开', base=[.6, -1.], angle=8., samples=240,
             explanation='直连局部路径通过检查，使用起点到目标的直接边。无需为了展示随机采样而强行建图。'),
        dict(key='detour', title='B：直连受阻，路线图绕行', base=[.6, -1.], angle=8., samples=500,
             model_xml=xml, obstacle=obstacle.tolist(),
             explanation='与 A 相同的站位和目标，增加一个小障碍。直连失败后采样关节姿态，搜索利用现有升降轴的绕行路径。'),
        dict(key='limited_budget', title='C：同一障碍，采样预算不足', base=[.6, -1.], angle=8., samples=0,
             model_xml=xml, obstacle=obstacle.tolist(),
             explanation='与 B 的物理场景完全相同，但关闭随机采样。直连失败且无替代边，返回本次未找到，不能据此判定位置绝对不可行。'),
        dict(key='partial', title='D：能到把手，开门中途受限', base=[.3, -.9], angle=0., samples=240,
             explanation='PRM 只证明到预抓取位置的几何路径。实际开门仍会遇到后续工作空间限制。'),
        dict(key='unreachable', title='E：目标超出当前分支工作空间', base=[2., -1.2], angle=8., samples=240,
             explanation='预抓取目标没有当前肘分支的合法逆运动学解，先排除该目标，不浪费采样。其他抓点或移动底座属于另一个问题。'),
    ]


def run_case(case, output, viewer=None):
    plan = GraspPlan((.8, -.11, .85), approach_angle_deg=case['angle'])
    exp = Experiment(base=case['base'], model_xml=case.get('model_xml'))
    roadmap = Roadmap(exp, PRMSettings(samples=case['samples'], neighbors=18))
    query = roadmap.plan(plan, include_graph=True)
    execution, outcome, frames = None, None, []
    folder = output/case['key']
    folder.mkdir(parents=True, exist_ok=True)
    if case.get('model_xml'):
        (folder/'scene.xml').write_text(case['model_xml'])
    if query['path_found']:
        execution, rows, states = execute_path(exp, query['path_q'], viewer=viewer, record=True)
        if rows:
            with (folder/'preparation.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            np.savez_compressed(folder/'preparation.npz', qpos=np.array(states),
                                time=np.array([r['time_s'] for r in rows]))
        for row, state in zip(rows[::5], states[::5]):
            frames.append([row['time_s'], *state[[exp.lift_q, *exp.arm_q]].tolist(),
                           float(state[exp.door_q])])
        if execution['success']:
            outcome, opening_rows, opening_states = exp.trial(plan=plan, reset_before_trial=False, viewer=viewer)
            save_trial('opening', outcome, opening_rows, opening_states, folder)
            for row, state in zip(opening_rows[::10], opening_states[::10]):
                frames.append([execution['duration_s']+row['time_s'],
                               *state[[exp.lift_q, *exp.arm_q]].tolist(), float(state[exp.door_q])])
    if not frames:
        frames = [[0., *list(roadmap.settings.initial_q), 0.]]
    data = dict(key=case['key'], title=case['title'], base=case['base'], angle=case['angle'],
        explanation=case['explanation'], obstacle=case.get('obstacle'), query=query,
        execution=execution, opening=outcome, frames=np.round(frames, 6).tolist())
    plan.save(folder/'plan.json')
    write_json(folder/'analysis.json', data)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'reproduced/prm_examples')
    p.add_argument('--view', choices=[c['key'] for c in cases()])
    p.add_argument('--render', choices=[c['key'] for c in cases()], help='Render actual saved trajectories as GIF')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.view:
        # Use the same model for rendering and execution, with passive door.
        import mujoco.viewer
        case = next(c for c in cases() if c['key'] == args.view)
        exp = Experiment(base=case['base'], model_xml=case.get('model_xml'))
        plan = GraspPlan((.8, -.11, .85), approach_angle_deg=case['angle'])
        query = Roadmap(exp, PRMSettings(samples=case['samples'], neighbors=18)).plan(plan)
        print(json.dumps(query, indent=2), flush=True)
        if query['path_found']:
            with mujoco.viewer.launch_passive(exp.m, exp.d) as window:
                execution, _, _ = execute_path(exp, query['path_q'], viewer=window)
                if execution['success']:
                    result, _, _ = exp.trial(plan=plan, reset_before_trial=False, viewer=window)
                    print(json.dumps(result, indent=2), flush=True)
        return
    results = []
    for case in cases():
        data = run_case(case, args.output)
        results.append(data)
        print(case['key'], data['query']['status'], (data['opening'] or {}).get('stop_reason'), flush=True)
    write_json(args.output/'summary.json', results)
    html = (ROOT/'prm_examples_template.html').read_text()
    html = html.replace('__PRM_DATA__', json.dumps(results, ensure_ascii=False).replace('</', '<\\/'))
    (args.output/'index.html').write_text(html, encoding='utf-8')
    lines = ['# PRM 预抓取路径：五个实际例子', '',
        '节点为升降、肩、肘、腕四个关节配置。初始姿态固定为 [0 m, −90°, −120°, 0°]，夹爪张开、门关闭。',
        '门和机器人未增加自由度。B/C 增加同一个小障碍，比较不同采样预算。', '',
        '| 例子 | 底座位置 m | PRM 状态 | 采样有效节点 | 路径点数 | 实际开门结果 |',
        '| --- | --- | --- | --- | --- | --- |']
    for d in results:
        q, o = d['query'], d['opening'] or {}
        lines.append(f"| {d['title']} | {d['base']} | {q['status']} | {q.get('accepted_nodes',0)} | {len(q['path_q'])} | {o.get('stop_reason','未执行开门')}；{o.get('final_angle_deg','—')}° |")
    lines += ['', '打开 index.html 查看采样点、候选连边、最终路径，以及真实物理积分得到的关节轨迹俯视重放。',
        '灰边是尚未验证的候选；绿色最终路径的每段已按配置空间步长检查。二维投影中的交叉不代表四维节点相交。',
        '碰撞检查使用有限步长，不是连续碰撞检测或全局不可行证明；路径找到后还要执行控制和开门验证。', '',
        '```bash', 'python prm_examples.py', 'mjpython prm_examples.py --view detour', '```', '']
    (args.output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    if args.render:
        render_recorded(next(c for c in cases() if c['key'] == args.render), args.output)


if __name__ == '__main__':
    main()
