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
        dict(key='direct', title='A: Direct path and complete opening', base=[.6, -1.], angle=8., samples=240,
             explanation='The direct local path passes, using the start-goal edge without unnecessary random sampling.'),
        dict(key='detour', title='B: Blocked direct path, roadmap detour', base=[.6, -1.], angle=8., samples=500,
             model_xml=xml, obstacle=obstacle.tolist(),
             explanation='Same station and target as A, with a small obstacle. After direct failure, sample joints and find a detour using the existing lift.'),
        dict(key='limited_budget', title='C: Same obstacle, insufficient sampling budget', base=[.6, -1.], angle=8., samples=0,
             model_xml=xml, obstacle=obstacle.tolist(),
             explanation='Same physics as B with sampling disabled. Direct failure leaves no alternative edge; no path found within budget does not prove infeasibility.'),
        dict(key='partial', title='D: Reachable handle, limited opening', base=[.3, -.9], angle=0., samples=240,
             explanation='PRM establishes a geometric pregrasp path only. Physical opening still encounters downstream workspace limits.'),
        dict(key='unreachable', title='E: Target beyond this branch workspace', base=[2., -1.2], angle=8., samples=240,
             explanation='No legal pregrasp IK exists on the selected elbow branch, so exclude this target before sampling. Other grasps or base motion are separate choices.'),
    ]


def run_case(case, output, viewer=None):
    plan = GraspPlan((.8, -.11, .85), approach_angle_deg=case['angle'])
    exp = Experiment(base=case['base'], model_xml=case.get('model_xml'))
    # Reproduce the published examples' historical sampling stream explicitly;
    # newly configured PRM experiments default to42.
    roadmap = Roadmap(exp, PRMSettings(samples=case['samples'], neighbors=18, seed=7))
    query = roadmap.plan(plan, include_graph=True)
    execution, outcome, frames = None, None, []
    folder = output/case['key']
    folder.mkdir(parents=True, exist_ok=True)
    if case.get('model_xml'):
        (folder/'scene.xml').write_text(case['model_xml'],encoding='utf-8')
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
        query = Roadmap(exp, PRMSettings(samples=case['samples'], neighbors=18, seed=7)).plan(plan)
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
    html = (ROOT/'prm_examples_template.html').read_text(encoding='utf-8')
    html = html.replace('__PRM_DATA__', json.dumps(results, ensure_ascii=False).replace('</', '<\\/'))
    (args.output/'index.html').write_text(html, encoding='utf-8')
    lines = ['# PRM pregrasp paths: five physical examples', '',
        'Nodes are lift/shoulder/elbow/wrist configurations. Initial joints are [0 m, -90 degrees, -120 degrees, 0 degrees], fingers open, door closed.',
        'Door and robot gain no degrees of freedom. B/C share one added obstacle with different sampling budgets.', '',
        '| Example | Base, m | PRM status | Valid sampled nodes | Waypoints | Physical opening outcome |',
        '| --- | --- | --- | --- | --- | --- |']
    for d in results:
        q, o = d['query'], d['opening'] or {}
        lines.append(f"| {d['title']} | {d['base']} | {q['status']} | {q.get('accepted_nodes',0)} | {len(q['path_q'])} | {o.get('stop_reason','opening not executed')}; {o.get('final_angle_deg','—')}° |")
    lines += ['', 'Open index.html for samples, candidate edges, final paths, and top-view replay of physically integrated joints.',
        'Gray edges are unverified candidates; every green final edge passes the configured discrete check. Crossings in projection need not connect four-dimensional nodes.',
        'Finite-step checks are not continuous collision detection or global infeasibility proofs. Execute preparation and opening after finding a path.', '',
        '```bash', 'python prm_examples.py', 'mjpython prm_examples.py --view detour', '```', '']
    (args.output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    if args.render:
        render_recorded(next(c for c in cases() if c['key'] == args.render), args.output)


if __name__ == '__main__':
    main()
