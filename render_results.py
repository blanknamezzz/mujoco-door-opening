"""Render saved physical trajectories; replay assignments do not run control."""
import argparse
import csv
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from simulate import Experiment, ROOT


def font(size):
    for name in ["/System/Library/Fonts/Helvetica.ttc", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_case(name, output, animated=True):
    base = (.3, -.9) if name == "workspace_limit" else (.6, -1.)
    e = Experiment(base, friction=.005 if name == "low_friction" else .8)
    arr = np.load(output/f"{name}.npz")
    rows = list(csv.DictReader((output/f"{name}.csv").open()))
    failure = name == "workspace_limit"
    elbow_limit = np.degrees(e.m.jnt_range[e.m.joint("elbow").id, 0])
    indices = list(range(0, len(arr["qpos"]), 20))
    if failure and indices[-1] != len(arr["qpos"])-1:
        indices.append(len(arr["qpos"])-1)
    frames = []
    with mujoco.Renderer(e.m, height=540, width=720) as r:
        for i in indices:
            e.d.qpos[:] = arr["qpos"][i]
            mujoco.mj_forward(e.m, e.d)
            r.update_scene(e.d, camera="overview")
            frame = Image.fromarray(r.render().copy())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 720, 75), fill=(18, 28, 40))
            alarm = rows[i]["slip_detected"] == "True"
            draw.text((18, 10), f"{name}   |   t = {arr['time'][i]:.1f} s   |   door = {float(rows[i]['door_deg']):.1f} deg", font=font(21), fill="white")
            draw.text((18, 42), f"Phase: {rows[i]['phase']}   |   slip alarm: {'YES' if alarm else 'no'}", font=font(18), fill=(255, 180, 80) if alarm else (170, 225, 220))
            if failure and i == indices[-1]:
                draw.rectangle((0, 75, 720, 135), fill=(80, 25, 25))
                draw.text((18, 79), "STOP: next target exceeds elbow joint range", font=font(20), fill=(255, 220, 190))
                draw.text((18, 108), f"Elbow lower limit: {elbow_limit:.1f} deg | grip maintained", font=font(18), fill="white")
            frames.append(frame)
        if animated:
            durations = [100]*len(frames)
            if failure:
                durations[-1] = 1800  # Display the termination state, not further physics.
            frames[0].save(output/f"{name}.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0)
        frames[-1].save(output/f"{name}_final.png")
        # Close-up at established grasp, with free camera independent of the
        # camera that generated the RGB-D perception input.
        idx = np.argmin(np.abs(arr["time"]-3.4))
        e.d.qpos[:] = arr["qpos"][idx]
        mujoco.mj_forward(e.m, e.d)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = e.d.xpos[e.hbody]
        cam.distance = .55
        cam.azimuth = 115
        cam.elevation = -20
        r.update_scene(e.d, camera=cam)
        Image.fromarray(r.render().copy()).save(output/f"{name}_grasp_closeup.png")
    return frames


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT/"results")
    p.add_argument("--case", choices=["all", "nominal", "low_friction", "workspace_limit"], default="all")
    args = p.parse_args()
    if args.case != "all":
        render_case(args.case, args.output)
        raise SystemExit(0)
    good = render_case("nominal", args.output)
    bad = render_case("low_friction", args.output)
    render_case("workspace_limit", args.output)
    panels = [good[0], good[17], good[45], good[-1]]
    board = Image.new("RGB", (1440, 1080), "white")
    for i, panel in enumerate(panels):
        board.paste(panel, ((i % 2)*720, (i//2)*540))
    board.save(args.output/"opening_sequence.png")
