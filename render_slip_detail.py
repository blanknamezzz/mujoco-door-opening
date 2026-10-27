"""Render the recorded grip-to-slip-to-loss transition at normal speed."""
import argparse
from result_paths import case_directory
import csv
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from render_results import font
from simulate import Experiment, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results",
                        help="Directory containing low_friction.npz and .csv")
    args = parser.parse_args()
    output = case_directory(args.output, "low_friction")
    states = np.load(output/"low_friction.npz")
    rows = list(csv.DictReader((output/"low_friction.csv").open()))
    result = json.loads((output/"low_friction.json").read_text())
    e = Experiment(friction=.005)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = e.d.xpos[e.hbody]
    cam.distance, cam.azimuth, cam.elevation = .65, 115, -20
    valid = np.flatnonzero((states["time"] >= 2.7) & (states["time"] <= 5.5))
    indices = valid[::4].tolist()
    frames = []
    with mujoco.Renderer(e.m, height=540, width=720) as renderer:
        for i in indices:
            row = rows[i]
            e.d.qpos[:] = states["qpos"][i]
            mujoco.mj_forward(e.m, e.d)
            renderer.update_scene(e.d, camera=cam)
            frame = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(frame)
            lost, slipping = row["contact_lost"] == "True", row["slip_detected"] == "True"
            status = "CONTACT LOST" if lost else "SLIP ALARM" if slipping else "BILATERAL CONTACT" if row["bilateral_contact"] == "True" else "CLOSING"
            colour = (255, 130, 100) if lost else (255, 200, 90) if slipping else (150, 240, 210)
            draw.rectangle((0, 0, 720, 105), fill=(18, 28, 40))
            draw.text((16, 8), f"mu = 0.005  |  t = {float(row['time_s']):.2f} s  |  {status}", fill=colour, font=font(21))
            draw.text((16, 42), f"Normal force L/R: {float(row['left_normal_N']):.1f} / {float(row['right_normal_N']):.1f} N", fill="white", font=font(19))
            speed = 1000*max(float(row["left_tangent_speed_m_s"]), float(row["right_tangent_speed_m_s"]))
            value = "n/a (no contact)" if lost else f"{speed:.1f} mm/s"
            draw.text((16, 73), f"Tangential relative speed: {value}", fill="white", font=font(19))
            frames.append(frame)
    durations = [40]*len(frames)  # Four 10 ms log samples per frame: 1x speed.
    durations[-1] = 1040
    frames[0].save(output/"low_friction_closeup.gif", save_all=True,
                   append_images=frames[1:], duration=durations, loop=0)
    frames[-1].save(output/"low_friction_detached.png")
    model = e.m
    params = dict(case="low_friction", mujoco_version=mujoco.__version__,
                  friction=dict(sliding_coefficient=.005, normal_case=.8,
                      scope="All geom sliding coefficients are set together; handle/pad contact is the grasp under test.",
                      torsional_coefficient=float(model.geom_friction[e.bar, 1]),
                      rolling_coefficient=float(model.geom_friction[e.bar, 2]), condim=4),
                  gripper=dict(position_kp_N_per_m=1800, position_kv_N_s_per_m=8,
                      actuator_force_limit_each_N=35, closed_joint_target_m=0,
                      open_joint_target_m=.04, finger_joint_range_m=[0,.045],
                      pad_dimensions_m=[.064,.020,.050]),
                  door=dict(panel_mass_kg=8, hinge_friction_Nm=1.2,
                      hinge_damping_Nm_s_per_rad=.6, handle_axis_initial_world_m=[.8,-.11,.85],
                      handle_radius_m=.015, handle_cylinder_length_m=.3),
                  base_xy_m=[.6,-1.0], timestep_s=float(model.opt.timestep),
                  log_interval_s=.01, opening_start_s=3.5, opening_reference_duration_s=10,
                  slip_detection=dict(min_normal_each_N=1, tangential_speed_m_s=.01,
                      relative_translation_m=.008, relative_rotation_deg=5, persistence_s=.08),
                  contact_loss_detection=dict(min_normal_each_N=1,persistence_s=.1),
                  diagnostic_continue_after_alarm=True,
                  observed=dict(first_slip_s=result["first_slip_s"],
                      first_contact_loss_s=result["first_contact_loss_s"],
                      max_door_angle_deg=result["max_angle_deg"]),
                  note="Deliberately extreme low-friction fault case; not a calibrated real material.")
    (output/"low_friction_parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(output/"low_friction_closeup.gif")


if __name__ == "__main__":
    main()
