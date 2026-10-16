"""Calibrated RGB-D baseline: gold colour mask + vertical cylinder fitting.

estimate() reads only RGB, depth and camera calibration. It never receives
MuJoCo geometry IDs, sites, handle positions, or segmentation ground truth.
Assumptions: one gold vertical rod, known radius, full initial visibility.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def capture(path):
    import mujoco
    from simulate import Experiment
    e = Experiment()
    width, height = 960, 720
    with mujoco.Renderer(e.m, height=height, width=width) as renderer:
        renderer.update_scene(e.d, camera="perception")
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(e.d, camera="perception")
        depth = renderer.render().copy()
    cid = e.m.camera("perception").id
    focal = height/(2*np.tan(np.deg2rad(e.m.cam_fovy[cid])/2))
    K = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1.]])
    T = np.eye(4)
    T[:3, :3] = e.d.cam_xmat[cid].reshape(3, 3)
    T[:3, 3] = e.d.cam_xpos[cid]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rgb=rgb, depth_m=depth, K=K, world_from_camera=T)
    Image.fromarray(rgb).save(path.with_suffix(".png"))


def estimate(path, radius=.015, write_preview=False):
    obs = np.load(path)
    rgb, depth, K, T = [obs[k] for k in ["rgb", "depth_m", "K", "world_from_camera"]]
    r, g, b = rgb.astype(float).transpose(2, 0, 1)
    mask = (r > 100) & (g > 70) & (b < .4*g) & (r > 1.15*g) & np.isfinite(depth) & (depth > 0)
    v, u = np.where(mask)
    if len(u) < 80:
        raise ValueError("Handle detection failed: too few gold RGB-D pixels.")
    z = depth[v, u]
    # MuJoCo camera looks down -Z; image row increases downwards.
    pc = np.column_stack(((u+.5-K[0, 2])*z/K[0, 0], -(v+.5-K[1, 2])*z/K[1, 1], -z))
    pw = pc@T[:3, :3].T+T[:3, 3]
    lo, hi = np.quantile(pw[:, 2], [.01, .99])
    core = pw[(pw[:, 2] > lo+.04) & (pw[:, 2] < hi-.04)]
    # Fit the visible cylindrical side, excluding spherical capsule ends.
    xy = core[:, :2]
    A = np.column_stack((2*xy, np.ones(len(xy))))
    cx, cy, const = np.linalg.lstsq(A, np.sum(xy*xy, axis=1), rcond=None)[0]
    fitted_radius = np.sqrt(max(0, const+cx*cx+cy*cy))
    residual = np.sqrt(np.mean((np.linalg.norm(xy-[cx, cy], axis=1)-radius)**2))
    if abs(fitted_radius-radius) > .004 or residual > .004:
        raise ValueError("Detected points are inconsistent with the known handle radius.")
    result = dict(center_world_m=[float(cx), float(cy), float((lo+hi)/2)],
                  axis_world=[0., 0., 1.], visible_z_bounds_m=[float(lo), float(hi)],
                  known_radius_m=radius, fitted_radius_m=float(fitted_radius),
                  fit_rms_m=float(residual), mask_pixels=len(u),
                  method="RGB gold mask + depth backprojection + vertical cylinder fit",
                  assumptions=["known vertical handle axis and nominal radius", "calibrated camera", "one gold handle", "both handle ends visible; partial middle occlusion allowed"])
    if write_preview:
        out = rgb.copy()
        out[mask] = (0.4*out[mask]+.6*np.array([0, 255, 80])).astype(np.uint8)
        Image.fromarray(out).save(Path(path).with_name("handle_detection.png"))
        Path(path).with_name("handle_estimate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["capture", "estimate"])
    p.add_argument("--observation", type=Path, default=ROOT/"results"/"observation.npz")
    args = p.parse_args()
    if args.command == "capture":
        capture(args.observation)
    print(json.dumps(estimate(args.observation, write_preview=True), indent=2))
