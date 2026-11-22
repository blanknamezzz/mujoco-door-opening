ALL SENSOR DEMOS
13 clips use preserved physical logs; two fresh inspection clips use seed42 and a known-map geometry reference. These new clips are illustrative experiments, not a learned-policy test and not additional samples for the 140/144 result.

GitHub origin/main was fetched at 0ae4c4d. Its committed tree did not contain the original mobile trajectory logs or checkpoints. Local main was not changed. The missing original policy was not fabricated.

Signals: measured contact force, elbow position and range, door angle, contact flags, relative drift; PRM additionally shows lift motion. The preparation logger did not record force, so that interval stays missing. Active inspection shows base pose/speed/path/cost, joint and door angles, and handle contact forces. Zero handle force during inspection is expected because grasping has not begun.

Safety interpretation: joint-limit stop occurs before the measured joint hits the hard bound. Slip and loss are different events. Relative pose drift after detachment is free separation, not continuing surface slip. Geometric sweep rejection is restricted to this finite candidate set and known-map checker.

Reproduce from repository root:
python render_sensor_demo.py
python render_all_sensor_demos.py
PYTHONHASHSEED=42 python record_inspection_sensor_demo.py
Dependencies: MuJoCo, NumPy, Pillow, imageio-ffmpeg; Torch for inspection seed and existing imports. macOS requires graphics-service access.
