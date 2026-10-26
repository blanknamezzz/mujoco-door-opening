# 抓握方案更新对比 Demo

当前目录已同步 `GraspPlan` 和参数化控制器。旧 `simulate.py`、`planning.py`
保存在 `reproduced/before_grasp_update/`，原始参考结果未改动。

## 直接观看

打开 `reproduced/grasp_update_demo/index.html`。可切换角度、夹持力、速度、能力边界四个章节，
观看双侧实测动画、参数和结果。`grasp_update_demo.webm` 是完整视频；每组也有独立 GIF。

## 生成与复现

在 Mac 上运行：

```bash
# 在仓库根目录运行，并先激活已安装 requirements.txt 的环境
source .venv/bin/activate
python demo_grasp_update.py
```

也可以分别运行物理计算与渲染：

```bash
python demo_grasp_update.py --physics-only
python demo_grasp_update.py --render-only
```

导出 WebM 需要 FFmpeg。程序优先使用 PATH 中的 `ffmpeg`，其次使用本机已有的
Playwright FFmpeg；也可以 `--ffmpeg /绝对路径/ffmpeg` 指定。没有 FFmpeg 时仍导出 GIF。
中文标题需要系统中文字体。当前 Mac 已有字体与编码器，无须额外安装。

## 在实时 MuJoCo 窗口中看某一侧

```bash
mjpython demo_grasp_update.py --view angle --variant before
mjpython demo_grasp_update.py --view angle --variant after
```

`angle` 可换成 `force`、`speed`、`limit`。`before/after` 是手动指定的两套方案，
**不是新旧神经网络的对照**。实时运行输出保存在 `reproduced/grasp_update_demo/live/`。

## 演示说明

1. **角度**：同一底座位置 `(0.8,-1.0)`，只改接近偏角 `0° → +8°`。
   原方案在接近全开时前臂与右手指自碰撞，调整后全开并保持。
   这个结果说明整段姿态的重要性；并不说明只有 8° 可行，或 8° 已是最优解。
2. **夹持力**：相同摩擦 `μ=0.2`，只改每指驱动力上限 `2 N → 25 N`。
   原方案拉动时滑移，调整后完成任务。该上限与实际接触力是两个量，CSV 记录实际接触力。
3. **速度**：同一抓点与同一圆弧，开门时间 `2 s → 14 s`。
   快速方案中途滑移，慢速方案成功。未声称越慢越好或 14 秒最优。
4. **能力边界**：在 `(0.3,-0.9)`，每指上限 `25 N → 35 N`。
   两者仍在约 42.5° 因后续关节目标越界停止，此时仍抓住；增加力不能扩大工作空间。

每组只改变一个方案字段，程序会自动检查；还检查预期成功/失败和关键终止原因。
视频为 2 倍模拟时间，失败侧停止后冻结，不能把画面冻结理解成继续运行并保持接触。
总结果、方案、CSV 和 NPZ 都保存在同一输出目录中，可独立核查。

## 已实现与后续能力

本次展示的是更丰富的动作参数、可追溯的对照试验及方案重放。
这有助于研究狭窄空间、有限夹持力、较滑接触表面和动作速度的取舍，
但没有验证真实材料或机器人。表面摩擦参数未经实测标定。

当前从预抓取姿态初始化，采用固定肘分支、直线接近和铰链圆弧控制。
本目录不包含 H-JEPA 学习流水线。自动候选搜索、全程可行性筛选、
学习成功概率和执行过程中的重抓纠错仍需进一步实现。
