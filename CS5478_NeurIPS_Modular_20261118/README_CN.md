# CS5478 Team 42 Final Report

本目录为已完成的英文最终报告，主文件是 `main.tex`，阅读版为 `main.pdf`。2026-11-21 根据仓库代码及已保存实验记录更新。

2026-11-22 完成图表位置与表达优化：Figure 2、Figure 3、Table 1、Table 2、Table 3 紧随对应介绍；Figure 4 展示正确率与耗时，Table 3 分组呈现检查方式与训练消融。四幅图统一为论文风格，实验数据保持不变。

随后压缩了正文中的重复核验说明、选模解释和自我限定，并将三批测试统一命名为 A、B、C。Figure 3 已换成测试集 C 中主策略的实际回放，方法部分补充了初始化来源。内部实验编号与检查点身份见 `EVIDENCE.md` 和 `COHORTS.json`。

最新版阅读文件为 `CS5478_Team42_Final_Report_Aligned_20261122.pdf`，`main.pdf` 同步更新；Overleaf 源码包为 `CS5478_Team42_Final_Report_Aligned_Source_20261122.zip`。

## 内容与结果

报告围绕连续的 Inspection、Positioning、Grasping 开门系统，说明物理执行、下游监督、分阶段 LoRA 训练及冻结测试。

- 最新测试集 C 的主策略：70/72 正确决策，50/52 候选可行场景开门，20/20 正确拒绝，零观测碰撞。
- 同批始终检查与自主检查的任务结果一致；自主检查 43 次，始终检查 72 次；平均回合为 38.31 与 55.68 仿真秒。
- 同一策略在测试集 A、B、C 中分别为 68/72、72/72、70/72，合计 210/216。
- 训练研究涵盖双分支选模、PCGrad、输出保持及独立的固定底座角度学习对照。

## 编辑与编译

正文 8 页，参考文献 1 页。

1. 标题与作者信息位于 `title.tex`，当前使用已有项目材料中的 Zheyu Zheng、Haoming Li。
2. 正文按 `sections/` 拆分，方法与实验分别调用其子目录。
3. 以根目录 `main.tex` 为主文件，使用 pdfLaTeX，文献由 BibTeX 生成。

```text
latexmk -pdf main.tex
```

或依次运行 `pdflatex main`、`bibtex main`、`pdflatex main`、`pdflatex main`。

`neurips_2026.sty` 保留原样；正文、图表和文献均通过源文件编译。`preamble.tex` 中的 `\showfigurestrue` 控制插图。

## 图表与依据

`figures/FIGURE_GUIDE.md` 说明当前四幅插图；`EVIDENCE.md` 将主要数值映射到实验记录。四幅插图均以矢量 PDF 组织，图像部分保留记录或渲染得到的像素。生成脚本为 `scripts/build_report_figures.py`，停止姿态渲染脚本为 `scripts/render_diagnostic_frame.py`。

Overleaf 上传与替换见 `OVERLEAF_UPDATE.md`。初始报告保存在 ignored `runs/final_report_revision_20261121/original_report/`；本轮图表优化前的版本保存在 `runs/final_report_visual_polish_20261122/before_polish/`。
