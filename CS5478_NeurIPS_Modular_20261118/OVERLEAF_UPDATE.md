# Overleaf 更新

将本目录的源文件与 `figures/`、`sections/` 按原有层级上传到项目。已打包的 source ZIP 可以直接导入新项目。

1. 将根目录 `main.tex` 设为 Main document。
2. Compiler 选择 pdfLaTeX，使用项目默认 BibTeX 文献流程。
3. 更新正文时编辑 `sections/`，标题与作者编辑 `title.tex`。
4. 点击 Recompile，检查图表、交叉引用及参考文献。

本次更新包括 `main.tex`、`preamble.tex`、`title.tex`、全部章节和方法／实验子模块、F01/F04 图注，以及新增的 `figures/methods/report_system.pdf` 和 `figures/results/report_inspection_cost.pdf`。完整 source ZIP 包含上述文件。

正文已由 Writing note 模板改成正式英文报告，共 8 页正文和 1 页参考文献。新增图表为矢量 PDF，其余图像保留原文件。`neurips_2026.sty` 和 `references.bib` 沿用原版本，文献仅列出正文引用项。

数值核对入口为 `EVIDENCE.md`，本地生成脚本位于 `scripts/`。编译出的 `main.pdf` 用于直接阅读。

2026-11-22 排版更新将 Figure 2、Figure 3、Table 1、Table 2、Table 3 固定在相应介绍之后。新版 source ZIP 包含所需的 `float` 宏包声明和图表定位设置，导入后按上述方式编译即可。

最新版使用 `CS5478_Team42_Final_Report_Aligned_Source_20261122.zip`。其中包含统一的 A/B/C 测试集命名、更新后的 Figure 3 和初始化说明，以及四幅矢量图与 Table 3 的分组结构。导入完整源码包可同步更新正文和图表，并保留已有定位设置。
