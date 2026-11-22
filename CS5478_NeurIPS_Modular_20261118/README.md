# CS5478 Team 42 Final Report

This directory contains the completed English final report. The main source is `main.tex`, and the compiled reading copy is `main.pdf`. The report was updated against the repository code and saved experiment records on 2026-11-21.

The 2026-11-22 revision improved figure placement and presentation. Figures 2 and 3 and Tables 1, 2, and 3 follow their respective introductions. Figure 4 compares accuracy and time, and Table 3 groups the inspection-mode comparison and training ablation separately. All four figures use a consistent scientific style, with the experimental data preserved.

The prose revision condensed repeated audit, checkpoint-selection, and scope statements and named the three test cohorts A, B, and C. Figure 3 now shows the primary policy's recorded replay from cohort C, and the methods section explains the initialization source. Archived experiment identifiers and checkpoint identities are listed in `EVIDENCE.md` and `COHORTS.json`.

The latest reading copy is `CS5478_Team42_Final_Report_Aligned_20261122.pdf`, with the same content in `main.pdf`. The Overleaf source package is `CS5478_Team42_Final_Report_Aligned_Source_20261122.zip`.

The 40.3% reduction in Section 5.2 refers to additional inspections. A clickable GitHub repository link appears at the bottom left of the first page.

## Content and results

The report presents a continuous door-opening system with inspection, positioning, and grasping. It describes physical execution, downstream supervision, staged LoRA training, and frozen evaluation.

- The primary policy on the latest test cohort C achieves 70/72 correct decisions, opens 50/52 candidate-feasible scenes, correctly refuses all 20 remaining scenes, and has zero observed collisions.
- Learned and always inspection have identical task outcomes on this cohort. Learned inspection uses 43 inspections versus 72, with mean episode durations of 38.31 versus 55.68 simulated seconds.
- The same policy scores 68/72, 72/72, and 70/72 on cohorts A, B, and C, respectively, for a total of 210/216.
- The training studies cover checkpoint selection across both branches, PCGrad, output retention, and an independent fixed-base grasp-angle comparison.

## Editing and compilation

The report has eight body pages and one reference page.

1. Edit the title and authors in `title.tex`. The current authors are Zheyu Zheng and Haoming Li, as listed in the existing project materials.
2. Edit the report in `sections/`; the methods and experiments load their respective subdirectories.
3. Use the root `main.tex` as the main document, pdfLaTeX as the compiler, and BibTeX for references.

```text
latexmk -pdf main.tex
```

Alternatively, run `pdflatex main`, `bibtex main`, `pdflatex main`, and `pdflatex main` in that order.

The original `neurips_2026.sty` is preserved. The text, figures, tables, and references compile from source. The `\showfigurestrue` setting in `preamble.tex` controls figure inclusion.

## Figures and evidence

`figures/FIGURE_GUIDE.md` describes the four active figures, and `EVIDENCE.md` maps the main numerical results to experiment records. The figures use vector PDFs with recorded or rendered image pixels embedded where needed. Use `scripts/build_report_figures.py` to generate the figures and `scripts/render_diagnostic_frame.py` to render the recorded stopping pose.

See `OVERLEAF_UPDATE.md` for upload and replacement instructions. The initial report is preserved in the ignored directory `runs/final_report_revision_20261121/original_report/`, and the version before the figure revision is in `runs/final_report_visual_polish_20261122/before_polish/`.
