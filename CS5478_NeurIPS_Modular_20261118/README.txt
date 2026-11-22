CS5478 Team 42 -- Final report, revised 2026-11-22

Figures 2/3 and Tables 1/2/3 follow their respective introductions.
The polished version redraws all four figures and groups the two comparison
types in Table 3. Experimental data are preserved. The report has eight body
pages and one reference page. The language revision condenses repeated audit,
selection, and scope statements while preserving definitions and results.

READ
  main.pdf is the compiled report. main.tex is the editable entry point.
  CS5478_Team42_Final_Report_Aligned_20261122.pdf is the named reading copy.
  title.tex contains the title and authors.
  sections/ contains completed English prose; methods and experiments are modular.
  EVIDENCE.md and COHORTS.json map paper cohorts and initialization to saved records.
  They also list the original experiment dates supplied by the project owner.

BUILD
  Select main.tex as the main document and pdfLaTeX as the compiler.
  latexmk -pdf main.tex
  Or: pdflatex main; bibtex main; pdflatex main; pdflatex main
  The original neurips_2026.sty is preserved.

RESULTS
  Latest cohort C primary: 70/72 correct, 50/52 feasible openings, 20/20 correct
  refusals, zero observed collisions; 43 inspections versus 72 for always
  inspecting, with identical per-scene task outcomes.
  Three disjoint geometry cohorts A/B/C: 68/72, 72/72, 70/72; total 210/216.

FIGURES
  Four figures: joint-limit diagnostic, shared encoder and supervision,
  cohort C inspection with linked viewpoint A/B images, and its accuracy/time tradeoff.
  Vector PDFs embed the preserved camera images and rendered scene where
  needed. scripts/build_report_figures.py reads the original saved data.
  scripts/render_diagnostic_frame.py renders the saved stopping pose.

OVERLEAF
  See OVERLEAF_UPDATE.md. Preserve the folder hierarchy when uploading.
  README.md provides detailed editing and build instructions.
