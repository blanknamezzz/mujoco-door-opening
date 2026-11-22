# Updating the Overleaf project

Upload this directory's source files, `figures/`, and `sections/` while preserving the folder hierarchy. The source ZIP can also be imported directly into a new project.

1. Set the root `main.tex` as the main document.
2. Select pdfLaTeX as the compiler and use the project's default BibTeX workflow.
3. Edit the report in `sections/` and the title and authors in `title.tex`.
4. Click Recompile and check the figures, tables, cross-references, and bibliography.

The update includes `main.tex`, `preamble.tex`, `title.tex`, all sections and their methods/experiments modules, the F01/F04 captions, and the added assets `figures/methods/report_system.pdf` and `figures/results/report_inspection_cost.pdf`. The complete source ZIP includes these files.

The writing-note template has been replaced with a completed English report of eight body pages and one reference page. New figures use vector PDFs, and the other images retain their original files. The original `neurips_2026.sty` and `references.bib` are preserved, and the bibliography lists only works cited in the text.

Use `EVIDENCE.md` to locate the sources of numerical results and `scripts/` for local figure-generation tools. The compiled `main.pdf` is the reading copy.

The 2026-11-22 layout revision places Figures 2 and 3 and Tables 1, 2, and 3 after their respective introductions. The source ZIP includes the required `float` package declaration and placement settings. Compile the imported project as described above.

Use `CS5478_Team42_Final_Report_Aligned_Source_20261122.zip` for the latest version. It includes the A/B/C cohort names, updated Figure 3, initialization description, four vector figures, and grouped comparisons in Table 3. Import the complete package to update the text and figures together while preserving their placement settings.
