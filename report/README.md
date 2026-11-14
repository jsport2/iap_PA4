# Report

`EF2039_PA04_20250795_JeonggukPark.pdf` is the final two-page report. The
remaining files are kept so that the submitted PDF can be reproduced:

- `main.tex`, `preamble.tex`, and `sec/` hold the LaTeX source.
- `main.bib` lists the STL-10 and Teachable Machine references.
- `figures/` contains the generated PNG and PDF figure.

Rebuild the result figure from the repository root:

```powershell
python .\code\visualization\make_result_figure.py
```

Then compile the report from this folder:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
