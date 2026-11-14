# Report

- `EF2039_PA04_20250795_JeonggukPark.pdf`: final two-page submission PDF.
- `main.tex`, `sec/`, and `figures/`: LaTeX source and generated figures.
- `main.bib`: STL-10 and Teachable Machine references.

Rebuild the result figure from the repository root with:

```powershell
python .\code\visualization\make_result_figure.py
```

Compile from this folder with:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
