# LAMP-Merge Final Manuscript

This directory is the sole final manuscript package. `final.pdf` is generated directly from `final.tex`, preserves all four client-average main tables in the paper body, includes the supplementary appendix, uses the updated full-scope hyperparameter figure, and contains the corrected four-backbone prediction-diagnostic values.

Compile with:

```bash
pdflatex final.tex
bibtex final
pdflatex final.tex
pdflatex final.tex
```

The `figures/` directory and the three `aaai2026.*` files are required build dependencies.
