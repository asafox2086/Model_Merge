# AAAI 2026 LAMP-Merge Draft

This directory contains an AAAI 2026 formatted wrapper for the LAMP-Merge draft.

Files:

- `lamp_merge_aaai26.tex`: paper wrapper that uses `aaai2026` and inputs `../lamp_merge_paper_sections.tex`.
- `lamp_merge_aaai26.pdf`: compiled PDF generated with `tectonic` in the `MM` conda environment.
- `anonymous-submission-latex-2026.tex`: AAAI 2026 template source extracted from the Overleaf template page.
- `aaai2026.sty` and `aaai2026.bst`: AAAI 2026 style and bibliography files used for local compilation.

Source note:

- The official AAAI author-kit link requested by the user is `https://aaai.org/authorkit26/`.
- On this server, direct access to `aaai.org` returned Cloudflare HTTP 403.
- The Overleaf template page for "AAAI 2026 Press Formatting Instructions for Authors Using LaTeX" states that its source is the official AAAI website: `https://aaai.org/authorkit26/`.

Compile command:

```bash
conda run -n MM tectonic My_merge_ret/aaai26_lamp/lamp_merge_aaai26.tex
```
