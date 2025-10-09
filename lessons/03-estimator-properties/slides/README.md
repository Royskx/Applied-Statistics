# Estimator Properties slides

This directory contains the Beamer slide source and supporting files for
the "03 - Estimator Properties" lecture.

Contents
- `main.tex` - Beamer entry point (uses `\input{sections/...}`)
- `sections/` - Per-section TeX files (modularized content)
- `figures/` - Generated figure assets (gitignored)
- `figure_scripts/` - Python figure generation modules
- `make_figures.py` - Orchestrator script to produce figures
- `compile-quiet.sh` / `Makefile` - Convenience build helpers

Build
1. Regenerate figures:

```bash
cd lessons/03-estimator-properties/slides
make figs
```

2. Build the PDF:

```bash
cd lessons/03-estimator-properties/slides
make
```

Notes
- The Makefile references a specific Python interpreter; if `make figs`
  fails, run the orchestrator directly with your environment Python:

```bash
python make_figures.py
```

- Keep `\pdfstringdefDisableCommands` in `main.tex` unchanged — do not
  place `\input{...}` lines inside it.

Figure development
- Figure scripts write into the local `figures/` folder. Add new figure
  modules under `figure_scripts/` and call them from `make_figures.py`.
