# Statistical Learning Slides

LaTeX Beamer presentation for Lesson 02: Statistical Learning - Parameter Estimation.

## Quick Start

```bash
make              # Build everything (figures + slides) → ../slides.pdf
```

## Directory Structure

```
slides/
├── main.tex              # Main presentation file
├── Makefile              # Build system
├── make_figures.py       # Figure orchestrator
├── sections/             # Modular slide sections (tracked)
├── figure_scripts/       # Figure generation code (tracked)
│   └── statistical_learning.py
├── figures/              # Generated figures (gitignored)
└── build/                # Build artifacts: .aux, .log, etc. (gitignored)
```

## Build Commands

### Standard Workflow
```bash
make              # Generate figures + compile PDF → ../slides.pdf
make clean        # Remove build artifacts (.aux, .log, etc.)
make clean-all    # Remove everything (build + slides.pdf + figures)
```

### Individual Steps
```bash
make figs         # Generate all figures only
make lesson-pdf   # Compile LaTeX only (requires figures)
make quick        # Quick compile (reuses artifacts, faster)
```

### Viewing
```bash
open ../slides.pdf  # View compiled presentation
```

## Development

### Making Changes
1. Edit `.tex` files in `sections/` or `main.tex`
2. Run `make` to rebuild (compiles directly to `../slides.pdf`)
3. View `../slides.pdf`
4. Commit when ready

### File Organization
- **main.tex**: Document structure, packages, macros, section includes
- **sections/*.tex**: Content organized by topic
- **figure_scripts/**: Python code for generating figures
- **figures/**: Output directory for generated PNG files (gitignored)
- **build/**: Temporary LaTeX files (gitignored)

### Build System
The Makefile:
- Uses `xelatex` for Unicode support
- Runs Python figure generation first
- Compiles LaTeX twice for cross-references
- Moves all build artifacts to `build/`
- Places final PDF in parent directory as `../slides.pdf`
- Keeps source directory clean

## Figure Generation

### Overview
Figures are generated programmatically using Python (NumPy, Matplotlib).

### Figure Scripts
- `statistical_learning.py`: All figure generation functions
  - MLE illustrations and likelihood profiles
  - Fisher information visualizations
  - Delta method examples
  - Method of moments comparisons
  - Simulation study results

### Adding New Figures
1. Add generation function in `figure_scripts/statistical_learning.py`
2. Call function in `make_figures.py`
3. Reference figure in LaTeX: `\includegraphics{figures/your_figure.png}`
4. Run `make figs` to generate
5. Commit the script, not the generated figure

### Figure Naming Convention
- Prefix with topic: `fig_mle_`, `fig_mom_`, `fig_fisher_`
- Be descriptive: `fig_bern_loglik.png`, `fig_delta_log_var.png`
- Use lowercase with underscores

## LaTeX Structure

### Main Document (main.tex)
- Beamer configuration and theme
- Package imports
- Custom macros and notation
- Title and metadata
- Section includes via `\input{sections/...}`

### Sections
Each section file contains:
- One or more `\section{}` or `\subsection{}` commands
- Frame content for that topic
- References to generated figures
- Examples and illustrations

### Section Organization
- `01-mle-likelihood.tex`: Likelihood functions and MLE basics
- `02-mle-properties.tex`: MLE properties, Fisher information, standard errors
- `03-method-of-moments.tex`: Method of moments, comparisons

## Requirements

### LaTeX Distribution
- **macOS**: MacTeX or BasicTeX
- **Linux**: TeXLive
- **Windows**: MiKTeX or TeXLive

### Required LaTeX Packages
- beamer
- amsmath, amssymb
- tikz (with arrows.meta, positioning, calc libraries)
- graphicx
- booktabs, array, arydshln
- hyperref
- mathtools

### Python Environment
- **Python 3.8+**
- **NumPy**: Numerical computing
- **Matplotlib**: Figure generation
- **SciPy**: Statistical functions

Install dependencies:
```bash
pip install numpy matplotlib scipy
```

### Build Tools
- `make` (for automated builds)
- `xelatex` (or `pdflatex` with modifications)

## Troubleshooting

### Build fails with missing packages
```bash
# Install missing packages via tlmgr (TeX Live)
tlmgr install <package-name>
```

### Figures not generated
```bash
# Check Python environment
python --version
python -c "import numpy, matplotlib, scipy"

# Generate figures manually
python make_figures.py
```

### Quick compile produces wrong output
```bash
# Do a full clean rebuild
make clean-all
make
```

### PDF not updating
```bash
# Check if build artifacts are stale
make clean
make
```

## Performance Tips

- Use `make quick` for fast iteration when only text changes
- Use `make lesson-pdf` to skip figure generation if figures haven't changed
- Full `make` ensures everything is up-to-date

## Notes

- Build artifacts are gitignored to keep repository clean
- Generated figures are gitignored (reproducible from scripts)
- The final PDF goes directly to the lesson root (`../slides.pdf`)
- For consistent formatting, follow `style-guide.md` in repository root
- Figure generation is deterministic (uses fixed random seed)

---

**Lesson:** 02 - Statistical Learning  
**Build System:** Makefile + XeLaTeX + Python  
**Last Updated:** October 15, 2025
