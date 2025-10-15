# Course Overview Presentation - Developer Guide

This directory contains the source files and build system for the Applied Statistics Course Overview presentation.

## Directory Structure

```
slides/
├── README.md              # This file - developer documentation
├── Makefile               # Build automation with multiple targets
├── .gitignore             # Excludes build artifacts
├── main.tex               # Main LaTeX document (uses modular sections)
├── build/                 # Build artifacts (*.aux, *.log, etc.) - gitignored
└── sections/              # Modular content sections
    ├── 01-goals.tex       # Course goals and learning outcomes
    ├── 02-structure.tex   # Six-lesson structure and topics
    ├── 03-assessment.tex  # Assessment methods and tools
    └── 04-philosophy.tex  # Course philosophy and AI guidance
```

## Build System

### Prerequisites

- **LaTeX Distribution**: TeX Live 2023+ or MacTeX 2023+
- **Compiler**: XeLaTeX (used instead of pdflatex for better Unicode support)
- **Packages**: Standard Beamer packages (automatically managed by LaTeX)

### Available Make Targets

```bash
make              # or 'make all' - Full compilation (2 passes, moves PDF to parent)
make quick        # Fast draft compilation (single pass, for rapid iteration)
make presentation-pdf  # Alias for 'make all'
make clean        # Remove build artifacts (keeps PDF)
make clean-all    # Remove everything including PDF
make help         # Show detailed help for all targets
```

### Build Process

The Makefile performs these steps:

1. **Clean**: Removes stray auxiliary files to prevent corruption
2. **First Pass**: Compiles with XeLaTeX to generate content
3. **Second Pass**: Recompiles to resolve cross-references and TOC
4. **Organize**: Moves build artifacts to `build/` directory
5. **Deploy**: Copies final PDF to `../slides.pdf` (presentation root)

### Example Workflows

**Quick editing cycle** (for rapid content changes):
```bash
make quick        # Fast single-pass compilation
open ../slides.pdf  # View result
# Edit sections/*, repeat
```

**Final production build**:
```bash
make clean        # Start fresh
make              # Full 2-pass compilation
open ../slides.pdf  # Verify final result
```

**Complete cleanup**:
```bash
make clean-all    # Remove all generated files
```

## Content Organization

### Modular Structure

Content is organized into logical sections in `sections/`:

- **`01-goals.tex`**: Course goals and learning outcomes
  - Primary goals (modeling, estimation, testing, reproducibility)
  - Learning outcomes and competencies

- **`02-structure.tex`**: Six-lesson course structure
  - Detailed lesson topics
  - Practical focus and tools

- **`03-assessment.tex`**: Assessment and tools
  - Practicals (50%) and Final Project (50%)
  - Technology stack (Python, Jupyter, datasets)
  - Prerequisites

- **`04-philosophy.tex`**: Course philosophy and responsible AI use
  - Real-world applications
  - Course values (theory-practice bridge, interpretation, visualization)
  - Guidance on using AI tools responsibly

### Main Document (`main.tex`)

The main document:
- Sets up the Beamer presentation structure
- Defines packages and macros
- Uses `\input{sections/...}` to include modular content
- Maintains consistent theming and formatting

## Editing Guidelines

### Adding New Content

1. **Identify the appropriate section file** in `sections/`
2. **Add frames** using standard Beamer syntax:
   ```latex
   \begin{frame}{Frame Title}
     \begin{block}{Block Title}
       Content here...
     \end{block}
   \end{frame}
   ```
3. **Rebuild** with `make quick` to preview changes
4. **Verify** the output in `../slides.pdf`

### Creating New Sections

To add a new section:

1. Create `sections/05-newsection.tex`
2. Add content with appropriate `\begin{frame}...\end{frame}` blocks
3. Include it in `main.tex`:
   ```latex
   \input{sections/05-newsection}
   ```
4. Rebuild and test

### Reordering Sections

Simply change the order of `\input` statements in `main.tex` and rebuild.

## LaTeX Conventions

### Safe Editing Practices

This presentation uses the course-wide LaTeX conventions:

- **Robust commands**: Use `\robustcmd{title}`, `\robustcmd{titlepage}` etc. to avoid issues with leading backslashes
- **Consistent notation**: Follow macros defined in preamble (`\P`, `\E`, `\Var`, etc.)
- **Theme consistency**: Madrid theme with default color scheme

For detailed LaTeX editing guidelines, see `../../lessons/safe-latex-edits.md`.

### Mathematical Notation

Standard macros are defined in `main.tex`:
```latex
\newcommand{\R}{\mathbb{R}}
\renewcommand{\P}{\mathbb{P}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\operatorname{Var}}
```

## Build Artifacts

### Ignored Files (`.gitignore`)

The following files are generated during compilation and should not be committed:

- `*.aux`, `*.log`, `*.nav`, `*.snm`, `*.toc` - LaTeX auxiliary files
- `*.out`, `*.fdb_latexmk`, `*.fls` - Additional build metadata
- `*.synctex.gz` - SyncTeX data for editor integration
- `build/` - Directory containing all build artifacts

### Committed Files

The compiled PDF (`../slides.pdf`) **is committed** to the repository so students can access it directly without building from source.

## Troubleshooting

### Common Issues

**"File not found" errors**:
- Ensure you're running `make` from the `slides/` directory
- Check that all section files exist in `sections/`

**Stale cross-references**:
- Run full `make` (2 passes) instead of `make quick`
- Try `make clean && make` for a fresh build

**Font/Unicode issues**:
- Verify XeLaTeX is installed and in your PATH
- Check that special characters in content are properly escaped

**PDF not updating**:
- Ensure PDF viewer isn't locking the file
- Try `make clean-all && make` for a complete rebuild

## Integration with Course Materials

This presentation is designed to be shown on the first day of class and should be updated whenever:

- Course structure changes (lessons added/removed/reordered)
- Assessment weights change
- New tools or technologies are adopted
- Course philosophy evolves

Coordinate with lesson materials in `../../lessons/` to ensure consistency.

## Questions?

For questions about the build system or technical issues, refer to:
- Course style guide: `../../style-guide.md`
- LaTeX editing guide: `../../lessons/safe-latex-edits.md`
- Repository documentation: `../../README.md`
