# Course Overview Presentation

This folder contains the **Applied Statistics Course Overview** presentation, which introduces the course structure, learning objectives, assessment methods, and philosophy to students at the beginning of the course.

## Overview

The course overview presentation provides:
- Course goals and learning outcomes
- Six-lesson structure with topics covered
- Assessment breakdown (practicals and final project)
- Tools and technology stack (Python, Jupyter notebooks)
- Real-world applications and course philosophy
- Guidance on responsible use of AI tools for learning

## Presentation Contents

### Main Topics

1. **Course Goals and Learning Outcomes**
   - Statistical modeling and data-generating processes
   - Parameter estimation and uncertainty quantification
   - Hypothesis testing with proper interpretation
   - Reproducible research using Python

2. **Course Structure**
   - Lesson 00: Welcome and Introduction (Probability foundations)
   - Lesson 01: Statistical Modeling (Random variables, LLN/CLT)
   - Lesson 02: Parameter Estimation (MLE, Method of Moments, Fisher Information)
   - Lesson 03: Estimator Properties (Bias, variance, MSE, confidence intervals)

> **Note:** Lessons 04–06 will be added to the public branch once they are released to students.

3. **Assessment and Tools**
   - Practicals (50%): Short labs and written tests
   - Final Project (50%): Complete statistical analysis with report
   - Python 3.9+ with scientific computing stack
   - Real datasets for practical applications

4. **Course Philosophy**
   - Bridge theory and practice
   - Emphasize interpretation over computation
   - Build intuition through visualization
   - Responsible use of AI tools for learning

## Folder Structure

```
presentation/
├── README.md              # This file - overview and documentation
└── course_overview.pdf    # Precompiled deck distributed to students
    ├── README.md          # Developer guide and build instructions
    ├── Makefile           # Build automation
    ├── .gitignore         # Ignore build artifacts
    ├── main.tex           # Main LaTeX document (modular)
    ├── build/             # Build artifacts (gitignored)
    └── sections/          # Modular content sections
        ├── 01-goals.tex
        ├── 02-structure.tex
        ├── 03-assessment.tex
        └── 04-philosophy.tex
```

## Quick Start

### For Students

To view the course overview presentation:

```bash
# View the compiled PDF
open course_overview.pdf
```

The presentation is typically shown on the first day of class to introduce the course structure and expectations.

### For Instructors/Maintainers

The development branch retains the full LaTeX sources. If you need to rebuild or update the deck, switch to `dev`, regenerate the PDF there, and copy the compiled result back to `presentation/course_overview.pdf` before syncing to `public-main`.

## Prerequisites

- LaTeX distribution (TeX Live 2023+ or MacTeX 2023+)
- XeLaTeX compiler
- Standard Beamer packages

## Usage Notes

- This presentation is designed for the **first class session**
- Duration: Approximately 15-20 minutes
- Interactive: Encourage questions about course structure and expectations
- Should be updated when course structure or assessment changes

## Related Files

- Course syllabus: `../syllabus.md`
- Individual lesson materials: `../lessons/*/`
- Course style guide: `../style-guide.md`

## Maintenance

When updating the course overview:

1. Update section files in `slides/sections/` as needed
2. Rebuild with `make` in the `slides/` directory
3. Verify the PDF looks correct
4. Commit both source files and the compiled PDF
5. Update this README if structural changes are made

## Questions?

For questions about the course structure or this presentation, contact the course instructor.
