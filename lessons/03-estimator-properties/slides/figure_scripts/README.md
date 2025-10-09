## Figure Scripts Directory Structure

This directory contains modular figure generation scripts for Lesson 03: Estimator Properties.

### Current Status

**Phase 1 (Complete)**: Integration with existing script
- `make_figures.py` orchestrates all figure generation
- Currently calls `../generate_figures_enhanced.py` for main figures
- `examples.py` provides simple example figures

**Phase 2 (Planned)**: Full Modularization
- Move functions from `generate_figures_enhanced.py` into `estimator_properties.py`
- Organize by topic:
  * Bias-variance tradeoff functions
  * Consistency demonstration functions
  * CRLB and asymptotic efficiency functions
  * Confidence interval functions
  * Bootstrap functions
  * Delta method functions

### File Organization

```
figure_scripts/
├── __init__.py              # Package initialization
├── examples.py              # Simple example figures
├── estimator_properties.py  # Main module (planned migration)
└── README.md               # This file
```

### Usage

From the slides directory:
```bash
python make_figures.py      # Generate all figures
make figs                   # Same via Makefile
```

### Migration Plan

To complete the refactoring:

1. **Extract functions**: Split `generate_figures_enhanced.py` into individual functions:
   - `bias_variance_tradeoff()` - 4-panel analysis
   - `consistency_demonstration()` - Convergence illustrations
   - `crlb_achievement()` - Lower bound demonstrations
   - `proportion_ci_coverage()` - CI method comparisons
   - `bootstrap_median_distribution()` - Bootstrap examples
   - `bias_variance_conceptual()` - Conceptual diagrams
   - `fisher_information_visualization()` - Information theory
   - `ci_interpretation()` - Interval interpretation
   - `delta_method_illustration()` - Transformation examples

2. **Move to module**: Place functions in `estimator_properties.py`

3. **Update orchestrator**: Modify `make_figures.py` to import from module:
   ```python
   from figure_scripts.estimator_properties import generate_all_estimator_property_figures
   ```

4. **Remove old script**: Delete `../generate_figures_enhanced.py` once migration complete

### Benefits of Modular Structure

- **Maintainability**: Each function is self-contained and testable
- **Readability**: Clear separation of concerns
- **Debugging**: Easy to run individual figure generators
- **Reusability**: Functions can be imported independently
- **Safety**: Smaller files reduce risk of corruption

### Design Principles

Following the pattern from lesson 02-statistical-learning:
- Keep orchestrator (`make_figures.py`) thin (< 50 lines)
- Each figure function is self-contained
- Common settings defined at module level
- Clear naming conventions
- Comprehensive docstrings
