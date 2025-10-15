# Lesson 2: Statistical Learning - Parameter Estimation

## Overview

This lesson covers the fundamental methods for parameter estimation in statistical learning, focusing on Maximum Likelihood Estimation (MLE) and Method of Moments. Students will learn how to derive estimators, compute standard errors using Fisher information and the delta method, and assess estimator quality through simulation studies.

**Estimated time:** 3–4 hours self-study

## Learning Objectives

- Derive estimators using **maximum likelihood** and **method of moments**
- Compute **standard errors** via Fisher information and the **delta method**
- Assess estimators: **bias**, **variance**, **MSE**
- Use **likelihood profiles** and **bootstrap** for uncertainty quantification
- Implement simulation studies to compare estimation procedures

## Folder Structure

```
lessons/02-statistical-learning/
├── README.md                    # This file - lesson overview
├── slides.pdf                   # Main presentation slides
├── material.md                  # Complete lesson content and theory
│
└── slides/                      # Slide source files
    ├── main.tex                # Main LaTeX file
    ├── Makefile                # Build system
    ├── README.md               # Developer guide
    ├── make_figures.py         # Figure generation orchestrator
    ├── sections/               # Modular slide sections
    ├── figure_scripts/         # Figure generation code
    ├── figures/                # Generated figures (gitignored)
    └── build/                  # Build artifacts (gitignored)
```

## Prerequisites

- **Lesson 1**: Random variables, distributions (PMF/PDF/CDF), LLN/CLT
- **Probability fundamentals**: Expectation, variance, conditional probability
- **Python programming**: NumPy, SciPy, Matplotlib
- **Basic calculus**: Derivatives, optimization

## How to Use This Lesson

### 1. Start with the Slides
Review `slides.pdf` for a comprehensive overview of all concepts covered in this lesson.

### 2. Read the Theory
Study `material.md` for detailed explanations, mathematical derivations, and theoretical foundations.

### 3. Practice Implementation
Work through examples implementing:
- MLE derivations for various distributions
- Method of moments estimators
- Standard error calculations using Fisher information
- Delta method for transformed parameters
- Simulation studies to assess estimator properties

### 4. Run Figure Generation
Explore the figure generation code in `slides/figure_scripts/` to understand visualizations.

## Viewing the Materials

### Slides
```bash
open slides.pdf              # macOS
xdg-open slides.pdf          # Linux
start slides.pdf             # Windows
```

## Topics Covered

The lesson is organized into two main sections:

### 1. Maximum Likelihood Estimation (MLE)

**Core Concepts:**
- Likelihood and log-likelihood functions
- MLE derivation and properties
- Fisher information and standard errors
- Likelihood profiles for uncertainty assessment

**Key Topics:**
- **Parameter Estimation Motivation**: Why estimate parameters?
- **Likelihood Function**: L(θ | data) = probability of data given parameters
- **Log-Likelihood**: Computational advantages, derivative properties
- **MLE Properties**: Consistency, asymptotic normality, efficiency
- **Fisher Information**: Measures information about parameters
- **Standard Errors**: Uncertainty quantification via Cramér-Rao bound
- **Delta Method**: Transforming parameter estimates and their variances

**Examples:**
- Bernoulli/Binomial parameter estimation
- Poisson rate estimation
- Normal distribution (μ, σ²) estimation
- Exponential rate parameter

### 2. Method of Moments

**Core Concepts:**
- Sample moments vs. population moments
- Equating moments to solve for parameters
- Comparison with MLE

**Key Topics:**
- **Moment Matching**: Equate sample and population moments
- **Derivation Strategy**: Solve system of equations
- **Advantages**: Simple, intuitive, no likelihood needed
- **Disadvantages**: May be less efficient than MLE

**Examples:**
- Method of moments for common distributions
- Comparison of MLE vs. MoM performance
- Cases where MoM is simpler than MLE

### Supporting Topics

- **Likelihood Profiles**: Visual assessment of parameter uncertainty
- **Bootstrap Methods**: Non-parametric uncertainty quantification
- **Simulation Studies**: Empirical comparison of estimators
- **Bias-Variance Trade-off**: Understanding estimator quality
- **Mean Squared Error (MSE)**: Combining bias and variance

## Building the Slides

For developers who want to rebuild the slides:

```bash
cd slides/
make              # Generate figures + compile slides → ../slides.pdf
```

The build system:
1. Runs `make_figures.py` to generate all figures
2. Compiles LaTeX twice for cross-references
3. Places final PDF in lesson root

See `slides/README.md` for detailed build instructions.

## Figures and Visualizations

All figures are generated programmatically using Python:
- **MLE visualizations**: Likelihood profiles, parameter surfaces
- **Fisher information**: Information matrix illustrations
- **Delta method**: Variance transformation examples
- **Simulation studies**: Empirical distributions, MSE comparisons
- **Comparison plots**: MLE vs. Method of Moments performance

Figure generation code is modular and well-documented in `slides/figure_scripts/statistical_learning.py`.

## Course Context

This is Lesson 2 of a 6-lesson course on Applied Statistics:
- Lesson 0: Welcome and Introduction
- Lesson 1: Statistical Modeling and Exploratory Data Analysis
- **Lesson 2:** Statistical Learning - Parameter Estimation (this lesson)
- Lesson 3: Properties of Estimators
- Lesson 4: Hypothesis Testing (Part 1)
- Lesson 5: Hypothesis Testing (Part 2)
- Lesson 6: Final Coding Project

## References

- **All of Statistics** by Larry Wasserman - Chapters 9-10
- **Statistical Inference** by Casella & Berger - Chapters 7-9
- **The Elements of Statistical Learning** - Chapter 2
- **Computer Age Statistical Inference** by Efron & Hastie - Chapters 4-5

## Key Takeaways

1. **MLE is powerful**: Optimal properties (consistency, efficiency, asymptotic normality)
2. **Fisher information quantifies knowledge**: Links to standard errors via Cramér-Rao bound
3. **Delta method enables inference**: Transform estimates and propagate uncertainty
4. **Method of moments is simpler**: Easy to derive, but may sacrifice efficiency
5. **Simulation validates theory**: Always test estimators empirically

## Next Steps

After completing this lesson:
1. Practice deriving MLEs for various distributions
2. Implement Fisher information calculations
3. Apply the delta method to transformed parameters
4. Run simulation studies to compare estimators
5. Move on to Lesson 3: Properties of Estimators

---

**Author:** Applied Statistics Course
**Last Updated:** October 15, 2025
