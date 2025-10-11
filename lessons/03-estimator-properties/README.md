# Lesson 3: Properties of Estimators

## Overview

This lesson covers the fundamental properties of statistical estimators, including consistency, bias, variance, and confidence intervals. Students will learn how to evaluate estimator quality, understand the bias-variance tradeoff, and construct various types of confidence intervals using both classical and modern (bootstrap) methods.

**Estimated time:** 3–4 hours self-study

## Learning Objectives

- Define and identify consistency, bias, and variance of estimators
- Explain bias–variance tradeoff and Mean Squared Error (MSE)
- Construct confidence intervals (CI) using CLT, t, and bootstrap methods
- Compare different CI approaches (Wald, Wilson, t-test, bootstrap)
- Apply these concepts to real statistical problems

## Folder Structure

```
lessons/03-estimator-properties/
├── README.md                           # This file - lesson overview
├── slides.pdf                          # Main presentation slides
├── material.md                         # Complete lesson content and theory
│
├── exercises/                          # Practice exercises
│   ├── 01-bias-variance-tradeoff.md
│   ├── 02-consistency.md
│   ├── 03-asymptotic-efficiency-crlb.md
│   ├── 04-confidence-intervals.md
│   └── 05-bootstrap.md
│
└── notebooks/                          # Interactive Jupyter notebooks
    ├── 01-bias-variance-tradeoff.ipynb
    ├── 02-consistency.ipynb
    ├── 03-asymptotic-efficiency-crlb.ipynb
    ├── 04-confidence-intervals.ipynb
    ├── 05-bootstrap.ipynb
    └── 06-practical-lab.ipynb
```

## Prerequisites

- Basic probability and statistics (random variables, distributions)
- Python programming experience with numpy, scipy, matplotlib
- Understanding of random variables, distributions, and expectation

## How to Use This Lesson

### 1. Start with the Slides
Review `slides.pdf` for a comprehensive overview of all concepts covered in this lesson.

### 2. Read the Theory
Study `material.md` for detailed explanations, mathematical derivations, and theoretical foundations.

### 3. Practice with Exercises
Work through the exercises in the `exercises/` directory:
- Start with bias-variance tradeoff concepts
- Progress through consistency and efficiency
- Practice confidence interval construction
- Master bootstrap methods

### 4. Interactive Learning
Use the Jupyter notebooks in `notebooks/` for hands-on practice:
- Run code examples
- Visualize concepts
- Experiment with different parameters
- Complete the practical lab

## Viewing the Materials

### Slides
```bash
open slides.pdf              # macOS
xdg-open slides.pdf          # Linux
start slides.pdf             # Windows
```

### Jupyter Notebooks
```bash
jupyter notebook notebooks/
# Or use VS Code, JupyterLab, etc.
```

## Topics Covered

The lesson is organized into 5 main sections:

### 1. Bias-Variance Tradeoff
Understanding the fundamental tradeoff in estimator design:
- Defining bias and variance
- Mean Squared Error (MSE) decomposition
- Practical examples and visualizations
- Shrinkage estimators

### 2. Consistency
Large-sample behavior of estimators:
- Definition and importance
- Convergence in probability
- Law of Large Numbers applications
- Examples with different estimators

### 3. Asymptotic Efficiency & CRLB
Cramér-Rao Lower Bound and efficiency concepts:
- Fisher Information
- CRLB theorem and interpretation
- Efficiency of estimators
- Maximum Likelihood Estimator properties

### 4. Confidence Intervals
Classical and modern approaches to interval estimation:
- Normal-based confidence intervals
- t-distribution intervals
- Proportion confidence intervals (Wald, Wilson)
- Coverage properties

### 5. Bootstrap Methods
Resampling-based inference:
- Bootstrap principle and algorithm
- Bootstrap confidence intervals
- Percentile, Basic, and BCa methods
- Practical applications and limitations

## Key Concepts

**By the end of this lesson, you should be able to:**
- ✓ Evaluate estimators based on bias, variance, and MSE
- ✓ Understand when and why estimators are consistent
- ✓ Apply the Cramér-Rao Lower Bound to assess efficiency
- ✓ Construct and interpret various types of confidence intervals
- ✓ Use bootstrap methods for inference in complex situations
- ✓ Make informed decisions about which estimation method to use

## Related Lessons

- **Lesson 1:** Statistical Modeling - Random variables, distributions, LLN, CLT
- **Lesson 2:** Statistical Learning - Estimation methods and model fitting
- **Lesson 4:** Hypothesis Testing - Part 1
- **Lesson 5:** Hypothesis Testing - Part 2

---

For questions or issues, please refer to the course instructor or teaching assistants.