# First Exam — Multiple Versions

This folder contains two versions of the first exam for the Applied Statistics course, covering Lessons 0 & 1.

## Purpose

Multiple exam versions help maintain academic integrity by:
- Reducing opportunities for copying during the exam
- Testing the same concepts with different numerical parameters
- Ensuring fair assessment across all students

## File Structure

```
Exam/
├── README.md                          # This file
├── src/                               # Markdown source for exams and solutions
│   ├── first-exam-version-A.md
│   ├── first-exam-version-A-solution.md
│   ├── first-exam-version-B.md
│   └── first-exam-version-B-solution.md
├── latex/                             # LaTeX wrappers + build scripts
│   ├── Makefile                       # Build PDFs with pdflatex
│   ├── first-exam-version-A.tex       # Wrapper for Version A (exam)
│   ├── first-exam-version-A-solution.tex  # Wrapper incl. exam + solutions
│   ├── first-exam-version-B.tex       # Wrapper for Version B (exam)
│   └── first-exam-version-B-solution.tex  # Wrapper incl. exam + solutions
└── build/                             # Generated PDFs + auxiliary files (ignored)
```

## Version Differences

Both versions test the same concepts at comparable difficulty while varying the context and parameters:

### Question 1: Probability Foundations
- **Version A**: Two coin tosses (fair/biased)
- **Version B**: Two die rolls (fair/biased)

### Question 2: Bayes' Rule
- **Version A**: Circuit board defect testing (suppliers $S_A$ and $S_B$)
- **Version B**: Fraud detection (flagged transactions with given sensitivity/specificity)

### Question 3: Distribution Derivations
- **Version A**: Exponential waiting time, $\lambda = \tfrac{1}{4}$ hours
- **Version B**: Uniform waiting time on $[0,8]$ minutes

### Question 4: Interpreting Summaries
- **Version A**: 1,000 samples from Exp(4), sample sizes 5/20/80, heavier-tailed QQ-plot
- **Version B**: 800 samples from Exp(3), sample sizes 10/40/160, lighter-tailed QQ-plot

## Distribution Strategy

### Random Distribution
Distribute versions randomly to students seated throughout the room:
```bash
# Example: For 40 students
Students 1, 3, 5, 7, ... (odd numbers) → Version A
Students 2, 4, 6, 8, ... (even numbers) → Version B
```

### Alternate Row Distribution
```
Row 1: A B A B A B ...
Row 2: B A B A B A ...
Row 3: A B A B A B ...
```

### Color-Coded Papers
- Version A: White paper
- Version B: Light blue/green paper (if allowed by institutional policy)

## Grading Guidelines

Both versions have:
- Total: 100 points
- Question 1: 25 points (probability foundations)
- Question 2: 25 points (Bayes' rule)
- Question 3: 25 points (exponential distribution)
- Question 4: 25 points (interpreting summaries)

### Important Notes for Graders

1. **Equivalent Reasoning**: Accept algebraically equivalent solutions even if steps differ
2. **Partial Credit**: Award partial credit as specified in solution files
3. **Version-Specific Answers**: Ensure answers are graded against the correct version's solution key
4. **Numerical Values**: Be strict on exact values (fractions vs decimals) as specified in instructions

## Student Instructions

Each exam version includes:
- Time limit: 60 minutes
- Allowed resources: Handwritten notes only
- Prohibited: Electronic devices, calculators, statistical tables
- Answer format: Exact values (fractions, radicals, exponentials) unless otherwise stated

## Creating Additional Versions

To create more versions (C, D, etc.):

1. **Keep the same structure**: 4 questions, same topics, same point distribution
2. **Vary the parameters**: Change numerical values, distributions, or contexts
3. **Maintain difficulty**: Ensure calculations have similar complexity
4. **Test thoroughly**: Verify solutions are correct before distributing

### Suggested Variations

**Question 1 Alternatives**:
- Three coin tosses
- Rolling two different dice
- Drawing cards with/without replacement

**Question 2 Alternatives**:
- Airport security screening
- Quality control in manufacturing
- Email spam filtering

**Question 3 Alternatives**:
- Different exponential rates: $\lambda = \frac{1}{2}, \frac{1}{5}, \frac{1}{6}$
- Different affine transformations: $Y = aT + b$ with various $a, b$

**Question 4 Alternatives**:
- Different sample sizes
- Different true means
- Different tail behaviors (uniform-like, skewed)

## Security and Storage

### Before Exam
- ⚠️ Store solution files securely (not in public repositories)
- Only distribute exam question files to students
- Keep versions in sealed envelopes if printing in advance

### After Exam
- Collect all exam papers to prevent sharing
- Consider different versions for makeup exams
- Archive exams for future reference but rotate versions annually

## Version History

- **2025-10-15**: Created Version A and Version B; updated Version B Q2 to fraud detection and Q3 to uniform model; added printable PDF build wrappers (now in `Exam/latex`, formerly `Exam/print`).
  Questions adapted from `first-exam.md` and `first-exam-solution.md`.

## Building PDFs

Inside `Exam/latex`, run:

```
make            # builds all four PDFs into ../build
make a          # builds Version A exam
make asol       # builds Version A solutions (includes full exam text)
make b          # builds Version B exam
make bsol       # builds Version B solutions (includes full exam text)
```

Generated PDFs and LaTeX artifacts are placed in `Exam/build/`. `distclean` removes the PDFs as well.

---

**Note**: The solutions in this folder are for instructor use only and should never be distributed to students before or during the exam period.
