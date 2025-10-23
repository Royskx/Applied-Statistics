# Exercise 5 — Multiple Testing and Error Control

## Goals

- Quantify false positives under repeated testing
- Apply Bonferroni, Holm, and Benjamini–Hochberg corrections
- Interpret adjustments in the context of product metrics and scientific studies

## Problems

### Problem A — False Positives Without Adjustment

1. Simulate 1,000 experiments where each computes p-values for 50 independent null hypotheses (use `np.random.uniform`).
2. For each experiment, record how many p-values are below 0.05 without adjustment.
3. Plot the distribution of false discoveries. Report the mean and compare with theoretical expectation.

### Problem B — Applying Corrections

1. Reuse the simulated p-values from part A. Apply the following adjustments using `statsmodels.stats.multitest.multipletests`:
   - Bonferroni
   - Holm
   - Benjamini–Hochberg (FDR)
2. For each method, estimate the average number of false discoveries and the proportion of experiments with at least one false discovery.

### Problem C — Mixed Null/Alternative Scenario

1. Simulate p-values for 100 hypotheses where 70% are null (uniform) and 30% follow a Beta(0.5, 1) distribution (strong signals).
2. Compare the number of true discoveries retained by each adjustment method versus the number of false positives.
3. Summarize which procedure you would recommend and why.

### Problem D — Communicating to Stakeholders

Draft a short memo (150–200 words) explaining to a product analytics team:
1. Why seeing a small p-value does not guarantee a real lift when many metrics are tracked.
2. The difference between controlling FWER and FDR in practice.
3. How you would incorporate multiplicity control into the release decision process.

## Deliverables

- Code snippets and summary tables for each simulation
- Memo from Problem D (plain language)

