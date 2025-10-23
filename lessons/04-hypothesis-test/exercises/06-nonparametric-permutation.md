# Exercise 6 — Nonparametric and Permutation Methods

## Goals

- Apply rank-based tests for non-normal data
- Implement permutation tests for mean differences
- Compare parametric and nonparametric conclusions to assess robustness

## Dataset

Use `shared/data/heights_weights_sample.csv`. Augment with simulated skewed data when directed.

## Problems

### Problem A — Mann–Whitney vs Welch

1. Compare male vs female heights using the Mann–Whitney U test. Report the U statistic, p-value, and rank-biserial correlation.
2. Compare with the Welch t-test results (from Exercise 2). Do conclusions agree? Discuss sensitivity to distributional assumptions.

### Problem B — Wilcoxon Signed-Rank

1. Create a “before vs after” metric using the first 25 participants’ weights (e.g., subtract 1.5 kg to simulate improvement).
2. Run the Wilcoxon signed-rank test and report the test statistic and p-value.
3. Compare with the paired t-test. When would you prefer Wilcoxon?

### Problem C — Permutation Test Implementation

1. Implement a function `perm_test_diff_means(a, b, n_perm=5000)`.
2. Apply it to the male vs female height difference. Report the two-sided permutation p-value.
3. Compare with p-values from Mann–Whitney and Welch tests.

### Problem D — Sensitivity to Skewness (Optional)

1. Simulate two groups of size 20 from a log-normal distribution with equal medians but different variances.
2. Estimate Type I error for Welch, Mann–Whitney, and the permutation test at $\alpha = 0.05$.
3. Discuss which method you would trust in skewed settings.

## Deliverables

- Code for rank-based and permutation tests
- Summary table comparing p-values and effect sizes for each method
- Interpretation paragraphs highlighting robustness considerations

