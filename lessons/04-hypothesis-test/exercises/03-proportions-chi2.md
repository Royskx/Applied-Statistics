# Exercise 3 — Proportions and Chi-Squared

## Goals

- Apply proportion tests to A/B experiments
- Analyze contingency tables with chi-squared tests and Fisher’s exact test
- Interpret lift, confidence intervals, and residuals driving decisions

## Dataset

Use `shared/data/ab_test_clicks.csv` and `shared/data/manufacturing_defects.csv`.

## Problems

### Problem A — CTR A/B Test

1. Aggregate impressions and clicks by variant (A/B). Compute click-through rates (CTR).
2. Perform a two-proportion z-test (pooled variance). Report test statistic, p-value, and 95% CI for the difference in proportions.
3. Compute the minimal detectable effect (MDE) at $n_A=n_B=500$ with power 0.8. Does the observed difference exceed this threshold?
4. Repeat the test using Fisher’s exact test. Compare p-values and discuss when Fisher’s method is necessary.

### Problem B — Proportion Quality Check

1. In `manufacturing_defects.csv`, estimate the defect rate for 100-unit batches (A*) vs 200-unit batches (B*).
2. Test whether the defect rates differ using a two-proportion z-test.
3. Construct Wilson score intervals for each rate and comment on overlap.

### Problem C — Chi-Squared Goodness-of-Fit

1. Suppose category counts `[68, 40, 52, 40]` should follow a uniform distribution under $H_0$. Compute the chi-squared statistic, degrees of freedom, and p-value.
2. Identify which categories contribute most to rejecting $H_0$ using standardized residuals.

### Problem D — Independence in Contingency Tables

1. Create a $2\times 2$ table from the A/B dataset: variant vs “clicked at least once”.
2. Run `scipy.stats.chi2_contingency` with and without Yates continuity correction; compare results.
3. Report odds ratio and a 95% confidence interval (use `statsmodels.stats.contingency_tables.Table2x2`). Interpret in business terms.

## Deliverables

- Calculations and code for each test
- Interpretation paragraphs (2–3 sentences each) summarizing statistical and practical conclusions
- Optional: include a bar chart showing observed vs expected counts for Problem C

