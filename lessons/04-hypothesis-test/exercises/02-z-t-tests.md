# Exercise 2 — z/t Tests for Means

## Goals

- Practice one-sample, paired, and two-sample tests
- Evaluate assumptions and compute effect sizes
- Relate test decisions to confidence intervals and business context

## Dataset

Use `shared/data/heights_weights_sample.csv` unless otherwise noted. You may supplement with simulated data for sensitivity analyses.

## Problems

### Problem A — One-Sample Inference

1. Test whether the average height in the sample differs from 170 cm using a one-sample t-test. Report:
   - Test statistic, degrees of freedom, and p-value
   - 95% confidence interval for the mean
   - Practical interpretation (is the difference meaningful?)
2. Repeat assuming the population variance is known and equal to $\sigma = 8$ cm. Compare z vs t results.

### Problem B — Independent Samples

1. Compare male vs female mean heights using Welch’s t-test. Report test statistic, df, p-value, CI, and Cohen’s $d$.
2. Diagnose assumptions:
   - Inspect histograms or QQ-plots for each group
   - Comment on whether Welch’s test is appropriate
3. Suppose stakeholders consider a 2 cm lift practically important. Does the observed CI support a meaningful difference?

### Problem C — Paired Measurements

1. Create a synthetic “before vs after training” scenario by subtracting 1.2 kg from each participant’s weight (first 20 rows). Perform a paired t-test.
2. Compare results with an unpaired test treating “before” and “after” as independent samples. Explain why the paired test is more powerful.

### Problem D — Sensitivity to Non-Normality (Optional Extension)

1. Simulate 1,000 datasets where each group follows a heavy-tailed $t_3$ distribution with true mean difference 0.
2. Estimate the empirical Type I error of the Welch t-test at $\alpha = 0.05$.
3. Discuss whether you would still trust the test with such data and what alternative you might use.

## Deliverables

- Python or R code for analyses (screenshots acceptable)
- Short written interpretations for each sub-problem
- Highlight any assumption checks performed

