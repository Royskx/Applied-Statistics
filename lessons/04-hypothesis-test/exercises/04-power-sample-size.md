# Exercise 4 — Power and Sample Size

## Goals

- Quantify power for mean and proportion tests
- Use analytical formulas and statsmodels solvers
- Communicate minimal detectable effects (MDE) and planning trade-offs

## Problems

### Problem A — Analytical Power Function

1. Derive the power function for the one-sample z-test with known variance $\sigma=4$, sample size $n=25$, and level $\alpha=0.05$ when testing $H_0: \mu = 50$ vs $H_1: \mu > 50$.
2. Evaluate the power for true means $\mu = 52$, 53, and 55.
3. Plot the power curve for $\mu \in [50, 56]$.

### Problem B — Required Sample Size (Means)

1. Using Statsmodels `TTestIndPower`, compute the per-group sample size required to detect a standardized effect size $d=0.35$ with 80% power and $\alpha=0.05$.
2. Re-run for power 0.9. How much larger is the sample size? Discuss diminishing returns.

### Problem C — Minimal Detectable Effect (Proportions)

1. Suppose current CTR is 9%. Using `NormalIndPower`, compute the MDE (absolute lift) detectable with 80% power, $\alpha=0.05$, and $n=2{,}000$ visitors per variant.
2. Plot power vs lift size for lifts between 1 and 6 percentage points.
3. Summarize the results for stakeholders: what lift would you guarantee to detect with this design?

### Problem D — Simulation Check (Optional)

1. Simulate 5,000 experiments for Problem C with true lift equal to the MDE from part 1.
2. Empirically estimate power (fraction of rejections). Does it match theoretical power?
3. Investigate the impact of variance misspecification by simulating higher variance scenarios.

## Deliverables

- Code, plots, and derived formulas for each part
- One paragraph summarizing how you would present the planning recommendations to a non-technical partner

