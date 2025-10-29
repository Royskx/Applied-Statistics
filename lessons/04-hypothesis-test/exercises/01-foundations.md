# Exercise 1 — Testing Foundations

## Goals

- Translate research questions into statistical hypotheses
- Compute rejection regions and p-values for simple test statistics
- Interpret Type I/II errors and communicate decisions in context

## Warm-Up (Concept Check)

1. For each scenario below, specify $H_0$, $H_1$, the direction of the test, and whether a Type I or Type II error is more costly.
   - (a) A manufacturer guarantees that the tensile strength of a cable is at least 120 MPa. A customer wants to verify the claim.
   - (b) A medical screening test aims to detect a rare disease with prevalence 1%. A positive result requires follow-up imaging.
   - (c) A product team will roll out a new onboarding flow only if the completion rate increases.

2. Explain the statement: “Reject $H_0$ at level $\alpha$ if and only if $\theta_0$ does not belong to the $(1-\alpha)$ confidence interval for $\theta$.” Give one concrete example.

## Analytical Practice

### Problem A — Critical Values and Rejection Regions

Let $X \sim \Normal(\mu, 1)$ and suppose we test $H_0: \mu = 0$ vs $H_1: \mu > 0$ based on a single observation.

1. Derive the rejection region for level $\alpha = 0.05$.
2. Compute $\beta(\mu)$, the probability of failing to reject $H_0$ when the true mean is $\mu = 0.8$.
3. Sketch the Type I and Type II error regions on the standard normal density (you can compare with the slide figure).

### Problem B — p-Values from Likelihood Ratios

Consider testing $H_0: \lambda = 2$ vs $H_1: \lambda > 2$ for a Poisson($\lambda$) variable $Y$ observed once.

1. Show that the likelihood ratio statistic simplifies to $\Lambda(y) = e^{-(\lambda_1-\lambda_0)} (\lambda_0/\lambda_1)^y$ for any fixed $\lambda_1 > \lambda_0$.
2. Argue that this likelihood ratio is monotone in $y$, so the LRT rejects for large counts.
3. Compute the p-value when $y=5$ under $H_0: \lambda=2$.
4. Interpret the result in words for a non-technical stakeholder.

## Communication Challenge

Write a 3–4 sentence summary (plain English, no formulas) explaining to a product manager:

- What $\alpha=0.05$ means in an A/B test with no observed lift
- Why a single experiment can produce a small p-value even if the true effect is zero
- How repeating many independent experiments without adjustment can lead to false discoveries

## Deliverables

- Show calculations or reasoning for all parts
- Include sketches or screenshots of any plots you create
- Optional: replicate the p-value histogram under $H_0$ using Python to build intuition

