# Lesson 4: Hypothesis Testing

**Author:** Applied Statistics Course Team  
**Prerequisites:** Lessons 1-3 (Statistical Modeling, Parameter Estimation, Estimator Properties)  
**Estimated time:** 3 hours  
**Slides:** `slides/main.pdf`  
**Test Selection Guide:** `TEST_SELECTION_GUIDE.md` ⭐ (Essential companion document)

This lesson builds upon:
- Lesson 1: Statistical Modeling — LLN, CLT, distributions
- Lesson 2: Parameter Estimation — MLE, MoM, Fisher information
- Lesson 3: Estimator Properties — consistency, efficiency, confidence intervals, bootstrap

## Learning Objectives

By the end of this lesson, you will be able to:

1. **Formulate** statistical hypotheses (H₀ vs. H₁) for real-world problems
2. **Understand** Type I/II errors, significance level (α), and statistical power (1-β)
3. **Interpret** p-values correctly and connect hypothesis tests to confidence intervals
4. **Select** appropriate tests based on data type and study design (using decision flowchart)
5. **Conduct** common tests: t-tests, proportion tests, chi-squared tests, non-parametric tests
6. **Report** results comprehensively (p-value + effect size + CI + practical interpretation)
7. **Recognize** common pitfalls and apply best practices

## Prerequisites

- Comfort with distributions, sampling, LLN/CLT
- Point estimation and asymptotics (MLE, delta method)
- Python: NumPy, SciPy, pandas, matplotlib/seaborn, statsmodels

---

## 1. Hypothesis Testing Foundations

### 1.1 Why Hypothesis Testing? A Motivating Example

Before diving into formal notation, let's understand **why** we need hypothesis testing through a concrete scenario.

#### The Drug Comparison Problem

Imagine you're a medical researcher testing whether a new drug (Drug A) reduces recovery time compared to a standard treatment (Drug B). You run a preliminary experiment with 3 patients on each drug:

- **Drug A patients:** Recover in 24, 28, and 32 hours (mean = 28 hours)
- **Drug B patients:** Recover in 40, 45, and 43 hours (mean = 42.7 hours)

Your initial result shows Drug A reduces recovery time by **15 hours on average**. Exciting! But should you conclude Drug A is better?

#### Why Individual Results Vary

Not everyone recovers in the same amount of time, even with identical treatment. Why? **Confounding factors:**

<img src="slides/figures/individual_variability.png" alt="Individual Variability" width="50%">

- **Exercise level:** More active patients may recover faster
- **Diet quality:** Nutrition affects healing
- **Sleep patterns:** Rest is crucial for recovery
- **Stress levels:** High stress can slow healing
- **Air pollution exposure:** Environmental factors matter
- **Baseline health:** Pre-existing conditions vary
- **Genetics and age:** Individual biological differences

These random factors create **variability** in outcomes, making it hard to isolate the drug's true effect.

#### The Problem with Single Experiments

What if you repeat the experiment with new patients? The second trial might show:
- Drug A: 45, 50, 48 hours (mean = 47.7 hours)
- Drug B: 30, 35, 33 hours (mean = 32.7 hours)

Now Drug B appears **15 hours faster**! If you keep repeating with small samples, results might consistently contradict your initial hypothesis. This teaches us: **one experiment is insufficient** when dealing with random variability.

#### The "Infinite Hypotheses" Problem

<img src="slides/figures/hypothesis_arbitrariness.png" alt="Hypothesis Arbitrariness" width="50%">

Suppose your repeated experiments show Drug A reduces recovery time by: 13 hours, then 12 hours, then 13.5 hours, then 12.25 hours...

**Which hypothesis should you test?**
- $H_0$: Drug A reduces time by 13 hours? (Why 13? That's arbitrary!)
- $H_0$: Drug A reduces time by 12 hours? (Why not 12.25?)
- $H_0$: Drug A reduces time by 13.1 hours? (Or 12.87?)

There are **infinitely many reasonable hypotheses** we could test. The value depends entirely on which experiment we happened to run first. This is unsatisfying and unscientific.

#### The Elegant Solution: Test "No Difference"

<img src="slides/figures/null_hypothesis_concept.png" alt="Null Hypothesis Concept" width="55%">

Since our goal is to determine whether Drug A is **different from** Drug B, we can simply test whether there is **NO difference** at all. This solves the arbitrary hypothesis problem because:

1. **Zero is the only unambiguous value** representing "no difference"
2. **Doesn't require preliminary data** to form the hypothesis
3. **Conservative approach:** We need strong evidence to claim a difference exists
4. **Clear decision framework:** Either reject this hypothesis (declare a difference) or fail to reject it (insufficient evidence)

This special hypothesis — that there is **no effect, no difference, no relationship** — is called the **null hypothesis** ($H_0$).

> 💡 **Key Insight:** We don't test specific effect sizes because they're arbitrary. We test whether the effect is zero, which is the only non-arbitrary reference point.

---

### 1.2 Problem Setup and Terminology

Now that we understand *why* hypothesis testing works this way, let's formalize the framework.

We test a claim about a parameter (or distribution) by specifying a **null hypothesis** $H_0$ and an **alternative** $H_1$. Examples:

- One-sided: $H_0: \mu \leq \mu_0$ vs. $H_1: \mu > \mu_0$
- Two-sided: $H_0: \mu = \mu_0$ vs. $H_1: \mu \neq \mu_0$
- Proportions: $H_0: p = p_0$ vs. $H_1: p \neq p_0$
- Independence: $H_0$: variables are independent in a contingency table

We choose a test statistic $T(X)$ whose sampling distribution under $H_0$ is known or well-approximated. A decision rule rejects $H_0$ in the rejection region (or for small p-values).

### 1.3 Making Decisions: Reject vs. Fail to Reject

<img src="slides/figures/reject_vs_fail_decision_tree.png" alt="Decision Framework" width="50%">

After conducting our hypothesis test, we arrive at one of two conclusions:

#### When to REJECT the Null Hypothesis

We **reject $H_0$** when our experimental data **consistently contradicts** the null hypothesis.

**Example:** If we test $H_0$: "Drug A and Drug B have no difference" and repeatedly find Drug A performs substantially better (e.g., 15-20 hours faster) across multiple experiments, we have strong evidence against $H_0$.

**What this means:** The data is incompatible with "no difference." We conclude there IS a real effect.

#### When to FAIL TO REJECT the Null Hypothesis

We **fail to reject $H_0$** when results are **consistent with the null hypothesis** or when differences are small enough that random variability could explain them.

**Example:** If small samples show Drug A is 0.5 hours faster in one trial, but Drug B is 0.3 hours faster in another trial, these tiny differences could easily be random fluctuations. We don't have sufficient evidence to claim a real difference.

**What this means:** The data doesn't contradict "no difference." We remain agnostic about whether a real effect exists.

#### Important: We NEVER "Accept" the Null Hypothesis

⚠️ **Critical Misconception to Avoid:**

We **never say** "accept $H_0$" — only **"fail to reject $H_0$"**. Why?

- **Absence of evidence ≠ evidence of absence**
- Failing to find a difference doesn't prove no difference exists
- Small samples might simply lack the **power** to detect real effects

Think of it like a court trial: "Not guilty" doesn't mean "innocent" — it means "insufficient evidence to convict."

---

### 1.4 Errors, Significance, and Power

<img src="slides/figures/type1_type2_regions.png" alt="Type I and Type II Errors" width="50%">

In hypothesis testing, two types of errors can occur:

- **Type I error:** Reject $H_0$ when $H_0$ is actually true (false positive)
  - Probability = significance level $\alpha$ (typically 0.05)
  - Example: Concluding Drug A is better when it's actually the same

- **Type II error:** Fail to reject $H_0$ when $H_1$ is actually true (false negative)
  - Probability = $\beta$
  - Example: Missing a real treatment benefit because sample size was too small

- **Power:** $1 - \beta$ = probability to correctly reject $H_0$ when $H_1$ is true
  - Target: typically 0.80 (80% chance to detect a real effect)

#### The Role of Sample Size

<img src="slides/figures/sample_size_effect.png" alt="Sample Size Effect" width="50%">

Sample size dramatically affects our ability to detect real effects:

- **Small samples (n=3):** Results vary wildly due to random factors. Hard to distinguish real effects from noise.
- **Medium samples (n=30):** Results become more stable. Real patterns start emerging.
- **Large samples (n=300):** Very stable results. Even small real effects become detectable.

**Key principle:** Larger samples → more stable estimates → higher statistical power → better ability to detect real effects.

Tuning $\alpha$ and designing for adequate power (typically 0.8) are core tasks in study planning.

### 1.5 p-Values and Interpretation

<img src="slides/figures/pvalue_under_null.png" alt="p-value Distribution Under Null" width="50%">

The **p-value** is the probability, under $H_0$, of observing data as extreme or more extreme than the realized data (as measured by test statistic $T$). For a right-tailed test:

$$\text{p-value} = \mathbb{P}_{H_0}(T \ge t_\text{obs}).$$

Two-sided tests typically double the tail probability (or use both tails appropriately for asymmetric distributions).

#### What p-values Tell Us

- **Small p-value (e.g., p < 0.05):** The observed data would be very unlikely if $H_0$ were true
  - This provides evidence **against** $H_0$
  - We reject $H_0$ and conclude there likely IS an effect

- **Large p-value (e.g., p > 0.05):** The observed data is consistent with $H_0$
  - This is **not** evidence **for** $H_0$ — just absence of evidence against it
  - We fail to reject $H_0$ (remain agnostic)

#### Common Misconceptions to Avoid

⚠️ **What p-values are NOT:**

1. ❌ The probability that $H_0$ is true
   - p-values assume $H_0$ is true, they don't test whether it's true

2. ❌ The probability that results are "due to chance"
   - This is informal and ambiguous phrasing

3. ❌ A measure of effect size or practical importance
   - A small p-value only indicates incompatibility with $H_0$
   - **Always report effect sizes and confidence intervals** alongside p-values

4. ❌ Proof of anything
   - Statistical tests provide evidence, not proof
   - Low p-values suggest data is inconsistent with $H_0$, not that $H_1$ is certain

#### Best Practice

When reporting results, include:
- The p-value and whether you reject/fail to reject
- The estimated effect size (e.g., difference in means)
- Confidence interval for the effect
- Context about practical significance

### 1.6 Understanding Variability and Confounders

<img src="slides/figures/confounder_illustration.png" alt="Sources of Variability" width="50%">

A crucial insight from our drug example: **even with identical treatment, outcomes vary** due to uncontrolled factors. Statistical hypothesis testing specifically accounts for this variability when determining whether observed differences are meaningful.

#### Why This Matters

When we observe a difference between groups (e.g., Drug A vs. Drug B), that difference could arise from:

1. **The treatment effect itself** (what we want to measure)
2. **Random variability** from confounding factors (noise)

Hypothesis testing helps us determine whether the observed difference is **larger than we'd expect from random variability alone**.

#### Practical Implications

- Small differences might not be statistically significant if variability is high
- Large samples help because they average out random fluctuations
- Controlling for confounders (through experimental design or statistical adjustment) increases statistical power

---

### 1.7 Duality with Confidence Intervals

For many parametric tests, rejecting $H_0: \theta = \theta_0$ at level $\alpha$ is equivalent to $\theta_0$ not lying in the $(1-\alpha)$ confidence interval for $\theta$. This equivalence ties Lesson 3 to hypothesis testing.

**Example:** If a 95% confidence interval for the difference in means is [2.5, 8.3] hours, then:
- We would reject $H_0: \mu_A - \mu_B = 0$ at $\alpha = 0.05$ (because 0 is not in the interval)
- We would fail to reject $H_0: \mu_A - \mu_B = 5$ (because 5 is in the interval)

This connection shows that confidence intervals provide richer information than p-values alone — they show not just whether an effect exists, but the range of plausible effect sizes.

---

### 1.8 Power and Sample Size (Brief Overview)

**Statistical Power** = 1 - β = Probability of correctly rejecting H₀ when H₁ is true.

**Key Insight:** Before collecting data, determine: *"How many samples do I need to reliably detect an effect of a given size?"*

**The Four Interconnected Quantities:**
1. **Effect Size (δ):** Magnitude of difference to detect (e.g., Cohen's d)
2. **Sample Size (n):** Number of observations needed
3. **Significance Level (α):** Type I error rate (typically 0.05)
4. **Power (1-β):** Typically target 0.80 (80% chance to detect real effect)

**Key Principle:** If you know any three quantities, you can solve for the fourth.

**Practical Workflow:**
1. Specify the minimal detectable effect you care about
2. Choose α (typically 0.05) and target power (typically 0.80)
3. Calculate required sample size using formulas or software
4. Assess feasibility and adjust if needed

**Python Example (Two-Sample t-Test):**

```python
from statsmodels.stats.power import TTestIndPower

# Calculate required sample size
effect_size = 0.5  # Cohen's d (medium effect)
alpha = 0.05
power = 0.80

analysis = TTestIndPower()
n_required = analysis.solve_power(
    effect_size=effect_size,
    alpha=alpha,
    power=power,
    alternative='two-sided'
)

print(f"Required sample size per group: {n_required:.0f}")
# Output: 64 per group
```

> 📘 **For detailed coverage:** See **Appendix A: Power Analysis** for comprehensive treatment including formulas, power curves, and advanced topics.

---

### 1.9 Optimality and Likelihood Ratios (Brief Mention)

**Neyman-Pearson Lemma:** For simple H₀ vs. simple H₁, the most powerful test at level α is a likelihood ratio test (LRT).

**Key Insights:**
- Classical z-, t-, χ², and F-tests arise as LRTs or their limits
- Wilks' theorem: For composite hypotheses, -2 log Λ → χ² asymptotically
- Provides unifying framework for understanding different tests

**Practical Implication:** The tests we use (t-test, chi-squared, etc.) are not arbitrary—they're theoretically optimal under certain conditions.

> 📘 **For theoretical details:** See advanced statistical inference textbooks (Casella & Berger, Lehmann & Romano).

---

## 2. Test Selection Guide

Before diving into specific tests, it's crucial to know **which test to use** for your data and research question.

### 2.1 Quick Decision Rules

1. **Continuous data, two groups:** Welch's t-test (default)
2. **Categorical data, two groups:** Two-proportion z-test (large n) or Fisher's exact (small n)
3. **Unsure about normality:** Use non-parametric test
4. **Paired/matched data:** Always use paired test
5. **Small samples + categorical:** Use exact tests

### 2.2 Decision Flowchart

**⭐ See `TEST_SELECTION_GUIDE.md` for comprehensive flowchart and detailed guidance.**

**Quick Overview:**

```
What type of data?
├─ CONTINUOUS → t-tests, ANOVA, or non-parametric alternatives
│  ├─ One group vs. value → One-sample t-test
│  ├─ Two independent groups → Welch's t-test (default)
│  ├─ Paired/matched data → Paired t-test
│  └─ Three+ groups → ANOVA
│
└─ CATEGORICAL → Proportion tests, chi-squared, or Fisher's exact
   ├─ One proportion → z-test for proportion
   ├─ Two proportions → Two-proportion z-test or Fisher's exact
   └─ Multiple categories → Chi-squared tests
```

### 2.3 Common Tests Summary

| Data Type | Scenario | Test | Python Function |
|-----------|----------|------|-----------------|
| Continuous | One sample | t-test | `ttest_1samp()` |
| Continuous | Two independent | Welch's t-test | `ttest_ind(equal_var=False)` |
| Continuous | Paired | Paired t-test | `ttest_rel()` |
| Categorical | One proportion | z-test | `proportions_ztest()` |
| Categorical | Two proportions | Two-proportion z-test | `proportions_ztest()` |
| Categorical | Multiple categories | Chi-squared | `chi2_contingency()` |
| Non-parametric | Two independent | Mann-Whitney U | `mannwhitneyu()` |
| Non-parametric | Paired | Wilcoxon signed-rank | `wilcoxon()` |

---

## 3. Tests for Means: z and t

### 3.1 One-Sample t-Test

**Purpose:** Test if population mean equals a specific value

**Hypotheses:**
- H₀: μ = μ₀
- H₁: μ ≠ μ₀ (or μ > μ₀, or μ < μ₀)

**Test Statistic:**

$$T = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t_{n-1}$$

**When to use:**
- Unknown population variance
- Approximately normal data (or large n by CLT)
- Single group compared to known value

**Python Example:**

```python
from scipy import stats

# Data: recovery times (hours)
data = [24, 28, 32, 26, 30, 29, 27, 31]

# Test H0: μ = 25
t_stat, p_value = stats.ttest_1samp(data, popmean=25)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject H₀: Mean is significantly different from 25")
else:
    print("Fail to reject H₀")
```

---

### 3.2 Two-Sample t-Tests

#### Welch's t-Test (Default) ⭐

**Purpose:** Compare means of two independent groups (unequal variances allowed)

**Test Statistic:**

$$T = \frac{\bar{X}_A - \bar{X}_B}{\sqrt{S_A^2/n_A + S_B^2/n_B}}$$

**Python Example:**

```python
# Drug A and Drug B recovery times
drug_a = [24, 28, 32, 26, 30, 29, 27, 31]
drug_b = [40, 45, 43, 38, 42, 44, 41, 39]

# Welch's t-test (equal_var=False is default)
t_stat, p_value = stats.ttest_ind(drug_a, drug_b, equal_var=False)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Effect size (Cohen's d)
import numpy as np
mean_diff = np.mean(drug_a) - np.mean(drug_b)
pooled_std = np.sqrt((np.var(drug_a, ddof=1) + np.var(drug_b, ddof=1)) / 2)
cohens_d = mean_diff / pooled_std

print(f"Mean difference: {mean_diff:.2f} hours")
print(f"Cohen's d: {cohens_d:.4f}")
```

#### Pooled t-Test

**Use only if:** Equal variances verified (via Levene's test)

```python
# Pooled t-test (equal_var=True)
t_stat, p_value = stats.ttest_ind(drug_a, drug_b, equal_var=True)
```

**Advantage:** Slightly more powerful when equal variance assumption holds  
**Disadvantage:** Can be misleading if variances differ

**Recommendation:** Use Welch's t-test as default (safer, robust)

---

### 3.3 Paired t-Test

**Purpose:** Compare means of paired/matched observations

**Examples:**
- Before/after measurements (same patients)
- Matched pairs (twins, siblings)
- Repeated measures

**Key Advantage:** Controls for individual differences → more powerful

**Test Statistic:**

$$T = \frac{\bar{D}}{S_D / \sqrt{n}}$$

where $D_i = X_{1i} - X_{2i}$

**Python Example:**

```python
# Blood pressure before and after treatment (same patients)
before = [140, 135, 150, 145, 138, 142, 148, 136]
after = [130, 128, 142, 135, 132, 138, 140, 130]

# Paired t-test
t_stat, p_value = stats.ttest_rel(before, after)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Mean difference
mean_diff = np.mean(np.array(before) - np.array(after))
print(f"Mean reduction: {mean_diff:.2f} mmHg")
```

**⚠️ Warning:** Don't use two-sample t-test on paired data (loses power)!

---

### 3.4 Effect Sizes and Reporting

**Cohen's d (Standardized Effect Size):**

$$d = \frac{\bar{X}_1 - \bar{X}_2}{S_p}$$

**Interpretation:**
- Small: d ≈ 0.2
- Medium: d ≈ 0.5
- Large: d ≈ 0.8

**Reporting Checklist:**
- ✓ Point estimate (mean difference)
- ✓ Confidence interval
- ✓ p-value
- ✓ Effect size (Cohen's d)
- ✓ Practical interpretation

**Example Report:**
> "Drug A reduced recovery time by 14.7 hours (95% CI: [8.2, 21.2], t(14)=4.23, p=0.002, Cohen's d=1.34). This represents a large and clinically meaningful improvement."

---

## 4. Tests for Proportions and Categorical Data

### 4.1 One-Proportion z-Test

**Purpose:** Test if proportion equals a specific value

**Example:** Is conversion rate 10%?

**Test Statistic:**

$$Z = \frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}}$$

**Python Example:**

```python
from statsmodels.stats.proportion import proportions_ztest

# 45 conversions out of 500 visitors
count = 45
nobs = 500
p0 = 0.10  # Test if p = 0.10

z_stat, p_value = proportions_ztest(count, nobs, value=p0)

print(f"z-statistic: {z_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Observed proportion: {count/nobs:.4f}")
```

**When to use:**
- Binary outcomes (success/failure)
- Large samples (np>10, n(1-p)>10)
- Testing against known value

---

### 4.2 Two-Proportion z-Test (A/B Testing)

**Purpose:** Compare proportions between two groups

**Example:** Does variant B have higher click-through rate than A?

**Test Statistic:**

$$Z = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{\hat{p}(1-\hat{p})(1/n_A + 1/n_B)}}$$

where $\hat{p} = (X_A + X_B)/(n_A + n_B)$ is the pooled proportion.

**Python Example:**

```python
# A/B test: click-through rates
count_a = 45  # clicks in variant A
nobs_a = 500  # visitors to A
count_b = 68  # clicks in variant B
nobs_b = 500  # visitors to B

z_stat, p_value = proportions_ztest([count_a, count_b], [nobs_a, nobs_b])

print(f"z-statistic: {z_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Conversion rates
ctr_a = count_a / nobs_a
ctr_b = count_b / nobs_b
lift = (ctr_b - ctr_a) / ctr_a * 100

print(f"CTR A: {ctr_a:.4f}")
print(f"CTR B: {ctr_b:.4f}")
print(f"Relative lift: {lift:.2f}%")
```

---

### 4.3 Fisher's Exact Test

**Purpose:** Test independence in 2×2 contingency tables (small samples)

**Advantage:** Exact p-values, no large-sample requirement

**Python Example:**

```python
# 2x2 contingency table
# Rows: Treatment (Success/Failure)
# Cols: Gender (Male/Female)
table = [[8, 2],   # Success: 8 males, 2 females
         [3, 7]]   # Failure: 3 males, 7 females

odds_ratio, p_value = stats.fisher_exact(table)

print(f"Odds ratio: {odds_ratio:.4f}")
print(f"p-value: {p_value:.4f}")
```

**When to use:**
- 2×2 tables
- Small samples (any expected count <5)
- Exact inference needed

---

### 4.4 Chi-Squared Tests

#### Chi-Squared Goodness-of-Fit

**Purpose:** Test if observed distribution matches expected

**Test Statistic:**

$$\chi^2 = \sum_{i=1}^k \frac{(O_i - E_i)^2}{E_i} \sim \chi^2_{k-1-r}$$

**Python Example:**

```python
# Dice rolls: observed frequencies
observed = [18, 22, 15, 20, 17, 18]  # 6 faces

# Expected frequencies (uniform)
expected = [18.33] * 6  # Total 110 rolls

chi2_stat, p_value = stats.chisquare(observed, expected)

print(f"χ² statistic: {chi2_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

#### Chi-Squared Test of Independence

**Purpose:** Test if two categorical variables are independent

**Python Example:**

```python
# Contingency table: Product preference × Age group
table = [[30, 20, 10],  # Product A: Young, Middle, Old
         [15, 25, 30]]  # Product B: Young, Middle, Old

chi2_stat, p_value, dof, expected = stats.chi2_contingency(table)

print(f"χ² statistic: {chi2_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")
```

**When to use:**
- Contingency tables (r × c)
- Testing independence of categorical variables
- All expected counts ≥5

---

## 5. Non-Parametric Tests

**When to use non-parametric tests:**
- Data is not normally distributed
- Small samples where CLT doesn't apply
- Ordinal data (rankings)
- Outliers present
- Robust alternative needed

### 5.1 Mann-Whitney U Test

**Purpose:** Compare distributions of two independent groups (non-parametric alternative to t-test)

**Python Example:**

```python
# Compare median income between two cities
city_a = [35000, 42000, 38000, 45000, 40000]
city_b = [50000, 55000, 48000, 52000, 60000]

u_stat, p_value = stats.mannwhitneyu(city_a, city_b, alternative='two-sided')

print(f"U statistic: {u_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

**When to use:**
- Two independent groups
- Non-normal data
- Ordinal data
- Robust to outliers

---

### 5.2 Wilcoxon Signed-Rank Test

**Purpose:** Compare paired observations (non-parametric alternative to paired t-test)

**Python Example:**

```python
# Satisfaction scores before and after training
before = [3, 4, 2, 5, 3, 4, 2]
after = [4, 5, 3, 5, 4, 5, 3]

w_stat, p_value = stats.wilcoxon(before, after)

print(f"W statistic: {w_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

**When to use:**
- Paired/matched data
- Non-normal differences
- Ordinal data
- Robust alternative to paired t-test

---

## 6. Practical Workflow & Common Pitfalls

### 6.1 Pre-Test Checklist

Before running any hypothesis test:

- [ ] **Clarify the research question:** What are you trying to test?
- [ ] **Formulate hypotheses:** Define H₀ and H₁ clearly
- [ ] **Check data type:** Continuous, categorical, ordinal?
- [ ] **Check independence:** Are observations independent or paired?
- [ ] **Verify assumptions:** Normality, equal variance, large sample?
- [ ] **Choose appropriate test:** Use decision flowchart
- [ ] **Set significance level:** Typically α = 0.05
- [ ] **Consider practical significance:** What effect size matters?

### 6.2 Common Pitfalls

#### 1. Using Wrong Test
❌ **Mistake:** t-test on categorical data  
✅ **Fix:** Use chi-squared or proportion tests

#### 2. Ignoring Assumptions
❌ **Mistake:** t-test on heavily skewed data (small n)  
✅ **Fix:** Check normality (QQ-plot), use non-parametric if needed

#### 3. Treating Paired Data as Independent
❌ **Mistake:** Two-sample t-test on before/after  
✅ **Fix:** Use paired t-test

#### 4. Multiple Testing Without Correction
❌ **Mistake:** Running 20 tests at α=0.05  
✅ **Fix:** Apply Bonferroni (α/m) or FDR control (see Appendix B)

#### 5. Confusing Statistical and Practical Significance
❌ **Mistake:** Reporting p<0.001 without mentioning tiny effect  
✅ **Fix:** Always report effect size and CI

#### 6. p-Hacking
❌ **Mistake:** Testing multiple hypotheses until one is significant  
✅ **Fix:** Pre-register analysis plan, correct for multiple testing

### 6.3 Reporting Best Practices

**Always include:**
1. **Test used:** "Welch's t-test"
2. **Sample sizes:** n_A = 50, n_B = 48
3. **Test statistic:** t = 3.45
4. **p-value:** p = 0.001
5. **Effect size:** Cohen's d = 0.67 (medium effect)
6. **Confidence interval:** 95% CI: [2.1, 7.3]
7. **Practical interpretation:** "Drug A reduced recovery time by 4.7 hours on average, a clinically meaningful improvement."

---

## 7. Summary and Key Takeaways

### Core Concepts

1. **Hypothesis testing formalizes evidence comparison** between H₀ (status quo) and H₁ (change)
2. **Type I/II errors quantify risk:** α (false positive), β (false negative), power = 1-β
3. **p-values measure surprise under H₀:** Must be contextualized with intervals and effect sizes
4. **Test selection depends on data type and design:** Use flowchart and decision rules
5. **Always report comprehensively:** p-value + effect size + CI + interpretation

### Practical Wisdom

- **Visualize first:** Plot your data before choosing a test
- **Check assumptions:** Don't blindly apply tests
- **When in doubt:** Non-parametric tests are safer (less powerful but more robust)
- **Think practically:** Does the effect size matter in real-world terms?
- **Be transparent:** Report all tests conducted, not just significant ones

### Next Steps

- **Lab 08:** Hands-on practice with hypothesis testing
- **Appendix A:** Power analysis (optional advanced topic)
- **Appendix B:** Multiple testing corrections (optional)
- **TEST_SELECTION_GUIDE.md:** Comprehensive reference for test selection

---

## Resources and Further Reading

### Textbooks

1. **Casella & Berger (2002):** *Statistical Inference* — Chapters 8-9 (theoretical foundation)
2. **Wasserman (2004):** *All of Statistics* — Concise treatment of tests and p-values
3. **Agresti (2019):** *An Introduction to Categorical Data Analysis* — Proportions and chi-squared
4. **Hollander & Wolfe (1999):** *Nonparametric Statistical Methods* — Rank-based tests

### Online Resources

- **SciPy documentation:** `scipy.stats` module for hypothesis testing
- **Statsmodels documentation:** Power analysis and proportion tests
- **StatQuest videos:** Intuitive explanations of hypothesis tests

### Datasets

- `shared/data/heights_weights_sample.csv` — For t-tests
- `shared/data/ab_test_clicks.csv` — For proportion tests
- `shared/data/manufacturing_defects.csv` — For chi-squared tests

---

# Appendices

## Appendix A: Power Analysis (Optional Advanced Topic)

**Note:** This is an optional advanced topic that may be skipped depending on course scope. For a focused course, prioritize Sections 1-7. Power analysis is important for study design but can be covered separately or in advanced courses.

### A.1 The Four Interconnected Quantities

<img src="slides/figures/power_visualization.png" alt="Power Visualization" width="60%">

Consider testing $H_0: \mu = \mu_0$ vs. $H_1: \mu = \mu_1$ where $\mu_1 > \mu_0$. The figure shows:

- **Blue distribution:** Sampling distribution of $\bar{X}$ under $H_0$
- **Red distribution:** Sampling distribution of $\bar{X}$ under $H_1$
- **Shaded region (right tail):** Rejection region for $H_0$ at level $\alpha$
- **Power:** Area under red curve in the rejection region

As sample size increases:
- Both distributions become narrower (less variance)
- Distributions separate more clearly
- Power increases (more red area in rejection region)

<img src="slides/figures/power_curve_sample_size.png" alt="Power vs Sample Size" width="55%">

**Power curves** show how power changes with sample size for a fixed effect size. Key observations:
- Power increases with $n$, but with diminishing returns
- Larger effect sizes require smaller samples to achieve target power
- The relationship is nonlinear: doubling power doesn't mean doubling $n$

#### 1.8.3 Sample Size Formulas

For common test scenarios, we can derive approximate sample size requirements.

**One-Sample z-Test ($\sigma$ known):**

To test $H_0: \mu = \mu_0$ vs. $H_1: \mu = \mu_1$ with power $1-\beta$:

$$n = \left(\frac{(z_{1-\alpha/2} + z_{1-\beta}) \cdot \sigma}{\delta}\right)^2$$

where $\delta = |\mu_1 - \mu_0|$ is the effect size.

**Two-Sample t-Test (equal variance, balanced groups):**

To detect difference $\delta = |\mu_A - \mu_B|$ with common standard deviation $\sigma$:

$$n_{\text{per group}} = 2 \left(\frac{(z_{1-\alpha/2} + z_{1-\beta}) \cdot \sigma}{\delta}\right)^2$$

Or in terms of Cohen's $d = \delta/\sigma$:

$$n_{\text{per group}} = 2 \left(\frac{z_{1-\alpha/2} + z_{1-\beta}}{d}\right)^2$$

**Poisson Rate Test:**

To test $H_0: \lambda = \lambda_0$ vs. $H_1: \lambda = \lambda_1$ using normal approximation:

$$n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \cdot \lambda_0}{(\lambda_1 - \lambda_0)^2}$$

**Two-Sample Proportions:**

To detect difference $\delta = |p_A - p_B|$ with pooled proportion $\bar{p} = (p_A + p_B)/2$:

$$n_{\text{per group}} = 2 \frac{\left(z_{1-\alpha/2}\sqrt{2\bar{p}(1-\bar{p})} + z_{1-\beta}\sqrt{p_A(1-p_A)+p_B(1-p_B)}\right)^2}{(p_B - p_A)^2}$$

**Note:** These formulas use normal approximations and assume two-sided tests. For one-sided tests, replace $z_{1-\alpha/2}$ with $z_{1-\alpha}$.

#### 1.8.4 Practical Workflow for Study Planning

**Step 1: Define the Minimum Detectable Effect (MDE)**

The MDE is the smallest effect that matters practically. This should be based on:
- Domain knowledge and subject-matter expertise
- Clinical or practical significance thresholds
- Previous research findings
- Cost-benefit considerations

**Example:** "We care if the new drug reduces recovery time by at least 8 hours, because shorter reductions wouldn't justify the higher cost."

⚠️ **Common mistake:** Choosing MDE based on what's statistically detectable with available budget, rather than what's practically meaningful.

**Step 2: Estimate Variability**

Obtain an estimate of $\sigma$ (or variance) from:
- Pilot studies or preliminary data
- Published literature on similar populations
- Historical data from previous studies
- Expert opinion (as last resort)

**Example:** "Previous studies show recovery time has $\sigma \approx 12$ hours."

**Step 3: Choose α and Target Power**

Standard choices:
- $\alpha = 0.05$ (5% false positive rate)
- Power = 0.80 (80% chance to detect real effect)

Adjust based on context:
- **Higher stakes** (e.g., drug approval): Lower $\alpha$ (0.01), higher power (0.90)
- **Exploratory research**: Standard values acceptable
- **Multiple testing**: Adjust $\alpha$ using Bonferroni or other corrections

**Step 4: Calculate Required Sample Size**

Use formulas or software:

```python
from statsmodels.stats.power import TTestIndPower

# Example: Two-sample t-test
effect_size = 8 / 12  # δ/σ = Cohen's d = 0.667
alpha = 0.05
power = 0.80

analysis = TTestIndPower()
n_required = analysis.solve_power(
    effect_size=effect_size,
    alpha=alpha,
    power=power,
    alternative='two-sided'
)
print(f"Required sample size per group: {n_required:.0f}")
# Output: Required sample size per group: 36
```

**Step 5: Adjust for Practical Constraints**

If required $n$ is too large:
- **Increase α:** Accept higher false positive rate (if justifiable)
- **Decrease power:** Accept lower detection probability (risky)
- **Increase MDE:** Only detect larger effects (if acceptable)
- **Reduce variability:** Improve measurement precision, control confounders
- **Seek more funding:** If effect is important enough

If required $n$ is achievable:
- **Add buffer:** Account for dropout, missing data (multiply by 1.1-1.2)
- **Document assumptions:** Record all decisions for transparency

#### 1.8.5 Python Implementation with Statsmodels

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestIndPower, zt_ind_solve_power
from scipy.stats import norm

# Example 1: Solve for required sample size
effect_size = 0.5  # Cohen's d (medium effect)
alpha = 0.05
power = 0.80

analysis = TTestIndPower()
n_required = analysis.solve_power(
    effect_size=effect_size,
    alpha=alpha,
    power=power,
    alternative='two-sided'
)
print(f"Required sample size per group: {n_required:.0f}")

# Example 2: Solve for achievable power with fixed n
n_available = 50
achievable_power = analysis.power(
    effect_size=effect_size,
    nobs1=n_available,
    alpha=alpha,
    alternative='two-sided'
)
print(f"Achievable power with n={n_available}: {achievable_power:.3f}")

# Example 3: Solve for minimum detectable effect with fixed n and power
mde = analysis.solve_power(
    nobs1=n_available,
    alpha=alpha,
    power=power,
    alternative='two-sided'
)
print(f"Minimum detectable effect (Cohen's d): {mde:.3f}")

# Example 4: Power curve visualization
sample_sizes = np.arange(10, 200, 5)
power_values = analysis.power(
    effect_size=effect_size,
    nobs1=sample_sizes,
    alpha=alpha,
    alternative='two-sided'
)

plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, power_values, linewidth=2)
plt.axhline(y=0.80, color='r', linestyle='--', linewidth=2, label='Target power = 0.80')
plt.axvline(x=n_required, color='g', linestyle='--', linewidth=2, 
            label=f'Required n = {n_required:.0f}')
plt.xlabel('Sample Size per Group', fontsize=12)
plt.ylabel('Statistical Power', fontsize=12)
plt.title('Power Curve for Two-Sample t-Test (d=0.5, α=0.05)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.show()

# Example 5: Effect size sensitivity
effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large
colors = ['blue', 'green', 'red']
labels = ['Small (d=0.2)', 'Medium (d=0.5)', 'Large (d=0.8)']

plt.figure(figsize=(10, 6))
for es, color, label in zip(effect_sizes, colors, labels):
    power_vals = analysis.power(
        effect_size=es,
        nobs1=sample_sizes,
        alpha=alpha,
        alternative='two-sided'
    )
    plt.plot(sample_sizes, power_vals, linewidth=2, color=color, label=label)

plt.axhline(y=0.80, color='black', linestyle='--', linewidth=1, alpha=0.5)
plt.xlabel('Sample Size per Group', fontsize=12)
plt.ylabel('Statistical Power', fontsize=12)
plt.title('Power Curves for Different Effect Sizes (α=0.05)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.show()
```

#### 1.8.6 Common Pitfalls and How to Avoid Them

⚠️ **Underpowered Studies**

**Problem:** Many published studies have power < 50%, meaning they're more likely to miss real effects than detect them.

**Consequences:**
- Wasted resources (time, money, participant burden)
- False negatives contribute to publication bias
- Irreproducible results

**Solution:** Always conduct power analysis during study design, not after data collection.

⚠️ **Post-hoc Power Analysis**

**Problem:** Computing "observed power" after seeing results is statistically meaningless.

**Why it's wrong:** Power depends on the true effect size (unknown). Using the observed effect size creates circular reasoning—studies with p > 0.05 will always have "low power," but this doesn't tell us anything new.

**What to do instead:** Report confidence intervals for effect sizes. Wide CIs indicate imprecision, which is more informative than post-hoc power.

⚠️ **Ignoring Practical Significance**

**Problem:** Focusing only on statistical significance without considering whether effects matter practically.

**Example:** With n=10,000, a difference of 0.5 hours in recovery time might be statistically significant (p < 0.001) but clinically irrelevant.

**Solution:** Always define MDE based on domain knowledge before collecting data. Report both statistical and practical significance.

⚠️ **Unrealistic Effect Size Assumptions**

**Problem:** Overestimating effect sizes leads to underpowered studies.

**Example:** Assuming d = 0.8 (large effect) when literature suggests d = 0.3 (small effect).

**Solution:** Use conservative estimates from:
- Meta-analyses of similar studies
- Pilot data
- Smallest effect size of practical interest (SESOI)

⚠️ **Ignoring Multiple Comparisons**

**Problem:** Planning sample size for one test but conducting many tests inflates Type I error.

**Solution:** Adjust $\alpha$ for multiple comparisons (e.g., Bonferroni: use $\alpha/m$ for $m$ tests) when calculating required $n$.

#### 1.8.7 Connection to Study Design

Power analysis informs multiple aspects of research planning:

**Budget Planning:**
- Cost per participant × required $n$ = total cost
- Helps justify funding requests
- Identifies when studies are infeasible

**Timeline Estimation:**
- Recruitment rate × required $n$ = study duration
- Informs project scheduling
- Identifies need for multi-site collaboration

**Feasibility Assessment:**
- Can we realistically recruit required $n$?
- Are there enough eligible participants?
- Do we have sufficient resources?

**Stopping Rules:**
- Sequential testing: Monitor accumulating data
- Early stopping for efficacy or futility
- Adaptive designs: Adjust $n$ based on interim results

**Pre-registration and Transparency:**
- Document power analysis before data collection
- Prevents p-hacking and HARKing (Hypothesizing After Results are Known)
- Increases credibility and reproducibility

#### 1.8.8 Poisson-Specific Power Analysis

For Poisson hypothesis testing (relevant to Lab 08), power analysis follows similar principles:

**One-Sample Poisson Test:**

Test $H_0: \lambda = \lambda_0$ vs. $H_1: \lambda = \lambda_1$ using normal approximation:

$$n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \cdot \lambda_0}{(\lambda_1 - \lambda_0)^2}$$

**Example:** Football goals
- Historical average: $\lambda_0 = 2.5$ goals per game
- Want to detect increase to $\lambda_1 = 3.5$ goals
- Target: $\alpha = 0.05$, power = 0.80

```python
from scipy.stats import norm

lambda_0 = 2.5
lambda_1 = 3.5
alpha = 0.05
power = 0.80

z_alpha = norm.ppf(1 - alpha/2)  # 1.96
z_beta = norm.ppf(power)          # 0.84

n_required = ((z_alpha + z_beta)**2 * lambda_0) / (lambda_1 - lambda_0)**2
print(f"Required number of games: {n_required:.0f}")
# Output: Required number of games: 20

# Verify with simulation
np.random.seed(42)
n_sims = 10000
reject_count = 0

for _ in range(n_sims):
    # Simulate data under H1
    data = np.random.poisson(lambda_1, size=int(n_required))
    
    # Perform test
    lambda_hat = np.mean(data)
    z_stat = (lambda_hat - lambda_0) / np.sqrt(lambda_0 / len(data))
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    
    if p_value < alpha:
        reject_count += 1

empirical_power = reject_count / n_sims
print(f"Empirical power: {empirical_power:.3f}")
# Output: Empirical power: 0.802
```

**Two-Sample Poisson Comparison:**

Comparing rates $\lambda_A$ and $\lambda_B$ requires similar calculations, treating the difference as approximately normal for large samples.

#### 1.8.9 Advanced Topics (Brief Mention)

For readers interested in going deeper:

**Adaptive Designs:**
- Adjust sample size based on interim results
- Group sequential designs with pre-specified stopping rules
- Requires careful control of Type I error inflation

**Bayesian Power:**
- Probability of achieving practical significance (not just statistical)
- Assurance: Average power over prior distribution of effect sizes
- More aligned with decision-making goals

**Simulation-Based Power:**
- For complex designs without closed-form solutions
- Monte Carlo simulation of data generation and analysis
- Essential for mixed models, survival analysis, etc.

**Equivalence and Non-Inferiority Testing:**
- Power to demonstrate similarity rather than difference
- Requires larger samples than superiority tests
- Common in pharmaceutical development

#### 1.8.10 Summary and Key Takeaways

Power analysis is not optional—it's a fundamental component of responsible research:

✓ **Plan ahead:** Conduct power analysis during study design, not after data collection  
✓ **Be realistic:** Use conservative effect size estimates from literature  
✓ **Consider context:** Adjust $\alpha$ and power based on consequences of errors  
✓ **Document decisions:** Pre-register analysis plans for transparency  
✓ **Report comprehensively:** Include effect sizes and CIs, not just p-values  
✓ **Think practically:** Statistical significance ≠ practical importance

**Connection to Lesson 3:** Power analysis builds directly on confidence intervals. A study with adequate power will produce CIs narrow enough to exclude the null value when a real effect exists.

**Connection to Lab 08:** You'll apply these concepts to Poisson hypothesis testing, calculating required sample sizes and visualizing power curves for real-world scenarios.

---

## Appendix B: Multiple Testing Corrections (Optional)

**Note:** This is an optional topic that may be skipped depending on course scope. Multiple testing becomes important when conducting many hypothesis tests simultaneously.

### B.1 The Multiple Testing Problem

When testing many hypotheses, false positives accumulate:
- With α = 0.05 and 20 independent tests, expect 1 false positive even if all H₀ are true
- With 100 tests, expect 5 false positives
- This inflates the family-wise error rate (FWER)

### B.2 Bonferroni Correction

**Most conservative approach:** Divide α by number of tests

**Adjusted significance level:** α_adj = α / m

**Python Example:**

```python
import numpy as np
from scipy import stats

# 10 hypothesis tests
p_values = [0.001, 0.03, 0.08, 0.15, 0.22, 0.35, 0.45, 0.60, 0.75, 0.90]
m = len(p_values)
alpha = 0.05

# Bonferroni correction
alpha_bonf = alpha / m
significant_bonf = [p < alpha_bonf for p in p_values]

print(f"Bonferroni threshold: {alpha_bonf:.4f}")
print(f"Significant tests: {sum(significant_bonf)}")
```

**Pros:** Simple, controls FWER strongly  
**Cons:** Very conservative, low power with many tests

### B.3 Benjamini-Hochberg (FDR Control)

**Less conservative:** Controls false discovery rate instead of FWER

**Procedure:**
1. Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
2. Find largest k where p_(k) ≤ (k/m) × α
3. Reject hypotheses 1, 2, ..., k

**Python Example:**

```python
from statsmodels.stats.multitest import multipletests

# Apply Benjamini-Hochberg
reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

print(f"Significant tests (BH): {sum(reject)}")
print(f"Adjusted p-values: {p_adj}")
```

**Pros:** More powerful than Bonferroni  
**Cons:** Controls FDR, not FWER (allows some false positives)

### B.4 When to Use Multiple Testing Corrections

**Use corrections when:**
- Conducting many tests on same dataset
- Exploratory data analysis with many variables
- Genome-wide association studies (GWAS)
- A/B testing with multiple metrics

**Don't need corrections when:**
- Single pre-specified hypothesis
- Confirmatory analysis of primary endpoint
- Tests are on independent datasets

---

**End of Lesson 04: Hypothesis Testing**

