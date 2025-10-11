# Lesson 3: Properties of Estimators

**Estimated time:** 3–4 hours self-study

This lesson builds upon:
- **Lesson 1**: Statistical Modeling — random variables, distributions, LLN, CLT
- **Lesson 2**: Parameter Estimation — MLE, MoM, Fisher information

## Learning Objectives

By the end of this lesson, you will be able to:

1. **Evaluate estimator quality** using bias, variance, consistency, and efficiency
2. **Understand the bias-variance tradeoff** and its implications for mean squared error (MSE)
3. **Assess consistency** of estimators and verify conditions for convergence
4. **Understand asymptotic efficiency** and the Cramér-Rao Lower Bound (CRLB)
5. **Construct confidence intervals** using multiple methods (Wald, t-distribution, Wilson, bootstrap)
6. **Compare CI methods** and understand their coverage properties
7. **Apply bootstrap methods** for inference on complex statistics

## Prerequisites

- Understanding of probability distributions and random variables (Lesson 1)
- Familiarity with point estimation methods (Lesson 2: MLE, MoM)
- Basic knowledge of the Central Limit Theorem and Law of Large Numbers
- Python programming with NumPy, SciPy, and data visualization libraries

---

## 1. Bias and Variance

### 1.1 Definitions

**Bias** measures the systematic error of an estimator:

$$\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta$$

- **Unbiased estimator**: $\text{Bias}(\hat{\theta}) = 0$, meaning $\mathbb{E}[\hat{\theta}] = \theta$
- **Biased estimator**: $\text{Bias}(\hat{\theta}) \neq 0$

**Variance** measures the spread or variability of an estimator:

$$\text{Var}(\hat{\theta}) = \mathbb{E}\left[(\hat{\theta} - \mathbb{E}[\hat{\theta}])^2\right]$$

**Mean Squared Error (MSE)** combines both bias and variance:

$$\text{MSE}(\hat{\theta}) = \mathbb{E}\left[(\hat{\theta} - \theta)^2\right] = \text{Bias}^2(\hat{\theta}) + \text{Var}(\hat{\theta})$$

### 1.2 The Bias-Variance Tradeoff

Key insight: **Minimizing MSE often requires balancing bias and variance**

- **Low bias, high variance**: Estimator centers around true value but spreads widely
- **High bias, low variance**: Estimator is precise but systematically off-target
- **Optimal MSE**: Often achieved with slight bias if variance reduction is substantial

**Example: Sample Variance Estimators**

For $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ i.i.d.:

1. **MLE (biased)**:
   $$\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2$$
   - Bias: $-\sigma^2/n$ (underestimates)
   - Variance: $\frac{2\sigma^4(n-1)}{n^2}$

2. **Unbiased estimator**:
   $$s^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$$
   - Bias: $0$
   - Variance: $\frac{2\sigma^4}{n-1}$

The MLE has lower MSE for small $n$ despite being biased!

### 1.3 Practical Examples

**Example 1: Estimating Population Mean**
- Sample mean $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$
- Bias: $0$ (unbiased)
- Variance: $\sigma^2/n$ (decreases with sample size)
- MSE: $\sigma^2/n$

**Example 2: Maximum of Uniform Distribution**
For $X_1, \ldots, X_n \sim \text{Uniform}(0, \theta)$:
- $\hat{\theta}_{\text{MLE}} = \max(X_1, \ldots, X_n)$
- Bias: $-\theta/(n+1)$ (underestimates)
- But: Variance decreases faster than bias squared
- Can correct: $\hat{\theta}_{\text{unbiased}} = \frac{n+1}{n}\max(X_i)$

---

## 2. Consistency

### 2.1 Definition

An estimator $\hat{\theta}_n$ is **consistent** for $\theta$ if:

$$\hat{\theta}_n \xrightarrow{p} \theta \quad \text{as } n \to \infty$$

This means: $\forall \epsilon > 0: \lim_{n \to \infty} P(|\hat{\theta}_n - \theta| > \epsilon) = 0$

**Intuition**: As we collect more data, the estimator gets arbitrarily close to the true parameter value.

### 2.2 Sufficient Conditions for Consistency

**Theorem**: If $\hat{\theta}_n$ is an estimator such that:
1. $\lim_{n \to \infty} \text{Bias}(\hat{\theta}_n) = 0$ (asymptotically unbiased)
2. $\lim_{n \to \infty} \text{Var}(\hat{\theta}_n) = 0$ (variance goes to zero)

Then $\hat{\theta}_n$ is consistent.

**Proof sketch**: By Chebyshev's inequality and MSE decomposition.

### 2.3 Examples

**Example 1: Sample Mean**
For i.i.d. $X_i$ with finite variance:
- $\mathbb{E}[\bar{X}_n] = \mu$ (unbiased)
- $\text{Var}(\bar{X}_n) = \sigma^2/n \to 0$
- Therefore: $\bar{X}_n \xrightarrow{p} \mu$ (consistent)

This is essentially the **Weak Law of Large Numbers (WLLN)**.

**Example 2: Sample Variance**
Both $\hat{\sigma}^2_{\text{MLE}}$ and $s^2$ are consistent:
- MLE: $\mathbb{E}[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2 \to \sigma^2$
- Unbiased: $\mathbb{E}[s^2] = \sigma^2$ always
- Both variances: $\to 0$ as $n \to \infty$

**Example 3: Maximum of Uniform(0, θ)**
$\hat{\theta}_n = \max(X_1, \ldots, X_n)$ is consistent:
- $\mathbb{E}[\hat{\theta}_n] = \frac{n}{n+1}\theta \to \theta$
- $\text{Var}(\hat{\theta}_n) = \frac{n\theta^2}{(n+1)^2(n+2)} \to 0$

**Example 4: Inconsistent Estimator (First Observation)**
For any i.i.d. sample $X_1, X_2, \ldots, X_n$, consider using only the first observation:
- $\hat{\theta}_n = X_1$ (ignores all other data!)
- $\mathbb{E}[X_1] = \theta$ (unbiased)
- $\text{Var}(X_1) = \sigma^2$ (constant, doesn't decrease with $n$!)
- **Not consistent**: variance doesn't vanish as $n \to \infty$

**Visual intuition**: Boxplots of $X_1$ across different sample sizes (n=10, 50, 100, 500, 1000) show:
- **Constant box size**: Interquartile range stays the same
- **Outliers** (black circles): Extreme values beyond whiskers, natural for skewed distributions like exponential
- **No convergence**: Distribution of $X_1$ remains identical regardless of total sample size
- This demonstrates that collecting more data doesn't help if we ignore it!

### 2.4 Consistency vs. Unbiasedness

**Important**: Consistency and unbiasedness are different properties!
- Unbiased but inconsistent: Possible (e.g., $X_1$ estimator - unbiased but variance stays constant)
- Biased but consistent: Common (e.g., MLE for variance - biased but converges)
- Both unbiased and consistent: Ideal (e.g., sample mean - best of both worlds)

---

## 3. Asymptotic Efficiency and the Cramér-Rao Lower Bound

### 3.1 The Cramér-Rao Lower Bound (CRLB)

Under regularity conditions, the variance of any unbiased estimator is bounded below:

$$\text{Var}(\hat{\theta}) \geq \frac{1}{n \mathcal{I}(\theta)}$$

where $\mathcal{I}(\theta)$ is the **Fisher Information**:

$$\mathcal{I}(\theta) = \mathbb{E}\left[\left(\frac{\partial}{\partial\theta}\log f(X; \theta)\right)^2\right] = -\mathbb{E}\left[\frac{\partial^2}{\partial\theta^2}\log f(X; \theta)\right]$$

**Key insights**:
- CRLB provides a theoretical lower bound on estimator variance
- Estimators achieving CRLB are called **efficient**
- MLEs are often asymptotically efficient (achieve CRLB as $n \to \infty$)

### 3.2 Efficiency

An unbiased estimator $\hat{\theta}$ is **efficient** if:

$$\text{Var}(\hat{\theta}) = \frac{1}{n \mathcal{I}(\theta)}$$

**Relative efficiency**: For two unbiased estimators $\hat{\theta}_1$ and $\hat{\theta}_2$:

$$\text{eff}(\hat{\theta}_1, \hat{\theta}_2) = \frac{\text{Var}(\hat{\theta}_2)}{\text{Var}(\hat{\theta}_1)}$$

### 3.3 Examples

**Example 1: Normal Mean**
For $X_i \sim N(\mu, \sigma^2)$ i.i.d.:
- Fisher information: $\mathcal{I}(\mu) = 1/\sigma^2$
- CRLB: $1/(n/\sigma^2) = \sigma^2/n$
- Sample mean variance: $\text{Var}(\bar{X}) = \sigma^2/n$
- **Sample mean achieves CRLB** → efficient!

**Example 2: Exponential Rate**
For $X_i \sim \text{Exp}(\lambda)$ i.i.d.:
- Fisher information: $\mathcal{I}(\lambda) = 1/\lambda^2$
- CRLB: $\lambda^2/n$
- MLE $\hat{\lambda}_{\text{MLE}} = 1/\bar{X}$
- Asymptotically achieves CRLB (asymptotically efficient)

### 3.4 Asymptotic Normality of MLEs

Under regularity conditions, as $n \to \infty$:

$$\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta) \xrightarrow{d} N\left(0, \frac{1}{\mathcal{I}(\theta)}\right)$$

This provides the foundation for constructing confidence intervals!

---

## 4. Confidence Intervals

### 4.1 Basic Concepts

A **$(1-\alpha)$ confidence interval** is a random interval $[L, U]$ such that:

$$P(\theta \in [L, U]) = 1 - \alpha$$

**Interpretation** (frequentist):
- If we repeat sampling many times and construct CI each time
- Approximately $(1-\alpha) \times 100\%$ of intervals contain the true $\theta$
- **NOT**: "Probability that $\theta$ is in this specific interval"

**Common confidence levels**:
- 90%: $\alpha = 0.10$, $z_{0.95} = 1.645$
- 95%: $\alpha = 0.05$, $z_{0.975} = 1.960$
- 99%: $\alpha = 0.01$, $z_{0.995} = 2.576$

### 4.2 Large-Sample CIs via Central Limit Theorem

If $\hat{\theta}_n$ satisfies:

$$\sqrt{n}(\hat{\theta}_n - \theta) \xrightarrow{d} N(0, \tau^2)$$

Then an approximate $(1-\alpha)$ CI is:

$$\hat{\theta}_n \pm z_{1-\alpha/2} \cdot \frac{\hat{\tau}}{\sqrt{n}}$$

This is called the **Wald confidence interval**.

**Example: Mean with known variance**
For $X_i \sim (\mu, \sigma^2)$ i.i.d. with known $\sigma^2$:

$$\bar{X} \pm z_{1-\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

### 4.3 CI for Mean with Unknown Variance (t-distribution)

For $X_i \sim N(\mu, \sigma^2)$ i.i.d. with **unknown** $\sigma^2$:

$$T = \frac{\bar{X} - \mu}{s/\sqrt{n}} \sim t_{n-1}$$

**t-based confidence interval**:

$$\bar{X} \pm t_{n-1, 1-\alpha/2} \cdot \frac{s}{\sqrt{n}}$$

where $s = \sqrt{\frac{1}{n-1}\sum_{i=1}^n(X_i - \bar{X})^2}$

**Key differences from z-interval**:
- Uses $t_{n-1}$ quantiles instead of $z$ (heavier tails for small $n$)
- Accounts for uncertainty in estimating $\sigma^2$
- Converges to z-interval as $n \to \infty$

### 4.4 CIs for Proportions

For Bernoulli data: $X_i \sim \text{Ber}(p)$ i.i.d., estimate $p$ with $\hat{p} = \bar{X}$

**Method 1: Wald Interval** (simple but problematic)

$$\hat{p} \pm z_{1-\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

**Problems**:
- Poor coverage when $p$ near 0 or 1
- Can produce invalid intervals $[<0, >1]$
- Actual coverage often far from nominal $(1-\alpha)$

**Method 2: Wilson Score Interval** (recommended)

$$\frac{\hat{p} + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \pm \frac{z}{1 + \frac{z^2}{n}} \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}$$

where $z = z_{1-\alpha/2}$

**Advantages**:
- Better coverage properties, especially for small $n$ or extreme $p$
- Always produces valid intervals $[0,1]$
- Recommended by statisticians for practical use

**Method 3: Agresti-Coull Interval** (simple approximation to Wilson)

Add 2 successes and 2 failures:
$$\tilde{p} = \frac{X + 2}{n + 4}$$

Then use Wald formula:
$$\tilde{p} \pm z_{1-\alpha/2} \sqrt{\frac{\tilde{p}(1-\tilde{p})}{n+4}}$$

**Method 4: Clopper-Pearson (Exact)**
Based on binomial distribution, guaranteed coverage $\geq 1-\alpha$ but conservative.

### 4.5 Comparing CI Methods

**Coverage probability**: Actual $P(\theta \in \text{CI})$ vs. nominal $1-\alpha$

Simulation studies show:
- **Wald**: Poor coverage for proportions (can be far below nominal)
- **Wilson**: Near-nominal coverage across wide range of $p$ and $n$
- **t-interval**: Exact for normal data, robust for near-normal data
- **Bootstrap**: Good for complex statistics, requires sufficient sample size

---

## 5. Bootstrap Methods

### 5.1 The Bootstrap Principle

**Key idea**: Use the sample as a proxy for the population

**Bootstrap procedure**:
1. Draw $B$ bootstrap samples (with replacement) from original data
2. Compute statistic $\hat{\theta}^{*}_{b}$ for each bootstrap sample
3. Use distribution of $\{\hat{\theta}^{*}_{1}, \ldots, \hat{\theta}^{*}_{B}\}$ to:
   - Estimate standard error
   - Construct confidence intervals
   - Assess sampling variability

### 5.2 Bootstrap Standard Error

Estimate the standard error of $\hat{\theta}$ using bootstrap:

$$\widehat{\text{SE}}(\hat{\theta}) = \sqrt{\frac{1}{B-1}\sum_{b=1}^{B} (\hat{\theta}^{*}_{b} - \bar{\theta}^{*})^2}$$

where $\bar{\theta}^{*} = \frac{1}{B}\sum_{b=1}^{B} \hat{\theta}^{*}_{b}$

### 5.3 Bootstrap Confidence Intervals

**Method 1: Percentile Bootstrap CI**

Simply use quantiles of bootstrap distribution:

$$\text{CI}_{1-\alpha} = [q_{\alpha/2}, q_{1-\alpha/2}]$$

where $q_p$ is the $p$-th quantile of $\{\hat{\theta}^{*}_{1}, \ldots, \hat{\theta}^{*}_{B}\}$

**Method 2: Basic Bootstrap CI (Reflection Method)**

The Basic CI reflects the bootstrap quantiles around the observed statistic $\hat{\theta}$:

$$\text{CI}_{1-\alpha} = [2\hat{\theta} - q_{1-\alpha/2}, 2\hat{\theta} - q_{\alpha/2}]$$

**Intuition**: If the bootstrap distribution is centered at $\bar{\theta}^*$ but the true parameter is at $\theta$, we need to "reflect" the quantiles to account for this shift. The formula $2\hat{\theta} - q$ performs this reflection, centering the interval around our estimate rather than the bootstrap mean.

**Method 3: Bootstrap-t CI**

1. Compute $t^{*}_{b} = \frac{\hat{\theta}^{*}_{b} - \hat{\theta}}{\widehat{\text{SE}}(\hat{\theta}^{*}_{b})}$ for each bootstrap
2. Find quantiles $t_{\alpha/2}$ and $t_{1-\alpha/2}$ of $\{t^{*}_{1}, \ldots, t^{*}_{B}\}$
3. CI: $[\hat{\theta} - t_{1-\alpha/2} \cdot \widehat{\text{SE}}(\hat{\theta}), \hat{\theta} - t_{\alpha/2} \cdot \widehat{\text{SE}}(\hat{\theta})]$

**Method 4: BCa (Bias-Corrected and Accelerated)**

The most sophisticated bootstrap CI method, adjusting for both bias and skewness:

**Bias correction (z₀)**: Measures how much the bootstrap distribution is shifted from the observed statistic
- $z_0 = \Phi^{-1}\left(\frac{\#\{\hat{\theta}^{*}_{b} < \hat{\theta}\}}{B}\right)$
- When $z_0 \neq 0$, the bootstrap distribution is asymmetric around $\hat{\theta}$

**Acceleration (a)**: Captures the rate at which the standard error changes
- Computed using jackknife: repeatedly leaving out one observation
- Corrects for skewness in the statistic's sampling distribution

**Adjusted percentiles**: Instead of using $(\alpha/2, 1-\alpha/2)$ quantiles, BCa uses adjusted values that account for both bias and acceleration, often providing better coverage than simpler methods, especially for skewed statistics.

### 5.4 When to Use Bootstrap

**Advantages**:
- Works for complex statistics (median, correlation, etc.)
- No assumption about sampling distribution
- Can handle non-standard situations

**Limitations**:
- Requires sufficient sample size ($n \geq 30$ typically)
- Computationally intensive (need $B \geq 1000$ resamples)
- May fail for extreme order statistics (min, max)
- Assumes original sample is representative

**Example applications**:
- Median and other quantiles
- Ratio of means
- Correlation coefficients
- Standard errors for complex regression models
- Custom statistics without known distributions

### 5.5 Bootstrap Algorithm (Python)

```python
import numpy as np

def bootstrap_ci(data, statistic, n_bootstrap=10000, alpha=0.05, random_state=None):
    """
    Compute bootstrap confidence interval

    Parameters:
    -----------
    data : array-like
        Original sample data
    statistic : callable
        Function to compute statistic of interest
    n_bootstrap : int
        Number of bootstrap resamples
    alpha : float
        Significance level (e.g., 0.05 for 95% CI)
    random_state : int or None
        Random seed for reproducibility

    Returns:
    --------
    ci : tuple
        (lower, upper) confidence interval bounds
    """
    rng = np.random.default_rng(random_state)
    n = len(data)

    # Generate bootstrap samples
    bootstrap_stats = np.array([
        statistic(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    # Percentile method
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return (lower, upper)

# Example usage
data = np.array([1.2, 3.4, 2.1, 4.5, 3.2, 2.8, 3.9, 2.5])
ci_median = bootstrap_ci(data, np.median, n_bootstrap=10000, random_state=2025)
print(f"95% CI for median: [{ci_median[0]:.3f}, {ci_median[1]:.3f}]")
```

---

## 6. Practical Examples and Applications

### 6.1 Example: A/B Test for Click-Through Rates

**Scenario**: Testing two website designs
- Design A: 45 clicks out of 500 visitors ($\hat{p}_A = 0.09$)
- Design B: 68 clicks out of 500 visitors ($\hat{p}_B = 0.136$)

**Question**: Is the difference statistically significant?

**Approach 1: Separate CIs using Wilson method**
- Construct 95% CI for each proportion
- Check if intervals overlap

**Approach 2: CI for difference**
$$(\hat{p}_B - \hat{p}_A) \pm z_{0.975} \sqrt{\frac{\hat{p}_A(1-\hat{p}_A)}{n_A} + \frac{\hat{p}_B(1-\hat{p}_B)}{n_B}}$$

**Approach 3: Bootstrap**
- Resample from each group independently
- Compute difference for each bootstrap sample
- Use percentile CI

### 6.2 Example: Comparing Estimator Quality

**Scenario**: Estimate variance of normal distribution

Compare three estimators:
1. MLE: $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum(X_i - \bar{X})^2$
2. Unbiased: $s^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$
3. Known-mean: $\tilde{\sigma}^2 = \frac{1}{n}\sum(X_i - \mu)^2$ (if $\mu$ known)

**Metrics**:
- Bias
- Variance
- MSE
- Coverage probability of CIs

**Simulation approach**:
```python
def compare_variance_estimators(mu, sigma, n, n_sim=10000):
    rng = np.random.default_rng(2025)

    results = {'MLE': [], 'unbiased': [], 'known_mean': []}

    for _ in range(n_sim):
        X = rng.normal(mu, sigma, n)
        xbar = X.mean()

        # Three estimators
        results['MLE'].append(np.sum((X - xbar)**2) / n)
        results['unbiased'].append(np.sum((X - xbar)**2) / (n - 1))
        results['known_mean'].append(np.sum((X - mu)**2) / n)

    # Compute bias, variance, MSE for each
    true_var = sigma**2
    for name, estimates in results.items():
        est_array = np.array(estimates)
        bias = est_array.mean() - true_var
        variance = est_array.var()
        mse = bias**2 + variance
        print(f"{name}: Bias={bias:.6f}, Var={variance:.6f}, MSE={mse:.6f}")
```

### 6.3 Example: Bootstrap for Median Income

**Scenario**: Estimating median household income with skewed data

**Why bootstrap?**:
- Income data is typically right-skewed
- CLT-based CI may not be accurate for median
- Bootstrap doesn't assume normality

**Implementation**:
```python
import pandas as pd

# Load data
income_data = pd.read_csv("shared/data/heights_weights_sample.csv")  # Example dataset

# Compute bootstrap CI for median
median_ci = bootstrap_ci(income_data['height'].values, np.median,
                         n_bootstrap=10000, random_state=2025)

print(f"Sample median: {np.median(income_data['height']):.2f}")
print(f"95% Bootstrap CI: [{median_ci[0]:.2f}, {median_ci[1]:.2f}]")
```

---

## 7. Summary and Key Takeaways

### Core Concepts

1. **Bias-Variance Tradeoff**
   - MSE = Bias² + Variance
   - Optimal estimators may have slight bias to reduce variance
   - Context-dependent: small samples vs. asymptotic behavior

2. **Consistency**
   - Estimator converges to true value as $n \to \infty$
   - Sufficient conditions: asymptotic unbiasedness + vanishing variance
   - Different from unbiasedness (finite-sample property)

3. **Efficiency**
   - Cramér-Rao Lower Bound provides theoretical minimum variance
   - MLEs often asymptotically efficient
   - Relative efficiency compares estimators

4. **Confidence Intervals**
   - Quantify uncertainty in estimates
   - Multiple construction methods (Wald, t, Wilson, bootstrap)
   - Coverage probability is key quality metric

5. **Bootstrap**
   - Powerful resampling technique
   - Applicable to complex statistics
   - Requires adequate sample size

### Practical Guidelines

**Choosing an estimator**:
- Consider bias, variance, and MSE (not just one metric)
- Asymptotic properties matter for large samples
- Finite-sample behavior matters for small samples

**Constructing CIs**:
- t-distribution for normal data with unknown variance
- Wilson or Agresti-Coull for proportions (not Wald!)
- Bootstrap for non-standard statistics
- Always check coverage in simulations when possible

**Bootstrap best practices**:
- Use $B \geq 1000$ resamples (more is better)
- Set random seed for reproducibility
- Verify assumptions (i.i.d. sampling, sufficient $n$)
- Consider BCa for skewed distributions

### Connection to Other Lessons

- **Lesson 1**: CLT justifies large-sample CIs; WLLN underlies consistency
- **Lesson 2**: Fisher information from MLE theory appears in CRLB
- **Future lessons**: These concepts foundational for hypothesis testing and regression

---

## 8. Practice Exercises

The following exercises are provided in two formats:
- **Written exercises**: Detailed problems with solutions in `exercises/` directory
  - `01-bias-variance-tradeoff.md`
  - `02-consistency.md`
  - `03-asymptotic-efficiency-crlb.md`
  - `04-confidence-intervals.md`
  - `05-bootstrap.md`
- **Computational notebooks**: Interactive Jupyter notebooks in `notebooks/` directory
  - `01-05.ipynb` - Topic-specific interactive examples
  - `06-practical-lab.ipynb` - Comprehensive lab session (see Section 10)

### Conceptual Questions

These questions are provided with detailed solutions in the `exercises/` directory:

1. **Bias-Variance Decomposition**
   - Prove that $\text{MSE}(\hat{\theta}) = \text{Bias}^2(\hat{\theta}) + \text{Var}(\hat{\theta})$
   - Give an example where a biased estimator has lower MSE than an unbiased one

2. **Consistency**
   - Show that the sample median is a consistent estimator for the population median under mild conditions
   - Prove that if $\hat{\theta}_n \xrightarrow{p} \theta$ and $g$ is continuous, then $g(\hat{\theta}_n) \xrightarrow{p} g(\theta)$

3. **Fisher Information**
   - Compute the Fisher information for $X \sim \text{Poisson}(\lambda)$
   - Verify that the sample mean achieves the CRLB for estimating $\lambda$

4. **Confidence Intervals**
   - Explain why the Wald interval for proportions can have poor coverage
   - Describe scenarios where bootstrap CIs are preferable to Wald CIs

### Computational Exercises

These exercises are fully implemented in the interactive Jupyter notebooks `01-05` in the `notebooks/` directory, with complete solutions and visualizations.

5. **Bias-Variance Simulation**
   - Simulate $n=20$ samples from $N(5, 4)$ repeatedly (10,000 times)
   - Compare bias, variance, and MSE of $\hat{\sigma}^2_{\text{MLE}}$ vs. $s^2$
   - Plot histograms of both estimators with the true value marked

6. **Coverage Study**
   - For $p \in \{0.05, 0.1, 0.3, 0.5\}$ and $n \in \{20, 50, 100, 200\}$:
     * Simulate 5,000 datasets from $\text{Ber}(p)$
     * Compute 95% CIs using Wald, Wilson, and Agresti-Coull methods
     * Calculate empirical coverage probability for each method
   - Create a heatmap showing coverage by method, $n$, and $p$

7. **Bootstrap Application**
   - Load the heights data from `shared/data/heights_weights_sample.csv`
   - Compute 95% bootstrap CI for:
     * Median height
     * Interquartile range (IQR)
     * Correlation between height and weight
   - Compare percentile vs. BCa methods

8. **Consistency Verification**
   - For $X_i \sim \text{Uniform}(0, \theta)$, estimate $\theta$ using $\hat{\theta}_n = \max(X_1, \ldots, X_n)$
   - Generate samples with increasing $n = 10, 20, 50, 100, 500, 1000$
   - Plot $\hat{\theta}_n$ vs. $n$ for multiple simulation runs (use $\theta = 10$)
   - Demonstrate convergence visually

### Advanced Problems

These advanced problems are provided with step-by-step guidance in both the written exercises and computational notebooks.

9. **Cramér-Rao Lower Bound**
   - For $X_i \sim \text{Exp}(\lambda)$ i.i.d.:
     * Derive the Fisher information $\mathcal{I}(\lambda)$
     * Compute the CRLB for estimating $\lambda$
     * Show that $\hat{\lambda}_{\text{MLE}} = 1/\bar{X}$ is asymptotically efficient
     * Verify through simulation that $\text{Var}(\hat{\lambda}_{\text{MLE}}) \approx \text{CRLB}$ for large $n$

10. **Two-Sample CI Comparison**
    - Generate two groups: $X \sim N(100, 15^2)$ ($n_X=30$) and $Y \sim N(110, 20^2)$ ($n_Y=40$)
    - Construct 95% CIs for $\mu_Y - \mu_X$ using:
      * Pooled t-interval (assuming equal variances)
      * Welch t-interval (unequal variances)
      * Bootstrap percentile method
    - Repeat 10,000 times and compare:
      * Average width
      * Coverage probability
      * Which method is most appropriate and why?

11. **Estimator Comparison**
    - For $X_i \sim N(\mu, 1)$ with $\mu$ unknown:
      * Compare sample mean, sample median, and trimmed mean (trim 10%)
      * Assess bias, variance, MSE via simulation
      * How do results change with contamination (1% of data from $N(\mu+10, 1)$)?
      * When would you prefer each estimator?

12. **Bootstrap Diagnostics**
    - Implement bootstrap for correlation coefficient using A/B test data
    - Create diagnostic plots:
      * Bootstrap distribution histogram
      * Q-Q plot vs. normal distribution
      * Trace plot showing stability with increasing $B$
    - Discuss when normality-based methods might be adequate vs. bootstrap necessary

---

## 9. Resources and Further Reading

### Textbook References

1. **Casella, G., & Berger, R. L.** (2002). *Statistical Inference* (2nd ed.). Duxbury Press.
   - Chapter 7: Point Estimation
   - Chapter 9: Interval Estimation
   - Comprehensive treatment of CRLB, consistency, and efficiency

2. **Wasserman, L.** (2004). *All of Statistics: A Concise Course in Statistical Inference*. Springer.
   - Chapter 6: Models, Statistical Inference and Learning
   - Chapter 8: The Bootstrap
   - Modern, concise presentation with computational focus

3. **Efron, B., & Tibshirani, R. J.** (1994). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
   - Comprehensive coverage of bootstrap methods
   - BCa and other advanced techniques

### Online Resources

- [StatQuest: Bootstrap Confidence Intervals](https://www.youtube.com/user/joshstarmer) - Excellent visual explanations
- [Seeing Theory: Confidence Intervals](https://seeing-theory.brown.edu/frequentist-inference/) - Interactive visualizations
- [Wilson Score Interval](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval) - Detailed technical discussion

### Datasets Used

- `shared/data/heights_weights_sample.csv` - Height and weight measurements
- `shared/data/ab_test_clicks.csv` - A/B test click-through rates
- `shared/data/manufacturing_defects.csv` - Quality control data

### Python Libraries

- **NumPy**: Array operations and random number generation
- **SciPy**: Statistical distributions and tests (`scipy.stats`)
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Visualization
- **Statsmodels**: Advanced statistical models (optional)

---

## 10. Practical Lab Session

### Overview

The practical lab exercises are implemented in the Jupyter notebook:

**📓 `notebooks/06-practical-lab.ipynb`**

This comprehensive lab notebook includes four major tasks with complete, executable code:

### Lab Contents

**Task 1: Bias-Variance Tradeoff Investigation**
- Compare MLE vs. unbiased estimator for variance
- Visualize distributions and compute MSE components
- Explore tradeoff with different sample sizes
- **Output**: Interactive visualizations and numerical results

**Task 2: Confidence Interval Coverage Study**
- Compare Wald, Wilson, and Agresti-Coull intervals
- Test coverage across different proportions and sample sizes
- Generate heatmaps showing actual vs. nominal coverage
- **Output**: Coverage probability matrices for all methods

**Task 3: Bootstrap Confidence Intervals with Diagnostics**
- Apply bootstrap to median and IQR estimation
- Compare percentile, normal approximation, and basic bootstrap CIs
- Generate diagnostic plots (Q-Q plots, convergence analysis)
- **Output**: Multiple CI methods with visual diagnostics

**Task 4: Real Data Application - A/B Testing**
- Analyze actual A/B test data from `shared/data/`
- Compare Wald, bootstrap, and hypothesis testing approaches
- Make business recommendations based on statistical evidence
- **Output**: Statistical conclusions and visualizations

### How to Use the Lab Notebook

1. **Prerequisites**: Complete theoretical notebooks 01-05 first
2. **Open the notebook**: `jupyter notebook notebooks/06-practical-lab.ipynb`
3. **Run cells sequentially**: Each task builds on previous setup
4. **Experiment**: Modify parameters and try your own variations
5. **Estimated time**: 2-3 hours for complete lab

### Lab Tasks - Quick Reference

All lab tasks are implemented in **`notebooks/06-practical-lab.ipynb`** with complete code, visualizations, and detailed explanations. Below is a brief overview of each task:

**Task 1: Bias-Variance Tradeoff Investigation**
- Simulates 10,000 samples to compare MLE vs. unbiased variance estimators
- Produces side-by-side histograms and MSE decomposition bar charts
- Key finding: Unbiased estimator has zero bias but higher variance

**Task 2: Confidence Interval Coverage Study**
- Compares Wald, Wilson, and Agresti-Coull CI methods for proportions
- Tests across 4 true proportions × 4 sample sizes (5,000 simulations each)
- Generates heatmaps showing actual coverage vs. 95% nominal level
- Key finding: Wald fails for extreme proportions; Wilson and Agresti-Coull more robust

**Task 3: Bootstrap Confidence Intervals with Diagnostics**
- Applies bootstrap to median and IQR estimation using real height data
- Compares three CI construction methods (percentile, normal approx, basic)
- Includes diagnostic plots: Q-Q plot for normality, convergence analysis
- Key finding: Bootstrap distribution may not be perfectly normal; percentile method most reliable

**Task 4: Real Data Application - A/B Testing**
- Analyzes click-through rate differences using `shared/data/ab_test_clicks.csv`
- Compares Wald CI and bootstrap CI for difference in proportions
- Includes z-test for hypothesis testing and business recommendation
- Key finding: Statistical significance should be combined with practical significance for business decisions

---

## Summary

This comprehensive lesson material covers:

1. **Theoretical foundations**: Bias, variance, MSE, consistency, efficiency, CRLB
2. **Confidence intervals**: Multiple construction methods with practical comparisons
3. **Bootstrap methods**: Resampling techniques for complex statistics
4. **Practical applications**: Real datasets and implementation examples
5. **Extensive exercises**: Both conceptual and computational
6. **Lab session**: Hands-on implementation with visualization

The material is designed to:
- Build on Lessons 1 and 2
- Provide both theory and practice
- Include reproducible Python code
- Connect to real-world applications
- Prepare students for hypothesis testing (future lessons)
