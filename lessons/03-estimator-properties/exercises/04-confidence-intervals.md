# Exercises: Confidence Intervals

## Exercise 1: Derive t-Interval Using Pivotal Method

**Problem:** Derive the t-interval for Normal mean with unknown variance using the pivotal method.

**Solution:**

### Setup

Let $X_1, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2)$ i.i.d., where **both $\mu$ and $\sigma^2$ are unknown**.

We want a $(1-\alpha)$ confidence interval for $\mu$.

### Step 1: Find a Pivot

A pivot is a function of the data and parameter that has a known distribution.

**Sample mean:**
$$\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)$$

**Sample variance:**
$$s^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$$

**Key facts:**
1. $\bar{X}$ and $s^2$ are independent
2. $\frac{(n-1)s^2}{\sigma^2} \sim \chi^2_{n-1}$
3. $\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim \mathcal{N}(0,1)$

### Step 2: Construct the Pivot

Since $\sigma$ is unknown, we standardize using $s$ instead:

$$T = \frac{\bar{X} - \mu}{s/\sqrt{n}} = \frac{\mathcal{N}(0,1)}{\sqrt{\chi^2_{n-1}/(n-1)}}$$

By definition of the t-distribution:
$$T \sim t_{n-1}$$

This is our **pivot** - it depends on $\mu$ but has a known distribution.

### Step 3: Find Critical Values

For confidence level $(1-\alpha)$, find $t_{\alpha/2, n-1}$ such that:
$$P\left(-t_{\alpha/2, n-1} \leq T \leq t_{\alpha/2, n-1}\right) = 1 - \alpha$$

### Step 4: Solve for μ

$$P\left(-t_{\alpha/2, n-1} \leq \frac{\bar{X} - \mu}{s/\sqrt{n}} \leq t_{\alpha/2, n-1}\right) = 1 - \alpha$$

Multiply through by $s/\sqrt{n}$:
$$P\left(-t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}} \leq \bar{X} - \mu \leq t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}\right) = 1 - \alpha$$

Subtract $\bar{X}$ and multiply by -1 (reversing inequalities):
$$P\left(\bar{X} - t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}} \leq \mu \leq \bar{X} + t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}\right) = 1 - \alpha$$

### Final Result: t-Interval

$$\boxed{\mu \in \left[\bar{X} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}\right]}$$

The simulation that implements this interval (including a reusable helper) now lives in `../notebooks/03-asymptotic-efficiency-crlb.ipynb` under the “Confidence Interval Simulations” section.

### Why t Instead of z?

When $\sigma$ is unknown:
- Using z-interval: $\bar{X} \pm z_{\alpha/2} \cdot \frac{s}{\sqrt{n}}$ **undercovers** (too narrow)
- Using t-interval: $\bar{X} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$ has **correct coverage**

The t-distribution has **heavier tails** than normal, accounting for estimation uncertainty in $\sigma$.

---

## Exercise 2: Wilson vs Wald Interval for p=0.5

**Problem:** Show that the Wilson interval for p=0.5 and large n reduces to the Wald interval.

**Solution:**

### Wald Interval (Standard Normal Approximation)

$$\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

For $\hat{p} = 0.5$:
$$0.5 \pm z_{\alpha/2} \sqrt{\frac{0.25}{n}} = 0.5 \pm \frac{z_{\alpha/2}}{2\sqrt{n}}$$

### Wilson Score Interval

The Wilson interval inverts the test statistic. For proportion $p$ with $X$ successes in $n$ trials:

$$\tilde{p} \pm z_{\alpha/2} \sqrt{\frac{\tilde{p}(1-\tilde{p})}{n + z_{\alpha/2}^2}}$$

where:
$$\tilde{p} = \frac{X + z_{\alpha/2}^2/2}{n + z_{\alpha/2}^2}$$

### Approximation for Large n

For large $n$, $z_{\alpha/2}^2/n \to 0$, so:
$$\tilde{p} = \frac{X + z_{\alpha/2}^2/2}{n + z_{\alpha/2}^2} \approx \frac{X}{n} = \hat{p}$$

The denominator in the SE:
$$n + z_{\alpha/2}^2 \approx n \quad \text{(for large } n)$$

Therefore:
$$\text{Wilson SE} = \sqrt{\frac{\tilde{p}(1-\tilde{p})}{n + z_{\alpha/2}^2}} \approx \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} = \text{Wald SE}$$

### Special Case: p=0.5

When $\hat{p} = 0.5$ (meaning $X = n/2$):

$$\tilde{p} = \frac{n/2 + z_{\alpha/2}^2/2}{n + z_{\alpha/2}^2} = \frac{n + z_{\alpha/2}^2}{2(n + z_{\alpha/2}^2)} = 0.5$$

So $\tilde{p} = \hat{p} = 0.5$ exactly, and:

$$\text{Wilson CI} = 0.5 \pm z_{\alpha/2} \sqrt{\frac{0.25}{n + z_{\alpha/2}^2}}$$

For large $n$:
$$\sqrt{\frac{0.25}{n + z_{\alpha/2}^2}} \approx \sqrt{\frac{0.25}{n}} = \frac{0.5}{\sqrt{n}}$$

Therefore:
$$\text{Wilson CI} \approx 0.5 \pm \frac{z_{\alpha/2}}{2\sqrt{n}} = \text{Wald CI}$$

Numerical comparisons for a grid of sample sizes (showing convergence of Wilson and Wald intervals at $p=0.5$) have been moved to the notebook’s confidence-interval section.

---

## Exercise 3: Compare Proportion CIs with A/B Test Data

**Problem:** Use A/B testing data from `shared/data/ab_test_clicks.csv` to compute and compare proportion CIs.

**Key Findings:**

1. **Wald interval:** Simplest but can go outside [0,1] and has poor coverage for extreme p
2. **Wilson interval:** Better coverage, especially for small n or extreme p
3. **Agresti-Coull:** Similar to Wilson, adds pseudo-counts
4. **Clopper-Pearson:** Exact coverage but conservative (too wide)

**Recommendation:** Use **Wilson** or **Agresti-Coull** for most applications. Full computations (including plots and the difference-in-proportions interval) are in the notebook’s confidence-interval section.

---

## Exercise 4: Delta Method for Coefficient of Variation

**Problem:** Apply the delta method to construct a CI for the coefficient of variation $\sigma/\mu$.

**Solution:**

### Coefficient of Variation

The coefficient of variation (CV) measures relative variability:
$$CV = \frac{\sigma}{\mu}$$

### Setup

Given $X_1, \ldots, X_n \sim F$ with mean $\mu$ and variance $\sigma^2$, we have estimators:
- $\bar{X} \xrightarrow{\mathsf{P}} \mu$
- $s^2 \xrightarrow{\mathsf{P}} \sigma^2$

### Asymptotic Distribution

By CLT:
$$\sqrt{n}(\bar{X} - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$

$$\sqrt{n}(s^2 - \sigma^2) \xrightarrow{d} \mathcal{N}(0, \tau^2)$$

where $\tau^2 = \mathbb{E}[(X - \mu)^4] - \sigma^4$.

### Delta Method

For $g(\mu, \sigma) = \sigma/\mu$, we need:

$$\nabla g = \left(\frac{\partial g}{\partial \mu}, \frac{\partial g}{\partial \sigma}\right) = \left(-\frac{\sigma}{\mu^2}, \frac{1}{\mu}\right)$$

Evaluated at $(\mu, \sigma)$:
$$\nabla g(\mu, \sigma) = \left(-\frac{CV}{\mu}, \frac{1}{\mu}\right)$$

### Asymptotic Variance

Using the delta method:
$$\text{Var}(\widehat{CV}) \approx \left(\frac{\partial g}{\partial \mu}\right)^2 \text{Var}(\bar{X}) + \left(\frac{\partial g}{\partial \sigma}\right)^2 \text{Var}(s)$$

Assuming independence:
$$\text{Var}(\widehat{CV}) \approx \frac{CV^2}{\mu^2} \cdot \frac{\sigma^2}{n} + \frac{1}{\mu^2} \cdot \frac{\sigma^2}{2n}$$

$$= \frac{1}{n}\left(CV^2 \cdot CV^2 + \frac{CV^2}{2}\right) = \frac{CV^2}{n}\left(CV^2 + \frac{1}{2}\right)$$

### Confidence Interval

$$\widehat{CV} \pm z_{\alpha/2} \cdot \sqrt{\frac{\widehat{CV}^2}{n}\left(\widehat{CV}^2 + \frac{1}{2}\right)}$$

where $\widehat{CV} = s/\bar{X}$.

The delta-method and bootstrap implementation (with visualisations) is now executed in the notebook; this section retains the theoretical derivation for reference. **Conclusion:** the delta-method CI gives a quick analytical approximation, while the bootstrap offers an empirical check on its accuracy.
