# Exercises: Consistency

## Exercise 1: Continuous Mapping Theorem

**Problem:** Prove that if $\hat{\theta}_n \xrightarrow{\mathsf{P}} \theta$ and $g$ is continuous, then $g(\hat{\theta}_n) \xrightarrow{\mathsf{P}} g(\theta)$.

**Solution:**

We need to show that for any $\epsilon > 0$:
$$\lim_{n \to \infty} P(|g(\hat{\theta}_n) - g(\theta)| > \epsilon) = 0$$

**Proof:**

Since $g$ is continuous at $\theta$, for any $\epsilon > 0$, there exists $\delta > 0$ such that:
$$|\hat{\theta}_n - \theta| < \delta \implies |g(\hat{\theta}_n) - g(\theta)| < \epsilon$$

This means:
$$\{|g(\hat{\theta}_n) - g(\theta)| > \epsilon\} \subseteq \{|\hat{\theta}_n - \theta| \geq \delta\}$$

Taking probabilities:
$$P(|g(\hat{\theta}_n) - g(\theta)| > \epsilon) \leq P(|\hat{\theta}_n - \theta| \geq \delta)$$

Since $\hat{\theta}_n \xrightarrow{\mathsf{P}} \theta$, we have:
$$\lim_{n \to \infty} P(|\hat{\theta}_n - \theta| \geq \delta) = 0$$

Therefore:
$$\lim_{n \to \infty} P(|g(\hat{\theta}_n) - g(\theta)| > \epsilon) = 0$$

**Conclusion:** This is the **Continuous Mapping Theorem** - continuity preserves convergence in probability.

**Examples:**
- If $\bar{X}_n \xrightarrow{\mathsf{P}} \mu$, then $\bar{X}_n^2 \xrightarrow{\mathsf{P}} \mu^2$ (since $g(x) = x^2$ is continuous)
- If $\hat{p}_n \xrightarrow{\mathsf{P}} p$, then $\sqrt{\hat{p}_n} \xrightarrow{\mathsf{P}} \sqrt{p}$ (for $p > 0$)

---

## Exercise 2: Sample Median Consistency

**Problem:** Show that the sample median is consistent for the population median under mild conditions.

**Solution:**

Let $X_1, \ldots, X_n$ be i.i.d. with continuous CDF $F$ and population median $m$ (i.e., $F(m) = 0.5$).

Let $M_n$ denote the sample median. We need to show $M_n \xrightarrow{\mathsf{P}} m$.

**Key Steps:**

1. **Order Statistics:** Let $X_{(k)}$ denote the $k$-th order statistic. For odd $n = 2k+1$:
   $$M_n = X_{(k+1)}$$

2. **Indicator Function:** Define $I_i = \mathbb{1}\{X_i \leq x\}$. Then:
   $$\sum_{i=1}^n I_i = \text{number of observations} \leq x$$

3. **Empirical CDF:** The empirical CDF is:
   $$F_n(x) = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{X_i \leq x\}$$

4. **Empirical CDF convergence:** For any fixed $x$, define $I_i(x) = \mathbb{1}\{X_i \leq x\}$. Each $I_i(x)$ is a Bernoulli random variable with
   $$\mathbb{E}[I_i(x)] = P(X_i \leq x) = F(x).$$
   The empirical CDF is just their average:
   $$F_n(x) = \frac{1}{n}\sum_{i=1}^n I_i(x).$$
   By the Strong Law of Large Numbers,
   $$F_n(x) = \frac{1}{n}\sum_{i=1}^n I_i(x) \xrightarrow{\text{a.s.}} \mathbb{E}[I_1(x)] = F(x) \quad \text{for every fixed } x.$$
   Monotonicity of $F_n$ lets us extend this pointwise convergence to any interval: once $F_n(m-\epsilon)$ is below $0.5$ and $F_n(m+\epsilon)$ is above $0.5$, the median must lie between them.

5. **Consistency of Median:** For any $\epsilon > 0$:
   - $F(m - \epsilon) < 0.5 < F(m + \epsilon)$ (by continuity and definition of median)
   - Eventually, $F_n(m - \epsilon) < 0.5 < F_n(m + \epsilon)$ (by convergence)
   - Therefore, the sample median $M_n$ must lie in $(m - \epsilon, m + \epsilon)$

**Formal Argument:**

For any $\epsilon > 0$:
$$P(|M_n - m| > \epsilon) \leq P(F_n(m - \epsilon) \geq 0.5) + P(F_n(m + \epsilon) \leq 0.5)$$

As $n \to \infty$:
- $F_n(m - \epsilon) \to F(m - \epsilon) < 0.5$, so first term $\to 0$
- $F_n(m + \epsilon) \to F(m + \epsilon) > 0.5$, so second term $\to 0$

Therefore: $M_n \xrightarrow{\mathsf{P}} m$.

**Conditions Required:**
- Continuity of $F$ at $m$ (ensures unique median)
- i.i.d. samples (so the strong law applies to the indicator averages)

**Asymptotic Distribution:** Under additional regularity:
$$\sqrt{n}(M_n - m) \xrightarrow{d} \mathcal{N}\left(0, \frac{1}{4f(m)^2}\right)$$
where $f(m)$ is the PDF at the median.

---

## Exercise 3: Uniform Maximum Consistency Verification

**Problem:** Use the `uniform_max_consistency()` function to verify that $\max\{X_i\}$ is consistent for $\theta$ in Uniform$[0,\theta]$.

**Solution:** Theory shows $\hat{\theta}_n = \max\{X_i\}$ has bias and variance that both vanish as $n \to \infty$, so it converges in probability to $\theta$. The full Monte Carlo verification, plots, and discussion appear in the notebook section `../notebooks/02-consistency.ipynb` under “3. Uniform Maximum Estimator (Biased but Consistent).”

---

## Exercise 4: Why X₁ is Inconsistent

**Problem:** Explain why $X_1$ is inconsistent for $\mu$ while $\bar{X}_n$ is consistent.

**Solution:**

Let $X_1, X_2, \ldots$ be i.i.d. with mean $\mu$ and variance $\sigma^2 > 0$.

### Why X₁ is Inconsistent:

For any fixed sample, $X_1$ is just a single random observation. No matter how large $n$ becomes, $X_1$ doesn't change—it's always the same random variable.

**Formally:**
$$P(|X_1 - \mu| > \epsilon) = P(|X_1 - \mu| > \epsilon) \text{ for all } n$$

This probability is **constant** (doesn't depend on $n$) and generally **non-zero**. For example:
- If $X_1 \sim \mathcal{N}(\mu, \sigma^2)$, then:
  $$P(|X_1 - \mu| > \epsilon) = 2\Phi\left(-\frac{\epsilon}{\sigma}\right) > 0$$

**Intuition:** $X_1$ is a single random draw—it has inherent randomness that doesn't decrease with sample size because we never use the additional data!

### Why X̄ₙ is Consistent:

The sample mean $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$ **uses all available data** and benefits from averaging.

**By the Weak Law of Large Numbers:**
$$\bar{X}_n \xrightarrow{\mathsf{P}} \mu$$

**Formally:** For any $\epsilon > 0$:
$$P(|\bar{X}_n - \mu| > \epsilon) \to 0 \text{ as } n \to \infty$$

**Why it works:**
1. **Unbiasedness:** $\mathbb{E}[\bar{X}_n] = \mu$ (no systematic error)
2. **Variance decreases:** $\text{Var}(\bar{X}_n) = \sigma^2/n \to 0$
3. **By Chebyshev's inequality:**
   $$P(|\bar{X}_n - \mu| > \epsilon) \leq \frac{\text{Var}(\bar{X}_n)}{\epsilon^2} = \frac{\sigma^2}{n\epsilon^2} \to 0$$

Numerical illustrations of these probabilities are provided in the notebook section `../notebooks/02-consistency.ipynb` under “4. Inconsistent Estimator: First Observation.”

### Key Insights:

| Property | X₁ | X̄ₙ |
|----------|----|----|
| **Uses data** | Only first observation | All n observations |
| **Variance** | σ² (constant) | σ²/n → 0 |
| **P(\|Est - μ\| > ε)** | Constant > 0 | → 0 as n → ∞ |
| **Consistent?** | ❌ No | ✅ Yes |

**Conclusion:** Consistency requires that estimation error vanishes as more data is collected. $X_1$ ignores additional data, while $\bar{X}_n$ effectively uses all information through averaging.
