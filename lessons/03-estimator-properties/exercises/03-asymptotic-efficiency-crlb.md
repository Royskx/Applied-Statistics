# Exercises: Asymptotic Efficiency and CRLB

## Exercise 1: Fisher Information for Bernoulli

**Problem:** Compute the Fisher information for Bernoulli$(p)$ and derive the CRLB for unbiased estimators of $p$.

**Solution:**

### Step 1: Setup

For $X \sim \text{Bernoulli}(p)$, the PMF is:
$$f(x; p) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$$

### Step 2: Log-likelihood

$$\log f(x; p) = x \log p + (1-x) \log(1-p)$$

### Step 3: Score function

The score function is:
$$U(p) = \frac{\partial}{\partial p} \log f(X; p) = \frac{X}{p} - \frac{1-X}{1-p}$$

Simplify:
$$U(p) = \frac{X(1-p) - p(1-X)}{p(1-p)} = \frac{X - p}{p(1-p)}$$

### Step 4: Fisher Information (Method 1 - Variance)

$$I(p) = \text{Var}(U(p)) = \mathbb{E}\left[\left(\frac{X - p}{p(1-p)}\right)^2\right]$$

Since $\mathbb{E}[X] = p$ and $\text{Var}(X) = p(1-p)$:
$$I(p) = \frac{\mathbb{E}[(X - p)^2]}{[p(1-p)]^2} = \frac{p(1-p)}{[p(1-p)]^2} = \frac{1}{p(1-p)}$$

### Step 5: Fisher Information (Method 2 - Second Derivative)

$$\frac{\partial^2}{\partial p^2} \log f(X; p) = -\frac{X}{p^2} - \frac{1-X}{(1-p)^2}$$

Taking expectation:
$$I(p) = -\mathbb{E}\left[\frac{\partial^2}{\partial p^2} \log f(X; p)\right] = \frac{p}{p^2} + \frac{1-p}{(1-p)^2} = \frac{1}{p} + \frac{1}{1-p} = \frac{1}{p(1-p)}$$

### Step 6: CRLB for n observations

For $n$ i.i.d. Bernoulli observations:
$$I_n(p) = n \cdot I(p) = \frac{n}{p(1-p)}$$

### Step 7: CRLB

For any unbiased estimator $\hat{p}$ of $p$:
$$\text{Var}(\hat{p}) \geq \frac{1}{I_n(p)} = \frac{p(1-p)}{n}$$

### Verification: Sample Proportion

The sample proportion $\hat{p} = \bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ is unbiased:
$$\mathbb{E}[\hat{p}] = p$$

Its variance is:
$$\text{Var}(\hat{p}) = \frac{p(1-p)}{n}$$

**Conclusion:** The sample proportion **achieves the CRLB** and is therefore asymptotically efficient!

### Interpretation:
- Information increases linearly with $n$
- Maximum information occurs when $p = 0.5$ (most uncertain)
- Minimum information at extremes ($p \approx 0$ or $p \approx 1$)

---

## Exercise 2: Sample Mean Achieves CRLB for Normal Distribution

**Problem:** Show that the sample mean achieves the CRLB for Normal$(\mu, \sigma^2)$ with $\sigma$ known.

**Solution:**

### Setup

Let $X_1, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2)$ i.i.d., where $\sigma^2$ is known.

PDF: 
$$f(x; \mu) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

### Step 1: Log-likelihood

$$\log f(x; \mu) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$$

### Step 2: Score function

$$U(\mu) = \frac{\partial}{\partial \mu} \log f(X; \mu) = \frac{X - \mu}{\sigma^2}$$

### Step 3: Fisher Information

**Method 1 (Variance of score):**
$$I(\mu) = \text{Var}(U(\mu)) = \text{Var}\left(\frac{X - \mu}{\sigma^2}\right) = \frac{\text{Var}(X)}{\sigma^4} = \frac{\sigma^2}{\sigma^4} = \frac{1}{\sigma^2}$$

**Method 2 (Expected second derivative):**
$$\frac{\partial^2}{\partial \mu^2} \log f(X; \mu) = -\frac{1}{\sigma^2}$$

Therefore:
$$I(\mu) = -\mathbb{E}\left[-\frac{1}{\sigma^2}\right] = \frac{1}{\sigma^2}$$

### Step 4: Fisher Information for n observations

$$I_n(\mu) = n \cdot I(\mu) = \frac{n}{\sigma^2}$$

### Step 5: CRLB

For any unbiased estimator $\hat{\mu}$:
$$\text{Var}(\hat{\mu}) \geq \frac{1}{I_n(\mu)} = \frac{\sigma^2}{n}$$

### Step 6: Verify Sample Mean Achieves CRLB

The sample mean is:
$$\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$$

**Unbiasedness:**
$$\mathbb{E}[\bar{X}] = \mu$$

**Variance:**
$$\text{Var}(\bar{X}) = \text{Var}\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}$$

**Comparison:**
$$\text{Var}(\bar{X}) = \frac{\sigma^2}{n} = \frac{1}{I_n(\mu)} = \text{CRLB}$$

**Conclusion:** The sample mean **achieves the CRLB exactly** for all $n$, making it a uniformly minimum variance unbiased estimator (UMVUE)!

### Additional Notes:

1. **Efficiency:** $\bar{X}$ has efficiency = 1 (100% efficient)
2. **Also the MLE:** $\bar{X}$ is also the MLE for $\mu$
3. **Optimality:** No other unbiased estimator can have smaller variance
4. **Rao-Blackwell:** Any unbiased estimator can be improved by conditioning on $\bar{X}$ (complete sufficient statistic)

---

## Exercise 3: Fisher Information for Poisson

**Problem:** Use the `fisher_info_poisson()` function to compute information for $\lambda = 2, 5, 10$.

**Solution:** The full implementation—covering Fisher information calculations, simulation checks, CRLB visualisations, and MLE efficiency diagnostics for $\lambda = 2, 5, 10$—is provided in the notebook `../notebooks/03-asymptotic-efficiency-crlb.ipynb` (see Section "3. Poisson Rate: MLE Efficiency"). Run the notebook to reproduce the results and plots.


---

## Exercise 4: CRLB Achievement for Uniform Distribution

**Problem:** Explain why the MLE for Uniform$[0, \theta]$ achieves the CRLB while other estimators may not.

**Solution:**

### The MLE: Maximum Order Statistic

For $X_1, \ldots, X_n \sim \text{Uniform}[0, \theta]$ i.i.d., the MLE is:
$$\hat{\theta}_{\text{MLE}} = \max\{X_1, \ldots, X_n\} = X_{(n)}$$

### Why It's Interesting

The Uniform distribution is **non-regular** - it doesn't satisfy standard CRLB regularity conditions because:
1. Support depends on parameter: $[0, \theta]$
2. PDF discontinuous in $\theta$
3. Standard Fisher information undefined

### Distribution of MLE

The CDF of $\hat{\theta}_{\text{MLE}}$ is:
$$F_{\hat{\theta}}(t) = P(\max\{X_i\} \leq t) = P(X_1 \leq t, \ldots, X_n \leq t) = \left(\frac{t}{\theta}\right)^n$$

The PDF is:
$$f_{\hat{\theta}}(t) = \frac{n t^{n-1}}{\theta^n}, \quad 0 \leq t \leq \theta$$

### Properties

**Mean:**
$$\mathbb{E}[\hat{\theta}_{\text{MLE}}] = \int_0^\theta t \cdot \frac{n t^{n-1}}{\theta^n} dt = \frac{n}{\theta^n} \cdot \frac{\theta^{n+1}}{n+1} = \frac{n}{n+1}\theta$$

**Bias:**
$$\text{Bias} = \frac{n}{n+1}\theta - \theta = -\frac{\theta}{n+1}$$

**Variance:**
$$\mathbb{E}[\hat{\theta}^2] = \int_0^\theta t^2 \cdot \frac{n t^{n-1}}{\theta^n} dt = \frac{n\theta^2}{n+2}$$

$$\text{Var}(\hat{\theta}) = \frac{n\theta^2}{n+2} - \left(\frac{n\theta}{n+1}\right)^2 = \frac{n\theta^2}{(n+1)^2(n+2)}$$

### MSE

$$\text{MSE}(\hat{\theta}_{\text{MLE}}) = \text{Var} + \text{Bias}^2 = \frac{n\theta^2}{(n+1)^2(n+2)} + \frac{\theta^2}{(n+1)^2} = \frac{2\theta^2}{(n+1)(n+2)}$$

### "Modified" CRLB for Non-Regular Cases

For the Uniform case, there's a modified information measure:
$$I(\theta) = \frac{n}{\theta^2}$$

The "CRLB" is:
$$\text{Var}(\text{unbiased estimator}) \geq \frac{\theta^2}{n}$$

### Unbiased Estimator

To make MLE unbiased, use:
$$\hat{\theta}_{\text{unbiased}} = \frac{n+1}{n} \cdot X_{(n)}$$

Its variance is:
$$\text{Var}(\hat{\theta}_{\text{unbiased}}) = \left(\frac{n+1}{n}\right)^2 \text{Var}(X_{(n)}) = \frac{(n+1)^2}{n^2} \cdot \frac{n\theta^2}{(n+1)^2(n+2)} = \frac{\theta^2}{n(n+2)}$$

### Comparison with CRLB

Standard CRLB: $\theta^2/n$

Unbiased estimator variance: $\theta^2/(n(n+2))$

$$\frac{\theta^2}{n(n+2)} < \frac{\theta^2}{n}$$

**The unbiased estimator actually beats the standard CRLB!**

### Why This Happens

1. **Non-regular problem:** Support depends on $\theta$
2. **Super-efficiency possible:** Can beat naive CRLB in non-regular cases
3. **Information definition:** Standard Fisher information doesn't apply
4. **Order statistics:** Contain more information than score function captures

### Key Insights

| Estimator | Bias | Variance | MSE |
|-----------|------|----------|-----|
| MLE: $X_{(n)}$ | $-\theta/(n+1)$ | $\frac{n\theta^2}{(n+1)^2(n+2)}$ | $\frac{2\theta^2}{(n+1)(n+2)}$ |
| Unbiased: $\frac{n+1}{n}X_{(n)}$ | 0 | $\frac{\theta^2}{n(n+2)}$ | $\frac{\theta^2}{n(n+2)}$ |

**Conclusion:** 
- MLE is biased but has smallest MSE for this problem
- Unbiased version achieves "better than CRLB" performance
- Non-regular cases require careful analysis beyond standard CRLB theory
- Order statistics are remarkably efficient for bounded distributions
