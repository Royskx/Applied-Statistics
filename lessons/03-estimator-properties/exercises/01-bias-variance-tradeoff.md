# Exercises: Bias-Variance Tradeoff

## Exercise 1: Derive MLE Variance Estimator Bias

**Problem:** Derive the bias of the MLE variance estimator $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n (X_i - \bar{X})^2$ for Normal data.

**Solution:**

Given $X_1, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2)$ i.i.d., we need to find $\mathbb{E}[\hat{\sigma}^2_{\text{MLE}}]$.

Starting with the definition:
$$\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n (X_i - \bar{X})^2$$

We know from theory that:
$$\sum_{i=1}^n (X_i - \bar{X})^2 = \sum_{i=1}^n (X_i - \mu)^2 - n(\bar{X} - \mu)^2$$

Taking expectations:
$$\mathbb{E}\left[\sum_{i=1}^n (X_i - \bar{X})^2\right] = \mathbb{E}\left[\sum_{i=1}^n (X_i - \mu)^2\right] - \mathbb{E}[n(\bar{X} - \mu)^2]$$

Since $\mathbb{E}[(X_i - \mu)^2] = \sigma^2$ and $\mathbb{E}[(\bar{X} - \mu)^2] = \text{Var}(\bar{X}) = \sigma^2/n$:

$$\mathbb{E}\left[\sum_{i=1}^n (X_i - \bar{X})^2\right] = n\sigma^2 - n \cdot \frac{\sigma^2}{n} = (n-1)\sigma^2$$

Therefore:
$$\mathbb{E}[\hat{\sigma}^2_{\text{MLE}}] = \frac{1}{n}(n-1)\sigma^2 = \frac{n-1}{n}\sigma^2$$

The bias is:
$$\text{Bias}(\hat{\sigma}^2_{\text{MLE}}) = \mathbb{E}[\hat{\sigma}^2_{\text{MLE}}] - \sigma^2 = \frac{n-1}{n}\sigma^2 - \sigma^2 = -\frac{\sigma^2}{n}$$

**Conclusion:** The MLE is biased downward by $-\sigma^2/n$, which vanishes as $n \to \infty$ (asymptotically unbiased).

---

## Exercise 2: Prove MSE Decomposition

**Problem:** Show that $\text{MSE}(\hat{\theta}) = \text{Bias}(\hat{\theta})^2 + \text{Var}(\hat{\theta})$ using the definition of variance.

**Solution:**

Start with the definition of MSE:
$$\text{MSE}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \theta)^2]$$

Let $\mu_{\hat{\theta}} = \mathbb{E}[\hat{\theta}]$ and add/subtract this quantity:
$$\text{MSE}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \mu_{\hat{\theta}} + \mu_{\hat{\theta}} - \theta)^2]$$

Expanding the square:
$$= \mathbb{E}[(\hat{\theta} - \mu_{\hat{\theta}})^2 + 2(\hat{\theta} - \mu_{\hat{\theta}})(\mu_{\hat{\theta}} - \theta) + (\mu_{\hat{\theta}} - \theta)^2]$$

Taking expectations term by term:
$$= \mathbb{E}[(\hat{\theta} - \mu_{\hat{\theta}})^2] + 2(\mu_{\hat{\theta}} - \theta)\mathbb{E}[\hat{\theta} - \mu_{\hat{\theta}}] + (\mu_{\hat{\theta}} - \theta)^2$$

Note that:
- $\mathbb{E}[(\hat{\theta} - \mu_{\hat{\theta}})^2] = \text{Var}(\hat{\theta})$ by definition
- $\mathbb{E}[\hat{\theta} - \mu_{\hat{\theta}}] = \mathbb{E}[\hat{\theta}] - \mu_{\hat{\theta}} = 0$
- $(\mu_{\hat{\theta}} - \theta)^2 = (\mathbb{E}[\hat{\theta}] - \theta)^2 = \text{Bias}(\hat{\theta})^2$

Therefore:
$$\text{MSE}(\hat{\theta}) = \text{Var}(\hat{\theta}) + 0 + \text{Bias}(\hat{\theta})^2 = \text{Bias}(\hat{\theta})^2 + \text{Var}(\hat{\theta})$$

**Conclusion:** MSE decomposes into squared bias (systematic error) plus variance (random error).

---

## Exercise 3: Simulate Bias-Variance Tradeoff

**Problem:** Simulate the bias-variance tradeoff for the shrinkage estimator $\delta_\alpha = \alpha \bar{X} + (1-\alpha) \mu_0$ with $\mu_0 = 170$ using heights data.

**Solution:** Refer to the Simulation Exercises section in `../notebooks/01-bias-variance-tradeoff.ipynb` for the full Monte Carlo implementation, plots, and discussion.

---

## Exercise 4: Compare MSE of Sample Variance Estimators

**Problem:** Compare MSE of $s^2$ vs $\hat{\sigma}^2_{\text{MLE}}$ across different sample sizes $n = 5, 10, 20, 50$.

**Solution:** Refer to the Simulation Exercises section in `../notebooks/01-bias-variance-tradeoff.ipynb` for complete code, output tables, and interpretation comparing $s^2$ and the MLE across sample sizes.
