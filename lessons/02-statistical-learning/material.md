# Lesson 2: Statistical Learning — Parameter Estimation

Author: Applied Statistics Course Team
Prerequisites: Lesson 1 (distributions, moments, LLN/CLT)
Estimated time: 4-5 hours self-study

## Learning Objectives
- Derive estimators using Maximum Likelihood Estimation (MLE)
- Apply Method of Moments (MoM) and compare to MLE
- Compute standard errors via observed Fisher information and the delta method
- Assess estimators: bias, variance, MSE; use likelihood profiles and bootstrap
- Implement simulation studies to compare procedures
- Understand theoretical properties and practical considerations

## 1. Parameter Estimation: Why Do We Care?

### The Central Question
Given observed data x₁, x₂, ..., x_n and a probabilistic model f(x|θ), what parameter values make this data most plausible?

**Applications**:
- **A/B Testing**: Estimate click-through rates for different variants
- **Quality Control**: Estimate defect rates in manufacturing
- **Risk Assessment**: Estimate failure probabilities
- **Scientific Inference**: Estimate physical constants from measurements

## 2. Likelihood and Maximum Likelihood Estimation

### Probability vs Likelihood: The Key Duality

**Same mathematical function, different interpretations**:

| Aspect | Probability | Likelihood |
|--------|-------------|------------|
| Fixed | θ (parameters) | x (data) |
| Varies | x (data) | θ (parameters) |
| Question | P(X = x \| θ) | L(θ \| x) ∝ P(X = x \| θ) |
| Purpose | Forward prediction | Parameter estimation |

### The Likelihood Function

For i.i.d. observations X₁, ..., X_n with density/pmf f(x|θ):
```
L(θ) = ∏_{i=1}^n f(x_i | θ)
```

**Interpretation**: Measures how well parameter θ explains the observed data. Higher likelihood means θ is more supported by the data.

### Log-Likelihood

For computational convenience, work with:
```
ℓ(θ) = log L(θ) = ∑_{i=1}^n log f(x_i | θ)
```

**Advantages**:
- Products become sums (easier optimization)
- Numerical stability (avoids underflow)
- Concavity/convexity often clearer
- Same maximum: argmax L(θ) = argmax ℓ(θ)

### Maximum Likelihood Principle

The **Maximum Likelihood Estimator (MLE)** is:
```
θ̂_MLE = argmax_θ L(θ) = argmax_θ ℓ(θ)
```

**Intuition**: Choose the parameter value that makes our observed data as likely as possible.

### Examples

#### Bernoulli(p) - Coin Flips
**Model**: X_i ~ Bernoulli(p), observe k heads in n flips

**Likelihood**: L(p) = p^k (1-p)^{n-k}

**Log-likelihood**: ℓ(p) = k log p + (n-k) log(1-p)

**MLE derivation**:
```
dℓ/dp = k/p - (n-k)/(1-p) = 0
k(1-p) = p(n-k)
k - k p = p n - k p
k = p n
p̂ = k/n = x̄
```

**Result**: θ̂_MLE = proportion of successes.

#### Normal(μ, σ²) - Location and Scale
**Model**: X_i ~ N(μ, σ²)

**Joint MLEs**:
```
μ̂_MLE = x̄_n
σ̂²_MLE = (1/n) ∑(x_i - x̄_n)²
```

**Note**: Uses n in denominator (biased for finite samples).

#### Poisson(λ) - Count Data
**Model**: X_i ~ Poisson(λ)

**Log-likelihood**: ℓ(λ) = ∑(x_i log λ - λ - log(x_i!))

**MLE derivation**:
```
dℓ/dλ = ∑(x_i / λ - 1) = 0
∑ x_i = n λ
λ̂ = x̄_n
```

**Result**: λ̂_MLE = sample mean.

### Theoretical Properties of MLE

#### Consistency
```
θ̂_MLE → θ₀ (true parameter) as n → ∞
```
**Justification**: By LLN, sample averages converge to expectations.

#### Asymptotic Normality
```
√n (θ̂_MLE - θ₀) → N(0, I(θ₀)⁻¹) in distribution
```
where I(θ) is the Fisher information.

#### Efficiency
MLE achieves the Cramér-Rao lower bound asymptotically (most efficient estimator).

### Fisher Information

**Definition**:
```
I(θ) = E[ - ∂²ℓ(θ)/∂θ² ]
```

**Interpretation**: Measures the amount of information about θ contained in the data. Higher information → sharper likelihood peak → lower variance.

**Observed information**:
```
J(θ) = - ∂²ℓ(θ)/∂θ² |_{data}
```

**Asymptotic variance**:
```
Var(θ̂_MLE) ≈ I(θ̂_MLE)⁻¹ / n
SE(θ̂_MLE) ≈ √(I(θ̂_MLE)⁻¹ / n)
```

### Strengths of MLE

1. **Asymptotically optimal**: Consistent, asymptotically normal, efficient
2. **Flexible**: Works for any probabilistic model
3. **Invariant**: If θ̂ is MLE of θ, then g(θ̂) is MLE of g(θ)
4. **Well-studied theory**: Rich literature on properties and extensions

### Limitations of MLE

1. **Small samples**: Can be biased and unstable
2. **Non-identifiability**: Multiple maxima possible
3. **Flat likelihoods**: Poor precision when information is low
4. **Model misspecification**: Sensitive to incorrect assumptions
5. **Computation**: May require numerical optimization

## 3. Method of Moments (MoM)

### Why Another Method?

- MLE powerful but sometimes hard to compute
- MoM offers simpler alternative
- **Idea**: Match sample moments with theoretical moments

### The Method of Moments

**Theoretical moments**:
```
m_k(θ) = E_θ[X^k]
```

**Empirical moments**:
```
m̂_k = (1/n) ∑_{i=1}^n X_i^k
```

**MoM estimator**: Solve
```
m_k(θ) = m̂_k
```

### Examples

#### Bernoulli(p)
**Theoretical**: E[X] = p
**Empirical**: m̂₁ = x̄_n
**Solution**: p̂_MoM = x̄_n
**Note**: Coincides with MLE.

#### Poisson(λ)
**Theoretical**: E[X] = λ
**Empirical**: m̂₁ = x̄_n
**Solution**: λ̂_MoM = x̄_n
**Note**: Coincides with MLE.

#### Normal(μ, σ²)
**System of equations**:
```
E[X] = μ = x̄_n
E[X²] = μ² + σ² = m̂₂ = (1/n) ∑ X_i²
```

**Solution**:
```
μ̂_MoM = x̄_n
σ̂²_MoM = m̂₂ - (x̄_n)²
```

**Note**: Different from MLE (which uses n in denominator).

### MLE vs MoM: Similarities and Differences

| Aspect | MLE | MoM |
|--------|-----|-----|
| **Principle** | Maximize likelihood | Match moments |
| **Computation** | Often requires optimization | Usually algebraic |
| **Consistency** | Yes (under regularity) | Yes (under conditions) |
| **Efficiency** | Asymptotically efficient | Generally not |
| **Bias** | Can be biased in small samples | Can be biased |
| **When they coincide** | Simple models (exponential family) | Often for location parameters |

### Normal Variance: MLE vs MoM

**MLE**:
```
σ̂²_MLE = (1/n) ∑(X_i - x̄_n)²
```

**MoM**:
```
σ̂²_MoM = m̂₂ - (x̄_n)²
```

**Note**: These are equal for normal data, but differ in general.

## 4. Fisher Information and Standard Errors

### Fisher Information

**Definition**:
```
I(θ) = E[ - ∂²ℓ(θ)/∂θ² ]
```

**Interpretation**: Expected curvature of log-likelihood, measures information about θ.

**Properties**:
- Higher I(θ) → sharper likelihood → lower variance
- For i.i.d. data: I_n(θ) = n I(θ)
- Asymptotic variance: Var(θ̂) ≈ I(θ)⁻¹/n

### Observed vs Expected Information

**Expected information**: I(θ) = E[J(θ)]
**Observed information**: J(θ) = -∂²ℓ/∂θ² |_{data}

**Usage**: Often use observed information evaluated at MLE for standard errors.

### Delta Method

For transformations g(θ), the asymptotic distribution of g(θ̂) is:
```
√n (g(θ̂) - g(θ)) → N(0, ∇g(θ) Σ ∇g(θ)ᵀ)
```

where Σ = I(θ)⁻¹.

**Application**: Standard errors for functions of parameters (ratios, logarithms, etc.).

### Parametric Bootstrap

**Algorithm**:
1. Fit model: compute θ̂
2. For b = 1 to B:
   - Simulate x^(b) ~ f(·|θ̂)
   - Compute θ̂^(b) from x^(b)
3. Use empirical distribution of {θ̂^(b)} for inference

**Uses**:
- Standard errors: SD of bootstrap replicates
- Confidence intervals: percentiles of bootstrap distribution
- Bias correction: compare average of replicates to θ̂

## 5. Model Assessment

### Bias-Variance-MSE Decomposition

For estimator θ̂ of parameter θ:
```
MSE(θ̂) = E[(θ̂ - θ)²] = Bias(θ̂)² + Var(θ̂)
```

**Bias**: E[θ̂] - θ (systematic error)
**Variance**: Var(θ̂) (random error)
**MSE**: Mean squared error (total error)

### Likelihood Profiles

**Definition**: Plot ℓ(θ) vs θ to visualize:
- Location of maximum (point estimate)
- Curvature (information about precision)
- Multiple modes (identifiability issues)
- Flat regions (poor information)

### Bootstrap for Inference

**Parametric bootstrap**:
- Resample from fitted model f(·|θ̂)
- Compute distribution of θ̂^(b)

**Nonparametric bootstrap**:
- Resample from empirical distribution
- More robust to model misspecification

### Regularization

When likelihood is flat or multiple maxima exist:
- **Ridge regression**: Add penalty λ Σ θⱼ²
- **Lasso**: Add penalty λ Σ |θⱼ|
- **Bayesian**: Incorporate prior beliefs

## 6. Worked Examples

### Exponential(λ) Distribution

**Model**: X_i ~ Exponential(λ), density λ e^{-λx}

**MLE derivation**:
```
ℓ(λ) = n log λ - λ ∑ x_i
∂ℓ/∂λ = n/λ - ∑ x_i = 0
λ̂ = n / ∑ x_i = 1/x̄
```

**Fisher information**:
```
I(λ) = E[ -∂²ℓ/∂λ² ] = E[ -(-n/λ²) ] = n/λ²
SE(λ̂) ≈ √(λ²/n) = λ/√n
```

### Normal(μ, σ²) Distribution

**Joint MLEs**:
```
μ̂ = x̄, σ̂² = (1/n) ∑(x_i - x̄)²
```

**Note**: σ̂² is biased (divide by n-1 for unbiased estimate).

**Asymptotic variances**:
```
Var(μ̂) ≈ σ²/n
Var(σ̂²) ≈ (2σ⁴)/n (from delta method)
```

## 7. Summary

Parameter estimation is fundamental to statistical learning:

1. **MLE**: Powerful, asymptotically optimal, but requires optimization
2. **MoM**: Simple, algebraic, but generally less efficient
3. **Fisher information**: Quantifies precision and enables standard errors
4. **Delta method**: Extends inference to parameter transformations
5. **Bootstrap**: Provides robust inference without asymptotic assumptions

Choose method based on:
- Computational feasibility
- Sample size
- Model complexity
- Desired properties (efficiency, robustness)

## 8. References

- Casella and Berger, "Statistical Inference"
- Wasserman, "All of Statistics"
- Efron and Tibshirani, "An Introduction to the Bootstrap"
- Lehmann and Casella, "Theory of Point Estimation"

## 9. Discussion Questions

1. When would you prefer MoM over MLE? When would you prefer MLE over MoM?
2. How does the delta method extend the CLT to functions of estimators?
3. What are the practical implications of the bias-variance tradeoff in parameter estimation?
4. How might bootstrap methods be more robust than asymptotic approximations?

---

## Practical Session 2: Parameter Estimation in Practice

### Objectives
- Implement MLE and MoM estimators
- Compare estimators via simulation
- Use bootstrap for standard errors and confidence intervals
- Assess estimator properties (bias, variance, MSE)

### Tasks

1. **Exponential MLE vs MoM**
   - Simulate n = 50, 100, 500 samples from Exponential(λ = 2)
   - Compare bias and variance of λ̂_MLE = 1/x̄ and λ̂_MoM = 1/x̄
   - Note: They coincide for exponential distribution

2. **Poisson Parameter Estimation**
   - Generate data from Poisson(λ = 5)
   - Estimate λ using MLE (sample mean)
   - Use bootstrap to estimate standard error
   - Compare to theoretical SE = √(λ/n)

3. **Normal Distribution Fitting**
   - Generate data from N(μ = 170, σ² = 100)
   - Compute MLEs for μ and σ²
   - Use delta method to get SE for σ
   - Compare biased vs unbiased variance estimates

### Starter Code (Python)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

def bootstrap_se(data, statistic_func, B=1000):
    """Compute bootstrap standard error for a statistic"""
    n = len(data)
    boot_stats = []
    for _ in range(B):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(statistic_func(sample))
    return np.std(boot_stats)

# 1) Exponential parameter estimation
print("=== Exponential Parameter Estimation ===")
true_lambda = 2.0

for n in [50, 100, 500]:
    # Generate data
    data = np.random.exponential(scale=1/true_lambda, size=n)

    # MLE/MoM estimate (same for exponential)
    lambda_hat = 1 / np.mean(data)

    # Bootstrap SE
    se_boot = bootstrap_se(data, lambda_func=lambda x: 1/np.mean(x))

    # Theoretical SE
    se_theory = true_lambda / np.sqrt(n)

    print(f"n = {n}: λ̂ = {lambda_hat:.4f}, Bootstrap SE = {se_boot:.4f}, Theory SE = {se_theory:.4f}")

# 2) Poisson parameter estimation
print("\n=== Poisson Parameter Estimation ===")
true_lambda = 5.0
n = 200
data = np.random.poisson(true_lambda, size=n)

# MLE
lambda_hat = np.mean(data)

# Bootstrap SE
se_boot = bootstrap_se(data, np.mean)

# Theoretical SE
se_theory = np.sqrt(true_lambda / n)

print(f"Poisson: λ̂ = {lambda_hat:.4f}, Bootstrap SE = {se_boot:.4f}, Theory SE = {se_theory:.4f}")

# 3) Normal distribution fitting
print("\n=== Normal Distribution Fitting ===")
true_mu = 170.0
true_sigma2 = 100.0
n = 200
data = np.random.normal(true_mu, np.sqrt(true_sigma2), size=n)

# MLEs
mu_hat = np.mean(data)
sigma2_hat_mle = np.mean((data - mu_hat)**2)  # MLE (biased)
sigma2_hat_unbiased = np.var(data, ddof=1)    # Unbiased

# Standard errors
se_mu = np.sqrt(sigma2_hat_mle / n)
se_sigma2 = 2 * sigma2_hat_mle / np.sqrt(n)  # Delta method approximation

print(f"Normal: μ̂ = {mu_hat:.4f}, SE(μ̂) = {se_mu:.4f}")
print(f"Normal: σ²̂_MLE = {sigma2_hat_mle:.4f}, SE(σ²̂) ≈ {se_sigma2:.4f}")
print(f"Normal: σ²̂_unbiased = {sigma2_hat_unbiased:.4f}")

# Likelihood profile for exponential
print("\n=== Likelihood Profile (Exponential) ===")
lambda_grid = np.linspace(1.0, 3.0, 100)
log_likelihoods = []

for lam in lambda_grid:
    # Log-likelihood for exponential
    ll = n * np.log(lam) - lam * np.sum(data)
    log_likelihoods.append(ll)

log_likelihoods = np.array(log_likelihoods)
# Normalize to max = 0
log_likelihoods -= np.max(log_likelihoods)

plt.figure(figsize=(10, 6))
plt.plot(lambda_grid, log_likelihoods)
plt.axvline(true_lambda, color='red', linestyle='--', label=f'True λ = {true_lambda}')
plt.axvline(lambda_hat, color='green', linestyle='--', label=f'MLE λ̂ = {lambda_hat:.3f}')
plt.xlabel('λ')
plt.ylabel('Log-Likelihood (normalized)')
plt.title('Likelihood Profile for Exponential Parameter')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Extension Exercises

1. **Bias-Variance Analysis**: For the normal variance estimator, compute the bias of the MLE and compare to the unbiased estimator.

2. **Confidence Intervals**: Use bootstrap percentiles to construct 95% confidence intervals for all parameters.

3. **Model Misspecification**: Generate data from a mixture distribution and see how MLE and MoM perform when the model is misspecified.

4. **Delta Method Application**: Use the delta method to get the standard error for √σ² (standard deviation) and compare to bootstrap.

### Key Learning Points
- MLE often coincides with MoM for simple models
- Bootstrap provides robust standard errors
- Likelihood profiles reveal information about parameter precision
- Asymptotic approximations improve with sample size
- Model assessment requires both theoretical and empirical evaluation
