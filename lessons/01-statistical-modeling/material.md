# Lesson 1: Statistical Modeling — Course Material (PDF Draft)

Author: Applied Statistics Course Team  
Prerequisites: Intro probability and calculus  
Estimated time: 3–4 hours self-study

## Learning Objectives
- Define random variables (discrete/continuous) and sample space.
- Interpret PMF, PDF, and CDF; compute probabilities and quantiles.
- Describe expectation, variance, moments, and moment generating functions.
- Explain Law of Large Numbers (LLN) and Central Limit Theorem (CLT) with implications.
- Compute and interpret descriptive statistics: mean, median, variance, quantiles.

## 1. Random Variables
A random variable (RV) X is a measurable function from the sample space Ω to ℝ.
- Discrete: countable support (e.g., Poisson, Binomial).
- Continuous: has density f(x) with F(x)=P(X≤x)=∫_{-∞}^x f(t)dt.
- Mixed: combination of discrete and continuous parts.

### Common Distributions
- Bernoulli(p): P(X=1)=p, P(X=0)=1−p; E[X]=p, Var(X)=p(1−p).
- Binomial(n,p): sum of n IID Bernoulli; E=np, Var=np(1−p).
- Poisson(λ): P(X=k)=e^{−λ} λ^k/k!; E=Var=λ.
- Geometric(p): P(X=k)=(1−p)^{k−1}p; E=1/p.
- Uniform(a,b): f(x)=1/(b−a); E=(a+b)/2; Var=(b−a)^2/12.
- Exponential(λ): f(x)=λe^{−λx}, x≥0; E=1/λ; memoryless.
- Normal(μ,σ²): f(x)= (1/(σ√{2π})) exp(−(x−μ)²/(2σ²)).

## 2. Distribution Functions
- PMF (discrete): p(x)=P(X=x).
- PDF (continuous): f(x)=dF/dx where F(x)=P(X≤x).
- CDF: F(x)=P(X≤x), nondecreasing, right-continuous, lim_{x→−∞}F=0, lim_{x→∞}F=1.
- Survival S(x)=1−F(x); quantile q_α=inf{x: F(x)≥α}.

## 3. Moments and Descriptive Statistics
- Expectation: E[g(X)]=∑ g(x)p(x) or ∫ g(x)f(x)dx.
- Variance: Var(X)=E[(X−E[X])²]=E[X²]−E[X]².
- k-th central moment μ_k = E[(X−E[X])^k]; skewness γ1=μ_3/σ^3; kurtosis γ2=μ_4/σ^4.
- MGF M_X(t)=E[e^{tX}]; if exists in a neighborhood of 0, characterizes distribution.
- Sample analogs: sample mean x̄, variance s², quantiles (median, IQR).

## 4. LLN and CLT
Let X₁,…,X_n IID with E[X_i]=μ and Var(X_i)=σ²<∞.
- Weak LLN: x̄_n = (1/n)∑X_i → μ in probability.
- Strong LLN: x̄_n → μ almost surely.
- CLT: √n (x̄_n − μ) ⇒ N(0, σ²) (convergence in distribution). Thus, for large n,
  x̄_n ≈ N(μ, σ²/n). Enables approximate CIs and hypothesis tests.

## 5. Examples
1) Exponential waiting times: P(X>t)=e^{−λt}; median ln 2 / λ.  
2) Sum of IID Bernoulli is Binomial; normal approximation via CLT when np(1−p) large.  
3) Heights: approximate Normal; quantile computation and z-scores.

## 6. Practical Notes
- Numerical stability: prefer log-probabilities for small p.  
- Checking distribution fit: histograms, Q–Q plots, empirical CDFs.

## 7. Summary
We defined RVs, distributions, moments, and key limit theorems (LLN, CLT), and connected them to descriptive statistics.

## Short Exercises (theory)
1) For X~Uniform(0,1), compute E[X], Var(X), median, and q_{0.9}.  
2) For Poisson(3), compute P(X≤1) and E[X(X−1)].  
3) Show that for Bernoulli(p), skewness γ1=(1−2p)/√(p(1−p)).

---

# Practical Session 1

## Objectives
- Compute empirical moments and quantiles; compare to theoretical values.  
- Visualize distributions and explore LLN/CLT empirically.

## Dataset
- `shared/data/heights_weights_sample.csv`
- `shared/data/manufacturing_defects.csv`

## Tasks
1) Load heights dataset; compute mean, sd, median, IQR by sex. Draw histograms, KDE, and Q–Q plot vs Normal.  
2) Simulate 1,000 samples of size n∈{5,20,100} from Exponential(λ=1). Plot sampling distribution of x̄ and compare to Normal with mean 1 and sd 1/√n.  
3) For manufacturing defects, model counts with Poisson: estimate λ by sample mean; compare empirical and fitted PMF.

## Starter Code (Python)
```python
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from shared.utils.io import load_csv

# 1) EDA on heights
heights = load_csv("heights_weights_sample.csv")
print(heights.groupby("sex")["height_cm"].agg(["mean","std","median","quantile"]))
# Q–Q plot
for sex, df in heights.groupby("sex"):
    stats.probplot(df["height_cm"], dist="norm", plot=plt)
    plt.title(f"QQ plot heights – {sex}")
    plt.show()

# 2) LLN/CLT simulation
for n in [5, 20, 100]:
    xbar = np.mean(np.random.exponential(scale=1.0, size=(1000, n)), axis=1)
    sns.histplot(xbar, stat="density", kde=True)
    xs = np.linspace(min(xbar), max(xbar), 200)
    plt.plot(xs, stats.norm.pdf(xs, loc=1, scale=1/np.sqrt(n)), 'r--')
    plt.title(f"Sampling distribution of mean (n={n})")
    plt.show()

# 3) Poisson fit
md = load_csv("manufacturing_defects.csv")
lam = md["defects"].mean()
xs = np.arange(0, md["defects"].max()+1)
emp = md["defects"].value_counts(normalize=True).sort_index()
plt.bar(emp.index, emp.values, alpha=0.5, label="empirical")
plt.plot(xs, stats.poisson.pmf(xs, lam), 'ro-', label=f"Poisson λ={lam:.2f}")
plt.legend(); plt.show()
```
