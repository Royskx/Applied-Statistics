# Lesson 1 — Statistical Modeling (Notes aligned with updated slides)

Author: Stéphane Rivaud
Prerequisites: Intro probability and calculus
Estimated time: 3–4 hours self-study

---

## Learning Objectives
- Define random variables (RVs) and distributions; map real-world uncertainty to math objects.
- Use PMF/PDF/CDF to compute probabilities and quantiles.
- Compute expectation, variance, and interpret higher moments.
- Understand LLN and CLT (intuition + statements) and why normal approximations work.
- Connect descriptive statistics to probabilistic modeling and assumptions.

---

## 1. Modeling Motivation

### Why Probability? From Questions to Models
Natural-language questions become events and random variables.
- A/B test: “Is B better than A?” → clicks as Bernoulli trials, compare p_A vs p_B.
- Manufacturing: “Are defects rare and independent?” → counts as Poisson(λ).

See slide figures:
- CTR comparison with uncertainty: <img src="slides/figures/ab_test_ctr.png" alt="A/B test CTR" width="40%" />

Language of sets and probability: events are sets A ∈ 𝓕; probabilities P(A) quantify uncertainty. Set operations encode logic: A ∪ B = “A or B”, A ∩ B = “A and B”, A^c = “not A”.

### From Natural Language to Events and RVs
- Die roll: X ∈ {1,…,6}, E = {X even}; model: uniform on {1,…,6}.
- User clicks: indicator X = 1{click}; Bernoulli(p), compare p across variants.
- Defects: count D ∈ {0,1,2,…}; Poisson(λ) under rare, independent events.
- Time to failure: T ≥ 0; Exponential(λ) (memoryless).

Quick translations (event logic → probability):
- “At least one match” = complement of none: P(≥1) = 1 − P(0).
- “At most k defects” = cumulative count: P(D ≤ k) = ∑_{j=0}^k P(D=j).

### Catchy Examples
- A/B CTR with CIs (binomial proportion): visualizes uncertainty from finite samples.
- Birthday paradox: P(at least one shared birthday) ≈ 0.5 at n = 23.
- Monty Hall: switching wins with probability ≈ 2/3; illustrates conditional probability.
- Defects vs Poisson fit: λ summarizes average defects per batch.
- Time-to-failure: survival S(t)=e^{−λt} for Exponential(λ).

See slide figures:
- Birthday paradox curve: <img src="slides/figures/birthday_paradox.png" alt="Birthday paradox" width="40%" />
- Monty Hall simulation: <img src="slides/figures/monty_hall.png" alt="Monty Hall" width="40%" />
- Poisson fit by batch: <img src="slides/figures/defects_poisson_by_batch.png" alt="Defects Poisson fit" width="40%" />
- Exponential survival illustration: <img src="slides/figures/time_to_failure.png" alt="Time to failure" width="40%" />

---

## 2. Foundations and Random Variables

### Probability Space
A probability space is (Ω, 𝓕, P): sample space, σ-algebra, and probability measure.
- Basic properties for events A, B: bounds; complement P(A^c)=1−P(A); monotonicity; union/intersection inclusion–exclusion; countable additivity for disjoint events.
- Examples: fair coin/two tosses; illustrate independence.

Key identities (for events A, B):
- Inclusion–exclusion: P(A ∪ B) = P(A) + P(B) − P(A ∩ B).
- Total probability: if (B_i) is a partition with P(B_i)>0, then P(A) = ∑_i P(A|B_i)P(B_i).
- Bayes’ rule: P(B_j|A) = P(A|B_j)P(B_j)/∑_i P(A|B_i)P(B_i).

### Independence
- Events: A ⟂ B iff P(A ∩ B) = P(A)P(B). Mutual independence extends to finite collections.
- Stability: complements preserve independence. Pairwise independence does not imply mutual.
- RVs: X ⟂ Y iff P(X ∈ B, Y ∈ C) = P(X ∈ B)P(Y ∈ C) for measurable B, C.
- Modeling relevance: i.i.d. sampling, Poisson processes, exponential waiting times, CLT.

Practical notes:
- Independence is a strong assumption. Always justify (e.g., randomized assignment, non-overlapping time windows).
- Conditional independence appears in many models (e.g., given a rate λ, Poisson counts in disjoint intervals are independent).

### Random Variables and Laws
- RV: measurable map X:(Ω, 𝓕) → (ℝ, 𝓑(ℝ)).
- Law (distribution): μ_X(B) = P(X ∈ B). CDF F_X(x)=P(X ≤ x): nondecreasing, right-continuous, limits 0/1.
- Types: discrete (PMF), absolutely continuous (PDF), mixed.

Interpretations:
- CDF F encodes thresholds: “≤ x” questions. Quantiles invert F to answer “what cutoff?”
- Survival S(x)=1−F(x) is common in reliability/queueing.

### Discrete RVs: PMF and CDF
- PMF p_X(x)=P(X=x), ∑ p_X(x)=1. Probabilities add over sets.
- Discrete CDF: step function; jump at x equals p_X(x). P(a < X ≤ b)=F(b)−F(a).
- Use CDF only when order makes sense (not for unordered nominal categories).

Worked example (Poisson):
- p(k) = e^{−λ} λ^k/k!, k=0,1,2,…; F(k) = ∑_{j=0}^k p(j).
- Mean/variance both equal λ; estimate λ with sample mean.

See slide figure: <img src="slides/figures/poisson_empirical_pmf.png" alt="Empirical Poisson PMF" width="40%" />

### Continuous RVs: PDF and CDF
- Continuous: P(X=a)=0 for any point; probabilities live on sets/intervals.
- PDF f_X ≥ 0, ∫ f_X = 1, and P(a < X ≤ b) = ∫_a^b f_X.
- CDF F_X(x)=∫_{−∞}^x f_X(t)dt; f_X = F′ a.e. Quantiles: for q∈(0,1), x with F(x)≥q; if F strictly increasing, x = F^{-1}(q).

See slide figures:
- Uniform density & CDF: <img src="slides/figures/uniform_density_cdf.png" alt="Uniform density/CDF" width="40%" />
- Exponential density & CDF: <img src="slides/figures/exponential_density_cdf.png" alt="Exponential density/CDF" width="40%" />
- Normal density & CDF: <img src="slides/figures/normal_density_cdf.png" alt="Normal density/CDF" width="40%" />

### Key Distributions
- Discrete: Bernoulli(p): E=p, Var=p(1−p). Binomial(n,p): E=np, Var=np(1−p). Poisson(λ): P(X=k)=e^{−λ}λ^k/k!, E=Var=λ.
- Uniform(a,b): f=1/(b−a) on [a,b]; E=(a+b)/2; Var=(b−a)^2/12.
- Exponential(λ): f(x)=λe^{−λx} for x≥0; F=1−e^{−λx}; E=1/λ; Var=1/λ^2; memoryless.
- Normal(μ,σ^2): bell-shaped; standardization (X−μ)/σ ~ N(0,1).

Formulas you’ll use often:
- Binomial PMF: P(X=k)=C(n,k) p^k (1−p)^{n−k}. Normal approx valid when np(1−p) is moderate/large.
- Quantiles: for continuous strictly increasing F, q_α = F^{-1}(α). For empirical data, use order statistics or interpolation.

---

## 3. Moments and Descriptive Statistics

### Expectation (LOTUS)
For measurable g with E[|g(X)|]<∞:
- Discrete: E[g(X)] = ∑ g(x)p_X(x).
- Continuous: E[g(X)] = ∫ g(x) f_X(x) dx.
- In general: E[g(X)] = ∫ g dμ_X.

Tips:
- Use indicator functions to turn event probabilities into expectations, e.g., P(X∈A)=E[1_{X∈A}].
- Swap sum/integral with linearity when justified (absolute convergence/integrability).

### Expectation and Variance
- E[X] = ∫ x dμ_X(x). If E[X^2]<∞, Var(X)=E[(X−E[X])^2] = E[X^2] − (E[X])^2.
- Linearity: E[aX+b]=aE[X]+b; E[X+Y]=E[X]+E[Y].
- Scaling: Var(aX+b)=a^2 Var(X); Var(X)≥0 with equality iff X is a.s. constant.

Common pitfalls:
- Don’t treat the PDF height as a probability; use areas (integrals) for probabilities.
- For discrete variables, P(X=x) = jump of F at x, not the value of F(x) itself.

### Higher Moments, Skewness, Kurtosis
- Central moments μ_k = E[(X−μ)^k]. Skewness γ₁=μ₃/σ³ (asymmetry). Kurtosis β₂=μ₄/σ⁴; excess γ₂=β₂−3 (tail weight/peakedness).

Use cases:
- Diagnose asymmetry and tail heaviness; normal reference: γ₁=0, γ₂(excess)=0.
- Both are sensitive to outliers; complement with robust summaries (median, IQR).

### Worked Moment Examples
- Bernoulli(p): E[X]=p; Var(X)=p(1−p).
- Exponential(λ): E[X]=1/λ; Var(X)=1/λ².
- Normal(μ,σ²): E[X]=μ; Var(X)=σ².

Sketches:
- Bernoulli: E[X]=1·p+0·(1−p)=p; X^2=X ⇒ E[X^2]=p ⇒ Var=p−p^2.
- Exponential: E[X]=∫_0^∞ x λe^{−λx}dx=1/λ; E[X^2]=2/λ^2 ⇒ Var=1/λ^2.

### Moment Generating Function (MGF)
- M_X(t)=E[e^{tX}] (when finite near 0) ⇒ determines distribution; M_X^{(k)}(0)=E[X^k].
- Affine and sums: M_{aX+b}(t)=e^{bt}M_X(at); if X ⟂ Y then M_{X+Y}=M_X M_Y.
- Caveat: may not exist near 0 for heavy tails.

Examples:
- Normal(μ,σ²): M(t)=exp(μt+½σ²t²). Sum of independent normals is normal.
- Poisson(λ): M(t)=exp(λ(e^t−1)). Sum of independent Poissons adds rates.

### Characteristic Function
- φ_X(t)=E[e^{itX}] always exists; |φ|≤1; unique characterization with inversion.
- If E[|X|^k]<∞ then φ_X^{(k)}(0)= i^k E[X^k]. Sums and affine maps as for MGFs.

Why use φ instead of M?
- Always exists and behaves well for limit theorems (e.g., CLT via Lévy continuity).

### Descriptive vs Inferential Statistics
- Descriptive: summarize a dataset (mean, median, s², range, IQR; plots: hist/box/scatter).
- Inferential: generalize to population with uncertainty (SEs, CIs, tests, model-based inference, bootstrap).

See slide figure: <img src="slides/figures/descriptive_stats.png" alt="Descriptive statistics" width="40%" />

### Descriptive Statistics (Samples)
- Sample mean x̄, variance s², quantiles (median, IQR). Robustness: median robust to outliers; mean sensitive.

Practice pointers:
- Report both central tendency (x̄ and median) and spread (s, IQR); visualize with histograms/boxplots.
- Use Q–Q plots to assess normality before applying normal-based methods.

---

## 4. Modes of Convergence and Limit Theorems

### Why Convergence Matters
Multiple notions of random convergence with a hierarchy: a.s. ⇒ in probability ⇒ in distribution (no converses in general).

See slide figure: <img src="slides/figures/modes_convergence.png" alt="Modes of convergence" width="40%" />

### Convergence Almost Surely (a.s.)
X_n → X a.s. iff P(lim_n X_n = X) = 1. Strongest notion; implies in probability.

Intuition: pathwise convergence — almost all sample paths settle to the limit.

### Convergence in Probability
X_n →_P X iff for all ε>0, P(|X_n−X|>ε)→0. Large deviations vanish; paths may oscillate.

Diagnostic idea: histograms of X_n concentrate around X as n grows.

### Convergence in Distribution
X_n ⇒ X iff F_{X_n}(t)→F_X(t) at continuity points of F_X; equivalently E[f(X_n)]→E[f(X)] for all bounded continuous f.

Reminder: convergence in distribution alone does not ensure convergence of moments unless additional conditions hold (e.g., uniform integrability for means).

### Comparing Modes (typical contrasts)
- →_P but not →_a.s.: X_n = 1{U ≤ 1/n} with U~Unif(0,1).
- ⇒ but not →_P: X_n ≡ X with Y = 1−X; then X_n ⇒ X and ⇒ Y.
- Typewriter sequence: X_n → 0 in probability (and L^p, p<∞) but not almost surely.

### Law of Large Numbers (LLN)
- Weak LLN: for i.i.d. with E[X_i]=μ, x̄_n →_P μ.
- Strong LLN: for i.i.d. with E[|X_i|]<∞, x̄_n →_a.s. μ.
- Intuition: Var(x̄_n)=σ²/n when Var(X_i)=σ²<∞; Chebyshev ⇒ WLLN.

Implications:
- Consistency of averages and many estimators; Monte Carlo averages stabilize as n increases.
- Requires some integrability; heavy tails can break classical LLN assumptions.

### Central Limit Theorem (CLT)
- Lindeberg–Lévy CLT: if E[X_i]=μ, Var(X_i)=σ²∈(0,∞), then √n (x̄_n−μ) ⇒ N(0,σ²).
- Implication: large-sample normal approximations for estimators; enables CIs/tests.
- Bernoulli example: √n( p̂ − p ) ⇒ N(0, p(1−p)).

Practical cautions:
- Normal approximations may be poor for tiny n, extreme p in binomial, or heavy-tailed data; consider exact or bootstrap methods.

---

## 5. Worked Examples (from slides)
- PMF vs empirical frequencies (Poisson counts).
- PDFs and shapes (Uniform/Exponential/Normal).
- QQ plots for normality (heights dataset).

See slide figures:
- Poisson empirical vs model: <img src="slides/figures/poisson_empirical_pmf.png" alt="Poisson PMF" width="40%" />
- Density/CDF galleries: <img src="slides/figures/densities.png" alt="Densities" width="40%" />
- Heights Q–Q plot: <img src="slides/figures/heights_qq.png" alt="Heights QQ" width="40%" />

---

## 6. Exercises (Theory)
1) If X ~ Uniform(0,1), compute E[X], Var(X), median, and q_{0.9}.
2) If X ~ Poisson(3), compute P(X ≤ 1) and E[X(X−1)].
3) Show that for X ~ Bernoulli(p), skewness γ₁ = (1−2p)/√(p(1−p)).
4) Prove WLLN using Chebyshev for i.i.d. with finite variance.

Stretch prompts:
5) For Binomial(n,p), derive an approximate 95% CI for p using the CLT and discuss when it is unreliable.
6) Let X ~ Exponential(λ). Find the 90% quantile and the median; comment on the ratio.

---

## 7. Practical Preview (repo pointers)
- EDA on heights dataset; CLT simulation; Poisson fit to defects.
- Datasets: `shared/data/heights_weights_sample.csv`, `shared/data/manufacturing_defects.csv`.
- See lesson materials and starter code in the repository (example below).

### Starter Code (Python, optional)
```python
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from scipy import stats
from shared.utils.io import load_csv

# 1) EDA on heights
df = load_csv("heights_weights_sample.csv")
print(df.groupby("sex")["height_cm"].agg(["mean","std","median"]))
for sex, g in df.groupby("sex"):
    stats.probplot(g["height_cm"], dist="norm", plot=plt)
    plt.title(f"QQ plot heights – {sex}")
    plt.show()

# 2) CLT simulation for Exponential(1)
for n in [5, 20, 100]:
    xbar = np.mean(np.random.exponential(scale=1.0, size=(1000, n)), axis=1)
    sns.histplot(xbar, stat="density", kde=True)
    xs = np.linspace(min(xbar), max(xbar), 200)
    plt.plot(xs, stats.norm.pdf(xs, loc=1, scale=1/np.sqrt(n)), 'r--')
    plt.title(f"Sampling distribution of mean (n={n})")
    plt.show()

# 3) Poisson fit to manufacturing defects
md = load_csv("manufacturing_defects.csv")
lam = md["defects"].mean()
xs = np.arange(0, md["defects"].max()+1)
emp = md["defects"].value_counts(normalize=True).sort_index()
plt.bar(emp.index, emp.values, alpha=0.5, label="empirical")
plt.plot(xs, stats.poisson.pmf(xs, lam), 'ro-', label=f"Poisson λ={lam:.2f}")
plt.legend(); plt.show()
```

Related slide figures:
- Descriptive stats overview: <img src="slides/figures/descriptive_stats.png" alt="Descriptive stats" width="40%" />
- Poisson by batch: <img src="slides/figures/defects_poisson_by_batch.png" alt="Defects by batch" width="40%" />

---

## 8. Summary and References

### Summary
- Bridged descriptive and probabilistic modeling; formalized RVs, distributions, and moments.
- Clarified modes of convergence and stated LLN/CLT.
- Illustrated with A/B tests, paradoxes, defects, and reliability examples.

### References
- Casella and Berger, Statistical Inference.
- Wasserman, All of Statistics.
- Grimmett and Stirzaker, Probability and Random Processes.
