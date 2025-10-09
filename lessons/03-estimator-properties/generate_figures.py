#!/usr/bin/env python3
"""
Generate external figures for Lesson 3: Estimator Properties slides.
This script runs the key plotting code from the notebooks to generate
figures that can be included in the LaTeX slides.

Usage: python generate_figures.py
"""

import sys
import os
sys.path.append('..')  # For shared data access

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style and random seed
sns.set_theme(context="talk", style="whitegrid")
sns.set_palette(["#000000", "#E69F00", "#56B4E9", "#009E73",
                 "#F0E442", "#0072B2", "#D55E00", "#CC79A7"])
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})
rng = np.random.default_rng(2025)

# Create figures directory
os.makedirs('figures', exist_ok=True)

print("Generating figures for Lesson 3 slides...")

# Figure 1: Bias-Variance Tradeoff
print("Generating bias-variance tradeoff figure...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Bias and variance vs alpha (shrinkage)
alpha_values = np.linspace(0, 1, 21)
true_mu = 170
mu_0 = 165
n_small = 20

results = []
for alpha in alpha_values:
    estimates = []
    for r in range(1000):
        sample = rng.normal(true_mu, 15, n_small)
        estimate = alpha * np.mean(sample) + (1 - alpha) * mu_0
        estimates.append(estimate)

    bias = np.mean(estimates) - true_mu
    variance = np.var(estimates, ddof=0)
    mse = np.mean((np.array(estimates) - true_mu)**2)

    results.append({
        'alpha': alpha,
        'bias': bias,
        'variance': variance,
        'mse': mse,
        'bias_squared': bias**2
    })

shrinkage_df = pd.DataFrame(results)

# Bias and variance vs alpha
axes[0,0].plot(shrinkage_df['alpha'], shrinkage_df['bias'], 'b-', linewidth=2, label='Bias')
axes[0,0].plot(shrinkage_df['alpha'], shrinkage_df['variance'], 'r-', linewidth=2, label='Variance')
axes[0,0].set_xlabel(r'$\alpha$')
axes[0,0].set_ylabel('Value')
axes[0,0].set_title('Bias and Variance vs Shrinkage Parameter')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# MSE decomposition
axes[1,0].plot(shrinkage_df['alpha'], shrinkage_df['bias_squared'], 'g-', linewidth=2, label='Bias²')
axes[1,0].plot(shrinkage_df['alpha'], shrinkage_df['variance'], 'r-', linewidth=2, label='Variance')
axes[1,0].plot(shrinkage_df['alpha'], shrinkage_df['mse'], 'b-', linewidth=3, label='MSE')
optimal_idx = shrinkage_df['mse'].argmin()
optimal_alpha = shrinkage_df.iloc[optimal_idx]['alpha']
axes[1,0].axvline(optimal_alpha, color='black', linestyle='--', alpha=0.7)
axes[1,0].set_xlabel(r'$\alpha$')
axes[1,0].set_ylabel('Value')
axes[1,0].set_title('MSE Decomposition')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Sample variance comparison
sample_sizes = [5, 10, 20, 50, 100]
var_comparison = []
for n in sample_sizes:
    # Unbiased estimator
    unbiased_ests = []
    mle_ests = []
    for r in range(2000):
        sample = rng.normal(5, 2, n)
        unbiased_ests.append(np.var(sample, ddof=1))
        mle_ests.append(np.var(sample, ddof=0))

    unbiased_mse = np.mean((np.array(unbiased_ests) - 4)**2)
    mle_mse = np.mean((np.array(mle_ests) - 4)**2)

    var_comparison.append({'n': n, 'unbiased_mse': unbiased_mse, 'mle_mse': mle_mse})

var_df = pd.DataFrame(var_comparison)

axes[1,1].plot(var_df['n'], var_df['unbiased_mse'], 'b-', linewidth=2, marker='o', label='Unbiased')
axes[1,1].plot(var_df['n'], var_df['mle_mse'], 'r-', linewidth=2, marker='s', label='MLE')
axes[1,1].set_xlabel('Sample Size')
axes[1,1].set_ylabel('MSE')
axes[1,1].set_title('Sample Variance Estimator MSE')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/bias_variance_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Generated bias-variance tradeoff figure")

# Figure 2: Consistency Demonstration
print("Generating consistency demonstration figure...")

# Sample mean consistency
n_values = [5, 20, 100]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, n in enumerate(n_values):
    # Generate multiple samples
    samples = rng.normal(5, 2, (1000, n))
    sample_means = np.mean(samples, axis=1)

    # Plot histogram
    axes[i].hist(sample_means, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')

    # Overlay normal approximation
    mu, sigma = 5.0, 2.0/np.sqrt(n)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    axes[i].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
    axes[i].set_title(f'n = {n}: Mean ≈ {mu:.3f}, SD ≈ {sigma:.3f}')
    axes[i].set_xlabel('Sample Mean')
    axes[i].set_ylabel('Density')

plt.tight_layout()
plt.savefig('figures/consistency_demonstration.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Generated consistency demonstration figure")

# Figure 3: CRLB Achievement
print("Generating CRLB achievement figure...")

# Normal mean CRLB achievement
n_values = [5, 10, 20, 50, 100, 200]
crlb_results = []

for n in n_values:
    estimates = []
    for r in range(5000):
        sample = rng.normal(10, 2, n)
        estimates.append(np.mean(sample))

    empirical_var = np.var(estimates, ddof=0)
    crlb = 4 / n  # sigma^2 / n = 4/n
    efficiency_ratio = empirical_var / crlb

    crlb_results.append({
        'n': n,
        'empirical_var': empirical_var,
        'crlb': crlb,
        'efficiency_ratio': efficiency_ratio
    })

crlb_df = pd.DataFrame(crlb_results)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Variance comparison
axes[0].plot(crlb_df['n'], crlb_df['empirical_var'], 'b-', linewidth=3, marker='o', label='Empirical Var')
axes[0].plot(crlb_df['n'], crlb_df['crlb'], 'r-', linewidth=3, marker='s', label='CRLB')
axes[0].set_xlabel('Sample Size')
axes[0].set_ylabel('Variance')
axes[0].set_title('Sample Mean Variance vs CRLB')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Efficiency ratio
axes[1].plot(crlb_df['n'], crlb_df['efficiency_ratio'], 'g-', linewidth=3, marker='D')
axes[1].axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Efficiency')
axes[1].set_xlabel('Sample Size')
axes[1].set_ylabel('Efficiency Ratio')
axes[1].set_title('Efficiency Ratio (Empirical/CRLB)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/crlb_achievement.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Generated CRLB achievement figure")

# Figure 4: Proportion CI Coverage
print("Generating proportion CI coverage figure...")

def ci_prop_wald(k, n, alpha=0.05):
    p = k/n
    z = stats.norm.ppf(1 - alpha/2)
    hw = z*np.sqrt(max(p*(1-p)/n, 0.0))
    return max(0.0, p-hw), min(1.0, p+hw)

def ci_prop_wilson(k, n, alpha=0.05):
    if n==0: return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha/2)
    p = k/n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    adj = p*(1-p)/n + z**2/(4*n**2)
    hw = z/denom * np.sqrt(adj)
    return max(0.0, center - hw), min(1.0, center + hw)

# Coverage simulation
ps = np.linspace(0.05, 0.95, 20)
n = 20
R = 2000

wald_coverage = []
wilson_coverage = []

for p in ps:
    wald_cov = 0
    wilson_cov = 0

    for r in range(R):
        k = rng.binomial(n, p)
        wald_lo, wald_hi = ci_prop_wald(k, n)
        wilson_lo, wilson_hi = ci_prop_wilson(k, n)

        wald_cov += (wald_lo <= p <= wald_hi)
        wilson_cov += (wilson_lo <= p <= wilson_hi)

    wald_coverage.append(wald_cov / R)
    wilson_coverage.append(wilson_cov / R)

plt.figure(figsize=(10, 6))

plt.plot(ps, wald_coverage, 'b-', linewidth=3, marker='o', label='Wald')
plt.plot(ps, wilson_coverage, 'r-', linewidth=3, marker='s', label='Wilson')
plt.axhline(0.95, color='black', linestyle='--', alpha=0.7, label='95% Target')
plt.xlabel('True Proportion')
plt.ylabel('Empirical Coverage')
plt.title('Proportion CI Coverage Comparison (n=20)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/proportion_ci_coverage.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Generated proportion CI coverage figure")

# Figure 5: Bootstrap Distribution
print("Generating bootstrap distribution figure...")

# Generate exponential data
true_lambda = 2.0
n = 100
x = rng.exponential(1/true_lambda, n)

# Bootstrap the median
B = 5000
boot_medians = np.empty(B)
for b in range(B):
    xb = x[rng.integers(0, n, n)]
    boot_medians[b] = np.median(xb)

# Plot bootstrap distribution
plt.figure(figsize=(10, 6))
plt.hist(boot_medians, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')

# Vertical lines
sample_median = np.median(x)
true_median = np.log(2)/true_lambda
plt.axvline(sample_median, color='red', linewidth=3, label=f'Sample Median = {sample_median:.3f}')
plt.axvline(true_median, color='green', linewidth=3, linestyle='--', label=f'True Median = {true_median:.3f}')

# CI bounds (percentile)
lo_p, hi_p = np.quantile(boot_medians, [0.025, 0.975])
plt.axvline(lo_p, color='orange', linewidth=2, linestyle='--', label='95% Bootstrap CI')
plt.axvline(hi_p, color='orange', linewidth=2, linestyle='--')

plt.xlabel('Median Value')
plt.ylabel('Density')
plt.title('Bootstrap Distribution of Sample Median\\n(Exponential Data)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/bootstrap_median_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Generated bootstrap distribution figure")

print("\nAll figures generated successfully!")
print("Figures saved in: lessons/03-estimator-properties/figures/")
print("\nGenerated figures:")
print("- bias_variance_tradeoff.png")
print("- consistency_demonstration.png")
print("- crlb_achievement.png")
print("- proportion_ci_coverage.png")
print("- bootstrap_median_distribution.png")