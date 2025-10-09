"""Figure generation for Bias Variance.

This module generates publication-quality figures for the estimator properties lesson.
"""

from .config import *
from scipy import stats
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import pandas as pd



def generate():
    """Generate all figures for this module."""
    # Figure 1: Bias-Variance Tradeoff (Enhanced Multi-Panel)
    # ============================================================================
    print("\n[1/8] Generating enhanced bias-variance tradeoff figure...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Bias-Variance Tradeoff in Estimators', fontsize=24, fontweight='bold', y=0.995)

    # Panel 1: Shrinkage parameter effect
    alpha_values = np.linspace(0, 1, 30)
    true_mu = 170
    mu_0 = 165
    n = 20

    biases, variances, mses = [], [], []
    for alpha in alpha_values:
        estimates = []
        for _ in range(2000):
            sample = rng.normal(true_mu, 15, n)
            estimate = alpha * np.mean(sample) + (1 - alpha) * mu_0
            estimates.append(estimate)

        bias = np.mean(estimates) - true_mu
        variance = np.var(estimates, ddof=0)
        mse = np.mean((np.array(estimates) - true_mu)**2)

        biases.append(abs(bias))
        variances.append(variance)
        mses.append(mse)

    ax = axes[0, 0]
    ax.plot(alpha_values, biases, color=COLORS['vermillion'], linewidth=3,
            marker='o', markersize=6, label='|Bias|', alpha=0.9)
    ax.plot(alpha_values, variances, color=COLORS['blue'], linewidth=3,
            marker='s', markersize=6, label='Variance', alpha=0.9)
    ax.set_xlabel('Shrinkage Parameter α', fontsize=16)
    ax.set_ylabel('Value', fontsize=16)
    ax.set_title('(A) Bias and Variance vs Shrinkage', fontsize=18, pad=10)
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    # Panel 2: MSE Decomposition
    ax = axes[0, 1]
    biases_sq = [b**2 for b in biases]
    ax.fill_between(alpha_values, 0, biases_sq, alpha=0.4, color=COLORS['vermillion'], label='Bias²')
    ax.fill_between(alpha_values, biases_sq, np.array(biases_sq) + np.array(variances),
                    alpha=0.4, color=COLORS['blue'], label='Variance')
    ax.plot(alpha_values, mses, color=COLORS['black'], linewidth=4,
            label='MSE = Bias² + Var', zorder=10)
    optimal_idx = np.argmin(mses)
    ax.axvline(alpha_values[optimal_idx], color=COLORS['green'], linewidth=2.5,
               linestyle='--', alpha=0.7, label=f'Optimal α = {alpha_values[optimal_idx]:.2f}')
    ax.scatter([alpha_values[optimal_idx]], [mses[optimal_idx]],
               s=150, color=COLORS['green'], zorder=15, edgecolors='black', linewidths=2)
    ax.set_xlabel('Shrinkage Parameter α', fontsize=16)
    ax.set_ylabel('MSE Components', fontsize=16)
    ax.set_title('(B) MSE Decomposition', fontsize=18, pad=10)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    # Panel 3: Sample Variance Estimators
    ax = axes[1, 0]
    sample_sizes = [5, 10, 20, 50, 100]
    true_sigma2 = 25
    biases_unbiased, biases_mle = [], []

    for n in sample_sizes:
        unbiased_ests = []
        mle_ests = []
        for _ in range(5000):
            sample = rng.normal(0, np.sqrt(true_sigma2), n)
            unbiased_ests.append(np.var(sample, ddof=1))
            mle_ests.append(np.var(sample, ddof=0))

        biases_unbiased.append(np.mean(unbiased_ests) - true_sigma2)
        biases_mle.append(np.mean(mle_ests) - true_sigma2)

    width = 0.35
    x = np.arange(len(sample_sizes))
    ax.bar(x - width/2, biases_unbiased, width, label='s² (unbiased)',
           color=COLORS['skyblue'], edgecolor='black', linewidth=1.2)
    ax.bar(x + width/2, biases_mle, width, label='σ̂²_MLE (biased)',
           color=COLORS['orange'], edgecolor='black', linewidth=1.2)
    ax.axhline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.7)
    ax.set_xlabel('Sample Size', fontsize=16)
    ax.set_ylabel('Bias', fontsize=16)
    ax.set_title('(C) Variance Estimator Bias', fontsize=18, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(sample_sizes)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: Visual representation of bias-variance tradeoff
    ax = axes[1, 1]
    np.random.seed(2025)
    n_points = 200
    true_val = 0

    # Low bias, high variance
    high_var = np.random.normal(true_val, 2, n_points)
    ax.scatter([1] * n_points, high_var, alpha=0.3, s=30, color=COLORS['blue'], label='High Var, Low Bias')

    # High bias, low variance
    low_var = np.random.normal(true_val + 3, 0.5, n_points)
    ax.scatter([2] * n_points, low_var, alpha=0.3, s=30, color=COLORS['vermillion'], label='Low Var, High Bias')

    # Balanced
    balanced = np.random.normal(true_val + 1, 1, n_points)
    ax.scatter([3] * n_points, balanced, alpha=0.3, s=30, color=COLORS['green'], label='Balanced')

    ax.axhline(true_val, color='black', linewidth=3, linestyle='--', label='True Value', zorder=10)
    ax.set_ylabel('Estimate Value', fontsize=16)
    ax.set_title('(D) Bias-Variance Visualization', fontsize=18, pad=10)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Type 1', 'Type 2', 'Type 3'])
    ax.set_xlim(0.5, 3.5)
    ax.legend(loc='upper right', frameon=True, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bias_variance_tradeoff.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Generated: bias_variance_tradeoff.png")

    # ============================================================================


    # Figure 6: Conceptual diagram for bias and variance
    print("[6/8] Generating bias-variance conceptual diagram...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Conceptual View: Bias vs Variance', fontsize=24, fontweight='bold')

    scenarios = [
        ('Low Bias\nLow Variance', 0, 0.3, COLORS['green']),
        ('Low Bias\nHigh Variance', 0, 1.5, COLORS['blue']),
        ('High Bias\nLow Variance', 3, 0.3, COLORS['vermillion'])
    ]

    for idx, (title, bias, var, color) in enumerate(scenarios):
        ax = axes[idx]

        # Draw target
        circle1 = Circle((0, 0), 3, color='lightgray', alpha=0.3)
        circle2 = Circle((0, 0), 2, color='gray', alpha=0.3)
        circle3 = Circle((0, 0), 1, color='darkgray', alpha=0.3)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.add_patch(circle3)

        # Draw center
        ax.plot(0, 0, 'r*', markersize=20, label='True Value')

        # Draw estimates
        estimates_x = rng.normal(bias, var, 100)
        estimates_y = rng.normal(0, var, 100)
        ax.scatter(estimates_x, estimates_y, alpha=0.6, s=40, color=color,
                   edgecolors='black', linewidths=0.5, label='Estimates')

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
        ax.legend(loc='upper right', frameon=True, fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bias_variance_conceptual.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Generated: bias_variance_conceptual.png")



