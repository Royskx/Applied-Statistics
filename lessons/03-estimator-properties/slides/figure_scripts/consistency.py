"""Figure generation for Consistency.

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
    # Figure 2: Consistency Demonstration (Enhanced)
    # ============================================================================
    print("[2/8] Generating enhanced consistency demonstration figure...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Consistency of Estimators', fontsize=24, fontweight='bold', y=0.995)

    true_mu = 5
    sample_sizes = [10, 30, 100, 500]

    # Panel 1: Sample Mean Consistency
    ax = axes[0, 0]
    for n in sample_sizes:
        means = []
        for _ in range(1000):
            sample = rng.exponential(true_mu, n)
            means.append(np.mean(sample))

        ax.hist(means, bins=40, alpha=0.5, density=True,
                label=f'n={n}', edgecolor='black', linewidth=0.5)

    ax.axvline(true_mu, color='red', linewidth=3, linestyle='--',
               label='True μ', zorder=10)
    ax.set_xlabel('Sample Mean', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.set_title('(A) Sample Mean → μ', fontsize=18, pad=10)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel 2: Convergence visualization
    ax = axes[0, 1]
    n_range = np.arange(5, 501, 5)
    mean_estimates = []
    std_estimates = []

    for n in n_range:
        means = []
        for _ in range(500):
            sample = rng.exponential(true_mu, n)
            means.append(np.mean(sample))
        mean_estimates.append(np.mean(means))
        std_estimates.append(np.std(means))

    mean_estimates = np.array(mean_estimates)
    std_estimates = np.array(std_estimates)

    ax.plot(n_range, mean_estimates, color=COLORS['blue'], linewidth=3, label='E[X̄ₙ]')
    ax.fill_between(n_range, mean_estimates - 2*std_estimates,
                    mean_estimates + 2*std_estimates,
                    alpha=0.3, color=COLORS['blue'], label='±2 SD')
    ax.axhline(true_mu, color='red', linewidth=3, linestyle='--', label='True μ')
    ax.set_xlabel('Sample Size n', fontsize=16)
    ax.set_ylabel('Estimate', fontsize=16)
    ax.set_title('(B) Convergence to True Value', fontsize=18, pad=10)
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(true_mu - 1, true_mu + 1)

    # Panel 3: Inconsistent Estimator (X₁)
    ax = axes[1, 0]
    n_values = [10, 50, 100, 500, 1000]
    x1_estimates = {n: [] for n in n_values}

    for n in n_values:
        for _ in range(1000):
            sample = rng.exponential(true_mu, n)
            x1_estimates[n].append(sample[0])  # First observation only

    positions = range(len(n_values))
    bp = ax.boxplot([x1_estimates[n] for n in n_values], positions=positions,
                     widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor=COLORS['orange'], alpha=0.7, linewidth=1.5),
                     medianprops=dict(color='black', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))

    ax.axhline(true_mu, color='red', linewidth=3, linestyle='--', label='True μ')
    ax.set_xlabel('Sample Size n', fontsize=16)
    ax.set_ylabel('X₁ (First Observation)', fontsize=16)
    ax.set_title('(C) Inconsistent Estimator', fontsize=18, pad=10)
    ax.set_xticks(positions)
    ax.set_xticklabels(n_values)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: Uniform Max Consistency
    ax = axes[1, 1]
    theta = 10
    n_values = [5, 20, 50, 200]

    for n in n_values:
        max_vals = []
        for _ in range(1000):
            sample = rng.uniform(0, theta, n)
            max_vals.append(np.max(sample))

        ax.hist(max_vals, bins=40, alpha=0.5, density=True,
                label=f'n={n}', edgecolor='black', linewidth=0.5)

    ax.axvline(theta, color='red', linewidth=3, linestyle='--',
               label='True θ', zorder=10)
    ax.set_xlabel('max{Xᵢ}', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.set_title('(D) max{Xᵢ} → θ for Uniform[0,θ]', fontsize=18, pad=10)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, theta + 1)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'consistency_demonstration.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Generated: consistency_demonstration.png")

    # ============================================================================


