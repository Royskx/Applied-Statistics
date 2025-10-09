"""Figure generation for Confidence Intervals.

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
    # Figure 4: Proportion CI Coverage (Enhanced)
    # ============================================================================
    print("[4/8] Generating enhanced proportion CI coverage figure...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Confidence Interval Coverage for Proportions', fontsize=24, fontweight='bold', y=0.995)

    n_sim = 5000
    ps = np.linspace(0.1, 0.9, 17)
    n_values = [20, 50, 100, 200]

    for idx, n in enumerate(n_values):
        ax = axes[idx//2, idx%2]

        wald_coverage = []
        wilson_coverage = []

        for p in ps:
            wald_count = 0
            wilson_count = 0

            for _ in range(n_sim):
                x = rng.binomial(n, p)
                phat = x / n

                # Wald interval
                se = np.sqrt(phat * (1 - phat) / n)
                lo_wald = phat - 1.96 * se
                hi_wald = phat + 1.96 * se
                if lo_wald <= p <= hi_wald:
                    wald_count += 1

                # Wilson interval
                z = 1.96
                denom = 1 + z**2/n
                center = (phat + z**2/(2*n)) / denom
                margin = z * np.sqrt(phat*(1-phat)/n + z**2/(4*n**2)) / denom
                lo_wilson = center - margin
                hi_wilson = center + margin
                if lo_wilson <= p <= hi_wilson:
                    wilson_count += 1

            wald_coverage.append(wald_count / n_sim)
            wilson_coverage.append(wilson_count / n_sim)

        ax.plot(ps, wald_coverage, color=COLORS['blue'], linewidth=3,
                marker='o', markersize=6, label='Wald', alpha=0.8)
        ax.plot(ps, wilson_coverage, color=COLORS['vermillion'], linewidth=3,
                marker='s', markersize=6, label='Wilson', alpha=0.8)
        ax.axhline(0.95, color='black', linewidth=2, linestyle='--',
                   alpha=0.7, label='95% Target')
        ax.fill_between(ps, 0.94, 0.96, alpha=0.2, color='green', label='±1% Band')

        ax.set_xlabel('True Proportion p', fontsize=16)
        ax.set_ylabel('Empirical Coverage', fontsize=16)
        ax.set_title(f'({chr(65+idx)}) n = {n}', fontsize=18, pad=10)
        ax.legend(loc='lower center', frameon=True, ncol=2, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.85, 1.0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'proportion_ci_coverage.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Generated: proportion_ci_coverage.png")

    # ============================================================================


    # Figure 8: CI interpretation diagram
    print("[8/8] Generating CI interpretation diagram...")
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('Confidence Interval Interpretation', fontsize=24, fontweight='bold')

    true_param = 50
    n_intervals = 20
    coverage_level = 0.95

    # Use a specific seed for CI visualization that shows proper ~95% coverage
    rng_ci = np.random.default_rng(101)  # Seed chosen to give 19/20 = 95% coverage
    intervals = []
    colors_int = []

    for i in range(n_intervals):
        # Simulate a sample mean from sampling distribution
        # True sampling distribution: N(true_param, sigma^2/n)
        sample_mean = rng_ci.normal(true_param, 3 / np.sqrt(30))
        se = 3 / np.sqrt(30)  # Known: n=30, sigma=3

        # 95% CI
        ci_low = sample_mean - 1.96 * se
        ci_high = sample_mean + 1.96 * se

        # Check if interval contains true parameter
        contains = ci_low <= true_param <= ci_high
        intervals.append((ci_low, ci_high, sample_mean, contains))

        if contains:
            colors_int.append(COLORS['green'])
        else:
            colors_int.append(COLORS['vermillion'])

    # Plot intervals
    for i, (low, high, mean, contains) in enumerate(intervals):
        ax.plot([low, high], [i, i], linewidth=3, color=colors_int[i], alpha=0.7)
        ax.plot([low, low], [i-0.15, i+0.15], linewidth=2, color=colors_int[i])
        ax.plot([high, high], [i-0.15, i+0.15], linewidth=2, color=colors_int[i])
        ax.scatter([mean], [i], s=80, color='black', zorder=10, marker='o')

    # True parameter line
    ax.axvline(true_param, color='red', linewidth=4, linestyle='--',
               label=f'True Parameter = {true_param}', zorder=5)

    # Count coverage
    n_covered = sum(1 for _, _, _, contains in intervals if contains)
    coverage_rate = n_covered / n_intervals

    ax.set_ylabel('Interval Number', fontsize=16)
    ax.set_xlabel('Parameter Value', fontsize=16)
    ax.set_title(f'20 Independent 95% CIs: {n_covered} contain true value ({coverage_rate:.0%})',
                 fontsize=18, pad=15)
    ax.set_ylim(-1, n_intervals)
    ax.grid(True, alpha=0.3, axis='x')

    # Legend
    green_patch = mpatches.Patch(color=COLORS['green'], label=f'Contains θ (n={n_covered})')
    red_patch = mpatches.Patch(color=COLORS['vermillion'], label=f'Misses θ (n={n_intervals-n_covered})')
    ax.legend(handles=[green_patch, red_patch,
                       Line2D([0], [0], color='red', linewidth=4, linestyle='--', label='True θ')],
              loc='lower right', frameon=True, fontsize=14)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ci_interpretation.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Generated: ci_interpretation.png")



