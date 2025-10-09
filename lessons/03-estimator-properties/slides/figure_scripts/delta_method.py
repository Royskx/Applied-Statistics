"""Figure generation for Delta Method.

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
    # Figure 9: Delta Method Illustration
    print("[9/9] Generating delta method illustration...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Delta Method: CI for Log-Odds', fontsize=24, fontweight='bold', y=0.995)

    # Simulation parameters
    true_p = 0.3
    n_sim = 100
    n_samples = 1000
    rng_delta = np.random.default_rng(2025)

    # Panel 1: Sampling distribution of p-hat
    ax = axes[0, 0]
    p_hats = []
    for _ in range(n_samples):
        x = rng_delta.binomial(n_sim, true_p)
        p_hats.append(x / n_sim)

    ax.hist(p_hats, bins=40, density=True, alpha=0.7, color=COLORS['blue'],
            edgecolor='black', linewidth=1.5, label='Empirical')

    # Theoretical normal distribution
    p_mean = true_p
    p_std = np.sqrt(true_p * (1 - true_p) / n_sim)
    x_p = np.linspace(true_p - 4*p_std, true_p + 4*p_std, 200)
    ax.plot(x_p, stats.norm.pdf(x_p, p_mean, p_std), linewidth=3,
            color=COLORS['vermillion'], label='Normal Approx')
    ax.axvline(true_p, color='red', linewidth=2, linestyle='--', label=f'True p={true_p}')

    ax.set_xlabel(r'$\hat{p}$', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.set_title(f'(A) Distribution of $\\hat{{p}}$ (n={n_sim})', fontsize=18, pad=10)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel 2: Transformation function g(p) = log(p/(1-p))
    ax = axes[0, 1]
    p_vals = np.linspace(0.05, 0.95, 200)
    log_odds = np.log(p_vals / (1 - p_vals))

    ax.plot(p_vals, log_odds, linewidth=3, color=COLORS['blue'], label=r'$g(p) = \log(p/(1-p))$')
    ax.axvline(true_p, color='red', linewidth=2, linestyle='--', alpha=0.7)
    ax.axhline(np.log(true_p / (1 - true_p)), color='red', linewidth=2, linestyle='--', alpha=0.7,
               label=f'$g({true_p})={np.log(true_p/(1-true_p)):.2f}$')

    # Show tangent line (derivative)
    g_prime = 1 / (true_p * (1 - true_p))
    tangent_x = np.array([true_p - 0.1, true_p + 0.1])
    tangent_y = np.log(true_p / (1 - true_p)) + g_prime * (tangent_x - true_p)
    ax.plot(tangent_x, tangent_y, linewidth=2, color=COLORS['orange'], linestyle='--',
            label=f"Tangent: $g'(p)={g_prime:.2f}$")

    ax.set_xlabel('p', fontsize=16)
    ax.set_ylabel(r'$\log(p/(1-p))$', fontsize=16)
    ax.set_title('(B) Log-Odds Transformation', fontsize=18, pad=10)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel 3: Sampling distribution of g(p-hat)
    ax = axes[1, 0]
    log_odds_hats = np.log(np.array(p_hats) / (1 - np.array(p_hats)))

    ax.hist(log_odds_hats, bins=40, density=True, alpha=0.7, color=COLORS['blue'],
            edgecolor='black', linewidth=1.5, label='Empirical')

    # Theoretical distribution via delta method
    true_log_odds = np.log(true_p / (1 - true_p))
    log_odds_std = np.sqrt(1 / (n_sim * true_p * (1 - true_p)))
    x_lo = np.linspace(true_log_odds - 4*log_odds_std, true_log_odds + 4*log_odds_std, 200)
    ax.plot(x_lo, stats.norm.pdf(x_lo, true_log_odds, log_odds_std), linewidth=3,
            color=COLORS['vermillion'], label='Delta Method Approx')
    ax.axvline(true_log_odds, color='red', linewidth=2, linestyle='--',
               label=f'True={true_log_odds:.2f}')

    ax.set_xlabel(r'$\log(\hat{p}/(1-\hat{p}))$', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.set_title('(C) Distribution of Log-Odds', fontsize=18, pad=10)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel 4: Confidence intervals comparison
    ax = axes[1, 1]

    # Simulate many CIs
    n_ci = 50
    ci_data = []
    for i in range(n_ci):
        x = rng_delta.binomial(n_sim, true_p)
        p_hat = x / n_sim

        # Standard CI for p
        p_ci_low = p_hat - 1.96 * np.sqrt(p_hat * (1 - p_hat) / n_sim)
        p_ci_high = p_hat + 1.96 * np.sqrt(p_hat * (1 - p_hat) / n_sim)
        p_contains = p_ci_low <= true_p <= p_ci_high

        # Delta method CI for log-odds
        log_odds_hat = np.log(p_hat / (1 - p_hat))
        se_log_odds = np.sqrt(1 / (n_sim * p_hat * (1 - p_hat)))
        lo_ci_low = log_odds_hat - 1.96 * se_log_odds
        lo_ci_high = log_odds_hat + 1.96 * se_log_odds
        lo_contains = lo_ci_low <= true_log_odds <= lo_ci_high

        ci_data.append((p_contains, lo_contains))

    p_coverage = sum(p for p, _ in ci_data) / n_ci
    lo_coverage = sum(lo for _, lo in ci_data) / n_ci

    categories = [r'Proportion' + '\n' + r'$\hat{p}$', r'Log-Odds' + '\n' + r'$\log(\hat{p}/(1-\hat{p}))$']
    coverages = [p_coverage, lo_coverage]
    colors_bar = [COLORS['blue'], COLORS['orange']]

    bars = ax.bar(categories, coverages, color=colors_bar, alpha=0.8,
                  edgecolor='black', linewidth=2)
    ax.axhline(0.95, color='red', linewidth=3, linestyle='--', label='95% Target')
    ax.fill_between([-0.5, 1.5], 0.93, 0.97, alpha=0.2, color='green')

    # Add percentage labels on bars
    for bar, cov in zip(bars, coverages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{cov:.1%}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Empirical Coverage', fontsize=16)
    ax.set_title(f'(D) CI Coverage (n={n_ci} trials)', fontsize=18, pad=10)
    ax.set_ylim(0.85, 1.0)
    ax.legend(loc='lower right', frameon=True, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'delta_method_illustration.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Generated: delta_method_illustration.png")

    print("\n" + "=" * 70)
    print("✓ ALL ENHANCED FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nFigures saved in: lessons/03-estimator-properties/figures/")
    print("\nGenerated 9 enhanced figures:")
    print("  1. bias_variance_tradeoff.png")
    print("  2. consistency_demonstration.png")
    print("  3. crlb_achievement.png")
    print("  4. proportion_ci_coverage.png")
    print("  5. bootstrap_median_distribution.png")
    print("  6. bias_variance_conceptual.png")
    print("  7. fisher_information_visualization.png")


