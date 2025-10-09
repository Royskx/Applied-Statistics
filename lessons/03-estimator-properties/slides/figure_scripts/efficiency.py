"""Figure generation for Efficiency.

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
    # Figure 3: CRLB Achievement (Enhanced)
    # ============================================================================
    print("[3/8] Generating enhanced CRLB achievement figure...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Cramér-Rao Lower Bound Achievement', fontsize=24, fontweight='bold', y=0.995)

    # Panel 1: Normal Mean
    ax = axes[0, 0]
    mu_true = 10
    sigma = 2
    n_vals = np.arange(10, 201, 5)
    sample_var = []
    crlb_var = []

    for n in n_vals:
        means = []
        for _ in range(1000):
            sample = rng.normal(mu_true, sigma, n)
            means.append(np.mean(sample))
        sample_var.append(np.var(means))
        crlb_var.append(sigma**2 / n)

    ax.plot(n_vals, sample_var, color=COLORS['blue'], linewidth=3,
            marker='o', markersize=5, label='Sample Variance', alpha=0.8)
    ax.plot(n_vals, crlb_var, color=COLORS['vermillion'], linewidth=3,
            linestyle='--', label='CRLB', alpha=0.8)
    ax.fill_between(n_vals, sample_var, crlb_var, where=np.array(sample_var)>=np.array(crlb_var),
                    alpha=0.2, color=COLORS['green'], label='Difference')
    ax.set_xlabel('Sample Size n', fontsize=16)
    ax.set_ylabel('Variance', fontsize=16)
    ax.set_title('(A) Normal Mean: MLE Achieves CRLB', fontsize=18, pad=10)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel 2: Poisson Rate
    ax = axes[0, 1]
    lambda_true = 5
    n_vals = np.arange(10, 201, 5)
    sample_var = []
    crlb_var = []

    for n in n_vals:
        lambdas = []
        for _ in range(1000):
            sample = rng.poisson(lambda_true, n)
            lambdas.append(np.mean(sample))
        sample_var.append(np.var(lambdas))
        crlb_var.append(lambda_true / n)

    ax.plot(n_vals, sample_var, color=COLORS['blue'], linewidth=3,
            marker='s', markersize=5, label='Sample Variance', alpha=0.8)
    ax.plot(n_vals, crlb_var, color=COLORS['vermillion'], linewidth=3,
            linestyle='--', label='CRLB', alpha=0.8)
    ax.fill_between(n_vals, sample_var, crlb_var, where=np.array(sample_var)>=np.array(crlb_var),
                    alpha=0.2, color=COLORS['green'])
    ax.set_xlabel('Sample Size n', fontsize=16)
    ax.set_ylabel('Variance', fontsize=16)
    ax.set_title('(B) Poisson Rate: MLE Achieves CRLB', fontsize=18, pad=10)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel 3: Exponential Rate
    ax = axes[1, 0]
    lambda_true = 2
    n_vals = np.arange(10, 201, 5)
    sample_var = []
    crlb_var = []

    for n in n_vals:
        lambdas = []
        for _ in range(1000):
            sample = rng.exponential(1/lambda_true, n)
            lambdas.append(n / np.sum(sample))  # MLE for rate
        sample_var.append(np.var(lambdas))
        crlb_var.append(lambda_true**2 / n)

    ax.plot(n_vals, sample_var, color=COLORS['blue'], linewidth=3,
            marker='^', markersize=5, label='Sample Variance', alpha=0.8)
    ax.plot(n_vals, crlb_var, color=COLORS['vermillion'], linewidth=3,
            linestyle='--', label='CRLB', alpha=0.8)
    ax.fill_between(n_vals, sample_var, crlb_var, where=np.array(sample_var)>=np.array(crlb_var),
                    alpha=0.2, color=COLORS['green'])
    ax.set_xlabel('Sample Size n', fontsize=16)
    ax.set_ylabel('Variance', fontsize=16)
    ax.set_title('(C) Exponential Rate: MLE Achieves CRLB', fontsize=18, pad=10)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel 4: Efficiency Comparison
    ax = axes[1, 1]
    n = 100
    distributions = ['Normal', 'Poisson', 'Exponential']
    efficiencies = []

    for dist in distributions:
        if dist == 'Normal':
            theta_true = 10
            sigma = 2
            estimates = [np.mean(rng.normal(theta_true, sigma, n)) for _ in range(5000)]
            crlb = sigma**2 / n
        elif dist == 'Poisson':
            theta_true = 5
            estimates = [np.mean(rng.poisson(theta_true, n)) for _ in range(5000)]
            crlb = theta_true / n
        else:  # Exponential
            theta_true = 2
            estimates = [n / np.sum(rng.exponential(1/theta_true, n)) for _ in range(5000)]
            crlb = theta_true**2 / n

        efficiency = crlb / np.var(estimates)
        efficiencies.append(efficiency * 100)

    colors_list = [COLORS['blue'], COLORS['orange'], COLORS['green']]
    bars = ax.bar(distributions, efficiencies, color=colors_list,
                  edgecolor='black', linewidth=2, alpha=0.8)
    ax.axhline(100, color='red', linewidth=3, linestyle='--',
               label='100% (Achieves CRLB)', zorder=10)
    ax.set_ylabel('Efficiency (%)', fontsize=16)
    ax.set_title('(D) MLE Efficiency (n=100)', fontsize=18, pad=10)
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(95, 105)

    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{eff:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'crlb_achievement.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Generated: crlb_achievement.png")

    # ============================================================================


    # Figure 7: Fisher Information visualization
    print("[7/8] Generating Fisher information visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Fisher Information and Precision', fontsize=24, fontweight='bold', y=0.995)

    # Panel 1: Normal distribution with different variances
    ax = axes[0, 0]
    x = np.linspace(-10, 10, 500)
    sigmas = [0.5, 1, 2, 3]
    colors_fi = [COLORS['vermillion'], COLORS['orange'], COLORS['blue'], COLORS['skyblue']]

    for sigma, color in zip(sigmas, colors_fi):
        pdf = stats.norm.pdf(x, 0, sigma)
        fisher_info = 1 / sigma**2
        ax.plot(x, pdf, linewidth=3, color=color, alpha=0.8,
                label=f'σ={sigma}, I={fisher_info:.2f}')

    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.set_title('(A) Fisher Info ∝ 1/σ² (Normal)', fontsize=18, pad=10)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel 2: Poisson distribution with different rates
    ax = axes[0, 1]
    x_vals = np.arange(0, 25)
    lambdas = [2, 5, 10, 15]
    colors_pois = [COLORS['vermillion'], COLORS['orange'], COLORS['blue'], COLORS['green']]

    for lam, color in zip(lambdas, colors_pois):
        pmf = stats.poisson.pmf(x_vals, lam)
        fisher_info = 1 / lam
        ax.plot(x_vals, pmf, linewidth=3, marker='o', markersize=5,
                color=color, alpha=0.8, label=f'λ={lam}, I={fisher_info:.3f}')

    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('Probability', fontsize=16)
    ax.set_title('(B) Fisher Info = 1/λ (Poisson)', fontsize=18, pad=10)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel 3: Likelihood curvature
    ax = axes[1, 0]
    theta_vals = np.linspace(8, 12, 200)
    n = 30
    x_bar = 10

    # Log-likelihood for Normal mean (known sigma=2)
    sigma = 2
    log_lik = -n/(2*sigma**2) * (theta_vals - x_bar)**2

    ax.plot(theta_vals, log_lik, linewidth=3, color=COLORS['blue'], label='Log-Likelihood')
    ax.axvline(x_bar, color='red', linewidth=2, linestyle='--', label='MLE')

    # Second derivative (curvature) = -n/sigma^2
    ax2 = ax.twinx()
    fisher_info_vals = np.full_like(theta_vals, n/sigma**2)
    ax2.plot(theta_vals, fisher_info_vals, linewidth=3, color=COLORS['orange'],
             linestyle='--', alpha=0.7, label='Fisher Information')

    ax.set_xlabel('θ (Mean)', fontsize=16)
    ax.set_ylabel('Log-Likelihood', fontsize=16, color=COLORS['blue'])
    ax2.set_ylabel('Fisher Information', fontsize=16, color=COLORS['orange'])
    ax.set_title('(C) Likelihood Curvature = Information', fontsize=18, pad=10)
    ax.legend(loc='lower left', frameon=True)
    ax2.legend(loc='lower right', frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel 4: CRLB vs sample size
    ax = axes[1, 1]
    n_vals = np.arange(5, 201, 5)
    sigma = 2
    crlb_vals = sigma**2 / n_vals

    ax.fill_between(n_vals, 0, crlb_vals, alpha=0.4, color=COLORS['blue'],
                    label='Var(θ̂) ≥ CRLB')
    ax.plot(n_vals, crlb_vals, linewidth=3, color=COLORS['vermillion'],
            label='CRLB = σ²/n')

    ax.set_xlabel('Sample Size n', fontsize=16)
    ax.set_ylabel('Minimum Variance (CRLB)', fontsize=16)
    ax.set_title('(D) CRLB Decreases with n', fontsize=18, pad=10)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fisher_information_visualization.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Generated: fisher_information_visualization.png")



