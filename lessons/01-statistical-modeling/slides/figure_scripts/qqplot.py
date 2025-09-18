import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

OUTDIR = Path(__file__).parent.parent / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Deterministic randomness for reproducible sampling
rng = np.random.default_rng(42)


def qq_plots():
    """Generate all QQ-plot figures for the slides"""

    # Slide 1: Normal CDF with quantiles highlighted
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    x = np.linspace(-3, 3, 1000)
    cdf = stats.norm.cdf(x)
    ax.plot(x, cdf, 'b-', linewidth=2, label='Normal(0,1) CDF')

    # Highlight 25%, 50%, 75% quantiles
    quantiles = [0.25, 0.5, 0.75]
    q_values = stats.norm.ppf(quantiles)
    colors = ['red', 'green', 'orange']

    for i, (p, q, color) in enumerate(zip(quantiles, q_values, colors)):
        # Horizontal line at p
        ax.hlines(p, -3, q, colors=color, linestyles='--', alpha=0.7)
        # Vertical line down to x-axis
        ax.vlines(q, 0, p, colors=color, linestyles='--', alpha=0.7)
        # Mark the quantile
        ax.plot(q, p, 'o', color=color, markersize=8)
        ax.text(q, -0.1, f'q_{{{int(p*100)}}}', ha='center', fontsize=10, color=color)

    ax.set_xlabel('x')
    ax.set_ylabel('F(x)')
    ax.set_title('Quantiles divide the distribution into portions')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUTDIR / "qq_quantiles_definition.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Slide 2: Comparing quantiles of Normal vs Uniform
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    x = np.linspace(-3, 3, 1000)
    cdf_norm = stats.norm.cdf(x)
    ax.plot(x, cdf_norm, 'b-', linewidth=2, label='Normal(0,1)')

    # Create proper Uniform(0,1) CDF across the full x range
    cdf_unif = np.zeros_like(x)
    cdf_unif[x < 0] = 0.0      # CDF = 0 for x < 0
    mask = (x >= 0) & (x <= 1)  # CDF = x for 0 <= x <= 1
    cdf_unif[mask] = x[mask]
    cdf_unif[x > 1] = 1.0      # CDF = 1 for x > 1
    ax.plot(x, cdf_unif, 'r-', linewidth=2, label='Uniform(0,1)')

    # Show quantiles for both
    quantiles = [0.25, 0.5, 0.75]
    q_norm = stats.norm.ppf(quantiles)
    q_unif = quantiles  # For Uniform(0,1), quantiles = probabilities

    for i, (p, qn, qu) in enumerate(zip(quantiles, q_norm, q_unif)):
        ax.plot(qn, p, 'bo', markersize=6)
        ax.plot(qu, p, 'ro', markersize=6)
        ax.text(qn, p + 0.05, f'{qn:.2f}', ha='center', fontsize=8, color='blue')
        ax.text(qu, p + 0.05, f'{qu:.2f}', ha='center', fontsize=8, color='red')

    ax.set_xlabel('x')
    ax.set_ylabel('F(x)')
    ax.set_title('Comparing quantiles = comparing distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "qq_comparing_quantiles.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Slide 3: QQ-Plot concept diagram
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    # Perfect diagonal line
    x_diag = np.linspace(-2, 2, 100)
    ax.plot(x_diag, x_diag, 'r--', linewidth=2, label='y = x (perfect match)')

    # Some sample points near the line
    n_points = 20
    x_points = np.linspace(-1.5, 1.5, n_points)
    y_points = x_points + rng.normal(0, 0.1, n_points)  # Small noise
    ax.scatter(x_points, y_points, alpha=0.6, s=30)

    ax.set_xlabel('Quantiles of Distribution A')
    ax.set_ylabel('Quantiles of Distribution B')
    ax.set_title('Straight line → matching distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(OUTDIR / "qq_concept_diagram.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Slide 4: Theoretical Normal vs Normal
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    quantiles = np.linspace(0.01, 0.99, 50)
    q_norm1 = stats.norm.ppf(quantiles, loc=0, scale=1)
    q_norm2 = stats.norm.ppf(quantiles, loc=0, scale=1)  # Same distribution

    ax.scatter(q_norm1, q_norm2, alpha=0.6, s=30)
    ax.plot([-3, 3], [-3, 3], 'r--', linewidth=2, label='y = x')
    ax.set_xlabel('Normal(0,1) Quantiles')
    ax.set_ylabel('Normal(0,1) Quantiles')
    ax.set_title('Same distributions → perfect straight line')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(OUTDIR / "qq_normal_vs_normal_theoretical.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Slide 5: Empirical Normal vs Normal (Sample Size Comparison)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=200)

    # Small sample size (n=50)
    n_small = 50
    sample1_small = rng.normal(0, 1, n_small)
    sample2_small = rng.normal(0, 1, n_small)

    q_empirical_small = np.linspace(0, 1, n_small+1)[1:-1]
    q1_small = np.quantile(sample1_small, q_empirical_small)
    q2_small = np.quantile(sample2_small, q_empirical_small)

    ax1.scatter(q1_small, q2_small, alpha=0.7, s=25, color='blue')
    ax1.plot([-3, 3], [-3, 3], 'r--', linewidth=2, label='y = x')
    ax1.set_xlabel('Sample 1 Quantiles')
    ax1.set_ylabel('Sample 2 Quantiles')
    ax1.set_title(f'Small Sample (n={n_small})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)

    # Large sample size (n=1000)
    n_large = 1000
    sample1_large = rng.normal(0, 1, n_large)
    sample2_large = rng.normal(0, 1, n_large)

    q_empirical_large = np.linspace(0, 1, n_large+1)[1:-1]
    q1_large = np.quantile(sample1_large, q_empirical_large)
    q2_large = np.quantile(sample2_large, q_empirical_large)

    ax2.scatter(q1_large, q2_large, alpha=0.7, s=8, color='green')
    ax2.plot([-3, 3], [-3, 3], 'r--', linewidth=2, label='y = x')
    ax2.set_xlabel('Sample 1 Quantiles')
    ax2.set_ylabel('Sample 2 Quantiles')
    ax2.set_title(f'Large Sample (n={n_large})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)

    plt.suptitle('Same distribution, sample noise decreases with larger n', y=1.02)
    fig.tight_layout()
    fig.savefig(OUTDIR / "qq_normal_vs_normal_empirical.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Slide 6: Theoretical Normal vs Uniform
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    quantiles = np.linspace(0.01, 0.99, 50)
    q_norm = stats.norm.ppf(quantiles)
    q_unif = stats.uniform.ppf(quantiles, loc=0, scale=1)

    ax.scatter(q_norm, q_unif, alpha=0.6, s=30)
    ax.plot([-3, 3], [-3, 3], 'r--', linewidth=2, label='y = x')
    ax.set_xlabel('Normal(0,1) Quantiles')
    ax.set_ylabel('Uniform(0,1) Quantiles')
    ax.set_title('Different distributions → systematic curvature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "qq_normal_vs_uniform_theoretical.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Slide 7: Empirical Normal vs Exponential (Sample Size Comparison)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=200)

    # Small sample size (n=50)
    n_small = 50
    sample_norm_small = rng.normal(0, 1, n_small)
    sample_exp_small = rng.exponential(1, n_small)

    q_empirical_small = np.linspace(0, 1, n_small+1)[1:-1]
    q_norm_emp_small = np.quantile(sample_norm_small, q_empirical_small)
    q_exp_emp_small = np.quantile(sample_exp_small, q_empirical_small)

    ax1.scatter(q_norm_emp_small, q_exp_emp_small, alpha=0.7, s=25, color='blue')
    ax1.plot([-3, 4], [-3, 4], 'r--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Normal Sample Quantiles')
    ax1.set_ylabel('Exponential Sample Quantiles')
    ax1.set_title(f'Small Sample (n={n_small})')
    ax1.grid(True, alpha=0.3)

    # Large sample size (n=1000)
    n_large = 1000
    sample_norm_large = rng.normal(0, 1, n_large)
    sample_exp_large = rng.exponential(1, n_large)

    q_empirical_large = np.linspace(0, 1, n_large+1)[1:-1]
    q_norm_emp_large = np.quantile(sample_norm_large, q_empirical_large)
    q_exp_emp_large = np.quantile(sample_exp_large, q_empirical_large)

    ax2.scatter(q_norm_emp_large, q_exp_emp_large, alpha=0.7, s=8, color='green')
    ax2.plot([-3, 4], [-3, 4], 'r--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Normal Sample Quantiles')
    ax2.set_ylabel('Exponential Sample Quantiles')
    ax2.set_title(f'Large Sample (n={n_large})')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Heavy-tail differences clearer with larger samples', y=1.02)
    fig.tight_layout()
    fig.savefig(OUTDIR / "qq_normal_vs_exponential_empirical.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # NEW Slide: Empirical vs Theoretical Normal (Sample Size Comparison)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=200)

    # Small sample empirical vs theoretical normal
    n_small = 50
    sample_norm_small = rng.normal(0, 1, n_small)

    # Get empirical quantiles
    q_levels = np.linspace(0.01, 0.99, n_small)
    empirical_quantiles_small = np.quantile(sample_norm_small, q_levels)
    theoretical_quantiles = stats.norm.ppf(q_levels)

    ax1.scatter(theoretical_quantiles, empirical_quantiles_small, alpha=0.7, s=25, color='orange')
    ax1.plot([-3, 3], [-3, 3], 'r--', linewidth=2, label='Perfect match')
    ax1.set_xlabel('Theoretical Normal Quantiles')
    ax1.set_ylabel('Empirical Sample Quantiles')
    ax1.set_title(f'Small Sample (n={n_small})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)

    # Large sample empirical vs theoretical normal
    n_large = 1000
    sample_norm_large = rng.normal(0, 1, n_large)

    q_levels_large = np.linspace(0.01, 0.99, 200)  # Subsample for visualization
    empirical_quantiles_large = np.quantile(sample_norm_large, q_levels_large)
    theoretical_quantiles_large = stats.norm.ppf(q_levels_large)

    ax2.scatter(theoretical_quantiles_large, empirical_quantiles_large, alpha=0.7, s=8, color='red')
    ax2.plot([-3, 3], [-3, 3], 'r--', linewidth=2, label='Perfect match')
    ax2.set_xlabel('Theoretical Normal Quantiles')
    ax2.set_ylabel('Empirical Sample Quantiles')
    ax2.set_title(f'Large Sample (n={n_large})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)

    plt.suptitle('Empirical vs Theoretical: checking normality assumption', y=1.02)
    fig.tight_layout()
    fig.savefig(OUTDIR / "qq_empirical_vs_theoretical_normal.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Slide 8: Four thumbnails summary
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=150)

    # Top left: Same vs same
    quantiles = np.linspace(0.01, 0.99, 30)
    q_norm1 = stats.norm.ppf(quantiles)
    q_norm2 = stats.norm.ppf(quantiles)
    axes[0,0].scatter(q_norm1, q_norm2, alpha=0.6, s=15)
    axes[0,0].plot([-3, 3], [-3, 3], 'r--', linewidth=1)
    axes[0,0].set_title('Same vs Same', fontsize=10)
    axes[0,0].grid(True, alpha=0.3)

    # Top right: Empirical vs empirical
    n = 100
    sample1 = rng.normal(0, 1, n)
    sample2 = rng.normal(0, 1, n)
    q_emp = np.linspace(0.05, 0.95, n//2)
    q1 = np.quantile(sample1, q_emp)
    q2 = np.quantile(sample2, q_emp)
    axes[0,1].scatter(q1, q2, alpha=0.6, s=15)
    axes[0,1].plot([-3, 3], [-3, 3], 'r--', linewidth=1)
    axes[0,1].set_title('Empirical vs Empirical', fontsize=10)
    axes[0,1].grid(True, alpha=0.3)

    # Bottom left: Different vs different
    q_norm = stats.norm.ppf(quantiles)
    q_unif = stats.uniform.ppf(quantiles)
    axes[1,0].scatter(q_norm, q_unif, alpha=0.6, s=15)
    axes[1,0].plot([-3, 3], [-3, 3], 'r--', linewidth=1)
    axes[1,0].set_title('Different Distributions', fontsize=10)
    axes[1,0].grid(True, alpha=0.3)

    # Bottom right: Normal vs exponential
    sample_norm = rng.normal(0, 1, n)
    sample_exp = rng.exponential(1, n)
    q_norm_emp = np.quantile(sample_norm, q_emp)
    q_exp_emp = np.quantile(sample_exp, q_emp)
    axes[1,1].scatter(q_norm_emp, q_exp_emp, alpha=0.6, s=15)
    axes[1,1].plot([-3, 4], [-3, 4], 'r--', linewidth=1, alpha=0.7)
    axes[1,1].set_title('Normal vs Exponential', fontsize=10)
    axes[1,1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.tick_params(labelsize=8)

    fig.suptitle('QQ-plots are diagnostics for distributional comparison', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / "qq_summary_thumbnails.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print("Generated all QQ-plot figures")