"""Figure generation for Bootstrap.

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
    # Bootstrap Algorithm Visual
    # ============================================================================
    print("[5a/8] Generating bootstrap algorithm visual...")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # Box 1: Original Sample
    box1_x, box1_y = 0.5, 1.5
    box1_width, box1_height = 2.5, 1.0
    rect1 = mpatches.FancyBboxPatch((box1_x, box1_y), box1_width, box1_height,
                                     boxstyle="round,pad=0.05",
                                     linewidth=3, edgecolor=COLORS['blue'],
                                     facecolor=COLORS['blue'], alpha=0.2)
    ax.add_patch(rect1)
    ax.text(box1_x + box1_width/2, box1_y + box1_height/2 + 0.3,
            'Original Sample', fontsize=18, fontweight='bold', ha='center', va='center')
    ax.text(box1_x + box1_width/2, box1_y + box1_height/2 - 0.2,
            r'$X_1, X_2, \ldots, X_n$', fontsize=16, ha='center', va='center')

    # Box 2: Bootstrap Resample
    box2_x, box2_y = 3.75, 1.5
    box2_width, box2_height = 2.5, 1.0
    rect2 = mpatches.FancyBboxPatch((box2_x, box2_y), box2_width, box2_height,
                                     boxstyle="round,pad=0.05",
                                     linewidth=3, edgecolor=COLORS['green'],
                                     facecolor=COLORS['green'], alpha=0.2)
    ax.add_patch(rect2)
    ax.text(box2_x + box2_width/2, box2_y + box2_height/2 + 0.3,
            'Bootstrap Resample', fontsize=18, fontweight='bold', ha='center', va='center')
    ax.text(box2_x + box2_width/2, box2_y + box2_height/2 - 0.2,
            r'$X_1^*, X_2^*, \ldots, X_n^*$', fontsize=16, ha='center', va='center')

    # Box 3: Bootstrap Distribution
    box3_x, box3_y = 7.0, 1.5
    box3_width, box3_height = 2.5, 1.0
    rect3 = mpatches.FancyBboxPatch((box3_x, box3_y), box3_width, box3_height,
                                     boxstyle="round,pad=0.05",
                                     linewidth=3, edgecolor=COLORS['vermillion'],
                                     facecolor=COLORS['vermillion'], alpha=0.2)
    ax.add_patch(rect3)
    ax.text(box3_x + box3_width/2, box3_y + box3_height/2 + 0.3,
            'Bootstrap Distribution', fontsize=18, fontweight='bold', ha='center', va='center')
    ax.text(box3_x + box3_width/2, box3_y + box3_height/2 - 0.2,
            r'$\hat{\theta}^{*1}, \hat{\theta}^{*2}, \ldots, \hat{\theta}^{*B}$',
            fontsize=16, ha='center', va='center')

    # Arrow 1: Original to Bootstrap
    arrow1_start = box1_x + box1_width
    arrow1_end = box2_x
    arrow1_y = box1_y + box1_height/2
    ax.annotate('', xy=(arrow1_end, arrow1_y), xytext=(arrow1_start, arrow1_y),
                arrowprops=dict(arrowstyle='->', lw=4, color='black'))

    # Arrow 2: Bootstrap to Distribution
    arrow2_start = box2_x + box2_width
    arrow2_end = box3_x
    arrow2_y = box2_y + box2_height/2
    ax.annotate('', xy=(arrow2_end, arrow2_y), xytext=(arrow2_start, arrow2_y),
                arrowprops=dict(arrowstyle='->', lw=4, color='black'))

    # Bottom text: Resampling description
    ax.text(5.0, 0.5, 'Resample with replacement',
            fontsize=16, ha='center', va='center', style='italic')
    ax.text(5.0, 0.1, 'from empirical distribution',
            fontsize=16, ha='center', va='center', style='italic')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bootstrap_algorithm_visual.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Generated: bootstrap_algorithm_visual.png")

    # Figure 5: Bootstrap Distribution (Enhanced)
    # ============================================================================
    print("[5/8] Generating enhanced bootstrap distribution figure...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Bootstrap Method for Inference', fontsize=24, fontweight='bold', y=0.995)

    # Panel 1: Bootstrap distribution of median
    ax = axes[0, 0]
    true_lambda = 2.0
    n = 100
    x = rng.exponential(1/true_lambda, n)
    B = 5000
    boot_medians = np.array([np.median(x[rng.integers(0, n, n)]) for _ in range(B)])

    counts, bins, patches = ax.hist(boot_medians, bins=50, density=True, alpha=0.7,
                                     color=COLORS['skyblue'], edgecolor='black', linewidth=0.8)
    sample_median = np.median(x)
    true_median = np.log(2) / true_lambda

    ax.axvline(sample_median, color=COLORS['vermillion'], linewidth=3,
               label=f'Sample Median = {sample_median:.3f}')
    ax.axvline(true_median, color=COLORS['green'], linewidth=3, linestyle='--',
               label=f'True Median = {true_median:.3f}')

    lo_p, hi_p = np.quantile(boot_medians, [0.025, 0.975])
    ax.axvline(lo_p, color=COLORS['orange'], linewidth=2.5, linestyle='--', alpha=0.8)
    ax.axvline(hi_p, color=COLORS['orange'], linewidth=2.5, linestyle='--', alpha=0.8,
               label=f'95% CI: [{lo_p:.3f}, {hi_p:.3f}]')

    ax.set_xlabel('Median Value', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.set_title('(A) Bootstrap Distribution of Median', fontsize=18, pad=10)
    ax.legend(loc='upper right', frameon=True, fontsize=12)
    ax.grid(True, alpha=0.3)

    # Panel 2: Bootstrap sample size effect
    ax = axes[0, 1]
    B_values = [100, 500, 1000, 5000]
    colors_boot = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['vermillion']]

    for B, color in zip(B_values, colors_boot):
        boot_medians = np.array([np.median(x[rng.integers(0, n, n)]) for _ in range(B)])
        ax.hist(boot_medians, bins=40, density=True, alpha=0.4,
                label=f'B={B}', color=color, edgecolor='black', linewidth=0.5)

    ax.axvline(true_median, color='black', linewidth=3, linestyle='--',
               label='True Median')
    ax.set_xlabel('Median Value', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.set_title('(B) Effect of Bootstrap Iterations B', fontsize=18, pad=10)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel 3: Comparison with parametric CI
    ax = axes[1, 0]
    true_mu = 50
    true_sigma = 10
    n = 30
    x_normal = rng.normal(true_mu, true_sigma, n)

    # Bootstrap CI for mean
    B = 5000
    boot_means = np.array([np.mean(x_normal[rng.integers(0, n, n)]) for _ in range(B)])
    boot_ci = np.quantile(boot_means, [0.025, 0.975])

    # Parametric t-CI
    sample_mean = np.mean(x_normal)
    sample_std = np.std(x_normal, ddof=1)
    t_crit = stats.t.ppf(0.975, n-1)
    t_ci = [sample_mean - t_crit * sample_std / np.sqrt(n),
            sample_mean + t_crit * sample_std / np.sqrt(n)]

    methods = ['Bootstrap\nPercentile', 't-interval']
    lowers = [boot_ci[0], t_ci[0]]
    uppers = [boot_ci[1], t_ci[1]]
    centers = [(boot_ci[0] + boot_ci[1])/2, (t_ci[0] + t_ci[1])/2]
    widths = [boot_ci[1] - boot_ci[0], t_ci[1] - t_ci[0]]

    y_pos = [0, 1]
    colors_ci = [COLORS['skyblue'], COLORS['orange']]

    for i, (method, lower, upper, center, color) in enumerate(zip(methods, lowers, uppers, centers, colors_ci)):
        ax.plot([lower, upper], [y_pos[i], y_pos[i]], linewidth=8, color=color, alpha=0.7)
        ax.plot([lower, lower], [y_pos[i]-0.1, y_pos[i]+0.1], linewidth=3, color=color)
        ax.plot([upper, upper], [y_pos[i]-0.1, y_pos[i]+0.1], linewidth=3, color=color)
        ax.scatter([center], [y_pos[i]], s=150, color='black', zorder=10, marker='o')

    ax.axvline(true_mu, color='red', linewidth=3, linestyle='--',
               label=f'True μ = {true_mu}', zorder=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=14)
    ax.set_xlabel('Value', fontsize=16)
    ax.set_title('(C) Bootstrap vs Parametric CI', fontsize=18, pad=10)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_ylim(-0.5, 1.5)

    # Panel 4: Coverage simulation
    ax = axes[1, 1]
    n_values = [10, 20, 50, 100]
    boot_coverage = []
    t_coverage = []

    for n in n_values:
        boot_hits = 0
        t_hits = 0

        for _ in range(1000):
            sample = rng.normal(true_mu, true_sigma, n)

            # Bootstrap
            boot_means_sim = np.array([np.mean(sample[rng.integers(0, n, n)]) for _ in range(1000)])
            boot_ci_sim = np.quantile(boot_means_sim, [0.025, 0.975])
            if boot_ci_sim[0] <= true_mu <= boot_ci_sim[1]:
                boot_hits += 1

            # t-interval
            mean_sim = np.mean(sample)
            std_sim = np.std(sample, ddof=1)
            t_crit_sim = stats.t.ppf(0.975, n-1)
            t_ci_sim = [mean_sim - t_crit_sim * std_sim / np.sqrt(n),
                        mean_sim + t_crit_sim * std_sim / np.sqrt(n)]
            if t_ci_sim[0] <= true_mu <= t_ci_sim[1]:
                t_hits += 1

        boot_coverage.append(boot_hits / 1000)
        t_coverage.append(t_hits / 1000)

    x = np.arange(len(n_values))
    width = 0.35
    ax.bar(x - width/2, boot_coverage, width, label='Bootstrap',
           color=COLORS['skyblue'], edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.bar(x + width/2, t_coverage, width, label='t-interval',
           color=COLORS['orange'], edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.axhline(0.95, color='black', linewidth=2, linestyle='--',
               label='95% Target')
    ax.fill_between([-0.5, len(n_values)-0.5], 0.94, 0.96,
                    alpha=0.2, color='green')

    ax.set_xlabel('Sample Size n', fontsize=16)
    ax.set_ylabel('Empirical Coverage', fontsize=16)
    ax.set_title('(D) Coverage Comparison', fontsize=18, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(n_values)
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.90, 1.0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bootstrap_median_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Generated: bootstrap_median_distribution.png")

    # ============================================================================

    # Figure 6: BCa Method Visual Explanation
    # ============================================================================
    print("[5b/8] Generating BCa method visual example...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('BCa Bootstrap: Bias-Corrected and Accelerated Intervals',
                 fontsize=24, fontweight='bold', y=0.995)

    # Panel 1: Skewed distribution example (ratio estimator)
    ax = axes[0, 0]
    np.random.seed(2025)
    n = 50
    # Generate skewed data: ratio of two variables
    x_data = rng.lognormal(0, 0.5, n)
    y_data = rng.lognormal(0.3, 0.6, n)
    ratio_obs = np.mean(x_data) / np.mean(y_data)

    # Bootstrap
    B = 5000
    boot_ratios = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        boot_ratios.append(np.mean(x_data[idx]) / np.mean(y_data[idx]))
    boot_ratios = np.array(boot_ratios)

    # Plot bootstrap distribution
    ax.hist(boot_ratios, bins=60, density=True, alpha=0.7,
            color=COLORS['skyblue'], edgecolor='black', linewidth=0.8)
    ax.axvline(ratio_obs, color='black', linewidth=3, linestyle='-',
               label=f'Observed ratio = {ratio_obs:.3f}')

    # Show bias
    boot_mean = np.mean(boot_ratios)
    bias = boot_mean - ratio_obs
    ax.axvline(boot_mean, color=COLORS['orange'], linewidth=3, linestyle='--',
               label=f'Bootstrap mean = {boot_mean:.3f}')

    # Annotate bias
    ax.annotate('', xy=(boot_mean, 0.5), xytext=(ratio_obs, 0.5),
                arrowprops=dict(arrowstyle='<->', lw=2.5, color=COLORS['vermillion']))
    ax.text((ratio_obs + boot_mean)/2, 0.6, f'Bias = {bias:.3f}',
            fontsize=14, ha='center', bbox=dict(boxstyle='round',
            facecolor=COLORS['vermillion'], alpha=0.3))

    ax.set_xlabel('Ratio Value', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.set_title('(A) Bootstrap Distribution Shows Bias', fontsize=18, pad=10)
    ax.legend(loc='upper right', frameon=True, fontsize=12)
    ax.grid(True, alpha=0.3)

    # Panel 2: Comparison of CI methods
    ax = axes[0, 1]

    # Percentile CI (naive)
    percentile_ci = np.quantile(boot_ratios, [0.025, 0.975])

    # Basic CI
    basic_ci = [2*ratio_obs - percentile_ci[1], 2*ratio_obs - percentile_ci[0]]

    # BCa CI (simplified calculation)
    # Bias correction factor
    z0 = stats.norm.ppf(np.mean(boot_ratios < ratio_obs))

    # Jackknife for acceleration
    jack_ratios = []
    for i in range(n):
        idx = np.delete(np.arange(n), i)
        jack_ratios.append(np.mean(x_data[idx]) / np.mean(y_data[idx]))
    jack_ratios = np.array(jack_ratios)
    jack_mean = np.mean(jack_ratios)

    # Acceleration constant
    numerator = np.sum((jack_mean - jack_ratios)**3)
    denominator = 6 * (np.sum((jack_mean - jack_ratios)**2))**1.5
    a = numerator / denominator if denominator != 0 else 0

    # Adjusted percentiles
    z_alpha = stats.norm.ppf([0.025, 0.975])
    alpha_adjusted = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    alpha_adjusted = np.clip(alpha_adjusted, 0.001, 0.999)
    bca_ci = np.quantile(boot_ratios, alpha_adjusted)

    # Plot intervals
    methods = ['Percentile', 'Basic', 'BCa']
    cis = [percentile_ci, basic_ci, bca_ci]
    y_positions = [2, 1, 0]
    colors_methods = [COLORS['blue'], COLORS['orange'], COLORS['green']]

    for i, (method, ci, y_pos, color) in enumerate(zip(methods, cis, y_positions, colors_methods)):
        ax.plot([ci[0], ci[1]], [y_pos, y_pos], linewidth=10,
                color=color, alpha=0.7, label=method)
        ax.plot([ci[0], ci[0]], [y_pos-0.15, y_pos+0.15], linewidth=3, color=color)
        ax.plot([ci[1], ci[1]], [y_pos-0.15, y_pos+0.15], linewidth=3, color=color)

        # Add CI text
        ax.text(ci[0] - 0.02, y_pos + 0.3, f'{ci[0]:.3f}',
                fontsize=11, ha='right', va='center')
        ax.text(ci[1] + 0.02, y_pos + 0.3, f'{ci[1]:.3f}',
                fontsize=11, ha='left', va='center')

    ax.axvline(ratio_obs, color='black', linewidth=3, linestyle='--',
               label=f'Observed = {ratio_obs:.3f}', zorder=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(methods, fontsize=14)
    ax.set_xlabel('Ratio Value', fontsize=16)
    ax.set_title('(B) Comparison of 95% Confidence Intervals', fontsize=18, pad=10)
    ax.legend(loc='upper left', frameon=True, fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_ylim(-0.5, 2.5)

    # Panel 3: Bias correction visualization
    ax = axes[1, 0]

    # Show how z0 shifts the percentiles
    x_vals = np.linspace(-3, 3, 1000)
    y_standard = stats.norm.pdf(x_vals)

    ax.plot(x_vals, y_standard, linewidth=3, color='black',
            label='Standard Normal', linestyle='--')

    # Mark standard percentiles
    z_025 = stats.norm.ppf(0.025)
    z_975 = stats.norm.ppf(0.975)
    ax.axvline(z_025, color=COLORS['blue'], linewidth=2.5, alpha=0.7,
               label='Standard 2.5%/97.5%')
    ax.axvline(z_975, color=COLORS['blue'], linewidth=2.5, alpha=0.7)

    # Mark BCa-adjusted percentiles
    ax.axvline(z0 + z_025, color=COLORS['green'], linewidth=2.5, alpha=0.7,
               label=f'BCa adjusted (z₀={z0:.2f})')
    ax.axvline(z0 + z_975, color=COLORS['green'], linewidth=2.5, alpha=0.7)

    # Shade regions
    x_fill = np.linspace(z_025, z_975, 100)
    ax.fill_between(x_fill, 0, stats.norm.pdf(x_fill),
                    alpha=0.2, color=COLORS['blue'], label='Standard 95%')

    x_fill_bca = np.linspace(z0 + z_025, z0 + z_975, 100)
    ax.fill_between(x_fill_bca, 0, stats.norm.pdf(x_fill_bca),
                    alpha=0.2, color=COLORS['green'], label='BCa 95%')

    ax.set_xlabel('Standardized Value', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.set_title('(C) Bias Correction: Shifting Percentiles', fontsize=18, pad=10)
    ax.legend(loc='upper right', frameon=True, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)

    # Panel 4: Coverage simulation for skewed data
    ax = axes[1, 1]

    # Simulate coverage for different methods
    n_sim = 500
    coverage_percentile = 0
    coverage_bca = 0
    true_ratio = 1.0  # Known for simulation

    for _ in range(n_sim):
        # Generate skewed data
        x_sim = rng.lognormal(0, 0.5, n)
        y_sim = rng.lognormal(0, 0.5, n)  # Same distribution
        ratio_sim = np.mean(x_sim) / np.mean(y_sim)

        # Bootstrap
        boot_ratios_sim = []
        for _ in range(1000):
            idx = rng.integers(0, n, n)
            boot_ratios_sim.append(np.mean(x_sim[idx]) / np.mean(y_sim[idx]))
        boot_ratios_sim = np.array(boot_ratios_sim)

        # Percentile CI
        perc_ci_sim = np.quantile(boot_ratios_sim, [0.025, 0.975])
        if perc_ci_sim[0] <= true_ratio <= perc_ci_sim[1]:
            coverage_percentile += 1

        # BCa CI (simplified)
        z0_sim = stats.norm.ppf(np.mean(boot_ratios_sim < ratio_sim))
        jack_ratios_sim = []
        for i in range(n):
            idx = np.delete(np.arange(n), i)
            jack_ratios_sim.append(np.mean(x_sim[idx]) / np.mean(y_sim[idx]))
        jack_ratios_sim = np.array(jack_ratios_sim)
        jack_mean_sim = np.mean(jack_ratios_sim)

        num = np.sum((jack_mean_sim - jack_ratios_sim)**3)
        den = 6 * (np.sum((jack_mean_sim - jack_ratios_sim)**2))**1.5
        a_sim = num / den if den != 0 else 0

        alpha_adj_sim = stats.norm.cdf(z0_sim + (z0_sim + z_alpha) / (1 - a_sim * (z0_sim + z_alpha)))
        alpha_adj_sim = np.clip(alpha_adj_sim, 0.001, 0.999)
        bca_ci_sim = np.quantile(boot_ratios_sim, alpha_adj_sim)

        if bca_ci_sim[0] <= true_ratio <= bca_ci_sim[1]:
            coverage_bca += 1

    coverage_percentile /= n_sim
    coverage_bca /= n_sim

    # Bar plot
    methods_cov = ['Percentile', 'BCa']
    coverages = [coverage_percentile, coverage_bca]
    colors_cov = [COLORS['blue'], COLORS['green']]

    bars = ax.bar(methods_cov, coverages, color=colors_cov,
                  edgecolor='black', linewidth=2, alpha=0.8, width=0.6)

    # Add value labels on bars
    for bar, cov in zip(bars, coverages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{cov:.1%}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.axhline(0.95, color='black', linewidth=3, linestyle='--',
               label='95% Target')
    ax.fill_between([-0.5, 1.5], 0.94, 0.96, alpha=0.2, color='green',
                    label='Acceptable range')

    ax.set_ylabel('Empirical Coverage', fontsize=16)
    ax.set_title('(D) Coverage Comparison (Skewed Data)', fontsize=18, pad=10)
    ax.legend(loc='lower right', frameon=True, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.85, 1.0)
    ax.set_xlim(-0.5, 1.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bca_method_visual.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Generated: bca_method_visual.png")

    # ============================================================================


