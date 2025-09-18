"""
Central Limit Theorem (CLT) visualization figures for educational slides.
Generates comprehensive visualizations showing CLT concepts progressively.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
import seaborn as sns

# Set style for consistent educational plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def generate_clt_figures():
    """Generate all CLT visualization figures."""
    np.random.seed(42)  # For reproducible results

    # Figure 1: Motivation - Single simulation trajectory and shrinking Normal curves
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Single coin toss simulation trajectory
    n_tosses = 200
    tosses = np.random.binomial(1, 0.5, n_tosses)
    cumulative_mean = np.cumsum(tosses) / np.arange(1, n_tosses + 1)

    ax1.plot(range(1, n_tosses + 1), cumulative_mean, 'b-', linewidth=2, alpha=0.8)
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='True mean μ = 0.5')
    ax1.set_xlabel('Number of tosses (n)')
    ax1.set_ylabel('Sample mean')
    ax1.set_title('LLN: Sample mean stabilizes around μ')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Shrinking Normal curves showing CLT variance decrease
    x = np.linspace(-0.3, 0.3, 1000)
    sample_sizes = [10, 50, 200]
    colors = ['red', 'orange', 'blue']

    for i, n in enumerate(sample_sizes):
        # For Bernoulli(0.5): variance = p(1-p) = 0.25, so std of sample mean = sqrt(0.25/n)
        std_sample_mean = np.sqrt(0.25 / n)
        density = stats.norm.pdf(x, loc=0, scale=std_sample_mean)
        ax2.plot(x, density, color=colors[i], linewidth=2.5, label=f'n = {n}')

    ax2.set_xlabel('Sample mean deviation from μ')
    ax2.set_ylabel('Density')
    ax2.set_title('CLT: Gaussian fluctuations shrink with 1/√n')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/clt_motivation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Visual demonstration - Histograms of normalized deviations
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    n_simulations = 10000
    sample_sizes = [5, 20, 50]

    for i, n in enumerate(sample_sizes):
        # Generate sample means for fair coin tosses
        sample_means = np.random.binomial(n, 0.5, n_simulations) / n

        # Compute normalized deviations: Z_n = √n * (X̄_n - μ) / σ
        # For Bernoulli(0.5): μ = 0.5, σ = 0.5
        normalized_deviations = np.sqrt(n) * (sample_means - 0.5) / 0.5

        # Create histogram
        axes[i].hist(normalized_deviations, bins=50, density=True, alpha=0.7,
                    color='skyblue', edgecolor='black', linewidth=0.5)

        # Overlay theoretical Normal(0,1) curve
        x_theory = np.linspace(-4, 4, 1000)
        y_theory = stats.norm.pdf(x_theory, 0, 1)
        axes[i].plot(x_theory, y_theory, 'r-', linewidth=3, label='N(0,1)')

        axes[i].set_title(f'n = {n}')
        axes[i].set_xlabel('Normalized deviation Z_n')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(-4, 4)

    fig2.suptitle('CLT Demonstration: Coin Toss Averages Become Gaussian', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/clt_demonstration.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: Informal statement - Clear stochastic process visualization
    fig3, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Generate multiple sample paths to show the stochastic process clearly
    np.random.seed(42)
    n_paths = 8
    n_samples = 100
    mu_true = 0.5  # True mean for Bernoulli(0.5)

    # Panel 1: Multiple sample paths showing convergence to μ
    sample_sizes = np.arange(1, n_samples + 1)
    colors = ['#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#f7fbff'][:n_paths]

    for i in range(n_paths):
        # Generate Bernoulli samples and compute running averages
        samples = np.random.binomial(1, mu_true, n_samples)
        running_averages = np.cumsum(samples) / sample_sizes
        axes[0].plot(sample_sizes, running_averages, color=colors[i], alpha=0.7, linewidth=1.5)

    axes[0].axhline(y=mu_true, color='red', linestyle='--', linewidth=3, label='True mean μ = 0.5')
    axes[0].set_xlabel('Sample size n')
    axes[0].set_ylabel('Sample mean X̄_n')
    axes[0].set_title('LLN: Sample paths\nconverge to μ')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.2, 0.8)

    # Panel 2: Same paths after centering and scaling by √n
    for i in range(n_paths):
        samples = np.random.binomial(1, mu_true, n_samples)
        running_averages = np.cumsum(samples) / sample_sizes
        # Center and scale: √n × (X̄_n - μ)
        scaled_deviations = np.sqrt(sample_sizes) * (running_averages - mu_true)
        axes[1].plot(sample_sizes, scaled_deviations, color=colors[i], alpha=0.7, linewidth=1.5)

    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=3)
    axes[1].set_xlabel('Sample size n')
    axes[1].set_ylabel('√n × (X̄_n - μ)')
    axes[1].set_title('CLT scaling: √n × (X̄_n - μ)\nFluctuations become stable')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-4, 4)

    # Panel 3: Histogram showing Gaussian shape at large n
    np.random.seed(42)
    n_large = 100
    n_simulations = 2000

    # Generate many scaled deviations at large sample size
    scaled_values = []
    for _ in range(n_simulations):
        samples = np.random.binomial(n_large, mu_true, 1)[0]
        sample_mean = samples / n_large
        scaled_deviation = np.sqrt(n_large) * (sample_mean - mu_true)
        scaled_values.append(scaled_deviation)

    axes[2].hist(scaled_values, bins=40, density=True, alpha=0.7, color='lightblue',
                edgecolor='black', linewidth=0.5, label='Empirical')

    # Overlay theoretical N(0, σ²) curve
    # For Bernoulli(p): Var(X) = p(1-p) = 0.5×0.5 = 0.25, so σ = 0.5
    x_theory = np.linspace(-4, 4, 1000)
    y_theory = stats.norm.pdf(x_theory, 0, 0.5)  # N(0, 0.25) has std = 0.5
    axes[2].plot(x_theory, y_theory, 'r-', linewidth=3, label='N(0, 0.25)')

    axes[2].set_xlabel('√n × (X̄_n - μ)')
    axes[2].set_ylabel('Density')
    axes[2].set_title('CLT result: Gaussian shape\nat large n')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(-4, 4)

    plt.tight_layout()
    plt.savefig('figures/clt_informal.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 4: Formal statement - Distribution convergence visualization
    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Sample mean distributions for different n
    x = np.linspace(1.5, 2.5, 1000)
    mu = 2.0  # Population mean
    sample_sizes = [5, 20, 100]
    colors = ['red', 'orange', 'blue']

    for i, n in enumerate(sample_sizes):
        # Sample mean distribution: X̄_n ~ N(μ, σ²/n)
        sigma = 0.5  # Population standard deviation
        std_sample_mean = sigma / np.sqrt(n)
        density = stats.norm.pdf(x, loc=mu, scale=std_sample_mean)
        ax1.plot(x, density, color=colors[i], linewidth=2.5, label=f'n = {n}')

    ax1.axvline(x=mu, color='black', linestyle='--', linewidth=2, label='μ')
    ax1.set_xlabel('Sample mean X̄_n')
    ax1.set_ylabel('Density')
    ax1.set_title('Sample mean distributions\n(narrower as n increases)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Normalized version converging to N(0,1)
    x_norm = np.linspace(-4, 4, 1000)
    standard_normal = stats.norm.pdf(x_norm, 0, 1)

    # Show the target N(0,1) distribution
    ax2.plot(x_norm, standard_normal, 'black', linewidth=4, label='N(0,1) limit')
    ax2.fill_between(x_norm, standard_normal, alpha=0.15, color='gray')

    # Generate empirical distributions for normalized sample means
    n_simulations = 10000
    colors = ['#d62728', '#ff7f0e', '#2ca02c']  # red, orange, green
    line_styles = ['-', '--', '-.']

    # Use exponential distribution with rate λ=0.5 for clear non-normal population
    # Mean = 2, Variance = 4, so standardized CLT should converge to N(0,1)
    population_mean = 2.0
    population_std = 2.0  # sqrt(4)

    for i, n in enumerate(sample_sizes):
        # Generate many samples and compute their means
        sample_means = []
        for _ in range(n_simulations):
            # Generate n samples from Exponential(0.5) distribution
            samples = np.random.exponential(scale=2.0, size=n)  # scale=1/rate, so scale=2 gives rate=0.5
            sample_means.append(np.mean(samples))

        sample_means = np.array(sample_means)

        # Normalize: (X̄_n - μ) / (σ/√n)
        normalized_means = (sample_means - population_mean) / (population_std / np.sqrt(n))

        # Create kernel density estimate for smooth curve
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(normalized_means)
        density = kde(x_norm)

        # Plot empirical distribution
        ax2.plot(x_norm, density, color=colors[i], linewidth=3,
                alpha=0.8, linestyle=line_styles[i],
                label=f'n = {n} (empirical)')

    ax2.set_xlabel('Normalized: (X̄_n - μ)/(σ/√n)')
    ax2.set_ylabel('Density')
    ax2.set_title('Empirical distributions\nconverge to N(0,1)')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.45)  # Set consistent y-axis for better comparison
    ax2.set_xlim(-3.5, 3.5)  # Focus on main region

    plt.tight_layout()
    plt.savefig('figures/clt_formal.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 5: Practical consequences - Shrinking confidence intervals
    fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Normal curves with decreasing variance
    x = np.linspace(1.0, 3.0, 1000)
    mu = 2.0  # Population mean
    sample_sizes = [10, 50, 200]
    colors = ['red', 'orange', 'blue']
    sigma = 0.8

    for i, n in enumerate(sample_sizes):
        std_sample_mean = sigma / np.sqrt(n)
        density = stats.norm.pdf(x, loc=mu, scale=std_sample_mean)
        ax1.plot(x, density, color=colors[i], linewidth=2.5, label=f'n = {n}')
        ax1.fill_between(x, density, alpha=0.2, color=colors[i])

    ax1.axvline(x=mu, color='black', linestyle='--', linewidth=2, label='μ')
    ax1.set_xlabel('Sample mean X̄_n')
    ax1.set_ylabel('Density')
    ax1.set_title('X̄_n ~ N(μ, σ²/n)\nLarger n → smaller variance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Confidence intervals shrinking
    z_95 = 1.96  # 95% confidence level
    y_pos = [0.8, 0.5, 0.2]

    for i, n in enumerate(sample_sizes):
        std_sample_mean = sigma / np.sqrt(n)
        margin_error = z_95 * std_sample_mean

        # Draw confidence interval
        ax2.plot([mu - margin_error, mu + margin_error], [y_pos[i], y_pos[i]],
                color=colors[i], linewidth=6, label=f'n = {n}')
        ax2.plot([mu - margin_error, mu - margin_error], [y_pos[i]-0.05, y_pos[i]+0.05],
                color=colors[i], linewidth=3)
        ax2.plot([mu + margin_error, mu + margin_error], [y_pos[i]-0.05, y_pos[i]+0.05],
                color=colors[i], linewidth=3)

    ax2.axvline(x=mu, color='black', linestyle='--', linewidth=2, label='μ')
    ax2.set_xlabel('Confidence interval')
    ax2.set_title('95% Confidence Intervals\nShrink as n increases')
    ax2.set_ylim(0, 1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f'n = {n}' for n in sample_sizes])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/clt_practical.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 6: Takeaways - LLN vs CLT comparison
    fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: LLN visual - trajectory converging to mean
    n_steps = 200
    np.random.seed(42)
    sample_path = np.random.binomial(1, 0.5, n_steps)
    cumulative_mean = np.cumsum(sample_path) / np.arange(1, n_steps + 1)

    ax1.plot(range(1, n_steps + 1), cumulative_mean, 'b-', linewidth=2, label='Sample mean trajectory')
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='True mean μ = 0.5')
    ax1.set_xlabel('Sample size n')
    ax1.set_ylabel('Sample mean')
    ax1.set_title('LLN: Destination\n(convergence to μ)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.3, 0.7)

    # Right: CLT visual - empirical distributions approaching N(0,1)
    # Using the same visualization as the formal CLT slide for better pedagogical impact
    x_norm = np.linspace(-4, 4, 1000)
    standard_normal = stats.norm.pdf(x_norm, 0, 1)

    # Show the target N(0,1) distribution
    ax2.plot(x_norm, standard_normal, 'black', linewidth=4, label='N(0,1) limit')
    ax2.fill_between(x_norm, standard_normal, alpha=0.15, color='gray')

    # Generate empirical distributions for normalized sample means
    n_simulations = 10000
    sample_sizes = [5, 20, 100]  # Better pedagogical progression
    colors = ['#d62728', '#ff7f0e', '#2ca02c']  # red, orange, green
    line_styles = ['-', '--', '-.']

    # Use exponential distribution with rate λ=0.5 for clear non-normal population
    # Mean = 2, Variance = 4, so standardized CLT should converge to N(0,1)
    population_mean = 2.0
    population_std = 2.0  # sqrt(4)

    for i, n in enumerate(sample_sizes):
        # Generate many samples and compute their means
        sample_means = []
        for _ in range(n_simulations):
            # Generate n samples from Exponential(0.5) distribution
            samples = np.random.exponential(scale=2.0, size=n)  # scale=1/rate, so scale=2 gives rate=0.5
            sample_means.append(np.mean(samples))

        sample_means = np.array(sample_means)

        # Normalize: (X̄_n - μ) / (σ/√n)
        normalized_means = (sample_means - population_mean) / (population_std / np.sqrt(n))

        # Create kernel density estimate for smooth curve
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(normalized_means)
        density = kde(x_norm)

        # Plot empirical distribution
        ax2.plot(x_norm, density, color=colors[i], linewidth=3,
                alpha=0.8, linestyle=line_styles[i],
                label=f'n = {n} (empirical)')

    ax2.set_xlabel('Normalized: √n(X̄_n - μ)/σ')
    ax2.set_ylabel('Density')
    ax2.set_title('CLT: Speed + Shape\n(Gaussian convergence)')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.45)  # Set consistent y-axis for better comparison
    ax2.set_xlim(-3.5, 3.5)  # Focus on main region

    plt.tight_layout()
    plt.savefig('figures/clt_takeaways.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Generated all CLT figures successfully!")

if __name__ == "__main__":
    generate_clt_figures()