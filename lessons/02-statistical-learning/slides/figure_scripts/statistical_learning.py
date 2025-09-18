"""Statistical learning figure generators.

This module contains all figure generation functions for the statistical learning lesson,
separated into logical groups for better organization and maintainability.
"""

import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

# Set up the output directory for figures
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility
rng = np.random.default_rng(0)


def bernoulli_likelihood_profile():
    """Generate Bernoulli log-likelihood profile figure."""
    p_grid = np.linspace(0.001, 0.999, 400)
    n = 100
    k = 62
    logL = k*np.log(p_grid) + (n-k)*np.log(1-p_grid)
    logL -= logL.max()

    plt.figure(figsize=(4.8, 3.2))
    plt.plot(p_grid, logL, lw=2)
    plt.axvline(k/n, color='tab:red', ls='--', label=f"MLE = {k/n:.2f}")
    plt.xlabel('p')
    plt.ylabel('log-likelihood (shifted)')
    plt.title('Bernoulli log-likelihood profile')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_bern_loglik.png', dpi=160)
    plt.close()

    print("Generated: fig_bern_loglik.png")


def delta_method_variance():
    """Generate delta method variance illustration figure."""
    v = 0.15**2
    theta_hat = np.linspace(0.5, 1.5, 100)
    var_log = (1/theta_hat**2) * v

    plt.figure(figsize=(4.8, 3.2))
    plt.plot(theta_hat, var_log, lw=2)
    plt.xlabel(r"$\hat{\theta}$")
    plt.ylabel(r"Approx Var$(\log\hat{\theta})$")
    plt.title('Delta method variance on log scale')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_delta_log_var.png', dpi=160)
    plt.close()

    print("Generated: fig_delta_log_var.png")


def parameter_estimation_motivation():
    """Generate parameter estimation motivation figure."""
    fig, ax = plt.subplots(figsize=(6, 3))  # Reduced height from 4 to 3

    # Generate sample data from unknown distribution
    np.random.seed(42)
    true_theta = 0.7
    n_samples = 20
    data_points = np.random.binomial(1, true_theta, n_samples)

    # Plot data points
    x_positions = np.arange(1, n_samples + 1)
    colors = ['red' if x == 1 else 'blue' for x in data_points]
    ax.scatter(x_positions, data_points, c=colors, s=50, alpha=0.7, edgecolors='black')

    # Add annotations - removed y-label as requested
    ax.set_xlabel('Observation index')
    # ax.set_ylabel('Outcome')  # Removed this line
    ax.set_title('Observed Data: Coin tosses with unknown parameter θ')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Tail (0)', 'Head (1)'])  # Simplified labels
    ax.grid(True, alpha=0.3)

    # Add text box - positioned at right-middle as requested
    success_rate = np.mean(data_points)
    textstr = f'Observed: {np.sum(data_points)}/{n_samples} = {success_rate:.2f}\nTrue θ = ?'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)  # Changed color from wheat to lightblue
    ax.text(0.98, 0.5, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='right', bbox=props)  # Changed to right-middle

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mle_motivation.png', dpi=160, bbox_inches='tight')
    plt.close()

    print("Generated: mle_motivation.png")


def probability_vs_likelihood_duality():
    """Generate figure showing probability vs likelihood dual viewpoint."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: Probability function of data (θ fixed)
    n = 10
    theta_fixed = 0.6
    k_values = np.arange(0, n+1)
    probabilities = [np.exp(k*np.log(theta_fixed) + (n-k)*np.log(1-theta_fixed))
                    for k in k_values]
    # Normalize (these are binomial probabilities)
    from scipy.special import comb
    probabilities = [comb(n, k) * p for k, p in zip(k_values, probabilities)]

    ax1.bar(k_values, probabilities, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(6, color='red', linestyle='--', linewidth=2, label='Observed: k=6')
    ax1.set_xlabel('Number of successes (k)')
    ax1.set_ylabel('P(X = k | θ = 0.6)')
    ax1.set_title('Probability: θ fixed, data varies')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right panel: Likelihood function of θ (data fixed)
    k_observed = 6
    theta_values = np.linspace(0.01, 0.99, 100)
    likelihoods = [np.exp(k_observed*np.log(theta) + (n-k_observed)*np.log(1-theta))
                  for theta in theta_values]

    ax2.plot(theta_values, likelihoods, 'b-', linewidth=2)
    ax2.axvline(k_observed/n, color='red', linestyle='--', linewidth=2,
                label=f'MLE = {k_observed/n:.1f}')
    ax2.set_xlabel('Parameter θ')
    ax2.set_ylabel('L(θ | k = 6)')
    ax2.set_title('Likelihood: data fixed, θ varies')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'probability_likelihood_duality.png', dpi=160, bbox_inches='tight')
    plt.close()

    print("Generated: probability_likelihood_duality.png")


def likelihood_function_concept():
    """Generate likelihood function concept diagram."""
    fig, ax = plt.subplots(figsize=(8, 4))  # Reduced height from 5 to 4

    # Parameters for Bernoulli example
    n = 10
    k = 7
    theta_values = np.linspace(0.01, 0.99, 200)

    # Calculate likelihood (up to proportionality constant)
    log_likelihoods = k * np.log(theta_values) + (n-k) * np.log(1-theta_values)
    likelihoods = np.exp(log_likelihoods - np.max(log_likelihoods))  # Normalize to max = 1

    # Plot likelihood curve
    ax.plot(theta_values, likelihoods, 'b-', linewidth=3, label='L(θ)')

    # Mark MLE
    mle = k/n
    mle_likelihood = 1.0  # Since we normalized
    ax.axvline(mle, color='red', linestyle='--', linewidth=2)
    ax.plot(mle, mle_likelihood, 'ro', markersize=10, label=f'MLE = {mle:.1f}')

    # Add annotations with larger font sizes
    ax.set_xlabel('Parameter θ', fontsize=14)  # Increased from default (~10) to 14
    ax.set_ylabel('Likelihood L(θ)', fontsize=14)  # Increased from default (~10) to 14
    ax.set_title(f'Likelihood Function: {k} successes in {n} trials', fontsize=16)  # Increased from default (~12) to 16
    ax.legend(fontsize=12)  # Increased from default (~10) to 12
    ax.grid(True, alpha=0.3)

    # Increase tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add text annotation with larger font
    textstr = f'L(θ) ∝ θ^{k} × (1-θ)^{n-k}'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=14,  # Increased from 12 to 14
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'likelihood_function_concept.png', dpi=160, bbox_inches='tight')
    plt.close()

    print("Generated: likelihood_function_concept.png")


def mle_principle_illustration():
    """Generate MLE principle illustration with clear maximum."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Bernoulli example with n=20, k=12
    n = 20
    k = 12
    theta_values = np.linspace(0.01, 0.99, 200)

    # Calculate log-likelihood
    log_likelihoods = k * np.log(theta_values) + (n-k) * np.log(1-theta_values)

    # Plot log-likelihood curve
    ax.plot(theta_values, log_likelihoods, 'b-', linewidth=3, label='ℓ(θ) = log L(θ)')

    # Mark MLE
    mle = k/n
    mle_loglik = k * np.log(mle) + (n-k) * np.log(1-mle)
    ax.axvline(mle, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax.plot(mle, mle_loglik, 'ro', markersize=12, label=f'θ̂ = {mle:.2f}')

    # Add shaded region to show "most probable" interpretation
    fill_region = np.abs(theta_values - mle) < 0.15
    ax.fill_between(theta_values[fill_region],
                   log_likelihoods[fill_region],
                   np.min(log_likelihoods),
                   alpha=0.2, color='red',
                   label='High likelihood region')

    ax.set_xlabel('Parameter θ')
    ax.set_ylabel('Log-likelihood ℓ(θ)')
    ax.set_title('MLE Principle: Choose θ that maximizes likelihood')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation arrow
    ax.annotate('Maximum\nlikelihood\nestimator',
                xy=(mle, mle_loglik), xytext=(mle + 0.2, mle_loglik - 5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mle_principle.png', dpi=160, bbox_inches='tight')
    plt.close()

    print("Generated: mle_principle.png")


def mle_workflow_diagram():
    """Generate MLE workflow schematic diagram."""
    fig, ax = plt.subplots(figsize=(10, 4.5))  # Reduced height from 6 to 4.5
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)  # Reduced from 6 to 4.5
    ax.axis('off')

    # Define boxes and arrows - made more compact vertically
    boxes = [
        {'pos': (1.5, 3.5), 'size': (2, 0.8), 'text': 'Observed\nData\n{x₁, ..., xₙ}', 'color': 'lightblue'},  # Moved up, smaller height
        {'pos': (5, 3.5), 'size': (2, 0.8), 'text': 'Likelihood\nFunction\nL(θ)', 'color': 'lightgreen'},  # Moved up, smaller height
        {'pos': (8.5, 3.5), 'size': (2, 0.8), 'text': 'Optimization\nmax L(θ)', 'color': 'lightyellow'},  # Moved up, smaller height
        {'pos': (5, 1.2), 'size': (2, 0.8), 'text': 'Estimator\nθ̂ = arg max L(θ)', 'color': 'lightcoral'}  # Moved up slightly, smaller height
    ]

    # Draw boxes
    for box in boxes:
        x, y = box['pos']
        w, h = box['size']
        rect = Rectangle((x - w/2, y - h/2), w, h,
                       facecolor=box['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, box['text'], ha='center', va='center', fontsize=12, weight='bold')  # Increased from 10 to 12

    # Draw arrows
    arrows = [
        {'start': (2.5, 3.5), 'end': (4, 3.5)},  # Data to Likelihood
        {'start': (6, 3.5), 'end': (7.5, 3.5)},  # Likelihood to Optimization
        {'start': (8.5, 2.9), 'end': (6, 2.0)},   # Optimization to Estimator (adjusted for new positions)
    ]

    for arrow in arrows:
        ax.annotate('', xy=arrow['end'], xytext=arrow['start'],
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Add title - moved down slightly
    ax.text(5, 4.2, 'Maximum Likelihood Estimation Workflow',
           ha='center', va='center', fontsize=16, weight='bold')  # Increased from 14 to 16

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mle_workflow.png', dpi=160, bbox_inches='tight')
    plt.close()

    print("Generated: mle_workflow.png")


def likelihood_to_log_likelihood():
    """Generate figure showing transition from likelihood to log-likelihood."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left plot: Product of small probabilities (likelihood)
    n_terms = 10
    x = np.arange(1, n_terms + 1)
    # Simulate decreasing probabilities that multiply to very small values
    probs = 0.3 ** np.arange(n_terms)  # Exponentially decreasing

    ax1.bar(x, probs, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Data point i')
    ax1.set_ylabel('f(xi | θ)')
    ax1.set_title('Likelihood: L(θ) = ∏ f(xi | θ)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Add product arrow and result
    ax1.text(0.5, 0.95, f'Product ≈ {np.prod(probs):.2e}',
             transform=ax1.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Right plot: Sum of log probabilities (log-likelihood)
    log_probs = np.log(probs)

    ax2.bar(x, log_probs, alpha=0.7, color='darkorange')
    ax2.set_xlabel('Data point i')
    ax2.set_ylabel('log f(xi | θ)')
    ax2.set_title('Log-likelihood: ℓ(θ) = Σ log f(xi | θ)')
    ax2.grid(True, alpha=0.3)

    # Add sum arrow and result
    ax2.text(0.5, 0.95, f'Sum ≈ {np.sum(log_probs):.1f}',
             transform=ax2.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'likelihood_to_log_likelihood.png', dpi=160, bbox_inches='tight')
    plt.close()

    print("Generated: likelihood_to_log_likelihood.png")


def log_likelihood_benefits():
    """Generate figure comparing likelihood vs log-likelihood curves."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Simulate Bernoulli likelihood for n=50, k=30
    p_grid = np.linspace(0.01, 0.99, 200)
    n, k = 50, 30

    # Likelihood function - can become very small
    likelihood = p_grid**k * (1-p_grid)**(n-k)

    # Log-likelihood function
    log_likelihood = k*np.log(p_grid) + (n-k)*np.log(1-p_grid)
    log_likelihood -= log_likelihood.max()  # Normalize to 0 at maximum

    # Top plot: Likelihood (compressed near 0)
    ax1.plot(p_grid, likelihood, 'b-', linewidth=2, label='L(p)')
    ax1.axvline(k/n, color='red', linestyle='--', alpha=0.7, label=f'MLE = {k/n:.2f}')
    ax1.set_ylabel('Likelihood L(p)')
    ax1.set_title('Likelihood Function (compressed scale)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Bottom plot: Log-likelihood (expanded, smoother)
    ax2.plot(p_grid, log_likelihood, 'r-', linewidth=2, label='ℓ(p)')
    ax2.axvline(k/n, color='red', linestyle='--', alpha=0.7, label=f'MLE = {k/n:.2f}')
    ax2.set_xlabel('Parameter p')
    ax2.set_ylabel('Log-likelihood ℓ(p)')
    ax2.set_title('Log-likelihood Function (expanded scale)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'log_likelihood_benefits.png', dpi=160, bbox_inches='tight')
    plt.close()

    print("Generated: log_likelihood_benefits.png")


def bernoulli_likelihood_comparison():
    """Generate figure overlaying L(p) and ℓ(p) for Bernoulli case."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Parameters for the example
    p_grid = np.linspace(0.01, 0.99, 200)
    n, k = 20, 14  # 14 heads out of 20 flips

    # Likelihood and log-likelihood
    likelihood = p_grid**k * (1-p_grid)**(n-k)
    log_likelihood = k*np.log(p_grid) + (n-k)*np.log(1-p_grid)

    # Normalize both to have maximum at 1 for comparison
    likelihood = likelihood / likelihood.max()
    log_likelihood = log_likelihood - log_likelihood.max()
    log_likelihood = np.exp(log_likelihood)  # Convert back to same scale

    # Plot both curves
    ax.plot(p_grid, likelihood, 'b-', linewidth=2.5, label='L(p) (normalized)', alpha=0.8)
    ax.plot(p_grid, log_likelihood, 'r--', linewidth=2.5, label='exp(ℓ(p)) (normalized)', alpha=0.8)

    # Mark the MLE
    mle_p = k/n
    ax.axvline(mle_p, color='black', linestyle=':', alpha=0.7, linewidth=2)
    ax.plot(mle_p, 1, 'ko', markersize=8, label=f'MLE = {mle_p:.2f}')

    ax.set_xlabel('Parameter p', fontsize=12)
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title('Bernoulli Likelihood: Both L(p) and ℓ(p) peak at same value', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Add text box with the equations
    textstr = f'L(p) = p^{k}(1-p)^{n-k}\nℓ(p) = {k}log(p) + {n-k}log(1-p)\nBoth maximize at p̂ = {k}/{n} = {mle_p:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bernoulli_likelihood_comparison.png', dpi=160, bbox_inches='tight')
    plt.close()

    print("Generated: bernoulli_likelihood_comparison.png")


def normal_mean_mle():
    """Generate MLE for normal mean (sigma known) example."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Parameters
    true_mu = 5.0
    sigma = 1.5
    n = 20

    # Generate sample data
    np.random.seed(42)
    data = np.random.normal(true_mu, sigma, n)
    sample_mean = np.mean(data)

    # Mu grid for likelihood
    mu_grid = np.linspace(2, 8, 200)
    log_likelihood = -(n/2) * np.log(2*np.pi*sigma**2) - (1/(2*sigma**2)) * np.sum((data[:, None] - mu_grid)**2, axis=0)
    log_likelihood -= log_likelihood.max()  # Normalize

    # Plot likelihood
    ax.plot(mu_grid, log_likelihood, 'b-', linewidth=2.5, label='Log-likelihood ℓ(μ)')
    ax.axvline(float(sample_mean), color='red', linestyle='--', linewidth=2, label=f'MLE μ̂ = {sample_mean:.2f}')
    ax.axvline(true_mu, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'True μ = {true_mu}')

    ax.set_xlabel('Parameter μ', fontsize=12)
    ax.set_ylabel('Log-likelihood ℓ(μ)', fontsize=12)
    ax.set_title('MLE for Normal Mean (σ² known)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'normal_mean_mle.png', dpi=160, bbox_inches='tight')
    plt.close()

    print("Generated: normal_mean_mle.png")


def normal_full_mle():
    """Generate MLE for normal with both parameters unknown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Parameters and data
    true_mu, true_sigma = 3.0, 1.2
    n = 50
    np.random.seed(42)
    data = np.random.normal(true_mu, true_sigma, n)

    mle_mu = np.mean(data)
    mle_sigma = np.sqrt(np.mean((data - mle_mu)**2))  # MLE formula (divisor n)

    # Left plot: Histogram with fitted normal
    ax1.hist(data, bins=12, density=True, alpha=0.7, color='lightblue', edgecolor='black')

    # Fitted normal curve
    x_curve = np.linspace(data.min()-1, data.max()+1, 200)
    fitted_curve = (1/(mle_sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_curve - mle_mu)/mle_sigma)**2)
    ax1.plot(x_curve, fitted_curve, 'r-', linewidth=2.5, label=f'Fitted N({mle_mu:.2f}, {mle_sigma:.2f}²)')

    ax1.set_xlabel('Value', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Sample Data + MLE Fit', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right plot: MLE vs Unbiased comparison
    unbiased_sigma = np.sqrt(np.var(data, ddof=1))  # Unbiased (divisor n-1)

    categories = ['μ̂', 'σ̂ (MLE)', 'σ̂ (Unbiased)']
    estimates = [mle_mu, mle_sigma, unbiased_sigma]
    true_values = [true_mu, true_sigma, true_sigma]

    x_pos = np.arange(len(categories))
    ax2.bar(x_pos - 0.2, estimates, 0.4, label='Estimates', color='lightcoral', alpha=0.8)
    ax2.bar(x_pos + 0.2, true_values, 0.4, label='True values', color='lightgreen', alpha=0.8)

    ax2.set_xlabel('Parameter', fontsize=11)
    ax2.set_ylabel('Value', fontsize=11)
    ax2.set_title('MLE vs True Parameters', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add text annotations
    for i, (est, true_val) in enumerate(zip(estimates, true_values)):
        if i < 2:  # For mu and MLE sigma
            ax2.text(i-0.2, est+0.05, f'{est:.2f}', ha='center', fontsize=9)
        ax2.text(i+0.2, true_val+0.05, f'{true_val:.2f}', ha='center', fontsize=9)
        if i == 2:  # For unbiased sigma
            ax2.text(i-0.2, est+0.05, f'{est:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'normal_full_mle.png', dpi=160, bbox_inches='tight')
    plt.close()

    print("Generated: normal_full_mle.png")


def poisson_mle():
    """Generate MLE for Poisson parameter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Parameters and data
    true_lambda = 3.5
    n = 100
    np.random.seed(42)
    data = np.random.poisson(true_lambda, n)

    mle_lambda = np.mean(data)

    # Left plot: Histogram with fitted Poisson PMF
    unique_vals, counts = np.unique(data, return_counts=True)
    ax1.bar(unique_vals, counts/n, alpha=0.7, color='lightblue', edgecolor='black', label='Sample data')

    # Fitted Poisson PMF
    x_vals = np.arange(0, max(data)+2)
    fitted_pmf = np.exp(-mle_lambda) * (mle_lambda**x_vals) / np.array([math.factorial(x) for x in x_vals])
    ax1.plot(x_vals, fitted_pmf, 'ro-', linewidth=2, markersize=6, label=f'Fitted Poisson(λ̂={mle_lambda:.2f})')

    ax1.set_xlabel('Count', fontsize=11)
    ax1.set_ylabel('Probability/Frequency', fontsize=11)
    ax1.set_title('Sample Data + MLE Fit', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right plot: Log-likelihood function
    lambda_grid = np.linspace(2, 5, 200)
    log_likelihood = np.sum(data) * np.log(lambda_grid) - n * lambda_grid - np.sum(np.log(np.array([math.factorial(x) for x in data])))
    log_likelihood -= log_likelihood.max()  # Normalize

    ax2.plot(lambda_grid, log_likelihood, 'b-', linewidth=2.5, label='Log-likelihood ℓ(λ)')
    ax2.axvline(mle_lambda, color='red', linestyle='--', linewidth=2, label=f'MLE λ̂ = {mle_lambda:.2f}')
    ax2.axvline(true_lambda, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'True λ = {true_lambda}')

    ax2.set_xlabel('Parameter λ', fontsize=11)
    ax2.set_ylabel('Log-likelihood ℓ(λ)', fontsize=11)
    ax2.set_title('Log-likelihood Function', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'poisson_mle.png', dpi=160, bbox_inches='tight')
    plt.close()

    print("Generated: poisson_mle.png")


def mle_asymptotic_properties():
    """Generate figure showing asymptotic properties of MLE."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Consistency: sampling distributions for different n
    true_theta = 2.0
    sample_sizes = [10, 50, 200]
    colors = ['red', 'blue', 'green']

    for i, (n, color) in enumerate(zip(sample_sizes, colors)):
        # Simulate MLE distribution (using normal approximation)
        variance = 1.0 / n  # Assume Fisher info scales with n
        theta_vals = np.linspace(true_theta - 3*np.sqrt(variance), true_theta + 3*np.sqrt(variance), 200)
        density = (1/np.sqrt(2*np.pi*variance)) * np.exp(-0.5*(theta_vals - true_theta)**2/variance)

        ax1.plot(theta_vals, density, color=color, linewidth=2, label=f'n = {n}')

    ax1.axvline(true_theta, color='black', linestyle='--', linewidth=2, label='True θ')
    ax1.set_xlabel('θ̂', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Consistency: θ̂ → θ as n → ∞', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Fisher Information illustration
    theta_grid = np.linspace(-1, 5, 200)

    # High information case (sharp peak)
    high_info_ll = -2 * (theta_grid - true_theta)**2
    # Low information case (flat peak)
    low_info_ll = -0.2 * (theta_grid - true_theta)**2

    ax2.plot(theta_grid, high_info_ll, 'b-', linewidth=2.5, label='High Fisher Information')
    ax2.plot(theta_grid, low_info_ll, 'r--', linewidth=2.5, label='Low Fisher Information')
    ax2.axvline(true_theta, color='black', linestyle=':', alpha=0.7, label='True θ')

    ax2.set_xlabel('Parameter θ', fontsize=12)
    ax2.set_ylabel('Log-likelihood ℓ(θ)', fontsize=12)
    ax2.set_title('Fisher Information: Sharp vs Flat Likelihood', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add text annotations
    ax2.text(2.5, -15, 'High I(θ)\n→ Low Var(θ̂)', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax2.text(2.5, -2, 'Low I(θ)\n→ High Var(θ̂)', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mle_asymptotic_properties.png', dpi=160, bbox_inches='tight')
    plt.close()

    print("Generated: mle_asymptotic_properties.png")


def fisher_information_concept():
    """Generate figure illustrating Fisher Information concept."""
    fig, ax = plt.subplots(figsize=(10, 6))

    theta_grid = np.linspace(-2, 6, 300)
    true_theta = 2.0

    # Three different information scenarios
    scenarios = [
        ('Very High Information', 8, 'blue', '-'),
        ('Moderate Information', 2, 'green', '--'),
        ('Low Information', 0.5, 'red', ':')
    ]

    for label, fisher_info, color, linestyle in scenarios:
        # Log-likelihood proportional to Fisher info
        log_likelihood = -0.5 * fisher_info * (theta_grid - true_theta)**2
        ax.plot(theta_grid, log_likelihood, color=color, linewidth=3,
                linestyle=linestyle, label=f'{label} (I = {fisher_info})')

    ax.axvline(true_theta, color='black', linestyle='-', alpha=0.6, linewidth=2, label='True θ')
    ax.set_xlabel('Parameter θ', fontsize=14)
    ax.set_ylabel('Log-likelihood ℓ(θ)', fontsize=14)
    ax.set_title('Fisher Information: How Much the Data Tell Us About θ', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add informative text boxes
    ax.text(4, -5, 'Higher Fisher Information:\n• Sharper likelihood peak\n• More precise estimates\n• Lower variance',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            verticalalignment='top')

    ax.text(-1, -2, 'Lower Fisher Information:\n• Flatter likelihood\n• Less precise estimates\n• Higher variance',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
            verticalalignment='top')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fisher_information_concept.png', dpi=160, bbox_inches='tight')
    plt.close()

    print("Generated: fisher_information_concept.png")


def mle_strengths():
    """Generate figure showing when MLE works well."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Large sample concentration
    theta_grid = np.linspace(-1, 5, 200)
    true_theta = 2.0

    sample_sizes = [10, 100, 1000]
    colors = ['red', 'orange', 'blue']
    alphas = [0.6, 0.7, 0.9]

    for n, color, alpha in zip(sample_sizes, colors, alphas):
        variance = 0.5 / n  # Fisher information scales with n
        log_likelihood = -0.5 * (theta_grid - true_theta)**2 / variance
        log_likelihood -= log_likelihood.max()  # Normalize

        ax1.plot(theta_grid, log_likelihood, color=color, linewidth=2.5,
                alpha=alpha, label=f'n = {n}')

    ax1.axvline(true_theta, color='black', linestyle='--', linewidth=2, label='True θ')
    ax1.set_xlabel('Parameter θ', fontsize=12)
    ax1.set_ylabel('Normalized Log-likelihood', fontsize=12)
    ax1.set_title('Large Samples: Likelihood Concentrates', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Invariance property illustration
    # Original parameter and its transform
    theta_orig = np.linspace(0.5, 3, 100)
    likelihood_orig = np.exp(-2*(theta_orig - 1.5)**2)  # Peaked at 1.5

    # Transform: g(theta) = theta^2
    theta_transformed = theta_orig**2
    likelihood_transformed = likelihood_orig  # Same likelihood shape

    ax2.plot(theta_orig, likelihood_orig, 'b-', linewidth=2.5, label='L(θ)')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(theta_transformed, likelihood_transformed, 'r--', linewidth=2.5, label='L(g(θ)) where g(θ)=θ²')

    # Mark MLEs
    mle_orig = 1.5
    mle_transformed = mle_orig**2

    ax2.axvline(mle_orig, color='blue', linestyle=':', alpha=0.7, label=f'θ̂ = {mle_orig}')
    ax2_twin.axvline(mle_transformed, color='red', linestyle=':', alpha=0.7, label=f'ĝ(θ̂) = {mle_transformed}')

    ax2.set_xlabel('θ (blue) / g(θ) = θ² (red)', fontsize=11)
    ax2.set_ylabel('Likelihood L(θ)', color='blue', fontsize=11)
    ax2_twin.set_ylabel('Likelihood L(g(θ))', color='red', fontsize=11)
    ax2.set_title('Invariance: If θ̂ is MLE, then g(θ̂) is MLE of g(θ)', fontsize=13)

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')

    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mle_strengths.png', dpi=160, bbox_inches='tight')
    plt.close()

    print("Generated: mle_strengths.png")


def mle_limitations():
    """Generate figure showing when MLE may fail."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Multimodal likelihood (multiple local maxima)
    theta_grid = np.linspace(-5, 5, 400)

    # Create bimodal likelihood
    mode1, mode2 = -1.5, 2.0
    likelihood1 = np.exp(-2*(theta_grid - mode1)**2)
    likelihood2 = np.exp(-3*(theta_grid - mode2)**2)

    # Combine with different heights
    multimodal_likelihood = 0.6*likelihood1 + 0.8*likelihood2
    log_likelihood = np.log(multimodal_likelihood + 1e-10)  # Add small constant for numerical stability

    ax1.plot(theta_grid, log_likelihood, 'b-', linewidth=2.5, label='Log-likelihood')
    ax1.axvline(mode1, color='red', linestyle='--', alpha=0.8, label='Local max')
    ax1.axvline(mode2, color='red', linestyle='-', linewidth=2, label='Global max (MLE)')
    ax1.scatter([mode1, mode2], [log_likelihood[np.argmin(np.abs(theta_grid - mode1))],
                                 log_likelihood[np.argmin(np.abs(theta_grid - mode2))]],
               color='red', s=80, zorder=5)

    ax1.set_xlabel('Parameter θ', fontsize=12)
    ax1.set_ylabel('Log-likelihood ℓ(θ)', fontsize=12)
    ax1.set_title('Multiple Maxima: Which One to Choose?', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Small sample bias illustration
    n_values = [5, 10, 25, 100]
    true_theta = 1.0
    bias_values = [0.8/n for n in n_values]  # Decreasing bias
    variance_values = [2.0/n for n in n_values]  # Decreasing variance

    ax2.errorbar(n_values, bias_values, yerr=np.sqrt(variance_values),
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=6,
                color='red', label='Bias ± √Var')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.7, label='Unbiased')

    ax2.set_xlabel('Sample Size n', fontsize=12)
    ax2.set_ylabel('Bias of θ̂', fontsize=12)
    ax2.set_title('Small Sample Bias', fontsize=13)
    ax2.set_xscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add annotation
    ax2.annotate('MLE can be biased\nfor small samples',
                xy=(5, bias_values[0]), xytext=(20, bias_values[0]*1.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mle_limitations.png', dpi=160, bbox_inches='tight')
    plt.close()

    print("Generated: mle_limitations.png")


def mle_summary_comparison():
    """Generate summary comparison of MLE strengths and limitations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Strengths (checkmarks)
    strengths = [
        "Large sample size (n → ∞)",
        "Correct model specification",
        "Identifiable parameters",
        "Regular likelihood surface",
        "Sufficient data"
    ]

    y_pos = np.arange(len(strengths))
    ax1.barh(y_pos, [1]*len(strengths), color='lightgreen', alpha=0.7, edgecolor='darkgreen')

    # Add checkmarks
    for i, strength in enumerate(strengths):
        ax1.text(0.05, i, f"✓ {strength}", fontsize=16, va='center', weight='bold')

    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.5, len(strengths)-0.5)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_title('✓ MLE Works Well When...', fontsize=18, color='darkgreen', weight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    # Right: Limitations (warning signs)
    limitations = [
        "Small sample size (finite n)",
        "Model misspecification",
        "Non-identifiable parameters",
        "Multimodal likelihood",
        "Insufficient/poor data"
    ]

    y_pos = np.arange(len(limitations))
    ax2.barh(y_pos, [1]*len(limitations), color='lightcoral', alpha=0.7, edgecolor='darkred')

    # Add warning signs
    for i, limitation in enumerate(limitations):
        ax2.text(0.05, i, f"⚠ {limitation}", fontsize=16, va='center', weight='bold')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.5, len(limitations)-0.5)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_title('⚠ MLE May Struggle When...', fontsize=18, color='darkred', weight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mle_summary_comparison.png', dpi=160, bbox_inches='tight')
    plt.close()

    print("Generated: mle_summary_comparison.png")


def generate_all_statistical_learning_figures():
    """Generate all statistical learning figures."""
    print("Generating statistical learning figures...")

    # Original figures
    bernoulli_likelihood_profile()
    delta_method_variance()

    # New MLE introduction figures
    parameter_estimation_motivation()
    probability_vs_likelihood_duality()
    likelihood_function_concept()
    mle_principle_illustration()
    mle_workflow_diagram()

    # New log-likelihood explanation figures
    likelihood_to_log_likelihood()
    log_likelihood_benefits()
    bernoulli_likelihood_comparison()

    # Extended MLE examples and properties figures
    normal_mean_mle()
    normal_full_mle()
    poisson_mle()
    mle_asymptotic_properties()
    fisher_information_concept()
    mle_strengths()
    mle_limitations()
    mle_summary_comparison()

    # Method of Moments figures
    mom_motivation()
    mom_principle_diagram()
    mom_bernoulli_example()
    mom_poisson_example()
    mom_normal_example()
    mle_vs_mom_comparison()
    normal_variance_mle_mom()
    mle_mom_summary_table()
    mom_takeaways()

    print("All statistical learning figures generated successfully!")


def mom_motivation():
    """Generate MoM motivation figure showing sample mean = population mean concept."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: Sample data
    np.random.seed(42)
    sample_data = np.random.normal(3, 1, 50)
    ax1.hist(sample_data, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(sample_data), color='red', linewidth=3, linestyle='--', label=f'Sample mean = {np.mean(sample_data):.2f}')
    ax1.set_title('Sample Data', fontsize=16)
    ax1.set_xlabel('Values', fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    ax1.legend(fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Right panel: Population concept
    x = np.linspace(-1, 7, 200)
    y = (1/np.sqrt(2*np.pi)) * np.exp(-0.5*(x-3)**2)
    ax2.plot(x, y, 'b-', linewidth=2, label='Population distribution')
    ax2.axvline(3, color='red', linewidth=3, linestyle='--', label='Population mean = 3')
    ax2.set_title('Population Model', fontsize=16)
    ax2.set_xlabel('Values', fontsize=14)
    ax2.set_ylabel('Density', fontsize=14)
    ax2.legend(fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mom_motivation.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("Generated: mom_motivation.png")


def mom_principle_diagram():
    """Generate MoM principle diagram linking sample moment → theoretical moment → parameter."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create boxes for the three components
    boxes = [
        {'xy': (1, 2), 'width': 2.5, 'height': 1.5, 'label': 'Sample\nMoment\n$\\hat{m}_k$', 'color': 'lightblue'},
        {'xy': (5, 2), 'width': 2.5, 'height': 1.5, 'label': 'Theoretical\nMoment\n$m_k(\\theta)$', 'color': 'lightgreen'},
        {'xy': (9, 2), 'width': 2.5, 'height': 1.5, 'label': 'Parameter\nEstimate\n$\\hat{\\theta}_{MoM}$', 'color': 'lightcoral'}
    ]

    for box in boxes:
        rect = Rectangle(box['xy'], box['width'], box['height'],
                        facecolor=box['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(box['xy'][0] + box['width']/2, box['xy'][1] + box['height']/2,
               box['label'], ha='center', va='center', fontsize=18, fontweight='bold')

    # Add arrows
    arrow_props = dict(arrowstyle='->', lw=3, color='black')
    ax.annotate('', xy=(4.8, 2.75), xytext=(3.7, 2.75), arrowprops=arrow_props)
    ax.annotate('', xy=(8.8, 2.75), xytext=(7.7, 2.75), arrowprops=arrow_props)

    # Add equation
    ax.text(6.25, 1, '$m_k(\\theta) = \\hat{m}_k$', ha='center', va='center',
           fontsize=20, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    # Add title
    ax.text(6.25, 4.2, 'Method of Moments Principle', ha='center', va='center',
           fontsize=22, fontweight='bold')

    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mom_principle_diagram.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("Generated: mom_principle_diagram.png")


def mom_bernoulli_example():
    """Generate MoM Bernoulli example with coin toss histogram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Coin toss data
    np.random.seed(123)
    n = 100
    p_true = 0.3
    data = np.random.binomial(1, p_true, n)

    # Create histogram with separate bars for tails and heads
    n_tails = n - np.sum(data)
    n_heads = np.sum(data)

    bars = ax1.bar([0, 1], [n_tails, n_heads], alpha=0.7,
                   color=['lightcoral', 'lightblue'],
                   edgecolor='black', linewidth=2)

    # Add proportion annotation
    p_hat = np.mean(data)
    ax1.text(1, n_heads + 2, f'Proportion = {p_hat:.2f}', ha='center', fontsize=16,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow'))

    ax1.set_xlabel('Outcome (0=Tails, 1=Heads)', fontsize=14)
    ax1.set_ylabel('Count', fontsize=14)
    ax1.set_title(f'Sample Data (n={n})', fontsize=16)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Tails', 'Heads'])
    ax1.grid(True, alpha=0.3)

    # Right panel: MoM calculation
    ax2.text(0.5, 0.8, 'Method of Moments:', fontsize=20, fontweight='bold',
            transform=ax2.transAxes, ha='center')
    ax2.text(0.5, 0.65, '$E[X] = p$', fontsize=18, transform=ax2.transAxes, ha='center')
    ax2.text(0.5, 0.5, '$\\hat{m}_1 = \\bar{X}_n$', fontsize=18, transform=ax2.transAxes, ha='center')
    ax2.text(0.5, 0.35, '$\\hat{p}_{MoM} = \\bar{X}_n$', fontsize=18, transform=ax2.transAxes, ha='center')
    ax2.text(0.5, 0.15, f'$\\hat{{p}}_{{MoM}} = {p_hat:.2f}$', fontsize=20,
            transform=ax2.transAxes, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen'))

    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mom_bernoulli_example.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("Generated: mom_bernoulli_example.png")
def mom_poisson_example():
    """Generate MoM Poisson example with histogram and fitted curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Poisson data
    np.random.seed(456)
    n = 200
    lambda_true = 2.5
    data = np.random.poisson(lambda_true, n)

    counts, bins, patches = ax1.hist(data, bins=range(0, max(data)+2), alpha=0.7,
                                    color='skyblue', edgecolor='black', density=True)

    # Overlay theoretical Poisson PMF with MoM estimate
    lambda_hat = np.mean(data)
    x_vals = range(0, max(data)+1)
    pmf_vals = [math.exp(-lambda_hat) * (lambda_hat**k) / math.factorial(k) for k in x_vals]
    ax1.plot(x_vals, pmf_vals, 'ro-', linewidth=2, markersize=6, label=f'Fitted Poisson($\\lambda={lambda_hat:.2f}$)')

    ax1.set_xlabel('Count', fontsize=14)
    ax1.set_ylabel('Probability/Density', fontsize=14)
    ax1.set_title(f'Sample Data (n={n})', fontsize=16)
    ax1.legend(fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Right panel: MoM calculation
    ax2.text(0.5, 0.8, 'Method of Moments:', fontsize=20, fontweight='bold',
            transform=ax2.transAxes, ha='center')
    ax2.text(0.5, 0.65, '$E[X] = \\lambda$', fontsize=18, transform=ax2.transAxes, ha='center')
    ax2.text(0.5, 0.5, '$\\hat{m}_1 = \\bar{X}_n$', fontsize=18, transform=ax2.transAxes, ha='center')
    ax2.text(0.5, 0.35, '$\\hat{\\lambda}_{MoM} = \\bar{X}_n$', fontsize=18, transform=ax2.transAxes, ha='center')
    ax2.text(0.5, 0.15, f'$\\hat{{\\lambda}}_{{MoM}} = {lambda_hat:.2f}$', fontsize=20,
            transform=ax2.transAxes, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen'))

    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mom_poisson_example.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("Generated: mom_poisson_example.png")


def mom_normal_example():
    """Generate MoM Normal example with histogram and fitted curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Normal data
    np.random.seed(789)
    n = 150
    mu_true, sigma_true = 2, 1.5
    data = np.random.normal(mu_true, sigma_true, n)

    # Calculate MoM estimates
    m1_hat = np.mean(data)
    m2_hat = np.mean(data**2)
    mu_hat = m1_hat
    sigma_hat = np.sqrt(m2_hat - m1_hat**2)

    ax1.hist(data, bins=20, alpha=0.7, color='lightcyan', edgecolor='black', density=True)

    # Overlay fitted Normal curve
    x_vals = np.linspace(data.min(), data.max(), 200)
    y_vals = (1/(sigma_hat*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_vals-mu_hat)/sigma_hat)**2)
    ax1.plot(x_vals, y_vals, 'r-', linewidth=3, label=f'Fitted N($\\mu={mu_hat:.2f}, \\sigma={sigma_hat:.2f}$)')

    ax1.set_xlabel('Values', fontsize=14)
    ax1.set_ylabel('Density', fontsize=14)
    ax1.set_title(f'Sample Data (n={n})', fontsize=16)
    ax1.legend(fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Right panel: MoM calculation
    ax2.text(0.5, 0.85, 'Method of Moments:', fontsize=20, fontweight='bold',
            transform=ax2.transAxes, ha='center')
    ax2.text(0.5, 0.7, '$E[X] = \\mu, \\quad E[X^2] = \\mu^2 + \\sigma^2$', fontsize=16,
            transform=ax2.transAxes, ha='center')
    ax2.text(0.5, 0.55, '$\\hat{m}_1 = \\bar{X}_n, \\quad \\hat{m}_2 = \\frac{1}{n}\\sum X_i^2$', fontsize=16,
            transform=ax2.transAxes, ha='center')
    ax2.text(0.5, 0.4, '$\\hat{\\mu}_{MoM} = \\bar{X}_n$', fontsize=18, transform=ax2.transAxes, ha='center')
    ax2.text(0.5, 0.25, '$\\hat{\\sigma}^2_{MoM} = \\hat{m}_2 - (\\bar{X}_n)^2$', fontsize=18,
            transform=ax2.transAxes, ha='center')
    ax2.text(0.5, 0.05, f'$\\hat{{\\mu}}_{{MoM}} = {mu_hat:.2f}, \\hat{{\\sigma}}_{{MoM}} = {sigma_hat:.2f}$',
            fontsize=18, transform=ax2.transAxes, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen'))

    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mom_normal_example.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("Generated: mom_normal_example.png")


def mle_vs_mom_comparison():
    """Generate MLE vs MoM comparison diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create two main sections
    sections = [
        {'xy': (1, 4), 'width': 5, 'height': 3, 'label': 'Maximum Likelihood\nEstimation (MLE)', 'color': 'lightblue'},
        {'xy': (8, 4), 'width': 5, 'height': 3, 'label': 'Method of Moments\n(MoM)', 'color': 'lightgreen'}
    ]

    for section in sections:
        rect = Rectangle(section['xy'], section['width'], section['height'],
                        facecolor=section['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(section['xy'][0] + section['width']/2, section['xy'][1] + section['height']/2 + 0.8,
               section['label'], ha='center', va='center', fontsize=20, fontweight='bold')

    # Add method descriptions
    mle_text = 'Optimization:\n$\\max_{\\theta} L(\\theta)$'
    mom_text = 'Equation solving:\n$m_k(\\theta) = \\hat{m}_k$'

    ax.text(3.5, 5, mle_text, ha='center', va='center', fontsize=18,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.text(10.5, 5, mom_text, ha='center', va='center', fontsize=18,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Add arrows pointing to common destination
    arrow_props = dict(arrowstyle='->', lw=3, color='red')
    ax.annotate('', xy=(6.8, 2.5), xytext=(3.5, 3.8), arrowprops=arrow_props)
    ax.annotate('', xy=(7.2, 2.5), xytext=(10.5, 3.8), arrowprops=arrow_props)

    # Add destination
    dest_rect = Rectangle((6, 1.5), 2, 2, facecolor='gold', edgecolor='black', linewidth=2)
    ax.add_patch(dest_rect)
    ax.text(7, 2.5, 'Parameter\nEstimate\n$\\hat{\\theta}$', ha='center', va='center',
           fontsize=18, fontweight='bold')

    # Add title
    ax.text(7, 8.5, 'Two Roads to Parameter Estimation', ha='center', va='center',
           fontsize=24, fontweight='bold')

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mle_vs_mom_comparison.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("Generated: mle_vs_mom_comparison.png")


def normal_variance_mle_mom():
    """Generate Normal variance estimate comparison (MLE vs MoM)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Generate sample data
    np.random.seed(101112)
    n = 80
    mu_true, sigma_true = 0, 2
    data = np.random.normal(mu_true, sigma_true, n)

    # Calculate estimates
    x_bar = np.mean(data)
    sigma2_mle = np.mean((data - x_bar)**2)
    m2_hat = np.mean(data**2)
    sigma2_mom = m2_hat - x_bar**2

    # Left panel: Data histogram with both fits
    ax1.hist(data, bins=15, alpha=0.6, color='lightgray', edgecolor='black', density=True, label='Data')

    x_vals = np.linspace(data.min(), data.max(), 200)

    # MLE fit
    y_mle = (1/(np.sqrt(sigma2_mle)*np.sqrt(2*np.pi))) * np.exp(-0.5*(x_vals-x_bar)**2/sigma2_mle)
    ax1.plot(x_vals, y_mle, 'b-', linewidth=3, label=f'MLE: $\\sigma^2 = {sigma2_mle:.2f}$')

    # MoM fit
    y_mom = (1/(np.sqrt(sigma2_mom)*np.sqrt(2*np.pi))) * np.exp(-0.5*(x_vals-x_bar)**2/sigma2_mom)
    ax1.plot(x_vals, y_mom, 'r--', linewidth=3, label=f'MoM: $\\sigma^2 = {sigma2_mom:.2f}$')

    ax1.set_xlabel('Values', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Normal Distribution Fits', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right panel: Formula comparison
    ax2.text(0.5, 0.8, 'Variance Estimators:', fontsize=16, fontweight='bold',
            transform=ax2.transAxes, ha='center')

    ax2.text(0.5, 0.65, 'MLE:', fontsize=14, fontweight='bold',
            transform=ax2.transAxes, ha='center', color='blue')
    ax2.text(0.5, 0.55, '$\\hat{\\sigma}^2_{MLE} = \\frac{1}{n}\\sum (X_i - \\bar{X}_n)^2$',
            fontsize=12, transform=ax2.transAxes, ha='center')

    ax2.text(0.5, 0.4, 'MoM:', fontsize=14, fontweight='bold',
            transform=ax2.transAxes, ha='center', color='red')
    ax2.text(0.5, 0.3, '$\\hat{\\sigma}^2_{MoM} = \\hat{m}_2 - (\\bar{X}_n)^2$',
            fontsize=12, transform=ax2.transAxes, ha='center')

    ax2.text(0.5, 0.15, f'In this case: MLE = {sigma2_mle:.3f}, MoM = {sigma2_mom:.3f}',
            fontsize=12, transform=ax2.transAxes, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    ax2.text(0.5, 0.05, 'Note: These are equal for Normal distribution!',
            fontsize=11, transform=ax2.transAxes, ha='center', style='italic')

    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'normal_variance_mle_mom.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("Generated: normal_variance_mle_mom.png")


def mle_mom_summary_table():
    """Generate MLE vs MoM summary table as figure."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create table data
    headers = ['Property', 'MLE', 'MoM']
    rows = [
        ['Computation', 'Requires optimization', 'Simple equations'],
        ['Large-sample behavior', 'Consistent, efficient', 'Consistent, less efficient'],
        ['Small-sample behavior', 'Biased but systematic', 'Can be unstable, nonsensical'],
        ['Flexibility', 'Works for many models', 'Requires finite moments'],
        ['Coincidence', 'Often equals MoM in simple cases', 'Same as MLE sometimes']
    ]

    # Create table
    table_data = [headers] + rows
    n_rows, n_cols = len(table_data), len(table_data[0])

    # Table styling
    cell_colors = [['lightblue', 'lightgreen', 'lightcoral']]
    for i in range(1, n_rows):
        cell_colors.append(['white', 'lightgray', 'lightgray'])

    table = ax.table(cellText=table_data, cellColours=cell_colors,
                    cellLoc='center', loc='center',
                    colWidths=[0.25, 0.375, 0.375])

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)

    # Style the header row
    for i in range(n_cols):
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.15)

    # Style data rows
    for i in range(1, n_rows):
        for j in range(n_cols):
            table[(i, j)].set_height(0.12)
            if j == 0:  # Property column
                table[(i, j)].set_text_props(weight='bold')

    ax.set_title('MLE vs MoM: Summary Comparison', fontsize=22, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mle_mom_summary_table.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("Generated: mle_mom_summary_table.png")


def mom_takeaways():
    """Generate MoM takeaways figure with cartoon illustration."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create two paths leading to the same destination
    # Path 1: MoM (top path)
    mom_path_x = np.array([1, 3, 5, 7, 9])
    mom_path_y = np.array([6, 6.2, 6, 5.8, 6])
    ax.plot(mom_path_x, mom_path_y, 'g-', linewidth=4, label='Method of Moments')
    ax.scatter(mom_path_x, mom_path_y, c='green', s=80, zorder=5)

    # Path 2: MLE (bottom path)
    mle_path_x = np.array([1, 3, 5, 7, 9])
    mle_path_y = np.array([3, 2.8, 3, 3.2, 3])
    ax.plot(mle_path_x, mle_path_y, 'b-', linewidth=4, label='Maximum Likelihood')
    ax.scatter(mle_path_x, mle_path_y, c='blue', s=80, zorder=5)

    # Starting point
    start_rect = Rectangle((0.2, 4), 1.6, 1, facecolor='lightcyan', edgecolor='black', linewidth=2)
    ax.add_patch(start_rect)
    ax.text(1, 4.5, 'Data', ha='center', va='center', fontsize=18, fontweight='bold')

    # Destination
    dest_rect = Rectangle((10, 4), 2, 1, facecolor='gold', edgecolor='black', linewidth=2)
    ax.add_patch(dest_rect)
    ax.text(11, 4.5, 'Parameter\nEstimate', ha='center', va='center', fontsize=18, fontweight='bold')

    # Add method labels along paths
    ax.text(5, 6.5, 'MoM: Simple, Intuitive', ha='center', va='center', fontsize=16,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    ax.text(5, 2.5, 'MLE: Powerful, Optimal', ha='center', va='center', fontsize=16,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    # Add arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='red')
    ax.annotate('', xy=(2.8, 6), xytext=(1.8, 4.8), arrowprops=arrow_props)
    ax.annotate('', xy=(2.8, 3), xytext=(1.8, 4.2), arrowprops=arrow_props)
    ax.annotate('', xy=(10, 4.8), xytext=(9.2, 6), arrowprops=arrow_props)
    ax.annotate('', xy=(10, 4.2), xytext=(9.2, 3), arrowprops=arrow_props)

    # Add title and takeaway text
    ax.text(6, 8.5, 'Method of Moments: Key Takeaways', ha='center', va='center',
           fontsize=22, fontweight='bold')

    takeaways = [
        '• MoM: intuitive, simple, often first attempt',
        '• MLE: more powerful, statistically optimal in large samples',
        '• Both are fundamental tools in parameter estimation'
    ]

    for i, takeaway in enumerate(takeaways):
        ax.text(6, 1.5 - 0.3*i, takeaway, ha='center', va='center', fontsize=16,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(0, 9)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mom_takeaways.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("Generated: mom_takeaways.png")
if __name__ == "__main__":
    generate_all_statistical_learning_figures()