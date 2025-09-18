import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path(__file__).parent.parent / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Deterministic randomness for reproducible sampling
rng = np.random.default_rng(42)

def mm_cdf():
    # Define categories and sample probabilities from Dirichlet distribution
    categories = ["red", "yellow", "blue", "green", "brown", "orange"]
    # Sample from symmetric Dirichlet (all alphas = 2.0) for moderate variability
    probs = rng.dirichlet([2.0] * len(categories))

    # --- PMF (bar chart) ---
    fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
    x_positions = np.arange(len(categories))
    bars = ax.bar(x_positions, probs, color=categories, edgecolor='black', alpha=0.9)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Probability')
    ax.set_title('Category probabilities (PMF)')
    ax.set_ylim(0, probs.max() * 1.25)
    for rect, p in zip(bars, probs):
        ax.text(rect.get_x() + rect.get_width() / 2, p + 0.01, f"{p:.2f}", ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "mm_pmf.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # --- Corresponding CDF using the same category order ---
    cumulative_probs = np.cumsum(probs)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
    # Build step function coordinates - jumps occur at the left edge of each bin
    x_positions = np.arange(len(categories))
    # CDF jumps at x_positions (left edge of bins), not after
    x_extended = np.concatenate([[-0.5], x_positions, [len(categories) - 0.5]])
    y_extended = np.concatenate([[0.0], cumulative_probs, [cumulative_probs[-1]]])
    
    # Draw step function with jumps at category positions
    ax.step(x_extended, y_extended, where='post', color='black', linewidth=2)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative (same order as PMF)')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.6, len(categories) - 0.4)
    ax.grid(True, alpha=0.3, axis='y')
    ax.text(0.5, 0.85, '?', transform=ax.transAxes, fontsize=20, ha='center', va='center', color='red', weight='bold')
    ax.text(0.5, 0.75, 'Meaningless for\\nnominal categories!', transform=ax.transAxes, fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    fig.tight_layout()
    fig.savefig(OUTDIR / "mm_cdf.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
