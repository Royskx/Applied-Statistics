import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path(__file__).parent.parent / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


def lln_coin_tosses():
    np.random.seed(0)
    trials = [10, 50, 200, 1000]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=200)

    for i, n in enumerate(trials):
        ax = axes[i // 2, i % 2]
        cum_props = []
        for _ in range(500):
            tosses = np.random.binomial(1, 0.5, size=n)
            cum_prop = np.cumsum(tosses) / (np.arange(1, n + 1))
            ax.plot(cum_prop, alpha=0.05, color='blue')
        ax.set_title(f'n = {n}')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Cumulative proportion')
        ax.grid(True, alpha=0.3)

    fig.suptitle('LLN: Coin Tosses — Trajectories of Sample Proportion')
    fig.tight_layout()
    fig.savefig(OUTDIR / "lln_coin_tosses.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def lln_multiple_paths():
    np.random.seed(1)
    n = 200
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    for _ in range(30):
        tosses = np.random.binomial(1, 0.5, size=n)
        cum_prop = np.cumsum(tosses) / (np.arange(1, n + 1))
        ax.plot(cum_prop, alpha=0.6)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Sample proportion')
    ax.set_title('LLN: Multiple Paths (n=200)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "lln_multiple_paths.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def lln_small_vs_large():
    np.random.seed(2)
    small_n = 20
    large_n = 200
    trials = 300

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
    # small sample
    for _ in range(trials):
        tosses = np.random.binomial(1, 0.5, size=small_n)
        axes[0].plot(np.cumsum(tosses) / np.arange(1, small_n + 1), alpha=0.03, color='blue')
    axes[0].set_title('Small n = 20')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    # large sample
    for _ in range(trials):
        tosses = np.random.binomial(1, 0.5, size=large_n)
        axes[1].plot(np.cumsum(tosses) / np.arange(1, large_n + 1), alpha=0.03, color='blue')
    axes[1].set_title('Large n = 200')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('LLN: Small vs Large Sample Paths')
    fig.tight_layout()
    fig.savefig(OUTDIR / "lln_small_vs_large.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
