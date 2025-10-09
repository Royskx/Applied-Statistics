"""Example figure builders for lesson 03.

These are intentionally minimal. They produce PNGs into the slides/figures
directory so that the Makefile's dependency graph finds them.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_sample_variance(outdir: Path):
    """Plot distribution of sample variance for samples from N(0,1).

    Produces `fig_sample_variance.png` in `outdir`.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2025)
    n = 30
    trials = 500
    s2 = np.array([rng.normal(0, 1, n).var(ddof=1) for _ in range(trials)])

    plt.figure(figsize=(4, 3))
    plt.hist(s2, bins=30, color="#4C72B0", edgecolor="white")
    plt.axvline(1.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Sample variance")
    plt.ylabel("Frequency")
    plt.tight_layout()
    outpath = outdir / "fig_sample_variance.png"
    plt.savefig(outpath, dpi=150)
    plt.close()

    print(f"Wrote {outpath}")
