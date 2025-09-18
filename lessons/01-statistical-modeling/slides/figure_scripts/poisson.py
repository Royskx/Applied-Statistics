import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

OUTDIR = Path(__file__).parent.parent / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)


def poisson_empirical():
    lam = 3.0
    sample = rng.poisson(lam=lam, size=2000)
    counts = pd.Series(sample).value_counts().sort_index()
    xs = np.arange(0, max(counts.index.max(), int(lam * 3)) + 1)

    plt.figure(figsize=(6, 3.5), dpi=200)
    emp = counts.values.astype(float) / float(counts.sum())
    plt.bar(counts.index.to_numpy(), emp, alpha=0.5, label="empirical")
    plt.plot(xs, stats.poisson.pmf(xs, lam), 'ro-', label=f"Poisson $\\lambda={lam}$")
    plt.xlabel("k")
    plt.ylabel("Probability")
    plt.title("Poisson: Empirical vs PMF")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTDIR / "poisson_empirical_pmf.png", bbox_inches="tight", pad_inches=0.02)
    plt.close()


def defects_poisson_by_batch():
    lam = 1.5
    sample_sizes = [10, 100]
    rng_local = np.random.default_rng(2025)

    k_max = int(max(8, np.ceil(lam * 4)))
    ks = np.arange(0, k_max + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=200)

    for ax, n in zip(axes.ravel(), sample_sizes):
        sample = rng_local.poisson(lam, size=n)
        counts = pd.Series(sample).value_counts().sort_index()

        emp = np.zeros_like(ks, dtype=float)
        total = float(counts.sum()) if counts.sum() else float(n)
        keys = counts.index.to_numpy().astype(int)
        vals = counts.values.astype(float)
        for k, v in zip(keys, vals):
            if 0 <= k <= k_max:
                emp[k] = v / total

        ax.bar(ks, emp, width=0.6, alpha=0.6, color="#54A24B", label=f"empirical (n={n})")
        ax.plot(ks, stats.poisson.pmf(ks, lam), 'ko-', label=f"Poisson(λ={lam})")
        ax.set_xlabel("Defects per batch")
        ax.set_title(f"Empirical vs Poisson (n = {n})")
        ax.set_ylim(0, max(emp.max(), stats.poisson.pmf(ks, lam).max()) * 1.2)
        ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(OUTDIR / "defects_poisson_by_batch.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
