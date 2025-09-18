import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

OUTDIR = Path(__file__).parent.parent / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


def density_plots():
    xs = np.linspace(-3, 6, 400)
    uni_a, uni_b = 0, 1
    exp_lam = 1.0
    norm_mu, norm_sigma = 1.5, 1.0

    plt.figure(figsize=(6, 3.5), dpi=200)
    # Uniform density
    ux = np.linspace(uni_a, uni_b, 2)
    uy = np.ones_like(ux) * (1 / (uni_b - uni_a))
    plt.plot(ux, uy, label="Uniform(0,1)")
    # Exponential density
    mask = xs >= 0
    plt.plot(xs[mask], exp_lam * np.exp(-exp_lam * xs[mask]), label="Exponential(1)")
    # Normal density
    plt.plot(xs, stats.norm.pdf(xs, loc=norm_mu, scale=norm_sigma), label="Normal(1.5,1)")
    plt.ylim(bottom=0)
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Selected Densities")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTDIR / "densities.png", bbox_inches="tight", pad_inches=0.02)
    plt.close()


def uniform_pdf_cdf():
    a, b = 0.0, 1.0
    x = np.linspace(a - 0.3, b + 0.3, 400)
    pdf = np.where((x >= a) & (x <= b), 1.0 / (b - a), 0.0)
    cdf = np.piecewise(x, [x < a, (x >= a) & (x <= b), x > b], [0.0, lambda t: (t - a) / (b - a), 1.0])

    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=200)
    ax2 = ax.twinx()
    ax.fill_between(x, 0, pdf, color="#4C78A8", alpha=0.25, label="PDF")
    ax.plot(x, pdf, color="#4C78A8", linewidth=1.8)
    ax2.plot(x, cdf, color="#333333", linewidth=2.6, label="CDF")
    ax.set_ylim(0, max(pdf) * 1.25 + 1e-9)
    ax2.set_ylim(0, 1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("Density f(x)")
    ax2.set_ylabel("CDF F(x)")
    plt.tight_layout()
    fig.savefig(OUTDIR / "uniform_pdf_cdf.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
