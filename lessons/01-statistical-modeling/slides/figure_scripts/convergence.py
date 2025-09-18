import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

OUTDIR = Path(__file__).parent.parent / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


def det_vs_random_convergence():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), dpi=200)
    n_vals = np.arange(1, 51)
    det_seq = 1 / n_vals
    ax1.plot(n_vals, det_seq, 'b-o', markersize=3, linewidth=2, label='$a_n = 1/n$')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Limit = 0')
    ax1.set_xlabel('n')
    ax1.set_ylabel('Value')
    ax1.set_title('Deterministic Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.1)

    np.random.seed(42)
    n_paths = 5
    for i in range(n_paths):
        random_seq = np.random.normal(0, 1, len(n_vals)) / np.sqrt(n_vals)
        ax2.plot(n_vals, random_seq, alpha=0.7, linewidth=1.5)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Limit = 0')
    ax2.set_xlabel('n')
    ax2.set_ylabel('Value')
    ax2.set_title('Random Convergence (Sample Paths)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 1)

    fig.tight_layout()
    fig.savefig(OUTDIR / "det_vs_random_convergence.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def as_convergence():
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    np.random.seed(123)
    n_vals = np.arange(1, 101)
    n_paths = 8

    for i in range(n_paths):
        noise = np.random.normal(0, 1, len(n_vals)) / (n_vals**0.6)
        path = 1 + noise
        ax.plot(n_vals, path, alpha=0.8, linewidth=1.2)

    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label='True limit = 1')
    ax.set_xlabel('n')
    ax.set_ylabel('$X_n(\\omega)$')
    ax.set_title('Almost Sure Convergence: Sample Paths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.5)
    plt.tight_layout()
    fig.savefig(OUTDIR / "as_convergence.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def in_prob_convergence():
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=200)
    np.random.seed(456)
    n_values = [10, 50, 200, 1000]

    for i, n in enumerate(n_values):
        ax = axes[i // 2, i % 2]
        samples = np.random.normal(0, 1 / np.sqrt(n), 1000)
        ax.hist(samples, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Target = 0')
        ax.set_title(f'n = {n}')
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, max(5, np.sqrt(n)))
        if i >= 2:
            ax.set_xlabel('Value')
        if i % 2 == 0:
            ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Convergence in Probability: Distributions Narrowing', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / "in_prob_convergence.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def in_dist_convergence():
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    x = np.linspace(-3, 3, 1000)
    n_values = [5, 10, 25, 100]
    colors = ['blue', 'green', 'orange', 'purple']

    for n, color in zip(n_values, colors):
        if n == 100:
            cdf_vals = stats.norm.cdf(x)
            ax.plot(x, cdf_vals, color=color, linewidth=2.5, label=f'n = {n} (limit)', linestyle='-')
        else:
            cdf_vals = stats.t.cdf(x, df=n)
            ax.plot(x, cdf_vals, color=color, linewidth=1.5, label=f'n = {n}', linestyle='--')

    ax.set_xlabel('x')
    ax.set_ylabel(r'$F_n(x)$')
    ax.set_title('Convergence in Distribution: CDFs Aligning')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(OUTDIR / "in_dist_convergence.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def convergence_table():
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    ax.axis('tight')
    ax.axis('off')

    table_data = [
        ['Mode', 'Definition', 'Intuition', 'Strength', 'Example'],
        ['Almost Sure', '$P(\\lim X_n = X) = 1$', 'Pathwise convergence', 'Strongest', '$X_n = 1 + \\frac{Z_n}{n}$'],
        ['In Probability', '$P(|X_n - X| > \\varepsilon) \\to 0$', 'Deviations become rare', 'Medium', '$X_n = \\mathbf{1}_{\\{U \\leq 1/n\\}}$'],
        ['In Distribution', '$F_{X_n}(t) \\to F_X(t)$', 'Shapes converge', 'Weakest', 'Central Limit Theorem']
    ]

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

    ax.set_title('Modes of Convergence Summary', fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(OUTDIR / "convergence_table.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
