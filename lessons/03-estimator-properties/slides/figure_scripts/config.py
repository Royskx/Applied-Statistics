"""Common configuration and utilities for figure generation.

This module contains shared settings, color palettes, and utility functions
used across all estimator properties figures.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Output directory
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe palette
COLORS = {
    'black': '#000000',
    'orange': '#E69F00',
    'skyblue': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'pink': '#CC79A7'
}

# Set up plotting defaults
sns.set_theme(context="talk", style="whitegrid")
sns.set_palette(list(COLORS.values()))

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "axes.titleweight": "bold",
    "legend.fontsize": 14,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.8",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2.5,
    "axes.linewidth": 1.2,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# Global random number generator with fixed seed
rng = np.random.default_rng(2025)
