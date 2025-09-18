"""Orchestrator for lesson-01 figure generation.

This script is intentionally thin: the heavy lifting lives in
`figure_scripts/*`. Import the specific functions and call them in the
correct order. Keep this file small so it's easy to maintain and run.
"""

from pathlib import Path

# Ensure OUTDIR exists for historical compatibility; modules also write
# into slides/figures (the directory is gitignored).
OUTDIR = Path(__file__).parent / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Import modular figure generators from the package we added.
from figure_scripts.poisson import poisson_empirical, defects_poisson_by_batch
from figure_scripts.densities import density_plots, uniform_pdf_cdf
from figure_scripts.mm import mm_cdf
from figure_scripts.qqplot import qq_plots
from figure_scripts.convergence import (
    det_vs_random_convergence,
    as_convergence,
    in_prob_convergence,
    in_dist_convergence,
    convergence_table,
)
from figure_scripts.lln import (
    lln_coin_tosses,
    lln_multiple_paths,
    lln_small_vs_large,
)
from figure_scripts.clt_figures import generate_clt_figures


def main():
    # Poisson / densities / miscellaneous
    poisson_empirical()
    density_plots()
    uniform_pdf_cdf()
    defects_poisson_by_batch()
    mm_cdf()

    # QQ-plot figures
    qq_plots()

    # Convergence visuals
    det_vs_random_convergence()
    as_convergence()
    in_prob_convergence()
    in_dist_convergence()
    convergence_table()

    # LLN visuals
    lln_coin_tosses()
    lln_multiple_paths()
    lln_small_vs_large()

    # CLT visuals
    generate_clt_figures()


if __name__ == "__main__":
    main()
