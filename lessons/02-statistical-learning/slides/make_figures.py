"""Orchestrator for lesson-02 figure generation.

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
from figure_scripts.statistical_learning import generate_all_statistical_learning_figures


def main():
    """Generate all statistical learning figures."""
    generate_all_statistical_learning_figures()


if __name__ == "__main__":
    main()