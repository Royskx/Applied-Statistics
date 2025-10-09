"""Orchestrator for lesson-03 figure generation.

This script is intentionally thin: the heavy lifting lives in
`figure_scripts/*`. Import the specific functions and call them in the
correct order. Keep this file small so it's easy to maintain and run.
"""

from pathlib import Path

# Ensure OUTDIR exists for historical compatibility; modules also write
# into slides/figures (the directory is gitignored).
OUTDIR = Path(__file__).parent / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Import programmatic figure builders from figure_scripts
from figure_scripts import examples as fig_examples


def main():
    """Generate all estimator properties figures."""
    print("=" * 70)
    print("LESSON 03: ESTIMATOR PROPERTIES - FIGURE GENERATION")
    print("=" * 70)

    # Example figures
    print("\n[Example] Generating sample variance figure...")
    fig_examples.plot_sample_variance(OUTDIR)

    # Enhanced figures from modular scripts
    # Import modules and call their generate() functions explicitly
    from figure_scripts import bias_variance, consistency, efficiency
    from figure_scripts import confidence_intervals, bootstrap, delta_method

    print("\n[1/6] Generating bias-variance figures...")
    bias_variance.generate()

    print("[2/6] Generating consistency figures...")
    consistency.generate()

    print("[3/6] Generating efficiency figures...")
    efficiency.generate()

    print("[4/6] Generating confidence interval figures...")
    confidence_intervals.generate()

    print("[5/6] Generating bootstrap figures...")
    bootstrap.generate()

    print("[6/6] Generating delta method figures...")
    delta_method.generate()

    print("\n" + "=" * 70)
    print("✓ ALL FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nFigures saved in: {OUTDIR.absolute()}")
if __name__ == "__main__":
    main()