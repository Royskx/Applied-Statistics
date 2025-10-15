# ==============================================================================
# Lesson 02: Statistical Learning - LaTeX Build System
# ==============================================================================
# Comprehensive build system for LaTeX Beamer presentation with figure generation
# Organizes build artifacts in dedicated directory to maintain clean source folder

# ==============================================================================
# CONFIGURATION VARIABLES
# ==============================================================================
# LaTeX engine with Unicode support for modern fonts and special characters
TEX=xelatex

# Main LaTeX source file containing document structure and content
SRC=main.tex

# Output PDF filename (placed in parent lesson directory)
OUT=../slides.pdf

# Build directory for containing all LaTeX auxiliary files and intermediate outputs
# This keeps the main slides directory clean and organized
BUILD=build

# Directory containing generated figure files (PNG format)
FIGDIR=figures

# Python script that generates all statistical figures used in the presentation
FIGSCRIPT=make_figures.py

# Python interpreter with required scientific computing packages installed
PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/applied-stats/bin/python

# Pattern to find all PNG figure files for dependency tracking
FIGS=$(wildcard $(FIGDIR)/*.png)

# ==============================================================================
# MAIN BUILD TARGETS
# ==============================================================================

# Default target: builds figures and compiles presentation
# Dependencies ensure figures are generated before LaTeX compilation
.PHONY: all clean figs lesson-pdf quick clean-outputs clean-all
all: figs lesson-pdf

# Quick compile target: reuses previous compilation artifacts for faster builds
# Use this when you've only made minor changes and want to test quickly
# WARNING: May produce incorrect output if structural changes were made
quick: $(SRC) $(FIGS)
	$(TEX) -interaction=nonstopmode -halt-on-error $(SRC)
	mv -f *.aux *.log *.out *.fdb_latexmk *.fls *.nav *.snm *.toc $(BUILD)/ 2>/dev/null || true
	mv -f *.synctex.gz $(BUILD)/ 2>/dev/null || true
	mv -f main.pdf $(OUT)
	@echo "⚡ Quick compiled slides to $(OUT)"

# ==============================================================================
# FIGURE GENERATION TARGET
# ==============================================================================
# Generates all statistical visualizations used in the presentation
# Uses Python scripts with NumPy/Matplotlib for reproducible figure generation
figs:
	$(PYTHON) $(FIGSCRIPT)

# ==============================================================================
# PDF COMPILATION TARGET
# ==============================================================================
# Compiles the LaTeX presentation and organizes all build artifacts
# Final PDF is placed directly in parent lesson directory
lesson-pdf: $(SRC) $(FIGS)
	# Clean any stray auxiliary files before compilation to prevent corruption
	# This ensures a fresh build every time and avoids issues with outdated .aux files
	rm -f *.aux *.log *.out *.fdb_latexmk *.fls *.nav *.snm *.toc *.synctex.gz

	# Create build directory if it doesn't exist
	mkdir -p $(BUILD)

	# First LaTeX compilation pass
	# Note: XeLaTeX creates auxiliary files in current directory regardless of flags
	# -interaction=nonstopmode: Prevents interactive prompts during batch compilation
	# -halt-on-error: Stops compilation immediately if errors are encountered
	$(TEX) -interaction=nonstopmode -halt-on-error $(SRC)

	# Second LaTeX compilation pass for table of contents and cross-references
	$(TEX) -interaction=nonstopmode -halt-on-error $(SRC)

	# Move all build artifacts to build directory to keep slides folder clean
	mv -f *.aux *.log *.out *.fdb_latexmk *.fls *.nav *.snm *.toc $(BUILD)/ 2>/dev/null || true
	mv -f *.synctex.gz $(BUILD)/ 2>/dev/null || true

	# Move PDF to parent lesson directory
	mv -f main.pdf $(OUT)
	@echo "✅ Compiled slides to $(OUT)"

# ==============================================================================
# CLEAN TARGETS
# ==============================================================================
# Remove LaTeX build artifacts from build directory
clean:
	rm -rf $(BUILD)
	@echo "🧹 Cleaned build artifacts"

# Remove generated figures only
clean-figures:
	rm -rf $(FIGDIR)
	@echo "🧹 Cleaned generated figures"

# Remove build artifacts and figures, but keep the final PDF
clean-outputs: clean clean-figures
	@echo "🧹 Cleaned build artifacts and figures"

# Remove all generated files including final PDF
clean-all: clean clean-figures
	rm -f $(OUT)
	# Also remove any stray PNG files in slides directory
	rm -f *.png
	@echo "🧹 Cleaned all generated files"

# ==============================================================================
# HELP TARGET
# ==============================================================================
.PHONY: help
help:
	@echo "Lesson 02: Statistical Learning - Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build figures + slides (default)"
	@echo "  make figs         - Generate figures only"
	@echo "  make lesson-pdf   - Compile LaTeX only (requires figures)"
	@echo "  make quick        - Quick rebuild (reuses artifacts)"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make clean-figures - Remove generated figures"
	@echo "  make clean-outputs - Remove build artifacts + figures"
	@echo "  make clean-all    - Remove all generated files"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Output: $(OUT)"