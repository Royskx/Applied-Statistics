# ==============================================================================
# Makefile for Applied Statistics Course Overview Presentation
# ==============================================================================
# This Makefile provides a robust build system for the course overview
# presentation using XeLaTeX. It handles compilation, artifact management,
# and provides multiple targets for different use cases.
#
# Quick Start:
#   make              - Full compilation (2 passes, outputs to ../slides.pdf)
#   make quick        - Fast draft compilation (single pass)
#   make clean        - Remove build artifacts (keeps PDF)
#   make clean-all    - Remove everything including PDF
#   make help         - Show detailed help
#
# Directory Structure:
#   main.tex          - Main LaTeX document
#   sections/*.tex    - Modular content sections
#   build/            - Build artifacts (gitignored)
#   ../slides.pdf     - Final output (presentation root)
# ==============================================================================

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

MAIN = main
TEX = $(MAIN).tex
PDF = $(MAIN).pdf
OUTPUT_PDF = ../slides.pdf

# LaTeX compiler (XeLaTeX for better Unicode support)
LATEX = xelatex
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

# Directories
BUILD_DIR = build
SECTIONS_DIR = sections

# Source files (for dependency tracking)
SECTION_FILES = $(wildcard $(SECTIONS_DIR)/*.tex)

# ------------------------------------------------------------------------------
# Phony Targets (don't correspond to actual files)
# ------------------------------------------------------------------------------

.PHONY: all quick presentation-pdf clean clean-all help

# ------------------------------------------------------------------------------
# Default Target
# ------------------------------------------------------------------------------

all: $(OUTPUT_PDF)

# ------------------------------------------------------------------------------
# Main Compilation Target
# ------------------------------------------------------------------------------
# Performs a complete 2-pass compilation for proper cross-references,
# table of contents, and navigation. Moves artifacts to build/ and
# copies final PDF to presentation root.

$(OUTPUT_PDF): $(TEX) $(SECTION_FILES)
	@echo "=================================="
	@echo "Building Course Overview Presentation"
	@echo "=================================="
	# Clean any stray auxiliary files before compilation to prevent corruption
	# This ensures a fresh build every time and avoids issues with outdated .aux files
	@rm -f *.aux *.log *.out *.fdb_latexmk *.fls *.nav *.snm *.toc *.synctex.gz
	# Create build directory if it doesn't exist
	@mkdir -p $(BUILD_DIR)
	# First LaTeX compilation pass
	# Note: XeLaTeX creates auxiliary files in current directory regardless of flags
	# -interaction=nonstopmode: Prevents interactive prompts during batch compilation
	# -halt-on-error: Stops compilation immediately if errors are encountered
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	# Second LaTeX compilation pass for table of contents and cross-references
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	# Move all build artifacts to build directory to keep slides folder clean
	@mv -f *.aux *.log *.out *.fdb_latexmk *.fls *.nav *.snm *.toc $(BUILD_DIR)/ 2>/dev/null || true
	@mv -f *.synctex.gz $(BUILD_DIR)/ 2>/dev/null || true
	# Move PDF to parent presentation directory
	@mv -f $(PDF) $(OUTPUT_PDF)
	@echo "✅ Compiled presentation to $(OUTPUT_PDF)"

# ------------------------------------------------------------------------------
# Quick Draft Compilation
# ------------------------------------------------------------------------------
# Single-pass compilation for rapid iteration during content development.
# Faster but may have incorrect cross-references or missing TOC entries.

quick: $(TEX) $(SECTION_FILES)
	@echo "=================================="
	@echo "Quick Draft Compilation"
	@echo "=================================="
	@rm -f *.aux *.log *.out *.fdb_latexmk *.fls *.nav *.snm *.toc *.synctex.gz
	@mkdir -p $(BUILD_DIR)
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	@mv -f *.aux *.log *.out *.fdb_latexmk *.fls *.nav *.snm *.toc $(BUILD_DIR)/ 2>/dev/null || true
	@mv -f *.synctex.gz $(BUILD_DIR)/ 2>/dev/null || true
	@mv -f $(PDF) $(OUTPUT_PDF)
	@echo "✅ Quick draft compiled to $(OUTPUT_PDF)"

# ------------------------------------------------------------------------------
# Alias Targets
# ------------------------------------------------------------------------------

presentation-pdf: all

# ------------------------------------------------------------------------------
# Cleanup Targets
# ------------------------------------------------------------------------------

# Remove build artifacts but keep the compiled PDF
clean:
	@echo "Cleaning build artifacts..."
	@rm -f *.aux *.log *.nav *.out *.snm *.toc *.fdb_latexmk *.fls *.synctex.gz
	@rm -rf $(BUILD_DIR)
	@echo "✅ Build artifacts removed"

# Remove everything including the compiled PDF
clean-all: clean
	@echo "Removing compiled PDF..."
	@rm -f $(OUTPUT_PDF) $(PDF)
	@echo "✅ All generated files removed"

# ------------------------------------------------------------------------------
# Help Target
# ------------------------------------------------------------------------------

help:
	@echo "======================================================================"
	@echo "Applied Statistics Course Overview Presentation - Build System"
	@echo "======================================================================"
	@echo ""
	@echo "OVERVIEW:"
	@echo "  This Makefile compiles the course overview presentation using XeLaTeX."
	@echo "  The presentation introduces course structure, learning objectives,"
	@echo "  assessment methods, and philosophy to students."
	@echo ""
	@echo "AVAILABLE TARGETS:"
	@echo "  make                  Full compilation (default)"
	@echo "                        - 2 compilation passes for cross-references"
	@echo "                        - Outputs to ../slides.pdf"
	@echo "                        - Moves build artifacts to build/"
	@echo ""
	@echo "  make quick            Fast draft compilation"
	@echo "                        - Single pass, faster for content iteration"
	@echo "                        - May have incomplete cross-references"
	@echo ""
	@echo "  make presentation-pdf Alias for 'make all'"
	@echo ""
	@echo "  make clean            Remove build artifacts"
	@echo "                        - Keeps compiled PDF"
	@echo "                        - Cleans *.aux, *.log, *.nav, etc."
	@echo ""
	@echo "  make clean-all        Remove everything"
	@echo "                        - Removes build artifacts AND PDF"
	@echo "                        - Complete cleanup"
	@echo ""
	@echo "  make help             Show this help message"
	@echo ""
	@echo "USAGE EXAMPLES:"
	@echo "  Quick editing workflow:"
	@echo "    $$ make quick && open ../slides.pdf"
	@echo ""
	@echo "  Final production build:"
	@echo "    $$ make clean && make"
	@echo ""
	@echo "  Verify changes:"
	@echo "    $$ make && open ../slides.pdf"
	@echo ""
	@echo "FILE STRUCTURE:"
	@echo "  main.tex              Main LaTeX document"
	@echo "  sections/*.tex        Modular content sections"
	@echo "  build/                Build artifacts (gitignored)"
	@echo "  ../slides.pdf         Final compiled presentation"
	@echo ""
	@echo "REQUIREMENTS:"
	@echo "  - LaTeX distribution (TeX Live 2023+ or MacTeX 2023+)"
	@echo "  - XeLaTeX compiler"
	@echo "  - Standard Beamer packages"
	@echo ""
	@echo "For more details, see README.md in this directory."
	@echo "======================================================================"
