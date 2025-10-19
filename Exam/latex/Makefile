# ==============================================================================
# Exam PDFs Build System
# ==============================================================================
# Generates exam and solution PDFs with auxiliary files isolated in build/
# Final PDFs are placed in the parent Exam directory for easy distribution.

EXAMS := first-exam-version-A first-exam-version-B
SOLUTIONS := first-exam-version-A-solution first-exam-version-B-solution

BUILD_DIR := build
OUT_DIR := ..

TARGETS := $(addsuffix .pdf,$(EXAMS) $(SOLUTIONS))
OUTPUTS := $(addprefix $(OUT_DIR)/,$(TARGETS))

.PHONY: all clean distclean a b asol bsol help

all: $(OUTPUTS)

a: $(OUT_DIR)/first-exam-version-A.pdf
b: $(OUT_DIR)/first-exam-version-B.pdf
asol: $(OUT_DIR)/first-exam-version-A-solution.pdf
bsol: $(OUT_DIR)/first-exam-version-B-solution.pdf

$(OUT_DIR)/%.pdf: %.tex
	@mkdir -p $(BUILD_DIR)
	pdflatex -interaction=nonstopmode -shell-escape -output-directory=$(BUILD_DIR) $<
	pdflatex -interaction=nonstopmode -shell-escape -output-directory=$(BUILD_DIR) $<
	mv -f $(BUILD_DIR)/$*.pdf $(OUT_DIR)/$*.pdf

clean:
	rm -rf $(BUILD_DIR)

distclean: clean
	rm -f $(OUTPUTS)

help:
	@echo "Exam PDF build system"
	@echo "Targets:"
	@echo "  make            - Build all exam and solution PDFs"
	@echo "  make a          - Build Version A exam"
	@echo "  make asol       - Build Version A solutions"
	@echo "  make b          - Build Version B exam"
	@echo "  make bsol       - Build Version B solutions"
	@echo "  make clean      - Remove LaTeX auxiliary files (build/)"
	@echo "  make distclean  - Remove all generated PDFs and build artifacts"
