TEX=xelatex
SRC=main.tex
OUT=main.pdf
FIGDIR=figures
FIGSCRIPT=make_figures.py
PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/applied-stats/bin/python
FIGS=$(wildcard $(FIGDIR)/*.png)

.PHONY: all clean figs

all: figs $(OUT)

figs:
	$(PYTHON) $(FIGSCRIPT)

$(OUT): $(SRC) $(FIGS)
	# Compile with XeLaTeX (fontspec/Unicode support) - first run
	$(TEX) -interaction=nonstopmode -halt-on-error $(SRC)
	# Second run for TOC/bookmarks
	$(TEX) -interaction=nonstopmode -halt-on-error $(SRC)

# Alternate: build with tectonic (uncomment to use)
tectonic: $(SRC) $(FIGS)
	tectonic -X compile $(SRC)

clean:
	rm -f $(OUT) *.aux *.log *.out