TEX=tectonic
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
	$(TEX) -X compile $(SRC)

clean:
	rm -f $(OUT) *.aux *.log *.out
