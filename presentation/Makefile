# Makefile for Applied Statistics Course Overview Presentation

MAIN = course_overview
TEX = $(MAIN).tex
PDF = $(MAIN).pdf

.PHONY: all clean view

all: $(PDF)

$(PDF): $(TEX)
	pdflatex $(TEX)
	pdflatex $(TEX)

view: $(PDF)
	open $(PDF)

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.fdb_latexmk *.fls *.synctex.gz

distclean: clean
	rm -f $(PDF)