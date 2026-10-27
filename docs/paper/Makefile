.PHONY: all figures snippets clean
PYTHON ?= python

all: jacobsrays.pdf

jacobsrays.pdf: jacobsrays.tex references.bib $(wildcard snippets/*.py) $(wildcard figures/*.pdf)
	latexmk -pdf -Werror -halt-on-error -interaction=nonstopmode -file-line-error -outdir=_build jacobsrays.tex
	cp _build/jacobsrays.pdf $@

figures:
	$(PYTHON) make_figures.py

snippets:
	$(PYTHON) snippets/scalar.py
	$(PYTHON) snippets/trt_trace.py

# Keep the checked-in reading copy and figures; remove only auxiliary files.
clean:
	latexmk -C -outdir=_build jacobsrays.tex
