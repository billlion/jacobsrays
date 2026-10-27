# User and methods guide

Read [jacobsrays.pdf](jacobsrays.pdf), or edit [jacobsrays.tex](jacobsrays.tex)
and [references.bib](references.bib). This is a first user-facing draft,
not a journal submission. No author list has been assigned.

The guide covers motivation, the Szotten-code history, coordinates and
ordered traversal, scalar/LRT/TRT operators, TTRT conventions, matched
adjoints, Python/MATLAB/CIL use, validation, the published Gaussian slice
example, and measured performance. ORT remains a proposed application
without an assumed definition or unpublished reconstruction material.

## Version scope

The documented baseline is main revision
`854c6aa60d64be38619be6d268ceb6b7b56da5f7` (version 0.1.0), including the
optional atomic adjoint merged in
[PR #7](https://github.com/billlion/jacobsrays/pull/7).
The atomic benchmark was measured on revision
`f6464f3cb3565c8a2f0021138e96efcd0e7958c0` before that merge.
The source and recorded JSON are in `benchmarks/`.

## Build and check

A TeX Live installation with `latexmk`, pdfLaTeX, BibTeX, Latin Modern,
`natbib`, `listings`, `microtype`, `booktabs`, `tabularx`, `hyperref` and
`bookmark` is sufficient. On Debian/Ubuntu these are available through
`latexmk`, `lmodern`, `texlive-latex-recommended`, `texlive-latex-extra`
and `texlive-fonts-recommended`.

```sh
make -C docs/paper
# Force a fresh build even when using the checked-in PDF:
make -B -C docs/paper
# After installing jacobsrays with its scipy extra:
make -C docs/paper snippets PYTHON=python
```

The output is `docs/paper/jacobsrays.pdf`. Auxiliary files stay in `_build/`.
`latexmk -Werror` rejects unresolved citations/references after the final
pass. The documentation workflow builds the PDF, runs the two complete
Python listings, and uploads a PDF artifact. Existing numerical CI is
unchanged. When editing, inspect rendered pages for layout problems as
well as checking the TeX log.

The two figures are committed so readers can build without Python graphics
dependencies. To regenerate them, install NumPy and Matplotlib, then run:

```sh
make -C docs/paper figures PYTHON=python
make -C docs/paper
```

`make_figures.py` disables variable PDF date metadata. The atomic plot reads
the recorded JSON in `benchmarks/`; it does not rerun a benchmark or use
hard-coded plotted measurements. No timings are CI pass/fail thresholds.
The geometry figure is schematic.

The main scalar benchmark is recorded at
`benchmarks/results_2026-10-26.json`; the atomic benchmark is recorded at
`benchmarks/atomic_results_2026-10-27.json`. Slice experiment settings and limitations
are also described in [../trt_slices.md](../trt_slices.md).
