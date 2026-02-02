# LaTeX draft

This folder contains a minimal, *sectioned* LaTeX draft for writing a paper about the PhackingDetect agent.

## Structure

- `tex/build/main.tex`: main entry point (compile from `tex/build/`)
- `tex/sections/`: section files (`\input{...}`)
- `tex/figures/`: figure assets
- `tex/tables/`: table assets
- `tex/references.bib`: BibTeX references (p-hacking / selective reporting methods)

## Build

From `tex/build/`:

```bash
# If you have latexmk:
# latexmk -pdf -interaction=nonstopmode main.tex

# Portable fallback:
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

If TeX tries to write to `/var/cache/fonts` (permission error), set a local cache:

```bash
mkdir -p .texlive-cache
export VARTEXFONTS="$PWD/.texlive-cache"
export TEXMFVAR="$PWD/.texlive-cache"
export TEXMFCONFIG="$PWD/.texlive-cache"
```
