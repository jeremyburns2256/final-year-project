# Report

LaTeX source for the final year project report.

## Building

```bash
cd report
latexmk -pdf main.tex     # produces build/main.pdf
latexmk -C                # empties build/
```

Everything generated — `main.pdf` along with the `.aux`/`.log`/`.toc` churn —
is written to `build/` by `.latexmkrc`, so this directory holds only source.
`build/` is gitignored.

If your editor has its own LaTeX integration, point its output directory at
`build/` too, or let it read `.latexmkrc`.

## Layout

| Path | Contents |
| --- | --- |
| `main.tex` | Document outline only: the `\input` list of chapters. Start here. |
| `.latexmkrc` | Build configuration; sends all output to `build/`. |
| `preamble.tex` | Packages, page geometry, caption/numbering style, title metadata. |
| `frontmatter.tex` | Title page, table of contents, lists of figures and tables. |
| `chapters/` | One file per chapter, prefixed in reading order. |
| `appendices/` | One file per appendix. |
| `backmatter.tex` | Bibliography style and `\bibliography` call. |
| `references.bib` | BibTeX database, grouped by topic with `% ──` banners. |
| `images/` | Figures included by the chapters. |
| `assets/` | Source PDFs: datasheets, network price lists, the Gantt chart. |
| `build/` | Generated output, including `main.pdf`. Gitignored; safe to delete. |

## Conventions

- **Add a chapter** by creating `chapters/NN-name.tex` and adding one
  `\input{chapters/NN-name}` line to `main.tex` in the right position.
- **Figures** are referenced as `\includegraphics{images/foo.png}` from any
  chapter file; `\graphicspath` in the preamble resolves the path.
- **Labels** are namespaced by kind: `ch:`, `sec:`, `sub:`, `fig:`, `tab:`, `eq:`,
  `app:`.
- **Unresolved items** are marked with a `% TODO` comment on the line concerned.

## Relationship to `code/`

The results chapter reports numbers produced by the simulations in
`code/trading/`. The source text and the figures behind
`chapters/05-results.tex` are `code/trading/RESULTS.md` and the Bokeh pages in
`code/trading/plots/`. When a simulation is re-run, both need updating.
