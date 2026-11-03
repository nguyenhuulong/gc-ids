# Result archive

`commit1` to `commit4` are the four Kaggle sessions behind the manuscript (see `../REPRODUCE.md`). Each folder has
`raw/` (one JSON line per dataset, seed, variant and probe with metrics, embedding metrics, generator-quality metrics, cost and
config hash), `tables/` (summary tables written by `gcids.report`), `figures/` and `logs/` where they exist.
Embedding files (`emb/*.npz`, about 40 MB) and cached generators are not included. `figures_paper/fig4_dial.png` is Fig. 4 of
the manuscript; Fig. 3 is in `commit4/figures/` and Fig. 2 in `commit1/figures/` (NSL-KDD) and `commit3/figures/` (UNSW-NB15).
Tables of a single session contain only the variants and seeds of that session; the manuscript combines sessions with `../analysis/`.
