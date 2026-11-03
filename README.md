# GC-IDS

[![DOI](https://zenodo.org/badge/1443840377.svg)](https://doi.org/10.5281/zenodo.23907846)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code and result archive for the article **"Generative-Augmented Contrastive Representation Learning for NIDS: A Comparison
of CVAE and CGAN as View Generators"** (Generative-Contrastive framework, GC). Version `1.0-revision`.

**Archived version.** The exact code and results behind the revised manuscript are archived on Zenodo, release
`v1.0-revision`: [10.5281/zenodo.23907847](https://doi.org/10.5281/zenodo.23907847). The badge above points to the concept DOI,
which always resolves to the latest version.

> **Status of this code.** This is a clean PyTorch re-implementation. The code of the first submission was not preserved,
> so every number in the revised manuscript was produced by *this* code, with leakage-free preprocessing and a **reduced,
> shared training budget** (`configs/fast.yaml`); they replace the numbers of the first submission. Hyper-parameters were
> not tuned per method.

## What is in the repository
| Path | Content |
|---|---|
| `gcids/` | data loading and preprocessing, models, losses, generators (cVAE, cWGAN-GP with auxiliary classifier), contrastive methods, probes, metrics, generator-quality metrics, baselines, statistics, figures |
| `configs/` | `base.yaml` (defaults and variant definitions), one file per dataset, `fast.yaml` (budget of all reported results), `ext.yaml` (longer budget, cGAN without auxiliary classifier), ablation files |
| `tests/` | 22 unit and end-to-end tests (leakage rules, losses, metrics, statistics, every method on toy data) |
| `results_archive/` | raw result lines (`*.jsonl`), summary tables, figures and run logs of the four Kaggle runs behind the manuscript |
| `analysis/` | scripts that recompute every number of the manuscript from `results_archive/` (outputs in `analysis/outputs/`) |
| `kaggle/` | the notebook cells that were run on Kaggle (T4 GPUs) |
| `REPRODUCE.md` | which command produced which result file |
| `DECISIONS.md` | every design choice that the design notes did not fix |

## Install
```bash
pip install -r requirements.txt       # install the CUDA build of torch matching your driver first
pip install --no-deps pytorch-tabnet  # optional, only for the TabNet baseline
python -m pytest -q tests             # 22 tests, about 30 s on a CPU
```

## Data (not redistributed)
Download the public datasets and place the files exactly as follows.
```
data/nslkdd/KDDTrain+.txt            data/nslkdd/KDDTest+.txt                      (NSL-KDD, Tavallaee et al. 2009)
data/unswnb15/UNSW_NB15_training-set.csv   data/unswnb15/UNSW_NB15_testing-set.csv   (UNSW-NB15, Moustafa and Slay 2015)
```
Several public copies of UNSW-NB15 carry the two partition files under swapped names (the file called *training-set* has
82,332 rows). The loader detects this and follows the dataset paper: training partition 175,341 rows, testing partition
82,332 rows. The 80/20 train/validation split is drawn from the training partition with the run seed; the one-hot
encoder and the min-max scaler are fitted on the 80 % training part only.

## Run
```bash
# one dataset, one seed, the reduced budget used in the manuscript
python -m gcids.run --dataset nslkdd --config configs/base.yaml configs/nslkdd.yaml configs/fast.yaml --seeds 42 --device cuda
python -m gcids.report --results results       # tables (CSV and LaTeX) from results/raw
```
Runs are resumable: a finished (dataset, seed, variant) is skipped; results are appended to `results/raw/*.jsonl`.
`--variants` selects variants, `--shard i/n` splits the seeds over processes, `--out` chooses the result file.
`REPRODUCE.md` lists the exact commands behind each file of `results_archive/`.

## Methods (variant ids as in the manuscript)
S0 and S1 (supervised, S1 = cGAN oversampling); C1 (random masking), C1b-noise, C1b-mixup; C2-VAE and C2-GAN (generated view A
plus noisy real view B); label-role controls SupCon, C3-realpair (same-class real partner), GC-mask-GAN/VAE (same-class negatives
removed); SCARF; LightGBM, FT-Transformer, TabNet; `GenQ-VAE` / `GenQ-GAN` (generator quality: MMD², Wasserstein, correlation
difference, precision/recall, density/coverage, classifier two-sample test, memorization); `*-long` and `*-noAux` variants of
`configs/ext.yaml`.

## Reproducibility notes
* Seeds fix Python, NumPy and PyTorch RNGs; within a seed all methods share the split and the cached generator, so the paired
  tests are valid. Each result line stores the config hash, GPU name and library versions (results were produced on Tesla T4 GPUs,
  torch 2.11.0+cu128, Python 3.13).
* **GPU training is not bit-wise reproducible.** Re-running the same configuration and seed on another machine changed the
  Macro-F1 of a seed by 0.032 on average (up to 0.045) over three seeds and three independent runs of C2-GAN on NSL-KDD. The
  statistics of the manuscript average over this noise; small per-seed differences should not be over-interpreted. Expect
  your numbers to differ from `results_archive/` by about this amount per seed.
* `analysis/` recomputes the tables of the manuscript from the archived raw files:
  ```bash
  python analysis/calc_numbers.py results_archive analysis/outputs/numbers.json        # NSL-KDD
  python analysis/calc_unsw.py    results_archive analysis/outputs/numbers_unsw.json   # UNSW-NB15
  python analysis/calc_ext.py     results_archive analysis/outputs/numbers_ext.json    # ten-seed tests, longer budget, noAux
  python analysis/fig_dial.py     analysis/outputs/numbers.json analysis/outputs/fig4_dial.png
  ```

## Scope and limitations
Only NSL-KDD and UNSW-NB15 were evaluated. A CIC-IDS2017 loader is included but was not used for the manuscript. The code of
Kang et al. (M2M-VAEGAN) is not included. See the manuscript for the limitations of the study.

## Licence and citation
MIT licence (`LICENSE`). If you use this code or the archived results, please cite the article and this archive
(see also `CITATION.cff`, or use "Cite this repository" on GitHub).

```bibtex
@software{nguyen_gcids_2026,
  author    = {Nguyễn, Hữu Long and Nguyễn, Việt Hùng},
  title     = {{GC-IDS}: code and results for ``Generative-Augmented Contrastive Representation
               Learning for NIDS: A Comparison of CVAE and CGAN as View Generators''},
  year      = {2026},
  version   = {1.0-revision},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23907847},
  url       = {https://doi.org/10.5281/zenodo.23907847}
}
```
