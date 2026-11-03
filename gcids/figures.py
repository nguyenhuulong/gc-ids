"""Figures: vector PDF + 300 dpi PNG, Okabe-Ito palette, paper method IDs on all axes.

    python -m gcids.figures --results results --dataset nslkdd
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stats import load_results
from .utils import get_logger

log = get_logger()
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000", "#999999", "#882255"]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.fontsize": 8, "figure.dpi": 100})


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def tsne_grid(results: Path, dataset: str, variants: list[str], class_names: list[str] | None, rare: list[str] | None, seed: int = 42) -> None:
    from sklearn.manifold import TSNE
    files = {v: results / "emb" / f"{dataset}__{v}__{seed}.npz" for v in variants}
    files = {v: f for v, f in files.items() if f.exists()}
    if not files:
        log.warning("no saved embeddings for %s", dataset)
        return
    n = len(files)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols + 1.8, 3.0 * rows), squeeze=False)
    first = np.load(next(iter(files.values())))
    ncls = int(first["y"].max()) + 1
    names = class_names or [f"class {i}" for i in range(ncls)]
    for ax, (v, f) in zip(axes.ravel(), files.items()):
        d = np.load(f)
        z, y = d["z"].astype(np.float32), d["y"]
        t = TSNE(2, random_state=0, init="pca", perplexity=min(30, len(z) // 4)).fit_transform(z)   # same subsample, same seed
        for k in range(ncls):
            m = y == k
            isr = rare and names[k] in rare
            ax.scatter(t[m, 0], t[m, 1], s=14 if isr else 4, c=OKABE[k % len(OKABE)], label=names[k], alpha=0.9 if isr else 0.6,
                       marker="D" if isr else "o", linewidths=0)
        ax.set_title(v); ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=2, frameon=False)
    fig.tight_layout()
    _save(fig, results / "figures", f"tsne_{dataset}")


def curves(results: Path, df: pd.DataFrame, dataset: str, family: str, order: list[str], labels: list[str], metric: str, name: str) -> None:
    d = df[(df.dataset == dataset) & (df.probe == "mlp") & df.variant.isin(order)]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    mean = [d[d.variant == v][metric].mean() for v in order]
    sd = [d[d.variant == v][metric].std(ddof=1) for v in order]
    n = [d[d.variant == v][metric].count() for v in order]
    ci = [1.96 * s / np.sqrt(max(k, 1)) if k > 1 else 0 for s, k in zip(sd, n)]
    x = np.arange(len(order))
    ax.plot(x, mean, "-o", c=OKABE[0]); ax.fill_between(x, np.array(mean) - ci, np.array(mean) + ci, alpha=0.25, color=OKABE[0])
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel(metric); ax.set_title(f"{family} ({dataset}, 95% CI)")
    _save(fig, results / "figures", f"{name}_{dataset}")


def cost_scatter(results: Path, df: pd.DataFrame, dataset: str) -> None:
    d = df[(df.dataset == dataset) & (df.probe == "mlp")]
    col = [c for c in d.columns if c.startswith("cost:") and c.endswith(":seconds")]
    if d.empty or not col:
        return
    d = d.assign(total=d[col].sum(axis=1, skipna=True))
    g = d.groupby("variant").agg(total=("total", "mean"), f1=("macro_f1", "mean"))
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for i, (v, r) in enumerate(g.iterrows()):
        ax.scatter(r.total, r.f1, c=OKABE[i % len(OKABE)], s=30)
        ax.annotate(v, (r.total, r.f1), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("total wall time per run (s, log)"); ax.set_ylabel("Macro-F1")
    _save(fig, results / "figures", f"cost_vs_f1_{dataset}")


def perclass_heatmap(results: Path, df: pd.DataFrame, dataset: str, variants: list[str]) -> None:
    d = df[(df.dataset == dataset) & (df.probe == "mlp") & df.variant.isin(variants)]
    cols = [c for c in d.columns if c.startswith("f1:")]
    if d.empty or not cols:
        return
    m = d.groupby("variant")[cols].mean().reindex([v for v in variants if v in set(d.variant)])
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(cols), 0.5 + 0.4 * len(m)))
    im = ax.imshow(m.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[3:] for c in cols], rotation=45, ha="right")
    ax.set_yticks(range(len(m))); ax.set_yticklabels(m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", color="w" if m.values[i, j] < 0.6 else "k", fontsize=7)
    fig.colorbar(im, ax=ax, label="F1")
    _save(fig, results / "figures", f"perclass_f1_{dataset}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tsne-variants", nargs="+", default=["S0", "C1", "C1b-noise", "C2-VAE", "C2-GAN", "SupCon", "C3-realpair"])
    a = ap.parse_args()
    res = Path(a.results)
    df = load_results(res)
    names = None
    cp = res / "tables" / f"dataset_stats_{a.dataset}.csv"
    rare = None
    if cp.exists():
        t = pd.read_csv(cp)
        names, rare = t["class"].tolist(), t[t.rare]["class"].tolist()
    tsne_grid(res, a.dataset, a.tsne_variants, names, rare)
    curves(res, df, a.dataset, "view-mix ratio", [f"pgen-{p}" for p in [0.0, 0.25, 0.5, 0.75, 1.0]], ["0", "0.25", "0.5", "0.75", "1"], "macro_f1", "pgen_macro_f1")
    curves(res, df, a.dataset, "view-mix ratio", [f"pgen-{p}" for p in [0.0, 0.25, 0.5, 0.75, 1.0]], ["0", "0.25", "0.5", "0.75", "1"], "silhouette", "pgen_silhouette")
    curves(res, df, a.dataset, "GAN checkpoints", [f"C2-GAN-it{k}" for k in [1000, 2000, 5000, 10000, 20000]], ["1k", "2k", "5k", "10k", "20k"], "macro_f1", "dial_gan_macro_f1")
    cost_scatter(res, df, a.dataset)
    perclass_heatmap(res, df, a.dataset, ["S0", "S1", "C1", "C1b-noise", "C1b-mixup", "C2-VAE", "C2-GAN", "SupCon", "C3-realpair"])


if __name__ == "__main__":
    main()
