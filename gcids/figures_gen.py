"""Real-versus-generated figures (Reviewer 1, comment 3).

Needs the cached generators of a finished run (cache/gen/*.pt) and the dataset. Run on the GPU machine:

    python -m gcids.figures_gen --dataset nslkdd --config configs/base.yaml configs/nslkdd.yaml configs/fast.yaml \
        --seed 42 --cache cache --results results --out results/figures

Figures (vector PDF + 300 dpi PNG, written to --out):
  realgen_pca_<ds>        one panel per class: real vs cVAE vs cGAN samples in a PCA plane fitted on real training data
  realgen_marginals_<ds>  histograms of the highest-variance features, real vs cVAE vs cGAN (all classes, and rarest class)
  realgen_corr_<ds>       |corr(real) - corr(generated)| over the highest-variance features, one map per generator
  genq_mmd_<ds>           per-class MMD^2 (mean +- std over seeds) for cVAE and cGAN, read from results/raw/*.jsonl

The generator config must be exactly the one used in the run (same --config list), otherwise the cache tag differs and
the script stops instead of silently training a new generator (pass --allow-train to train one anyway).
The plotting functions take plain numpy arrays, so they can be re-used and tested without torch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"Real": "#7F7F7F", "cVAE": "#0072B2", "cGAN": "#D55E00"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "figure.dpi": 100})


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------
# sampling (torch side)
# ----------------------------------------------------------------------------------------------------------------
def load_and_sample(dataset: str, configs: list[str], seed: int, cache: str, data_root: str, n_max: int,
                    allow_train: bool, device: str = "auto") -> dict:
    """Return {'real': (X, y), 'cVAE': (X, y), 'cGAN': (X, y), 'class_names', 'feature_names', 'rare_ids'}.
    Real = a stratified subsample (<= n_max per class) of the TRAINING split the generators were fitted on."""
    import torch

    from .data import prepare
    from .generators import gen_tag, get_generator
    from .utils import get_device, load_config

    cfg = load_config(*configs)
    dev = get_device(device)
    sp = prepare(dataset, data_root, seed, cfg.get("data", {}).get("val_frac", 0.2),
                 cfg.get("data", {}).get("rare_thresh", 0.015), cfg.get("data", {}).get("strict_counts", True))
    rng = np.random.RandomState(seed)
    idx = np.concatenate([rng.permutation(np.where(sp.y_tr == k)[0])[:n_max] for k in range(sp.K)])
    out = {"real": (sp.X_tr[idx], sp.y_tr[idx]), "class_names": sp.class_names,
           "feature_names": list(sp.feature_names), "rare_ids": list(sp.rare_ids)}
    yt = torch.as_tensor(sp.y_tr[idx], dtype=torch.long, device=dev)
    for kind, label in (("vae", "cVAE"), ("gan", "cGAN")):
        gcfg = {k: v for k, v in cfg[kind].items() if k != "snapshots"}
        path = Path(cache) / "gen" / f"{dataset}_{kind}_{gen_tag(kind, gcfg)}_{seed}.pt"
        if not path.exists() and not allow_train:
            raise FileNotFoundError(f"{path} not found. Use the same --config list as the run, point --cache at the run's "
                                    f"cache folder, or pass --allow-train.")
        gen = get_generator(kind, dataset, seed, sp.X_tr, sp.y_tr, sp.K, gcfg, cache, dev, cfg[kind].get("snapshots"))
        torch.manual_seed(seed)
        with torch.no_grad():
            xg = gen.sample(yt).float().cpu().numpy()
        out[label] = (xg, sp.y_tr[idx])
    return out


# ----------------------------------------------------------------------------------------------------------------
# plotting (numpy only)
# ----------------------------------------------------------------------------------------------------------------
def _pca_fit(X: np.ndarray, k: int = 2):
    mu = X.mean(0)
    _, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    var = s ** 2 / max(1, len(X) - 1)
    return mu, vt[:k], var[:k] / var.sum()


def plot_pca_panels(real, gens: dict, class_names: list[str], rare_ids: list[int], out: Path, name: str,
                    max_points: int = 1500, seed: int = 0) -> None:
    Xr, yr = real
    mu, comps, evr = _pca_fit(Xr)
    proj = lambda X: (X - mu) @ comps.T
    K = len(class_names)
    ncol = min(5, K)
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.0 * nrow), squeeze=False)
    rng = np.random.RandomState(seed)
    for k in range(nrow * ncol):
        ax = axes[k // ncol][k % ncol]
        if k >= K:
            ax.axis("off")
            continue
        # generated first, real last (on top, dark, small) so real support stays visible
        layers = [*gens.items(), ("Real", real)]
        pts = []
        for label, (X, y) in layers:
            P = proj(X[y == k])
            if len(P) > max_points:
                P = P[rng.choice(len(P), max_points, replace=False)]
            pts.append(P)
            ax.scatter(P[:, 0], P[:, 1], s=1.2 if label == "Real" else 2.5, alpha=0.8 if label == "Real" else 0.35,
                       c="#222222" if label == "Real" else COLORS[label], label=label, linewidths=0, rasterized=True)
        allp = np.concatenate(pts)                         # zoom to this class (real + generated)
        lo, hi = np.percentile(allp, 1, 0), np.percentile(allp, 99, 0)
        pad = 0.15 * (hi - lo) + 1e-3
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        title = class_names[k] + (" (rare)" if k in rare_ids else "") + f"  n={int((yr == k).sum())}"
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    h = [plt.Line2D([], [], marker="o", ls="", color="#222222" if l == "Real" else COLORS[l], label=l)
         for l in ["Real", *gens]]
    fig.suptitle(f"Real training data (black) and generated samples in the PCA plane of the real data "
                 f"(PC1 {evr[0]:.0%}, PC2 {evr[1]:.0%} of variance); each panel zoomed to its class")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False)
    _save(fig, out, name)


def top_var_features(X: np.ndarray, k: int) -> np.ndarray:
    v = X.var(0)
    return np.argsort(-v)[:k]


def plot_marginals(real, gens: dict, feature_names: list[str], out: Path, name: str, k: int = 6,
                   cls: int | None = None, cls_name: str | None = None) -> None:
    Xr, yr = real
    sel_r = (yr == cls) if cls is not None else np.ones(len(yr), bool)
    feats = top_var_features(Xr[sel_r], k)
    fig, axes = plt.subplots(1, k, figsize=(1.9 * k, 1.8), squeeze=False)
    bins = np.linspace(0, 1, 31)
    for j, f in enumerate(feats):
        ax = axes[0][j]
        for label, (X, y) in [("Real", real), *gens.items()]:
            sel = (y == cls) if cls is not None else np.ones(len(y), bool)
            ax.hist(np.clip(X[sel, f], 0, 1), bins=bins, density=True, histtype="step", lw=1.2, color=COLORS[label],
                    label=label)
        fn = feature_names[f] if f < len(feature_names) else f"f{f}"
        ax.set_title(fn if len(fn) <= 22 else fn[:20] + "…")
        ax.set_yticks([])
    axes[0][0].legend(frameon=False, loc="upper right")
    fig.suptitle("Marginal distributions, " + (f"class {cls_name}" if cls is not None else "all classes") +
                 " (highest-variance features, min-max scaled)", y=1.04)
    fig.tight_layout()
    _save(fig, out, name)


def plot_corr_error(real, gens: dict, out: Path, name: str, k: int = 30) -> dict:
    Xr, _ = real
    feats = top_var_features(Xr, k)
    def corr(X):
        C = np.corrcoef(X[:, feats], rowvar=False)
        return np.nan_to_num(C)
    Cr = corr(Xr)
    fig, axes = plt.subplots(1, len(gens), figsize=(3.0 * len(gens), 2.8), squeeze=False)
    errs = {}
    for j, (label, (X, _)) in enumerate(gens.items()):
        E = np.abs(Cr - corr(X))
        errs[label] = float(E[np.triu_indices(len(feats), 1)].mean())
        im = axes[0][j].imshow(E, vmin=0, vmax=1, cmap="viridis")
        axes[0][j].set_title(f"{label}: mean |Δcorr| = {errs[label]:.3f}")
        axes[0][j].set_xticks([]); axes[0][j].set_yticks([])
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="|corr(real) − corr(generated)|")
    _save(fig, out, name)
    return errs


def collect_perclass_mmd(results_dir: Path, dataset: str) -> dict:
    """{'cVAE'|'cGAN': {class: [mmd2 per seed]}} from GenQ-VAE / GenQ-GAN records (last record per seed wins)."""
    recs = {}
    for p in sorted((Path(results_dir) / "raw").glob("*.jsonl")):
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("dataset") == dataset and r.get("variant") in ("GenQ-VAE", "GenQ-GAN") and "genq" in r:
                recs[(r["variant"], r["seed"])] = r
    out: dict = {"cVAE": {}, "cGAN": {}}
    for (v, _), r in sorted(recs.items()):
        lab = "cVAE" if v.endswith("VAE") else "cGAN"
        for c, d in r["genq"]["per_class"].items():
            out[lab].setdefault(c, []).append(d["mmd2"])
    return out


def plot_mmd_bars(mmd: dict, class_names: list[str], out: Path, name: str) -> None:
    labels = [c for c in class_names if any(c in mmd[g] for g in mmd)]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(3.2, 0.55 * len(labels) + 1.5), 2.4))
    for j, g in enumerate(["cVAE", "cGAN"]):
        m = [np.mean(mmd[g].get(c, [np.nan])) for c in labels]
        s = [np.std(mmd[g].get(c, [np.nan]), ddof=1) if len(mmd[g].get(c, [])) > 1 else 0 for c in labels]
        n = max([len(v) for v in mmd[g].values()] or [0])
        ax.bar(x + (j - 0.5) * w, m, w, yerr=s, capsize=2, color=COLORS[g], label=f"{g} (n={n} seeds)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("MMD² (lower is better)")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--n-per-class", type=int, default=2000)
    ap.add_argument("--allow-train", action="store_true")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    out = Path(a.out)
    d = load_and_sample(a.dataset, a.config, a.seed, a.cache, a.data_root, a.n_per_class, a.allow_train, a.device)
    gens = {"cVAE": d["cVAE"], "cGAN": d["cGAN"]}
    names, rare = d["class_names"], d["rare_ids"]
    plot_pca_panels(d["real"], gens, names, rare, out, f"realgen_pca_{a.dataset}")
    plot_marginals(d["real"], gens, d["feature_names"], out, f"realgen_marginals_{a.dataset}")
    yr = d["real"][1]
    rarest = int(np.argmin([(yr == k).sum() for k in range(len(names))]))
    plot_marginals(d["real"], gens, d["feature_names"], out, f"realgen_marginals_{a.dataset}_{names[rarest]}",
                   cls=rarest, cls_name=names[rarest])
    errs = plot_corr_error(d["real"], gens, out, f"realgen_corr_{a.dataset}")
    mmd = collect_perclass_mmd(Path(a.results), a.dataset)
    if any(mmd[g] for g in mmd):
        plot_mmd_bars(mmd, names, out, f"genq_mmd_{a.dataset}")
    print("figures written to", out, "| mean |Δcorr|:", {k: round(v, 4) for k, v in errs.items()})


if __name__ == "__main__":
    main()
