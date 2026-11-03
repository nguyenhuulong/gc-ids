"""Loading results, paired statistics (t, exact Wilcoxon, bootstrap CI, Cohen's d_z, Holm) and table writers."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from .utils import read_jsonl

# direction: +1 = higher is better, -1 = lower is better
DIRECTION = {
    "accuracy": 1, "balanced_accuracy": 1, "macro_f1": 1, "g_mean": 1, "g_mean_clip": 1, "mcc": 1, "macro_pr_auc": 1,
    "rare_recall": 1, "rare_f1": 1, "silhouette": 1, "davies_bouldin": -1, "intra_inter": -1, "alignment": -1,
    "uniformity": -1, "knn_acc": 1,
}
CLS_KEYS = ["accuracy", "balanced_accuracy", "macro_f1", "g_mean", "g_mean_clip", "mcc", "macro_pr_auc", "rare_recall", "rare_f1"]
EMB_KEYS = ["silhouette", "davies_bouldin", "intra_inter", "alignment", "uniformity", "knn_acc"]


# --------------------------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------------------------
def load_results(results_dir: str | Path = "results") -> pd.DataFrame:
    """Long table: one row per (dataset, seed, variant, probe) with flattened metrics, embedding metrics and costs.
    If a (dataset, seed, variant) was run twice the LAST record wins."""
    rows = []
    for p in sorted((Path(results_dir) / "raw").glob("*.jsonl")):
        recs = {}
        for r in read_jsonl(p):
            recs[(r["dataset"], r["seed"], r["variant"])] = r
        for r in recs.values():
            if "metrics" not in r:
                continue
            for probe, m in r["metrics"].items():
                row = {"dataset": r["dataset"], "seed": r["seed"], "variant": r["variant"], "method": r["method"], "probe": probe,
                       "val_f1": (r.get("val_f1") or {}).get(probe, np.nan), "checkpoint": r.get("checkpoint")}
                for k in CLS_KEYS:
                    row[k] = m.get(k, np.nan)
                for c, d in m.get("per_class", {}).items():
                    row[f"f1:{c}"], row[f"recall:{c}"] = d["f1"], d["recall"]
                for k, v in (r.get("embedding") or {}).items():
                    row[k] = v
                for stage, c in (r.get("cost") or {}).items():
                    if isinstance(c, dict):
                        row[f"cost:{stage}:seconds"] = c.get("seconds", np.nan)
                        row[f"cost:{stage}:peak_gpu_mb"] = c.get("peak_gpu_mb", np.nan)
                        row[f"cost:{stage}:n_params"] = c.get("n_params", np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


def load_genq(results_dir: str | Path = "results") -> pd.DataFrame:
    rows = []
    for p in sorted((Path(results_dir) / "raw").glob("*.jsonl")):
        for r in read_jsonl(p):
            if "genq" not in r:
                continue
            for scope in ("overall", "macro_mean"):
                row = {"dataset": r["dataset"], "seed": r["seed"], "variant": r["variant"], "method": r["method"],
                       "checkpoint": r.get("checkpoint"), "scope": scope}
                row.update({k: v for k, v in r["genq"][scope].items()})
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------------------
# paired statistics
# --------------------------------------------------------------------------------------------------------------
def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input; NaN stays NaN)."""
    p = np.array(pvals, dtype=float)
    ok = ~np.isnan(p)
    adj = np.full_like(p, np.nan)
    idx = np.argsort(p[ok])
    m = ok.sum()
    run = 0.0
    vals = p[ok][idx]
    out = np.empty(m)
    for i, v in enumerate(vals):
        run = max(run, (m - i) * v)
        out[i] = min(1.0, run)
    tmp = np.empty(m)
    tmp[idx] = out
    adj[ok] = tmp
    return adj.tolist()


def bootstrap_ci(d: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired comparison A vs B over seeds (arrays aligned by seed)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    out = {"n": n, "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)), "mean_diff": float(d.mean()) if n else np.nan}
    if n < 2:
        return {**out, "t_p": np.nan, "wilcoxon_p": np.nan, "ci_boot": (np.nan, np.nan), "ci_t": (np.nan, np.nan), "dz": np.nan,
                "min_wilcoxon_p": np.nan}
    sd = d.std(ddof=1)
    if sd == 0:
        t_p = 1.0 if d.mean() == 0 else 0.0
        ci_t = (float(d.mean()), float(d.mean()))
    else:
        t_p = float(sps.ttest_rel(a, b).pvalue)
        h = sps.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
        ci_t = (float(d.mean() - h), float(d.mean() + h))
    nz = d[d != 0]
    if len(nz) == 0:
        w_p = 1.0
    else:
        try:
            w_p = float(sps.wilcoxon(nz, method="exact" if len(nz) <= 25 else "approx").pvalue)
        except ValueError:
            w_p = np.nan
    return {**out, "t_p": t_p, "wilcoxon_p": w_p, "ci_boot": bootstrap_ci(d), "ci_t": ci_t,
            "dz": float(d.mean() / sd) if sd > 0 else (0.0 if d.mean() == 0 else float("inf")),
            "min_wilcoxon_p": float(2 / 2 ** n)}   # smallest attainable two-sided exact p with n pairs


def _vals(df: pd.DataFrame, dataset: str, variant: str, metric: str, probe: str, seeds: list[int]) -> np.ndarray:
    s = df[(df.dataset == dataset) & (df.variant == variant) & (df.probe == probe)].set_index("seed")[metric]
    return np.array([s[x] for x in seeds])


def paired_table(df: pd.DataFrame, dataset: str, pairs: list[tuple[str, str]], metrics: list[str], probe: str = "mlp") -> pd.DataFrame:
    """Rows: (A, B, metric). Holm correction is applied within each metric over all listed pairs (one family)."""
    rows = []
    for metric in metrics:
        fam = []
        for A, B in pairs:
            sa = set(df[(df.dataset == dataset) & (df.variant == A) & (df.probe == probe)].seed)
            sb = set(df[(df.dataset == dataset) & (df.variant == B) & (df.probe == probe)].seed)
            seeds = sorted(sa & sb)
            if len(seeds) < 2 or metric not in df:
                continue
            a, b = _vals(df, dataset, A, metric, probe, seeds), _vals(df, dataset, B, metric, probe, seeds)
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            fam.append({"dataset": dataset, "metric": metric, "A": A, "B": B, "probe": probe, **paired_test(a, b)})
        for key in ("t_p", "wilcoxon_p"):
            adj = holm([r[key] for r in fam])
            for r, v in zip(fam, adj):
                r[key + "_holm"] = v
        rows += fam
    out = pd.DataFrame(rows)
    if len(out):
        out["ci_boot_lo"] = out.ci_boot.map(lambda t: t[0]); out["ci_boot_hi"] = out.ci_boot.map(lambda t: t[1])
        out["ci_t_lo"] = out.ci_t.map(lambda t: t[0]); out["ci_t_hi"] = out.ci_t.map(lambda t: t[1])
        out = out.drop(columns=["ci_boot", "ci_t"])
    return out


# --------------------------------------------------------------------------------------------------------------
# aggregate tables
# --------------------------------------------------------------------------------------------------------------
def aggregate(df: pd.DataFrame, dataset: str, metrics: list[str], probe: str = "mlp", variants: list[str] | None = None) -> pd.DataFrame:
    d = df[(df.dataset == dataset) & (df.probe == probe)]
    if variants:
        d = d[d.variant.isin(variants)]
    g = d.groupby("variant", sort=False)
    mean, std = g[metrics].mean(), g[metrics].std(ddof=1)
    res = mean.copy()
    for m in metrics:
        res[m + "_std"] = std[m]
    res["n_seeds"] = g.seed.nunique()
    if variants:
        res = res.reindex([v for v in variants if v in res.index])
    return res


def to_latex(agg: pd.DataFrame, metrics: list[str], daggers: dict[tuple[str, str], bool] | None = None, caption: str = "",
             label: str = "", digits: int = 4) -> str:
    """booktabs table: mean ± std, best per column in bold (direction aware), dagger where the method differs from the
    reference after Holm correction (``daggers[(variant, metric)] = True``)."""
    best = {}
    for m in metrics:
        col = agg[m]
        best[m] = col.idxmax() if DIRECTION.get(m, 1) > 0 else col.idxmin()
    head = " & ".join(["Method"] + [m.replace("_", r"\_") + (r"$\uparrow$" if DIRECTION.get(m, 1) > 0 else r"$\downarrow$") for m in metrics])
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small",
             r"\begin{tabular}{l" + "c" * len(metrics) + "}", r"\toprule", head + r" \\", r"\midrule"]
    for v, row in agg.iterrows():
        cells = []
        for m in metrics:
            s = f"{row[m]:.{digits}f}$\\pm${row[m + '_std']:.{digits}f}" if not np.isnan(row[m + "_std"]) else f"{row[m]:.{digits}f}"
            if best[m] == v:
                s = r"\textbf{" + s + "}"
            if daggers and daggers.get((v, m)):
                s += r"$^\dagger$"
            cells.append(s)
        lines.append(" & ".join([str(v).replace("_", r"\_")] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def dial_correlation(df: pd.DataFrame, genq: pd.DataFrame, dataset: str, family: str, outcomes: list[str], probe: str = "mlp") -> pd.DataFrame:
    """Spearman correlation between generator MMD^2 and downstream outcomes across checkpoints x seeds.
    family: 'GAN' or 'VAE' (variants named C2-GAN-it*, GenQ-GAN-it*, C2-VAE-ep*, GenQ-VAE-ep*)."""
    tag = "it" if family == "GAN" else "ep"
    g = genq[(genq.dataset == dataset) & genq.variant.str.startswith(f"GenQ-{family}-{tag}") & (genq.scope == "overall")]
    d = df[(df.dataset == dataset) & df.variant.str.startswith(f"C2-{family}-{tag}") & (df.probe == probe)]
    if g.empty or d.empty:
        return pd.DataFrame()
    g = g.assign(ckpt=g.checkpoint.astype(int))[["seed", "ckpt", "mmd2"]]
    d = d.assign(ckpt=d.checkpoint.astype(int))
    m = d.merge(g, on=["seed", "ckpt"])
    rows = []
    for o in outcomes:
        mm = m[["mmd2", o]].dropna()
        if len(mm) >= 4:
            r = sps.spearmanr(mm.mmd2, mm[o])
            rows.append({"dataset": dataset, "family": family, "outcome": o, "n": len(mm), "spearman": float(r.statistic), "p": float(r.pvalue)})
    return pd.DataFrame(rows)
