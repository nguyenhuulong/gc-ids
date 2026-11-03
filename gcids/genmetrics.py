"""Generator-quality metrics, overall (pooled over classes) and per class.

MMD^2 (unbiased RBF, bandwidth = median heuristic x {0.5,1,2,4}), mean per-feature Wasserstein-1, correlation-matrix
difference (Frobenius), precision/recall (k=5, Kynkaanniemi 2019), density/coverage (k=5, Naeem 2020), C2ST AUROC,
diversity (mean pairwise distance) and a memorisation indicator.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .generators import GenWrapper


def _pdist2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.cdist(A, B).pow(2)


def median_bandwidth(X: torch.Tensor, max_n: int = 2000) -> float:
    if len(X) > max_n:
        X = X[torch.randperm(len(X), device=X.device)[:max_n]]
    d = torch.cdist(X, X)
    d = d[torch.triu(torch.ones_like(d, dtype=torch.bool), 1)]
    return float(d.median().clamp(min=1e-6))


def mmd2_unbiased(X: torch.Tensor, Y: torch.Tensor, sigma: float) -> float:
    n, m = len(X), len(Y)
    kxx = torch.exp(-_pdist2(X, X) / (2 * sigma ** 2))
    kyy = torch.exp(-_pdist2(Y, Y) / (2 * sigma ** 2))
    kxy = torch.exp(-_pdist2(X, Y) / (2 * sigma ** 2))
    sxx = (kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
    syy = (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
    return float(sxx + syy - 2 * kxy.mean())


def mmd_multi(real: torch.Tensor, fake: torch.Tensor, mults=(0.5, 1, 2, 4)) -> dict:
    s0 = median_bandwidth(real)
    vals = {f"mmd2_x{m}": mmd2_unbiased(real, fake, s0 * m) for m in mults}
    vals["mmd2"] = float(np.mean(list(vals.values())))
    return vals


def kth_nn(A: torch.Tensor, B: torch.Tensor, k: int, exclude_self: bool, chunk: int = 2048) -> torch.Tensor:
    out = []
    kk = k + 1 if exclude_self else k
    for i in range(0, len(A), chunk):
        d = torch.cdist(A[i:i + chunk], B)
        out.append(d.topk(min(kk, B.shape[0]), largest=False).values[:, -1])
    return torch.cat(out)


def prdc(real: torch.Tensor, fake: torch.Tensor, k: int = 5) -> dict:
    rr = kth_nn(real, real, k, True)
    ff = kth_nn(fake, fake, k, True)
    d_rf = torch.cdist(real, fake)           # real x fake
    precision = (d_rf <= rr[:, None]).any(0).float().mean()          # fake inside some real ball
    recall = (d_rf.t() <= ff[:, None]).any(0).float().mean()         # real inside some fake ball
    density = (d_rf <= rr[:, None]).float().sum(0).mean() / k
    coverage = (d_rf.min(1).values <= rr).float().mean()
    return {"precision": float(precision), "recall": float(recall), "density": float(density), "coverage": float(coverage)}


def c2st_auc(real: np.ndarray, fake: np.ndarray, min_real: int = 200, seed: int = 0) -> float:
    if len(real) < min_real:
        return float("nan")
    X = np.vstack([real, fake])
    y = np.r_[np.zeros(len(real)), np.ones(len(fake))]
    clf = LogisticRegression(max_iter=300)
    return float(cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean())


def mean_wasserstein(real: np.ndarray, fake: np.ndarray) -> float:
    return float(np.mean([wasserstein_distance(real[:, j], fake[:, j]) for j in range(real.shape[1])]))


def corr_diff(real: np.ndarray, fake: np.ndarray) -> float:
    def c(a):
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.corrcoef(a, rowvar=False)
        return np.nan_to_num(m)
    return float(np.linalg.norm(c(real) - c(fake), "fro"))


def diversity(X: torch.Tensor, max_n: int = 1000) -> float:
    if len(X) > max_n:
        X = X[:max_n]
    d = torch.cdist(X, X)
    n = len(X)
    return float(d.sum() / max(n * (n - 1), 1))


def memorisation(real: torch.Tensor, fake: torch.Tensor) -> float:
    """Share of fake samples whose nearest real neighbour is closer than the MEDIAN real-to-real nearest-neighbour distance."""
    rn = kth_nn(real, real, 1, True)
    fn = kth_nn(fake, real, 1, False)
    return float((fn < rn.median()).float().mean())


def _one(real_t: torch.Tensor, fake_t: torch.Tensor, c2st: bool, do_prdc: bool = True) -> dict:
    real, fake = real_t.cpu().numpy(), fake_t.cpu().numpy()
    out = mmd_multi(real_t, fake_t)
    out["wasserstein"] = mean_wasserstein(real, fake)
    out["corr_diff"] = corr_diff(real, fake) if len(real) > 5 else float("nan")
    if do_prdc and len(real) > 6 and len(fake) > 6:
        out.update(prdc(real_t, fake_t, k=min(5, len(real) - 1, len(fake) - 1)))
    out["diversity_real"], out["diversity_fake"] = diversity(real_t), diversity(fake_t)
    out["memorisation"] = memorisation(real_t, fake_t)
    out["c2st_auc"] = c2st_auc(real, fake) if c2st else float("nan")
    return out


@torch.no_grad()
def generator_quality(gen: GenWrapper, X: torch.Tensor, y: torch.Tensor, class_names: list[str], M: int = 2000,
                      seed: int = 0, pooled_cap: int = 5000, do_c2st: bool = True) -> dict:
    """Per class (M = min(N_k, 2000) fake vs real train samples) and overall (pooled over classes)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    K = len(class_names)
    per, reals, fakes = {}, [], []
    for k in range(K):
        idx = torch.where(y == k)[0]
        if len(idx) < 3:
            continue
        m = min(len(idx), M)
        sel = idx[torch.randperm(len(idx), generator=g)[:m].to(idx.device)]
        real = X[sel]
        fake = gen.sample(torch.full((m,), k, dtype=torch.long, device=X.device))
        per[class_names[k]] = {"n": int(m), **_one(real, fake, do_c2st)}
        reals.append(real); fakes.append(fake)
    R, F_ = torch.cat(reals), torch.cat(fakes)
    if len(R) > pooled_cap:
        sel = torch.randperm(len(R), generator=g)[:pooled_cap].to(R.device)
        R, F_ = R[sel], F_[sel]
    overall = {"n": int(len(R)), **_one(R, F_, do_c2st)}
    keys = [k for k in overall if k != "n"]
    macro = {k: float(np.nanmean([v[k] for v in per.values()])) for k in keys}
    return {"overall": overall, "macro_mean": macro, "per_class": per}


@torch.no_grad()
def gan_mode_collapse(gen: GenWrapper, K: int, device, n_z: int = 500, per_class_cap: int = 10) -> dict:
    """For fixed y and many z: mean pairwise distance (diversity) per class and its minimum (mode-collapse indicator)."""
    d = []
    for k in range(min(K, per_class_cap)):
        s = gen.sample(torch.full((n_z,), k, dtype=torch.long, device=device))
        d.append(diversity(s, n_z))
    return {"div_per_class": d, "div_min": float(np.min(d)), "div_mean": float(np.mean(d))}


@torch.no_grad()
def vae_latent_stats(gen: GenWrapper, X: torch.Tensor, y: torch.Tensor, K: int, n: int = 20000) -> dict:
    """cVAE: aggregated posterior vs prior N(0, I): mean/var of mu and KL per latent dimension."""
    from .generators import onehot
    m = gen.model
    idx = torch.randperm(len(X), device=X.device)[:n]
    mu, lv = m.encode(X[idx], onehot(y[idx], K))
    kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp())).mean(0)
    return {"mu_mean": float(mu.mean()), "mu_var_mean": float(mu.var(0).mean()), "kl_per_dim": kl.cpu().tolist(),
            "active_dims": int((kl > 0.01).sum())}
