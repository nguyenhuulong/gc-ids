"""All metrics live here once (SPEC rule 8). Classification, embedding quality, and helpers."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, davies_bouldin_score,
                             matthews_corrcoef, silhouette_score)


# --------------------------------------------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------------------------------------------
def confusion(y: np.ndarray, p: np.ndarray, K: int) -> np.ndarray:
    return np.bincount(y * K + p, minlength=K * K).reshape(K, K)


def per_class_prf(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    rec = tp / np.maximum(cm.sum(1), 1)
    prec = tp / np.maximum(cm.sum(0), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return prec, rec, f1


def macro_f1(y: np.ndarray, p: np.ndarray, K: int) -> float:
    """Macro-F1 over classes that are present in y_true or y_pred (used for model selection on validation)."""
    cm = confusion(y, p, K)
    _, _, f1 = per_class_prf(cm)
    present = (cm.sum(1) + cm.sum(0)) > 0
    return float(f1[present].mean())


def g_mean(rec: np.ndarray, clip: float | None = None) -> float:
    """Geometric mean of per-class recalls; with ``clip`` the recalls are floored first (avoids exact zeros)."""
    r = rec if clip is None else np.clip(rec, clip, None)
    if np.any(r <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(r))))


def classification_metrics(y: np.ndarray, p: np.ndarray, proba: np.ndarray | None, K: int, class_names: list[str],
                           rare_ids: list[int]) -> dict:
    cm = confusion(y, p, K)
    prec, rec, f1 = per_class_prf(cm)
    out = {
        "accuracy": float((y == p).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "macro_f1": float(f1.mean()),
        "g_mean": g_mean(rec),
        "g_mean_clip": g_mean(rec, 1e-3),
        "mcc": float(matthews_corrcoef(y, p)),
        "rare_recall": float(rec[rare_ids].mean()) if rare_ids else float("nan"),
        "rare_f1": float(f1[rare_ids].mean()) if rare_ids else float("nan"),
        "per_class": {class_names[k]: {"precision": float(prec[k]), "recall": float(rec[k]), "f1": float(f1[k])}
                      for k in range(K)},
        "confusion": cm.tolist(),
    }
    if proba is not None:
        y1 = np.eye(K)[y]
        ap = [average_precision_score(y1[:, k], proba[:, k]) for k in range(K) if y1[:, k].sum() > 0]
        out["macro_pr_auc"] = float(np.mean(ap))
    return out


# --------------------------------------------------------------------------------------------------------------
# embedding quality (inputs are L2-normalised embeddings)
# --------------------------------------------------------------------------------------------------------------
def intra_inter_ratio(z: np.ndarray, y: np.ndarray) -> float:
    """mean distance to own class centroid / mean pairwise distance between class centroids."""
    ks = np.unique(y)
    cents = np.stack([z[y == k].mean(0) for k in ks])
    intra = np.mean([np.linalg.norm(z[y == k] - cents[i], axis=1).mean() for i, k in enumerate(ks)])
    d = np.linalg.norm(cents[:, None] - cents[None], axis=2)
    inter = d[np.triu_indices(len(ks), 1)].mean()
    return float(intra / max(inter, 1e-12))


def alignment(z: np.ndarray, y: np.ndarray, n_pairs: int = 50000, seed: int = 0) -> float:
    """Mean squared distance of random SAME-CLASS pairs (alpha=2). LOWER IS BETTER."""
    rng = np.random.RandomState(seed)
    order = np.argsort(y, kind="stable")
    ys = y[order]
    ks, start, cnt = np.unique(ys, return_index=True, return_counts=True)
    ok = cnt >= 2
    ks, start, cnt = ks[ok], start[ok], cnt[ok]
    c = rng.choice(len(ks), size=n_pairs)
    a = order[start[c] + (rng.rand(n_pairs) * cnt[c]).astype(int)]
    b = order[start[c] + (rng.rand(n_pairs) * cnt[c]).astype(int)]
    keep = a != b
    return float(((z[a[keep]] - z[b[keep]]) ** 2).sum(1).mean())


def uniformity(z: np.ndarray, n_pairs: int = 50000, seed: int = 0, t: float = 2.0) -> float:
    """log mean exp(-t ||u-v||^2) over random pairs (Wang & Isola 2020). LOWER IS BETTER."""
    rng = np.random.RandomState(seed)
    a, b = rng.randint(0, len(z), n_pairs), rng.randint(0, len(z), n_pairs)
    keep = a != b
    d2 = ((z[a[keep]] - z[b[keep]]) ** 2).sum(1)
    return float(np.log(np.mean(np.exp(-t * d2))))


@torch.no_grad()
def knn_accuracy(ztr: np.ndarray, ytr: np.ndarray, zte: np.ndarray, yte: np.ndarray, k: int = 20, K: int | None = None,
                 device: str = "cpu", chunk: int = 2048) -> float:
    K = K or int(max(ytr.max(), yte.max()) + 1)
    A = torch.as_tensor(ztr, device=device)
    yt = torch.as_tensor(ytr, device=device)
    pred = []
    kk = min(k, len(A))
    for i in range(0, len(zte), chunk):
        q = torch.as_tensor(zte[i:i + chunk], device=device)
        idx = torch.cdist(q, A).topk(kk, largest=False).indices
        votes = torch.zeros(len(q), K, device=device).scatter_add_(1, yt[idx], torch.ones_like(idx, dtype=torch.float))
        pred.append(votes.argmax(1).cpu().numpy())
    return float((np.concatenate(pred) == yte).mean())


def stratified_subsample(y: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Fixed-seed subsample of at most n points that keeps every class (proportional, min 5 per class)."""
    if len(y) <= n:
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    ks, cnt = np.unique(y, return_counts=True)
    take = np.maximum(np.minimum(cnt, 5), np.floor(cnt * n / len(y)).astype(int))
    idx = [rng.choice(np.where(y == k)[0], size=t, replace=False) for k, t in zip(ks, take)]
    return np.sort(np.concatenate(idx))


def embedding_metrics(z_te: np.ndarray, y_te: np.ndarray, z_tr: np.ndarray, y_tr: np.ndarray, K: int,
                      n_eval: int = 20000, seed: int = 0, device: str = "cpu") -> dict:
    idx = stratified_subsample(y_te, n_eval, seed)
    z, y = z_te[idx].astype(np.float64), y_te[idx]
    out = {"n_eval": int(len(idx))}
    n_cls = len(np.unique(y))
    if n_cls >= 2:
        out["silhouette"] = float(silhouette_score(z, y))
        out["davies_bouldin"] = float(davies_bouldin_score(z, y))
        out["intra_inter"] = intra_inter_ratio(z, y)
    out["alignment"] = alignment(z, y, seed=seed)
    out["uniformity"] = uniformity(z, seed=seed)
    itr = stratified_subsample(y_tr, n_eval, seed)
    out["knn_acc"] = knn_accuracy(z_tr[itr].astype(np.float32), y_tr[itr], z_te.astype(np.float32), y_te, 20, K, device)
    return out


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()
