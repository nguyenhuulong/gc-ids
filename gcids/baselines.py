"""Non-contrastive reference baselines: LightGBM, FT-Transformer, TabNet (optional dependency)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .data import Split
from .probes import fit_classifier, predict_proba


def run_lightgbm(sp: Split, cfg: dict, seed: int) -> tuple[np.ndarray, dict]:
    import lightgbm as lgb

    clf = lgb.LGBMClassifier(
        objective="multiclass", class_weight="balanced", n_estimators=cfg.get("n_estimators", 1000),
        learning_rate=cfg.get("lr", 0.05), num_leaves=cfg.get("num_leaves", 63), subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=seed, n_jobs=cfg.get("n_jobs", -1), verbose=-1)
    clf.fit(sp.X_tr, sp.y_tr, eval_set=[(sp.X_val, sp.y_val)],
            callbacks=[lgb.early_stopping(cfg.get("early_stopping", 50), verbose=False)])
    proba = np.zeros((len(sp.X_te), sp.K), dtype=np.float32)
    proba[:, clf.classes_] = clf.predict_proba(sp.X_te)
    return proba, {"best_iteration": int(clf.best_iteration_ or 0)}


class FTTransformer(nn.Module):
    """Compact FT-Transformer (Gorishniy et al. 2021): every (one-hot or numeric) input column becomes a token."""

    def __init__(self, d_in: int, K: int, d: int = 64, heads: int = 4, layers: int = 2, ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_in, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(d_in, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x[:, :, None] * self.w[None] + self.b[None]
        t = torch.cat([self.cls.expand(len(x), -1, -1), t], 1)
        return self.out(self.norm(self.tr(t)[:, 0]))


def run_ftt(sp: Split, cfg: dict, seed: int, device: torch.device) -> tuple[np.ndarray, dict]:
    torch.manual_seed(seed)
    Xtr, ytr = torch.as_tensor(sp.X_tr, device=device), torch.as_tensor(sp.y_tr, device=device)
    Xva, yva = torch.as_tensor(sp.X_val, device=device), torch.as_tensor(sp.y_val, device=device)
    Xte = torch.as_tensor(sp.X_te, device=device)
    m = FTTransformer(sp.dim, sp.K, cfg.get("d", 64), cfg.get("heads", 4), cfg.get("layers", 2)).to(device)
    m, info = fit_classifier(m, Xtr, ytr, Xva, yva, sp.K, cfg.get("epochs", 30), cfg.get("lr", 3e-4),
                             cfg.get("batch", 256), weight_decay=1e-5, seed=seed, optimizer="adamw")
    info["n_params"] = int(sum(p.numel() for p in m.parameters()))
    return predict_proba(m, Xte), info


def run_tabnet(sp: Split, cfg: dict, seed: int, device: torch.device) -> tuple[np.ndarray, dict]:
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install pytorch-tabnet to run the TabNet baseline") from e
    cnt = np.bincount(sp.y_tr, minlength=sp.K).astype(float)
    w = {k: float(len(sp.y_tr) / (sp.K * max(c, 1))) for k, c in enumerate(cnt)}
    clf = TabNetClassifier(seed=seed, device_name=str(device), verbose=0)
    clf.fit(sp.X_tr, sp.y_tr, eval_set=[(sp.X_val, sp.y_val)], eval_metric=["balanced_accuracy"],
            max_epochs=cfg.get("epochs", 100), patience=cfg.get("patience", 15), batch_size=cfg.get("batch", 1024),
            virtual_batch_size=128, weights=w if cfg.get("class_weights", True) else 0)
    proba = np.zeros((len(sp.X_te), sp.K), dtype=np.float32)
    proba[:, clf.classes_] = clf.predict_proba(sp.X_te)
    return proba, {"best_epoch": int(clf.best_epoch)}
