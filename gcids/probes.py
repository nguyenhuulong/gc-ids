"""Stage 2: probes on frozen embeddings, and the generic class-weighted classifier trainer (S0/S1/baselines)."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import macro_f1
from .models import LinearHead, MLPHead


def class_weights(y: torch.Tensor, K: int) -> torch.Tensor:
    """w_k = N_total / (K * N_k); absent classes get weight 0."""
    cnt = torch.bincount(y, minlength=K).float()
    w = len(y) / (K * cnt.clamp(min=1))
    return torch.where(cnt > 0, w, torch.zeros_like(w))


@torch.no_grad()
def extract(enc: nn.Module, X: torch.Tensor, normalize: bool = True, bs: int = 8192) -> torch.Tensor:
    enc.eval()
    out = []
    for i in range(0, len(X), bs):
        h = enc(X[i:i + bs])
        out.append(F.normalize(h, dim=1) if normalize else h)
    return torch.cat(out)


@torch.no_grad()
def predict_proba(model: nn.Module, X: torch.Tensor, bs: int = 8192) -> np.ndarray:
    model.eval()
    return torch.cat([F.softmax(model(X[i:i + bs]), 1) for i in range(0, len(X), bs)]).cpu().numpy()


def fit_classifier(model: nn.Module, Xtr: torch.Tensor, ytr: torch.Tensor, Xva: torch.Tensor, yva: torch.Tensor, K: int,
                   epochs: int = 50, lr: float = 1e-3, bs: int = 256, weight_decay: float = 0.0, seed: int = 0,
                   optimizer: str = "adam") -> tuple[nn.Module, dict]:
    """Class-weighted CE; the epoch with the best VALIDATION Macro-F1 is returned (the test set is never touched)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = Xtr.device
    opt = (torch.optim.AdamW if optimizer == "adamw" else torch.optim.Adam)(model.parameters(), lr=lr,
                                                                           weight_decay=weight_decay)
    w = class_weights(ytr, K)
    yva_np = yva.cpu().numpy()
    best, best_state, best_ep, hist = -1.0, None, -1, []
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx], weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        pv = predict_proba(model, Xva).argmax(1)
        f1 = macro_f1(yva_np, pv, K)
        hist.append(f1)
        if f1 > best:
            best, best_ep, best_state = f1, ep, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_macro_f1": best, "best_epoch": best_ep, "val_curve": hist}


def train_probe(kind: str, Ztr, ytr, Zva, yva, K: int, cfg: dict, seed: int):
    """kind: 'mlp' (128-64-K, main) or 'linear' (128-K)."""
    torch.manual_seed(seed)
    head = (MLPHead(Ztr.shape[1], K, cfg.get("probe_hidden", 64), cfg.get("probe_dropout", 0.1)) if kind == "mlp"
            else LinearHead(Ztr.shape[1], K)).to(Ztr.device)
    return fit_classifier(head, Ztr, ytr, Zva, yva, K, cfg.get("probe_epochs", 50), cfg.get("probe_lr", 1e-3),
                          cfg.get("probe_batch", 256), seed=seed)
