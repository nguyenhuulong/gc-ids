"""Contrastive losses. All take two aligned views (z1[i], z2[i]) and return a scalar."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.5, labels: torch.Tensor | None = None,
            mask_same_class: bool = False) -> torch.Tensor:
    """Symmetric NT-Xent on 2N embeddings: each anchor has 1 positive and 2N-2 negatives.

    mask_same_class=True (GC-mask): other samples of the SAME class as the anchor (excluding the positive)
    are removed from the denominator, which removes class collisions caused by label-driven pairing.
    """
    n = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], 0), dim=1)
    sim = z @ z.t() / tau
    eye = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    drop = eye.clone()
    if mask_same_class:
        if labels is None:
            raise ValueError("labels required for mask_same_class")
        y = torch.cat([labels, labels], 0)
        same = y[:, None] == y[None, :]
        pos = torch.zeros_like(eye)
        idx = torch.arange(n, device=z.device)
        pos[idx, idx + n] = True
        pos[idx + n, idx] = True
        drop = drop | (same & ~pos)
    sim = sim.masked_fill(drop, float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def supcon(z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020), L_out form: positives = all same-class samples in the 2N batch."""
    n = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], 0), dim=1)
    y = torch.cat([labels, labels], 0)
    sim = z @ z.t() / tau
    eye = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~eye
    npos = pos.sum(1).clamp(min=1)
    loss = -(logp.masked_fill(~pos, 0.0).sum(1) / npos)
    return loss.mean()
