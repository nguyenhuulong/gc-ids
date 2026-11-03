"""Encoder, projection head and classifier heads (shared by all methods for a fair comparison)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """MLP [512, 256, 128]: two (Linear-BN-ReLU-Dropout) blocks and a linear 128-d output (see DECISIONS.md D3)."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (512, 256), out_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Contrastive embedding: L2-normalised encoder output."""
        return F.normalize(self.net(x), dim=1)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 128, hid: int = 64, out: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, out))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class MLPHead(nn.Module):
    """128-64-K, ReLU, dropout 0.1 (main probe)."""

    def __init__(self, in_dim: int, K: int, hid: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid, K))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, K: int):
        super().__init__()
        self.net = nn.Linear(in_dim, K)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class SupervisedNet(nn.Module):
    """S0 / S1: same encoder + MLP head, trained end-to-end."""

    def __init__(self, in_dim: int, K: int, cfg: dict):
        super().__init__()
        self.encoder = Encoder(in_dim, tuple(cfg.get("hidden", (512, 256))), cfg.get("out_dim", 128), cfg.get("dropout", 0.3))
        self.head = MLPHead(self.encoder.out_dim, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def n_params(*mods: nn.Module) -> int:
    return int(sum(p.numel() for m in mods for p in m.parameters()))
