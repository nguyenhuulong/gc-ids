"""Contrastive pretraining for all view-construction strategies, plus the S1 oversampling recipe."""
from __future__ import annotations

import math

import torch

from .generators import GenWrapper
from .losses import nt_xent, supcon
from .models import Encoder, ProjectionHead
from .utils import get_logger

log = get_logger()

# method id -> how views / loss are built. IDs follow the paper.
METHOD_SPECS: dict[str, dict] = {
    "C1": {"views": "mask"},
    "C1b-noise": {"views": "noise"},
    "C1b-mixup": {"views": "mixup"},
    "C2-VAE": {"views": "gen", "gen": "vae"},
    "C2-GAN": {"views": "gen", "gen": "gan"},
    "SupCon": {"views": "noise", "loss": "supcon"},
    "C3-realpair": {"views": "realpair"},
    "GC-mask-GAN": {"views": "gen", "gen": "gan", "loss": "ntxent_mask"},
    "GC-mask-VAE": {"views": "gen", "gen": "vae", "loss": "ntxent_mask"},
    "SCARF": {"views": "scarf"},
}


def needs_generator(method: str) -> str | None:
    return METHOD_SPECS.get(method, {}).get("gen")


class ViewMaker:
    def __init__(self, method: str, X: torch.Tensor, y: torch.Tensor, K: int, cfg: dict, gen: GenWrapper | None):
        self.spec = METHOD_SPECS[method]
        self.kind = self.spec["views"]
        self.X, self.y, self.K, self.gen = X, y, K, gen
        self.sigma = cfg.get("sigma", 0.05)
        self.keep = cfg.get("keep_prob", 0.7)
        self.alpha = cfg.get("mixup_alpha", 0.2)
        self.p_gen = cfg.get("p_gen", 1.0)
        self.scarf_frac = cfg.get("scarf_frac", 0.6)
        if self.kind in ("mixup", "realpair"):
            order = torch.argsort(y)
            cnt = torch.bincount(y, minlength=K)
            self.order, self.cnt = order, cnt
            self.start = torch.cumsum(cnt, 0) - cnt

    def noise(self, x: torch.Tensor) -> torch.Tensor:
        return (x + torch.randn_like(x) * self.sigma).clamp_(0.0, 1.0)

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        return x * (torch.rand_like(x) < self.keep).float()

    def partner(self, y: torch.Tensor) -> torch.Tensor:
        """A random REAL sample of the same class as each y."""
        r = (torch.rand(len(y), device=y.device) * self.cnt[y]).long()
        r = torch.minimum(r, self.cnt[y] - 1)
        return self.X[self.order[self.start[y] + r]]

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        k = self.kind
        if k == "mask":
            return self.mask(x), self.mask(x)
        if k == "noise":
            return self.noise(x), self.noise(x)
        if k == "mixup":
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample((len(x), 1)).to(x.device)
            lam = torch.maximum(lam, 1 - lam)  # x stays the dominant component
            return x, lam * x + (1 - lam) * self.partner(y)
        if k == "gen":
            a = self.gen.sample(y)
            if self.p_gen < 1.0:
                use = (torch.rand(len(x), 1, device=x.device) < self.p_gen).float()
                a = use * a + (1 - use) * self.noise(x)
            return a, self.noise(x)
        if k == "realpair":
            return self.noise(x), self.noise(self.partner(y))
        if k == "scarf":
            B, D = x.shape
            idx = torch.randint(0, B, (B, D), device=x.device)
            other = x[idx, torch.arange(D, device=x.device)[None, :].expand(B, D)]
            m = (torch.rand(B, D, device=x.device) < self.scarf_frac).float()
            return x, m * other + (1 - m) * x
        raise ValueError(k)


def pretrain(method: str, X: torch.Tensor, y: torch.Tensor, K: int, cfg: dict, gen: GenWrapper | None,
             seed: int) -> tuple[Encoder, ProjectionHead, list[dict]]:
    dev = X.device
    torch.manual_seed(seed)
    enc = Encoder(X.shape[1], tuple(cfg.get("hidden", (512, 256))), cfg.get("out_dim", 128), cfg.get("dropout", 0.3)).to(dev)
    proj = ProjectionHead(cfg.get("out_dim", 128), 64, 64).to(dev)
    epochs, bs, tau = cfg.get("epochs", 100), cfg.get("batch", 256), cfg.get("tau", 0.5)
    opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=cfg.get("lr", 1e-3),
                           weight_decay=cfg.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    vm = ViewMaker(method, X, y, K, cfg, gen)
    loss_kind = METHOD_SPECS[method].get("loss", "ntxent")
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    n = len(X)
    hist = []
    for ep in range(epochs):
        enc.train(); proj.train()
        perm = torch.randperm(n, generator=g).to(dev)
        tot, nb = 0.0, 0
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            x, yy = X[idx], y[idx]
            with torch.no_grad():
                v1, v2 = vm(x, yy)
            z = proj(enc(torch.cat([v1, v2], 0)))
            z1, z2 = z[:bs], z[bs:]
            if loss_kind == "supcon":
                loss = supcon(z1, z2, yy, tau)
            elif loss_kind == "ntxent_mask":
                loss = nt_xent(z1, z2, tau, labels=yy, mask_same_class=True)
            else:
                loss = nt_xent(z1, z2, tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() if (nb % 50 == 0) else 0.0
            nb += 1
        sched.step()
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            hist.append({"epoch": ep, "loss_sampled": tot / max(1, math.ceil(nb / 50))})
            log.info("%s ep %d loss~%.4f", method, ep, hist[-1]["loss_sampled"])
    enc.eval(); proj.eval()
    return enc, proj, hist


@torch.no_grad()
def build_s1_data(gen: GenWrapper, X: torch.Tensor, y: torch.Tensor, K: int, rule: str = "A", ratio: float = 1.0,
                  cap_mult: float = 50.0) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """S1 = real train + synthetic samples from the cGAN.
    rule A: every class is raised to the median class count (added samples capped at cap_mult x N_k).
    rule B (original-style): ratio x N_train synthetic samples spread uniformly over the classes."""
    cnt = torch.bincount(y, minlength=K)
    if rule == "A":
        target = int(cnt.float().median().item())
        add = [int(min(max(target - int(c), 0), cap_mult * int(c))) if c > 0 else 0 for c in cnt]
    elif rule == "B":
        per = int(round(ratio * len(y) / K))
        add = [per if c > 0 else 0 for c in cnt]
    else:
        raise ValueError(rule)
    ys = torch.cat([torch.full((a,), k, dtype=torch.long, device=y.device) for k, a in enumerate(add) if a > 0]) \
        if sum(add) > 0 else torch.empty(0, dtype=torch.long, device=y.device)
    xs = gen.sample(ys) if len(ys) else X[:0]
    return torch.cat([X, xs]), torch.cat([y, ys]), {"rule": rule, "ratio": ratio, "added": add, "n_added": int(sum(add))}
