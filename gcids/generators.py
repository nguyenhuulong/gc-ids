"""Class-conditional generators: cVAE (beta-VAE) and cWGAN-GP (optionally with an ACGAN auxiliary classifier).

Both are trained on the TRAIN split only, with class-aware sampling (probability proportional to N_k^0.5),
cached on disk per (dataset, generator, config-hash, seed), and exposed through ``GenWrapper.sample(y)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import config_hash, get_logger

log = get_logger()


def onehot(y: torch.Tensor, K: int) -> torch.Tensor:
    return F.one_hot(y, K).float()


def class_aware_weights(y: torch.Tensor, K: int, power: float = 0.5) -> torch.Tensor:
    """Per-sample sampling weight so that class k is drawn with probability proportional to N_k**power."""
    cnt = torch.bincount(y, minlength=K).float().clamp(min=1)
    return (cnt ** power / cnt)[y]


def _mlp(dims: list[int], act: nn.Module, final: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act if isinstance(act, nn.ReLU) else nn.LeakyReLU(0.2))
    if final is not None:
        layers.append(final)
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------------------------------------------
# cVAE
# --------------------------------------------------------------------------------------------------------------
class CVAE(nn.Module):
    def __init__(self, x_dim: int, K: int, latent: int = 32, hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.x_dim, self.K, self.latent = x_dim, K, latent
        h = list(hidden)
        self.enc = nn.Sequential(*[m for i, (a, b) in enumerate(zip([x_dim + K] + h[:-1], h)) for m in (nn.Linear(a, b), nn.ReLU())])
        self.mu = nn.Linear(h[-1], latent)
        self.logvar = nn.Linear(h[-1], latent)
        dd = [latent + K] + h
        self.dec = nn.Sequential(*[m for a, b in zip(dd[:-1], dd[1:]) for m in (nn.Linear(a, b), nn.ReLU())],
                                 nn.Linear(h[-1], x_dim), nn.Sigmoid())

    def encode(self, x: torch.Tensor, yh: torch.Tensor):
        h = self.enc(torch.cat([x, yh], 1))
        return self.mu(h), self.logvar(h)

    def decode(self, z: torch.Tensor, yh: torch.Tensor) -> torch.Tensor:
        return self.dec(torch.cat([z, yh], 1))

    def forward(self, x: torch.Tensor, yh: torch.Tensor):
        mu, lv = self.encode(x, yh)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.decode(z, yh), mu, lv

    @torch.no_grad()
    def sample(self, y: torch.Tensor) -> torch.Tensor:
        z = torch.randn(len(y), self.latent, device=y.device)
        return self.decode(z, onehot(y, self.K))


def train_cvae(X: torch.Tensor, y: torch.Tensor, K: int, cfg: dict, seed: int,
               snapshots: list[int] | None = None) -> tuple[CVAE, dict]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = X.device
    m = CVAE(X.shape[1], K, cfg.get("latent", 32), tuple(cfg.get("hidden", (256, 256)))).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.get("lr", 1e-3))
    epochs, bs, beta = cfg.get("epochs", 200), cfg.get("batch", 512), cfg.get("beta", 4.0)
    warm = max(1, int(cfg.get("warmup_frac", 0.2) * epochs))
    w = class_aware_weights(y, K, cfg.get("sampling_power", 0.5)).cpu()
    n = len(X)
    hist = []
    snaps: dict[int, dict] = {}
    for ep in range(epochs):
        b = beta * min(1.0, (ep + 1) / warm)
        idx_all = torch.multinomial(w, n, replacement=True, generator=g).to(dev)
        tot = rec_t = kl_t = 0.0
        nb = 0
        m.train()
        for i in range(0, n - bs + 1, bs):
            idx = idx_all[i:i + bs]
            x, yh = X[idx], onehot(y[idx], K)
            xr, mu, lv = m(x, yh)
            rec = ((xr - x) ** 2).sum(1).mean()
            kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
            loss = rec + b * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); rec_t += rec.item(); kl_t += kl.item(); nb += 1
        if snapshots and (ep + 1) in snapshots:
            snaps[ep + 1] = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        if ep % 20 == 0 or ep == epochs - 1:
            hist.append({"epoch": ep, "loss": tot / max(nb, 1), "rec": rec_t / max(nb, 1), "kl": kl_t / max(nb, 1)})
            log.info("cVAE ep %d loss %.3f rec %.3f kl %.3f", ep, hist[-1]["loss"], hist[-1]["rec"], hist[-1]["kl"])
    m.eval()
    return m, {"history": hist, "snapshots": snaps}


# --------------------------------------------------------------------------------------------------------------
# cWGAN-GP (+ optional ACGAN auxiliary classifier)
# --------------------------------------------------------------------------------------------------------------
class GANG(nn.Module):
    def __init__(self, x_dim: int, K: int, zdim: int = 32, hidden: tuple[int, ...] = (256, 256, 512)):
        super().__init__()
        self.K, self.zdim = K, zdim
        self.net = _mlp([zdim + K, *hidden, x_dim], nn.ReLU(), nn.Sigmoid())

    def forward(self, z: torch.Tensor, yh: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, yh], 1))


class GAND(nn.Module):
    def __init__(self, x_dim: int, K: int, hidden: tuple[int, ...] = (512, 256), aux: bool = True):
        super().__init__()
        self.body = nn.Sequential(*[m for a, b in zip([x_dim + K, *hidden[:-1]], hidden)
                                    for m in (nn.Linear(a, b), nn.LeakyReLU(0.2))])
        self.critic = nn.Linear(hidden[-1], 1)
        self.aux = nn.Linear(hidden[-1], K) if aux else None

    def forward(self, x: torch.Tensor, yh: torch.Tensor):
        h = self.body(torch.cat([x, yh], 1))
        return self.critic(h).squeeze(1), (self.aux(h) if self.aux is not None else None)


def _gp(D: GAND, real: torch.Tensor, fake: torch.Tensor, yh: torch.Tensor) -> torch.Tensor:
    a = torch.rand(real.size(0), 1, device=real.device)
    xh = (a * real + (1 - a) * fake).requires_grad_(True)
    out, _ = D(xh, yh)
    gr = torch.autograd.grad(out.sum(), xh, create_graph=True)[0]
    return ((gr.norm(2, dim=1) - 1) ** 2).mean()


def train_cgan(X: torch.Tensor, y: torch.Tensor, K: int, cfg: dict, seed: int,
               snapshots: list[int] | None = None) -> tuple[GANG, dict]:
    """WGAN-GP. Optional ACGAN classifier head in D (cfg.aux_classifier). Optional two-phase training:
    phase 1 on majority classes (train share >= majority_share) for pretrain_frac of the iterations, phase 2 on
    ``finetune_scope`` in {all, minority}."""
    dev = X.device
    gcpu = torch.Generator(device=dev).manual_seed(seed)   # sampling generator lives on the data device (fast)
    zdim, bs = cfg.get("zdim", 32), cfg.get("batch", 256)
    iters, ncrit, lam = cfg.get("iters", 20000), cfg.get("n_critic", 5), cfg.get("lambda_gp", 10.0)
    aux_on, aux_w = bool(cfg.get("aux_classifier", True)), cfg.get("aux_weight", 1.0)
    G = GANG(X.shape[1], K, zdim, tuple(cfg.get("g_hidden", (256, 256, 512)))).to(dev)
    D = GAND(X.shape[1], K, tuple(cfg.get("d_hidden", (512, 256))), aux_on).to(dev)
    oG = torch.optim.Adam(G.parameters(), lr=cfg.get("lr", 1e-4), betas=(0.5, 0.9))
    oD = torch.optim.Adam(D.parameters(), lr=cfg.get("lr", 1e-4), betas=(0.5, 0.9))
    share = torch.bincount(y, minlength=K).float() / len(y)
    major = share >= cfg.get("majority_share", 0.05)
    two_phase = bool(cfg.get("pretrain_majority", True)) and 0 < int(major.sum()) < K
    scope = cfg.get("finetune_scope", "all")
    switch = int(cfg.get("pretrain_frac", 0.5) * iters) if two_phase else 0
    base_w = class_aware_weights(y, K, cfg.get("sampling_power", 0.5))
    ycpu = y

    def pool_weights(phase: int) -> torch.Tensor:
        if phase == 1 and two_phase:
            keep = major[ycpu]
        elif phase == 2 and two_phase and scope == "minority":
            keep = ~major[ycpu]
        else:
            keep = torch.ones_like(ycpu, dtype=torch.bool)
        return base_w * keep.float()

    weights = {1: pool_weights(1), 2: pool_weights(2)}
    snaps: dict[int, dict] = {}
    hist = []

    def batch(phase: int):
        idx = torch.multinomial(weights[phase], bs, replacement=True, generator=gcpu)
        return X[idx], y[idx]

    for it in range(1, iters + 1):
        phase = 1 if (two_phase and it <= switch) else 2
        for _ in range(ncrit):
            xr, yr = batch(phase)
            yh = onehot(yr, K)
            z = torch.randn(bs, zdim, device=dev)
            with torch.no_grad():
                xf = G(z, yh)
            dr, cr = D(xr, yh)
            df, _ = D(xf, yh)
            loss_d = df.mean() - dr.mean() + lam * _gp(D, xr, xf, yh)
            if aux_on:
                loss_d = loss_d + aux_w * F.cross_entropy(cr, yr)
            oD.zero_grad(set_to_none=True)
            loss_d.backward()
            oD.step()
        _, yg = batch(phase)
        yh = onehot(yg, K)
        xf = G(torch.randn(bs, zdim, device=dev), yh)
        dg, cg = D(xf, yh)
        loss_g = -dg.mean()
        if aux_on:
            loss_g = loss_g + aux_w * F.cross_entropy(cg, yg)
        oG.zero_grad(set_to_none=True)
        loss_g.backward()
        oG.step()
        if it % max(1, iters // 10) == 0 or it == iters:
            hist.append({"iter": it, "loss_d": float(loss_d), "loss_g": float(loss_g), "phase": phase})
            log.info("cGAN it %d/%d phase %d d %.3f g %.3f", it, iters, phase, float(loss_d), float(loss_g))
        if snapshots and it in snapshots:
            snaps[it] = {k: v.detach().cpu().clone() for k, v in G.state_dict().items()}
    G.eval()
    return G, {"history": hist, "snapshots": snaps, "two_phase": two_phase, "switch_iter": switch}


# --------------------------------------------------------------------------------------------------------------
# wrapper + cache
# --------------------------------------------------------------------------------------------------------------
@dataclass
class GenWrapper:
    kind: str                      # "vae" | "gan"
    model: nn.Module
    K: int
    x_dim: int
    meta: dict = field(default_factory=dict)

    @torch.no_grad()
    def sample(self, y: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        if self.kind == "vae":
            return self.model.sample(y)
        z = torch.randn(len(y), self.model.zdim, device=y.device)
        return self.model(z, onehot(y, self.K))

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))


def gen_tag(kind: str, cfg: dict) -> str:
    return config_hash({"kind": kind, **cfg}, 6)


def build_generator(kind: str, K: int, x_dim: int, cfg: dict, device: torch.device) -> GenWrapper:
    if kind == "vae":
        m = CVAE(x_dim, K, cfg.get("latent", 32), tuple(cfg.get("hidden", (256, 256)))).to(device)
    else:
        m = GANG(x_dim, K, cfg.get("zdim", 32), tuple(cfg.get("g_hidden", (256, 256, 512)))).to(device)
    return GenWrapper(kind, m, K, x_dim)


def get_generator(kind: str, dataset: str, seed: int, X: np.ndarray, y: np.ndarray, K: int, cfg: dict,
                  cache_dir: str | Path, device: torch.device, snapshots: list[int] | None = None,
                  checkpoint: int | None = None) -> GenWrapper:
    """Train (or load from cache) a generator. ``checkpoint`` selects a saved snapshot (GAN iterations / VAE epochs)."""
    from .cost import Stage  # local import to avoid a cycle

    cache_dir = Path(cache_dir) / "gen"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = gen_tag(kind, cfg)
    path = cache_dir / f"{dataset}_{kind}_{tag}_{seed}.pt"
    if path.exists():
        blob = torch.load(path, map_location=device, weights_only=False)
        w = build_generator(kind, K, X.shape[1], cfg, device)
        w.model.load_state_dict(blob["state"])
        w.meta = blob["meta"]
        snaps = blob.get("snapshots", {})
        w.meta["snapshots_available"] = sorted(snaps)
        if checkpoint is not None:
            if checkpoint not in snaps:
                raise KeyError(f"checkpoint {checkpoint} not in cache; available {sorted(snaps)}")
            w.model.load_state_dict({k: v.to(device) for k, v in snaps[checkpoint].items()})
        w.model.eval()
        return w
    Xt = torch.as_tensor(X, device=device)
    yt = torch.as_tensor(y, device=device)
    log.info("training %s generator for %s seed %d (tag %s)", kind, dataset, seed, tag)
    torch.manual_seed(seed)
    st = Stage("generator_train")
    with st:
        if kind == "vae":
            model, extra = train_cvae(Xt, yt, K, cfg, seed, snapshots)
        else:
            model, extra = train_cgan(Xt, yt, K, cfg, seed, snapshots)
    meta = {"cost": st.result(n_params=int(sum(p.numel() for p in model.parameters()))), "history": extra["history"],
            "cfg": cfg, "tag": tag}
    torch.save({"state": model.state_dict(), "meta": meta, "snapshots": extra.get("snapshots", {})}, path)
    w = GenWrapper(kind, model, K, X.shape[1], meta)
    w.meta["snapshots_available"] = sorted(extra.get("snapshots", {}))
    if checkpoint is not None:
        w.model.load_state_dict({k: v.to(device) for k, v in extra["snapshots"][checkpoint].items()})
    w.model.eval()
    return w
