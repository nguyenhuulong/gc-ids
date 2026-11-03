"""Experiment runner. One finished (dataset, seed, variant) run = one JSON line in results/raw/<dataset>.jsonl.

    python -m gcids.run --dataset nslkdd --config configs/base.yaml configs/nslkdd.yaml --seeds 42 123 456

Resumable: runs that already exist are skipped. Within one seed every method shares the same train/val split and the
same cached generator, so paired statistics are valid.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import baselines
from .cost import Stage, hardware_info
from .data import Split, dataset_report, prepare
from .generators import get_generator
from .genmetrics import gan_mode_collapse, generator_quality, vae_latent_stats
from .methods import METHOD_SPECS, build_s1_data, needs_generator, pretrain
from .metrics import classification_metrics, embedding_metrics, stratified_subsample
from .models import SupervisedNet, n_params
from .probes import extract, fit_classifier, predict_proba, train_probe
from .utils import (append_jsonl, config_hash, deep_update, done_keys, get_device, get_logger, load_config,
                    seed_everything)

log = get_logger()
SECTIONS = ("pretrain", "probe", "vae", "gan", "s1", "eval", "supervised", "baseline")
BASELINES = {"LightGBM", "FT-Transformer", "TabNet"}


def variant_cfg(base: dict, vspec: dict) -> dict:
    return deep_update(base, {k: v for k, v in vspec.items() if k in SECTIONS})


def _tens(sp: Split, dev: torch.device):
    f = lambda a, dt=None: torch.as_tensor(a, device=dev) if dt is None else torch.as_tensor(a, device=dev, dtype=dt)
    return (f(sp.X_tr), f(sp.y_tr, torch.long), f(sp.X_val), f(sp.y_val, torch.long), f(sp.X_te), f(sp.y_te, torch.long))


def _gen_for(kind: str, dataset: str, seed: int, sp: Split, cfg: dict, cache: str, dev: torch.device, ckpt: int | None):
    sec = "vae" if kind == "vae" else "gan"
    gcfg = {k: v for k, v in cfg[sec].items() if k != "snapshots"}
    snaps = cfg[sec].get("snapshots")
    return get_generator(kind, dataset, seed, sp.X_tr, sp.y_tr, sp.K, gcfg, cache, dev, snaps, ckpt)


def _cls_pack(y, proba, sp: Split) -> dict:
    return classification_metrics(y, proba.argmax(1), proba, sp.K, sp.class_names, sp.rare_ids)


def run_variant(dataset: str, seed: int, name: str, vspec: dict, base: dict, sp: Split, dev: torch.device) -> dict:
    method = vspec["method"]
    cfg = variant_cfg(base, vspec)
    Xtr, ytr, Xva, yva, Xte, yte = _tens(sp, dev)
    cache = base.get("cache_dir", "cache")
    rec: dict = {"dataset": dataset, "seed": seed, "variant": name, "method": method, "hw": hardware_info(),
                 "cfg_hash": config_hash({k: cfg.get(k) for k in SECTIONS}), "rare_classes": [sp.class_names[i] for i in sp.rare_ids]}
    cost: dict = {}
    ckpt = vspec.get("checkpoint")
    seed_everything(seed)

    # ------------------------------------------------------------------ generator-only records
    if method in ("GenQ-VAE", "GenQ-GAN"):
        kind = "vae" if method.endswith("VAE") else "gan"
        gen = _gen_for(kind, dataset, seed, sp, cfg, cache, dev, ckpt)
        q = generator_quality(gen, Xtr, ytr, sp.class_names, seed=seed, do_c2st=cfg["eval"].get("c2st", True))
        q["extra"] = gan_mode_collapse(gen, sp.K, dev) if kind == "gan" else vae_latent_stats(gen, Xtr, ytr, sp.K)
        rec.update({"genq": q, "gen_meta": {k: v for k, v in gen.meta.items() if k in ("cost", "tag", "snapshots_available")},
                    "checkpoint": ckpt})
        return rec

    # ------------------------------------------------------------------ non-deep baselines
    if method in BASELINES:
        bcfg = cfg.get("baseline", {}).get(method, {})
        st = Stage("train")
        with st:
            if method == "LightGBM":
                proba, info = baselines.run_lightgbm(sp, bcfg, seed)
            elif method == "FT-Transformer":
                proba, info = baselines.run_ftt(sp, bcfg, seed, dev)
            else:
                proba, info = baselines.run_tabnet(sp, bcfg, seed, dev)
        cost["train"] = st.result(info.get("n_params"))
        rec.update({"metrics": {"mlp": _cls_pack(sp.y_te, proba, sp)}, "info": {k: v for k, v in info.items() if k != "val_curve"},
                    "cost": cost})
        return rec

    # ------------------------------------------------------------------ supervised S0 / S1
    if method in ("S0", "S1"):
        s1info = None
        X_fit, y_fit = Xtr, ytr
        if method == "S1":
            gen = _gen_for("gan", dataset, seed, sp, cfg, cache, dev, ckpt)
            cost["generator_train"] = gen.meta.get("cost", {})
            st = Stage("generator_sample")
            with st:
                X_fit, y_fit, s1info = build_s1_data(gen, Xtr, ytr, sp.K, cfg["s1"].get("rule", "A"),
                                                     cfg["s1"].get("ratio", 1.0), cfg["s1"].get("cap_mult", 50.0))
            cost["generator_sample"] = st.result()
        scfg = cfg["supervised"]
        net = SupervisedNet(sp.dim, sp.K, cfg["pretrain"]).to(dev)
        st = Stage("train")
        with st:
            net, info = fit_classifier(net, X_fit, y_fit, Xva, yva, sp.K, scfg.get("epochs", 50), scfg.get("lr", 1e-3),
                                       scfg.get("batch", 256), seed=seed)
        cost["train"] = st.result(n_params(net))
        st = Stage("inference")
        with st:
            proba = predict_proba(net, Xte)
            Zte = extract(net.encoder, Xte).cpu().numpy()
        cost["inference"] = st.result()
        Ztr = extract(net.encoder, Xtr)
        Zva = extract(net.encoder, Xva)
        lin, linfo = train_probe("linear", Ztr, ytr, Zva, yva, sp.K, cfg["probe"], seed)
        lproba = predict_proba(lin, torch.as_tensor(Zte, device=dev))
        emb = embedding_metrics(Zte, sp.y_te, Ztr.cpu().numpy(), sp.y_tr, sp.K, cfg["eval"].get("n_eval", 20000), 0, str(dev))
        rec.update({"metrics": {"mlp": _cls_pack(sp.y_te, proba, sp), "linear": _cls_pack(sp.y_te, lproba, sp)},
                    "embedding": emb, "val_f1": {"mlp": info["best_val_macro_f1"], "linear": linfo["best_val_macro_f1"]},
                    "best_epoch": info["best_epoch"], "s1": s1info, "cost": cost})
        _maybe_save_emb(base, dataset, name, seed, Zte, sp.y_te)
        return rec

    # ------------------------------------------------------------------ contrastive methods
    if method not in METHOD_SPECS:
        raise ValueError(f"unknown method {method}")
    gen = None
    kind = needs_generator(method)
    if kind:
        gen = _gen_for(kind, dataset, seed, sp, cfg, cache, dev, ckpt)
        cost["generator_train"] = gen.meta.get("cost", {})
        st = Stage("generator_sample")
        with st:
            gen.sample(ytr)  # cost of producing one training-set-sized batch of synthetic samples
        cost["generator_sample"] = st.result(gen.n_params())
    st = Stage("pretrain")
    with st:
        enc, proj, hist = pretrain(method, Xtr, ytr, sp.K, cfg["pretrain"], gen, seed)
    cost["pretrain"] = st.result(n_params(enc, proj))
    Ztr, Zva = extract(enc, Xtr), extract(enc, Xva)
    st = Stage("probe_train")
    with st:
        mlp, minfo = train_probe("mlp", Ztr, ytr, Zva, yva, sp.K, cfg["probe"], seed)
        lin, linfo = train_probe("linear", Ztr, ytr, Zva, yva, sp.K, cfg["probe"], seed)
    cost["probe_train"] = st.result(n_params(mlp))
    st = Stage("inference")
    with st:
        Zte_t = extract(enc, Xte)
        pm, pl = predict_proba(mlp, Zte_t), predict_proba(lin, Zte_t)
    cost["inference"] = st.result()
    Zte = Zte_t.cpu().numpy()
    emb = embedding_metrics(Zte, sp.y_te, Ztr.cpu().numpy(), sp.y_tr, sp.K, cfg["eval"].get("n_eval", 20000), 0, str(dev))
    rec.update({"metrics": {"mlp": _cls_pack(sp.y_te, pm, sp), "linear": _cls_pack(sp.y_te, pl, sp)}, "embedding": emb,
                "val_f1": {"mlp": minfo["best_val_macro_f1"], "linear": linfo["best_val_macro_f1"]},
                "pretrain_history": hist, "cost": cost, "checkpoint": ckpt,
                "gen_meta": ({k: v for k, v in gen.meta.items() if k in ("tag",)} if gen else None)})
    _maybe_save_emb(base, dataset, name, seed, Zte, sp.y_te)
    return rec


def _maybe_save_emb(base: dict, dataset: str, name: str, seed: int, Z: np.ndarray, y: np.ndarray) -> None:
    ev = base.get("eval", {})
    if seed not in ev.get("save_emb_seeds", []):
        return
    idx = stratified_subsample(y, ev.get("emb_save_n", 5000), 0)
    out = Path(base.get("results_dir", "results")) / "emb"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{dataset}__{name}__{seed}.npz", z=Z[idx].astype(np.float16), y=y[idx])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["nslkdd", "unswnb15", "cicids2017", "toy"])
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--variants", nargs="+", help="subset of variant names (default: cfg.run_list or all)")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--shard", default="0/1", help="i/n: process seeds with index % n == i (several processes per GPU)")
    ap.add_argument("--out", default=None, help="results jsonl (default results/raw/<dataset>.jsonl)")
    ap.add_argument("--overwrite", action="store_true", help="re-run even if the record exists (appends a new line)")
    a = ap.parse_args(argv)

    base = load_config(*a.config)
    dev = get_device(a.device)
    seeds = a.seeds or base.get("seeds", [42, 123, 456])
    i, n = (int(t) for t in a.shard.split("/"))
    seeds = [s for k, s in enumerate(seeds) if k % n == i]
    variants = base["variants"]
    names = a.variants or base.get("run_list") or list(variants)
    out = a.out or Path(base.get("results_dir", "results")) / "raw" / f"{a.dataset}.jsonl"
    done = set() if a.overwrite else done_keys(out)
    strict = base.get("data", {}).get("strict_counts", True) and a.dataset != "toy"
    log.info("dataset=%s seeds=%s variants=%s device=%s", a.dataset, seeds, names, dev)

    for seed in seeds:
        todo = [v for v in names if (a.dataset, seed, v) not in done]
        if not todo:
            continue
        sp = prepare(a.dataset, a.data_root, seed, base.get("data", {}).get("val_frac", 0.2),
                     base.get("data", {}).get("rare_thresh", 0.015), strict)
        if seed == seeds[0]:
            dataset_report(sp, Path(base.get("results_dir", "results")) / "tables" / f"dataset_stats_{a.dataset}.csv")
        for v in todo:
            t0 = time.perf_counter()
            try:
                rec = run_variant(a.dataset, seed, v, variants[v], base, sp, dev)
            except Exception:
                log.exception("FAILED %s seed %d variant %s (continuing)", a.dataset, seed, v)
                continue
            rec["wall_seconds"] = round(time.perf_counter() - t0, 1)
            append_jsonl(out, rec)
            m = rec.get("metrics", {}).get("mlp", {})
            log.info("DONE %s s%d %-18s macroF1=%s G=%s (%.0fs)", a.dataset, seed, v,
                     f"{m.get('macro_f1', float('nan')):.4f}", f"{m.get('g_mean', float('nan')):.4f}", rec["wall_seconds"])


if __name__ == "__main__":
    main()
