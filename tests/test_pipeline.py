"""End-to-end smoke test on synthetic toy data (a few seconds on CPU)."""
import numpy as np
import torch

from gcids.data import prepare
from gcids.generators import get_generator
from gcids.methods import METHOD_SPECS, build_s1_data, pretrain
from gcids.run import run_variant
from gcids.utils import deep_update, load_config


def _cfg():
    return load_config("configs/base.yaml", "configs/smoke.yaml", overrides={"pretrain": {"epochs": 1}, "vae": {"epochs": 3, "snapshots": [3]},
                                                                         "gan": {"iters": 30, "snapshots": [30]}})


def test_generators_train_cache_and_sample(tmp_path):
    cfg = _cfg()
    sp = prepare("toy", seed=1)
    dev = torch.device("cpu")
    for kind, sec in (("vae", "vae"), ("gan", "gan")):
        g = {k: v for k, v in cfg[sec].items() if k != "snapshots"}
        w = get_generator(kind, "toy", 1, sp.X_tr, sp.y_tr, sp.K, g, tmp_path, dev, cfg[sec]["snapshots"])
        y = torch.as_tensor(sp.y_tr[:50])
        s = w.sample(y)
        assert s.shape == (50, sp.dim) and float(s.min()) >= 0 and float(s.max()) <= 1
        w2 = get_generator(kind, "toy", 1, sp.X_tr, sp.y_tr, sp.K, g, tmp_path, dev, cfg[sec]["snapshots"])   # cache hit
        torch.manual_seed(0); a = w.sample(y); torch.manual_seed(0); b = w2.sample(y)
        assert torch.allclose(a, b, atol=1e-6)


def test_s1_rules():
    cfg = _cfg()
    sp = prepare("toy", seed=1)
    w = get_generator("gan", "toy", 1, sp.X_tr, sp.y_tr, sp.K, {k: v for k, v in cfg["gan"].items() if k != "snapshots"}, "cache_smoke", torch.device("cpu"))
    X, y = torch.as_tensor(sp.X_tr), torch.as_tensor(sp.y_tr)
    Xa, ya, ia = build_s1_data(w, X, y, sp.K, "A")
    cnt = torch.bincount(ya, minlength=sp.K)
    med = int(torch.bincount(y).float().median())
    assert (cnt[torch.bincount(y) < med] >= torch.bincount(y)[torch.bincount(y) < med]).all()
    Xb, yb, ib = build_s1_data(w, X, y, sp.K, "B", ratio=1.0)
    assert abs(ib["n_added"] - len(y)) <= sp.K


def test_every_method_runs_end_to_end(tmp_path):
    cfg = _cfg()
    cfg["cache_dir"] = str(tmp_path / "cache"); cfg["results_dir"] = str(tmp_path / "res")
    sp = prepare("toy", seed=1)
    dev = torch.device("cpu")
    names = list(METHOD_SPECS) + ["S0", "S1", "GenQ-VAE", "GenQ-GAN", "LightGBM"]
    for m in names:
        rec = run_variant("toy", 1, m, {"method": m}, cfg, sp, dev)
        if m.startswith("GenQ"):
            assert "genq" in rec and np.isfinite(rec["genq"]["overall"]["mmd2"])
        else:
            mm = rec["metrics"]["mlp"]
            assert 0 <= mm["accuracy"] <= 1 and np.isfinite(mm["macro_f1"])
            if m != "LightGBM":
                assert "silhouette" in rec["embedding"]
