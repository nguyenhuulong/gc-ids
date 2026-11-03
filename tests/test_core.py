import numpy as np
import pytest
import torch

from gcids import data as D
from gcids.genmetrics import corr_diff, mmd2_unbiased, prdc
from gcids.losses import nt_xent, supcon
from gcids.metrics import (alignment, classification_metrics, confusion, g_mean, intra_inter_ratio, macro_f1, stratified_subsample,
                           uniformity)
from gcids.probes import class_weights
from gcids.stats import holm, paired_test


# ---------------------------------------------------------------- data / leakage
def test_scaler_statistics_come_from_train_only():
    D.load_raw.cache_clear()
    sp = D.prepare("toy", seed=1)
    raw = D.load_raw("toy", "data", False)
    # the 80% train part defines min/max; val/test may exceed [0,1] before clipping but are clipped after
    assert sp.X_tr.min() >= 0 and sp.X_tr.max() <= 1
    assert sp.X_val.min() >= 0 and sp.X_val.max() <= 1 and sp.X_te.min() >= 0 and sp.X_te.max() <= 1
    # train numeric columns reach exactly 0 and 1 (fit on train), test shifted data is clipped, not rescaled
    num = [i for i, n in enumerate(sp.feature_names) if n.startswith("f")]
    assert np.allclose(sp.X_tr[:, num].min(0), 0) and np.allclose(sp.X_tr[:, num].max(0), 1)
    assert (sp.X_te[:, num].max(0) <= 1).all()


def test_unseen_category_is_all_zero():
    import pandas as pd
    D.load_raw.cache_clear()
    raw = D.load_raw("toy", "data", False)
    # emulate an unseen category by editing the cached test frame in place on a copy
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(raw.train[["proto"]])
    z = enc.transform(pd.DataFrame({"proto": ["never_seen"]}))
    assert z.sum() == 0


def test_split_is_stratified_and_seed_dependent():
    D.load_raw.cache_clear()
    a, b = D.prepare("toy", seed=1), D.prepare("toy", seed=2)
    assert not np.array_equal(a.y_tr[:200], b.y_tr[:200])
    assert set(np.unique(a.y_val)) == set(np.unique(a.y_tr))
    assert a.rare_ids == [i for i in range(a.K) if a.counts("tr")[i] / len(a.y_tr) < 0.015]


def test_nsl_label_map_complete():
    assert set(D.NSL_MAP.values()) == set(D.NSL_CLASSES)
    assert len(D.NSL_COLS) == 43


# ---------------------------------------------------------------- losses
def test_ntxent_perfect_alignment_lower_than_random():
    torch.manual_seed(0)
    z = torch.randn(64, 16)
    perfect = nt_xent(z, z.clone(), 0.5)
    rand = nt_xent(z, torch.randn(64, 16), 0.5)
    assert perfect < rand


def test_ntxent_known_value_two_samples():
    # N=2, orthogonal unit embeddings, positives identical -> sim(pos)=1/tau, sim(neg)=0
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = nt_xent(z1, z1.clone(), tau=0.5)
    pos, neg = 2.0, 0.0
    expected = -(pos - np.log(np.exp(pos) + 2 * np.exp(neg)))
    assert abs(loss.item() - expected) < 1e-5


def test_mask_same_class_removes_collisions():
    z1 = torch.tensor([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0]])
    z2 = z1.clone()
    y = torch.tensor([0, 0, 1])
    plain = nt_xent(z1, z2, 0.5)
    masked = nt_xent(z1, z2, 0.5, labels=y, mask_same_class=True)
    assert masked < plain      # near-duplicate same-class sample is no longer a hard negative


def test_supcon_runs_and_prefers_class_structure():
    torch.manual_seed(0)
    y = torch.arange(4).repeat(8)
    good = torch.nn.functional.one_hot(y, 4).float() + 0.01 * torch.randn(32, 4)
    bad = torch.randn(32, 4)
    assert supcon(good, good, y) < supcon(bad, bad, y)


# ---------------------------------------------------------------- metrics (known answers)
def test_macro_f1_and_gmean_known():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 0, 1, 0, 2, 2])
    cm = confusion(y, p, 3)
    assert cm.tolist() == [[2, 0, 0], [1, 1, 0], [0, 0, 2]]
    assert abs(macro_f1(y, p, 3) - np.mean([0.8, 2 / 3, 1.0])) < 1e-9
    assert abs(g_mean(np.array([1.0, 0.5, 1.0])) - 0.5 ** (1 / 3)) < 1e-9
    assert g_mean(np.array([1.0, 0.0])) == 0.0 and g_mean(np.array([1.0, 0.0]), 1e-3) > 0


def test_classification_metrics_fields():
    y = np.array([0, 0, 1, 1, 2, 2]); p = np.array([0, 0, 1, 0, 2, 2])
    proba = np.eye(3)[p]
    m = classification_metrics(y, p, proba, 3, ["a", "b", "c"], [2])
    assert abs(m["accuracy"] - 5 / 6) < 1e-9 and m["rare_recall"] == 1.0 and "macro_pr_auc" in m


def test_alignment_lower_is_better_and_uniformity():
    rng = np.random.RandomState(0)
    y = np.repeat(np.arange(3), 100)
    tight = np.eye(3)[y] + 0.01 * rng.randn(300, 3)
    loose = rng.randn(300, 3)
    tight /= np.linalg.norm(tight, axis=1, keepdims=True); loose /= np.linalg.norm(loose, axis=1, keepdims=True)
    assert alignment(tight, y) < alignment(loose, y)
    collapsed = np.tile(np.array([[1.0, 0, 0]]), (300, 1)) + 1e-6 * rng.randn(300, 3)
    spread = loose
    assert uniformity(spread) < uniformity(collapsed)      # lower = more uniform


def test_intra_inter_ratio():
    rng = np.random.RandomState(0)
    y = np.repeat([0, 1], 50)
    z = np.vstack([rng.randn(50, 2) * 0.1, rng.randn(50, 2) * 0.1 + 5])
    assert intra_inter_ratio(z, y) < 0.1


def test_class_weights_formula():
    y = torch.tensor([0] * 8 + [1] * 2)
    w = class_weights(y, 2)
    assert torch.allclose(w, torch.tensor([10 / (2 * 8), 10 / (2 * 2)]))


def test_stratified_subsample_keeps_classes():
    y = np.r_[np.zeros(10000, int), np.ones(10000, int), np.full(7, 2)]
    idx = stratified_subsample(y, 1000, 0)
    assert len(idx) <= 1100 and set(y[idx]) == {0, 1, 2}


# ---------------------------------------------------------------- generator metrics
def test_mmd_zero_for_same_distribution_and_large_for_shifted():
    torch.manual_seed(0)
    a, b = torch.randn(400, 5), torch.randn(400, 5)
    assert abs(mmd2_unbiased(a, b, 2.0)) < 0.02
    assert mmd2_unbiased(a, b + 3, 2.0) > 0.3


def test_prdc_identical_sets_high_and_far_sets_zero():
    torch.manual_seed(0)
    a, b = torch.randn(300, 4), torch.randn(300, 4)
    r = prdc(a, b)
    assert r["precision"] > 0.8 and r["recall"] > 0.8
    r2 = prdc(a, b + 20)
    assert r2["precision"] == 0 and r2["coverage"] == 0


def test_corr_diff_zero_for_identical():
    x = np.random.RandomState(0).randn(200, 5)
    assert corr_diff(x, x) < 1e-9


# ---------------------------------------------------------------- statistics
def test_holm_known():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_min_wilcoxon_p_and_paired_test():
    r3 = paired_test(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0]))
    assert abs(r3["min_wilcoxon_p"] - 0.25) < 1e-12 and r3["wilcoxon_p"] >= 0.25 - 1e-12
    r6 = paired_test(np.arange(1, 7, dtype=float), np.zeros(6))
    assert r6["wilcoxon_p"] < 0.05 and r6["t_p"] < 0.05 and r6["mean_diff"] == 3.5
