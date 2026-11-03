"""Dataset loading and leak-free preprocessing.

Rules (SPEC section 1.2): the one-hot encoder, the min-max scaler (and the CIC log1p column choice) are fit on the
TRAIN part only (the 80% part of the official train set that remains after the seeded train/val split). Validation
and test are only transformed (and clipped to [0, 1]). Categories unseen in train become all-zeros.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from .utils import get_logger

log = get_logger()

# --------------------------------------------------------------------------------------------------------------
# NSL-KDD
# --------------------------------------------------------------------------------------------------------------
NSL_COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level",
]
NSL_CAT = ["protocol_type", "service", "flag"]
NSL_MAP = {"normal": "Normal"}
for _a in ["mailbomb", "back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "udpstorm", "processtable"]:
    NSL_MAP[_a] = "DoS"
for _a in ["ipsweep", "satan", "nmap", "portsweep", "mscan", "saint"]:
    NSL_MAP[_a] = "Probe"
for _a in ["guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster", "warezclient", "spy", "xlock",
           "xsnoop", "snmpguess", "snmpgetattack", "sendmail", "named", "worm"]:
    NSL_MAP[_a] = "R2L"
for _a in ["buffer_overflow", "loadmodule", "httptunnel", "rootkit", "perl", "sqlattack", "xterm", "ps"]:
    NSL_MAP[_a] = "U2R"
NSL_CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

UNSW_CLASSES = ["Normal", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance",
                "Shellcode", "Worms"]
UNSW_CAT = ["proto", "service", "state"]


@dataclass
class RawData:
    name: str
    train: pd.DataFrame           # features + "__y__" (string class name)
    test: pd.DataFrame
    cat_cols: list[str]
    class_names: list[str]
    log1p: bool = False
    notes: list[str] = field(default_factory=list)


def _read_nsl_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=NSL_COLS)  # header=None: do NOT lose the first row
    df = df.drop(columns=["difficulty_level"])
    lab = df["label"].astype(str).str.strip().str.lower().str.rstrip(".")
    unk = sorted(set(lab) - set(NSL_MAP))
    if unk:
        raise ValueError(f"Unmapped NSL-KDD labels in {path.name}: {unk}")
    df["__y__"] = lab.map(NSL_MAP)
    return df.drop(columns=["label"])


def load_nslkdd(root: str | Path, strict_counts: bool = True) -> RawData:
    root = Path(root)
    tr = _read_nsl_file(root / "KDDTrain+.txt")
    te = _read_nsl_file(root / "KDDTest+.txt")
    if strict_counts:
        assert len(tr) == 125973, f"KDDTrain+ rows {len(tr)} != 125973"
        assert len(te) == 22544, f"KDDTest+ rows {len(te)} != 22544"
    notes = ["KDDTest+ contains attack subtypes that do not occur in KDDTrain+ (known NSL-KDD property)."]
    return RawData("nslkdd", tr, te, list(NSL_CAT), list(NSL_CLASSES), False, notes)


def load_unswnb15(root: str | Path, strict_counts: bool = True) -> RawData:
    root = Path(root)

    def rd(fn: str) -> pd.DataFrame:
        df = pd.read_csv(root / fn, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        y = df["attack_cat"].astype(str).str.strip().replace({"Backdoors": "Backdoor", "nan": "Normal", "": "Normal"})
        df = df.drop(columns=[c for c in ["id", "label", "attack_cat"] if c in df.columns])
        df["__y__"] = y.values
        for c in UNSW_CAT:
            df[c] = df[c].astype(str).str.strip()
        return df

    tr = rd("UNSW_NB15_training-set.csv")
    te = rd("UNSW_NB15_testing-set.csv")
    if len(tr) == 82332 and len(te) == 175341:
        # Several public copies carry the two partition files under swapped names. The dataset paper (Moustafa & Slay)
        # defines the training partition as 175,341 records and the testing partition as 82,332; follow the paper.
        log.warning("UNSW-NB15 partition files are swapped (training-set.csv has 82,332 rows); swapping to the "
                    "paper's convention: train 175,341 / test 82,332")
        tr, te = te, tr
    if strict_counts:
        assert len(tr) == 175341, f"UNSW train rows {len(tr)} != 175341"
        assert len(te) == 82332, f"UNSW test rows {len(te)} != 82332"
    bad = set(tr["__y__"]) | set(te["__y__"])
    bad -= set(UNSW_CLASSES)
    if bad:
        raise ValueError(f"Unexpected UNSW-NB15 classes: {sorted(bad)}")
    log.info("UNSW-NB15 raw feature count (without id/label/attack_cat): %d", tr.shape[1] - 1)
    return RawData("unswnb15", tr, te, list(UNSW_CAT), list(UNSW_CLASSES), False, [])


def load_cicids2017(root: str | Path, min_class: int = 2000, cap: int = 50000, test_size: float = 0.30,
                    split_seed: int = 0) -> RawData:
    """CIC-IDS2017 MachineLearningCVE/*.csv. No official split: ONE fixed stratified 70/30 split (seed 0)."""
    root = Path(root)
    files = sorted(glob.glob(str(root / "MachineLearningCVE" / "*.csv")) + glob.glob(str(root / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files under {root}/MachineLearningCVE")
    frames = []
    for f in files:
        d = pd.read_csv(f, encoding="latin1", low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        d = d.loc[:, ~d.columns.duplicated()]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["Label"] = (df["Label"].astype(str).str.strip().str.replace("�", "-", regex=False)
                   .str.replace(r"\s+-\s+", " - ", regex=True))
    num_cols = [c for c in df.columns if c != "Label"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    const = [c for c in num_cols if df[c].nunique() <= 1]
    df = df.drop(columns=const)
    vc = df["Label"].value_counts()
    keep = vc[vc >= min_class].index.tolist()
    df = df[df["Label"].isin(keep)]
    rng = np.random.RandomState(split_seed)
    parts = []
    for k, g in df.groupby("Label"):
        if len(g) > cap:
            g = g.iloc[rng.permutation(len(g))[:cap]]
        parts.append(g)
    df = pd.concat(parts).reset_index(drop=True).rename(columns={"Label": "__y__"})
    tr, te = train_test_split(df, test_size=test_size, stratify=df["__y__"], random_state=split_seed)
    classes = sorted(df["__y__"].unique(), key=lambda c: (c != "BENIGN", c))
    log.info("CIC-IDS2017 class table after cleaning:\n%s", df["__y__"].value_counts().to_string())
    notes = [f"dropped constant columns: {const}", f"classes >= {min_class} rows kept, capped at {cap}"]
    return RawData("cicids2017", tr.reset_index(drop=True), te.reset_index(drop=True), [], classes, True, notes)


def load_toy(n_train: int = 12000, n_test: int = 4000) -> RawData:
    """Synthetic imbalanced tabular data used ONLY for smoke tests and unit tests (no real-data claims)."""
    rng = np.random.RandomState(0)
    shares = np.array([0.50, 0.25, 0.15, 0.07, 0.02, 0.01])
    names = ["Normal", "DoS", "Probe", "R2L", "U2R", "Rare"]
    d_num, centers = 10, rng.randn(6, 10) * 1.2

    def gen(n: int, shift: float) -> pd.DataFrame:
        y = rng.choice(6, size=n, p=shares)
        x = centers[y] + rng.randn(n, d_num) * (1.0 + 0.2 * (y % 3))[:, None] + shift
        x[:, 0] = np.abs(x[:, 0]) ** 1.5
        df = pd.DataFrame(x, columns=[f"f{i}" for i in range(d_num)])
        df["proto"] = np.where(rng.rand(n) < 0.7, np.array(["tcp", "udp", "icmp"])[y % 3], "tcp")
        df["svc"] = np.where(rng.rand(n) < 0.6, np.array(["http", "ftp", "dns", "ssh"])[y % 4], "other")
        df["__y__"] = np.array(names)[y]
        return df

    return RawData("toy", gen(n_train, 0.0), gen(n_test, 0.15), ["proto", "svc"], names, False, ["synthetic toy data"])


@lru_cache(maxsize=4)
def load_raw(name: str, root: str, strict_counts: bool = True) -> RawData:
    if name == "toy":
        return load_toy()
    if name == "nslkdd":
        return load_nslkdd(Path(root) / "nslkdd", strict_counts)
    if name == "unswnb15":
        return load_unswnb15(Path(root) / "unswnb15", strict_counts)
    if name == "cicids2017":
        return load_cicids2017(Path(root) / "cicids2017")
    raise ValueError(f"unknown dataset {name}")


@dataclass
class Split:
    name: str
    seed: int
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray
    class_names: list[str]
    feature_names: list[str]
    rare_ids: list[int]
    raw_feature_dim: int

    @property
    def K(self) -> int:
        return len(self.class_names)

    @property
    def dim(self) -> int:
        return self.X_tr.shape[1]

    def counts(self, part: str = "tr") -> np.ndarray:
        return np.bincount(getattr(self, "y_" + part), minlength=self.K)


def _split_idx(y: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split; classes with < 2 samples go entirely to train."""
    cnt = np.bincount(y)
    tiny = np.where(cnt[y] < 2)[0]
    rest = np.where(cnt[y] >= 2)[0]
    tr, va = train_test_split(rest, test_size=val_frac, stratify=y[rest], random_state=seed)
    return np.sort(np.concatenate([tr, tiny])), np.sort(va)


def prepare(name: str, data_root: str = "data", seed: int = 42, val_frac: float = 0.2, rare_thresh: float = 0.015,
            strict_counts: bool = True) -> Split:
    raw = load_raw(name, str(data_root), strict_counts)
    cidx = {c: i for i, c in enumerate(raw.class_names)}
    y_all = raw.train["__y__"].map(cidx).to_numpy()
    y_te = raw.test["__y__"].map(cidx).to_numpy()
    if np.isnan(y_all.astype(float)).any() or np.isnan(y_te.astype(float)).any():
        raise ValueError("labels outside class list")
    y_all, y_te = y_all.astype(np.int64), y_te.astype(np.int64)
    feats = [c for c in raw.train.columns if c != "__y__"]
    cat = [c for c in raw.cat_cols if c in feats]
    num = [c for c in feats if c not in cat]
    itr, iva = _split_idx(y_all, val_frac, seed)
    dtr, dva, dte = raw.train.iloc[itr], raw.train.iloc[iva], raw.test

    # ---- numeric: optional log1p (columns chosen on TRAIN), then min-max fit on TRAIN
    ntr, nva, nte = (d[num].to_numpy(dtype=np.float64) for d in (dtr, dva, dte))
    if raw.log1p:
        cols = np.where((ntr.min(0) >= 0) & (ntr.max(0) > 1e3))[0]
        for a in (ntr, nva, nte):
            a[:, cols] = np.log1p(np.clip(a[:, cols], 0, None))
    sc = MinMaxScaler().fit(ntr)
    ntr, nva, nte = (np.clip(sc.transform(a), 0.0, 1.0) for a in (ntr, nva, nte))
    parts_tr, parts_va, parts_te, names = [ntr], [nva], [nte], list(num)
    if cat:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(dtr[cat].astype(str))
        parts_tr.append(enc.transform(dtr[cat].astype(str)))
        parts_va.append(enc.transform(dva[cat].astype(str)))
        parts_te.append(enc.transform(dte[cat].astype(str)))
        names += list(enc.get_feature_names_out(cat))
    X_tr, X_va, X_te = (np.hstack(p).astype(np.float32) for p in (parts_tr, parts_va, parts_te))
    y_tr, y_va = y_all[itr], y_all[iva]
    share = np.bincount(y_tr, minlength=len(raw.class_names)) / len(y_tr)
    rare = [int(i) for i in np.where(share < rare_thresh)[0]]
    return Split(name, seed, X_tr, y_tr, X_va, y_va, X_te, y_te, list(raw.class_names), names, rare, len(feats))


def dataset_report(sp: Split, out_csv: str | Path | None = None) -> pd.DataFrame:
    tr, va, te = sp.counts("tr"), sp.counts("val"), sp.counts("te")
    df = pd.DataFrame({"class": sp.class_names, "train": tr, "val": va, "test": te,
                       "train_share": tr / tr.sum(), "rare": [i in sp.rare_ids for i in range(sp.K)]})
    df["dataset"], df["feature_dim"], df["raw_features"] = sp.name, sp.dim, sp.raw_feature_dim
    log.info("\n%s", df.to_string(index=False))
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df
