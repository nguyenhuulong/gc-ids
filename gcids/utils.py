"""Small shared helpers: seeding, logging, config, resumable JSONL results."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


def get_logger(name: str = "gcids") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(False)  # some scatter ops have no deterministic kernel


def get_device(prefer: str | None = None) -> torch.device:
    if prefer and prefer != "auto":
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(*paths: str | Path, overrides: dict | None = None) -> dict:
    cfg: dict[str, Any] = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def config_hash(obj: Any, n: int = 8) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:n]


def append_jsonl(path: str | Path, rec: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def done_keys(path: str | Path, fields: Iterable[str] = ("dataset", "seed", "variant")) -> set[tuple]:
    return {tuple(r.get(k) for k in fields) for r in read_jsonl(path)}
