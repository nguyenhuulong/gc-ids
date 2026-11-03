"""Cost logging: wall time (after cuda synchronize), peak GPU memory, parameter count, hardware info."""
from __future__ import annotations

import platform
import time

import torch


def hardware_info() -> dict:
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
    }


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Stage:
    """``with Stage('pretrain') as s: ...`` then ``s.result(n_params=...)``."""

    def __init__(self, name: str):
        self.name = name
        self.seconds = 0.0
        self.peak_mb = 0.0

    def __enter__(self) -> "Stage":
        _sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        _sync()
        self.seconds = time.perf_counter() - self._t
        if torch.cuda.is_available():
            self.peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20

    def result(self, n_params: int | None = None) -> dict:
        r = {"seconds": round(self.seconds, 3), "peak_gpu_mb": round(self.peak_mb, 1)}
        if n_params is not None:
            r["n_params"] = int(n_params)
        return r
