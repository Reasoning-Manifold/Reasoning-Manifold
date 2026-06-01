"""Common utilities (seeding, GPU allocation, logging)."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def configure_logging(level: int = logging.INFO, prefix: str | None = None) -> None:
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    if prefix:
        fmt = f"%(asctime)s - [{prefix}] - %(levelname)s - %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_short_name(model_id: str) -> str:
    """Return the trailing path component of a HF id or local path."""
    return os.path.basename(model_id.rstrip("/")) or os.path.basename(os.path.dirname(model_id))


def allocate_gpus(tp: int, dp: int) -> list[list[int]]:
    """Split contiguous GPUs into ``dp`` groups of ``tp`` devices each."""
    total = torch.cuda.device_count()
    if tp * dp != total:
        raise ValueError(f"GPU allocation: TP({tp})·DP({dp}) = {tp * dp} != {total} GPUs available")
    return [list(range(i * tp, (i + 1) * tp)) for i in range(dp)]


def allocate_repeats(total: int, dp: int) -> list[list[int]]:
    if total % dp:
        raise ValueError(f"repeats({total}) must be divisible by dp({dp})")
    per = total // dp
    return [list(range(i * per, (i + 1) * per)) for i in range(dp)]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
