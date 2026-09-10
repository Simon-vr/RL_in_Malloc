"""Small shared utilities: seeding, RNG construction, IO helpers."""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Iterable, List

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if available) PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch optional at import time
        pass


def make_rng(seed: int, round_idx: int) -> np.random.Generator:
    """Deterministic per-round RNG so a round is identical across policies."""
    return np.random.default_rng((seed, round_idx))


def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(rows: Iterable[Dict[str, Any]], path: str,
              fieldnames: List[str] | None = None) -> None:
    import csv

    rows = list(rows)
    ensure_dir(os.path.dirname(path))
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
