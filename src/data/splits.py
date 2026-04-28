"""Train/val/test splits.

v1: seeded random 70/15/15 keyed on canonical SMILES so the same molecule
always lands in the same split across tasks (no leakage between toxicity,
permeability, and IRI heads sharing rows).

v2 (Phase 2): cluster-aware splits using Tanimoto similarity on Morgan
fingerprints. Stub left here; full implementation lands later.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from ..utils import get_logger

log = get_logger("data.splits")


def _stable_bucket(smiles: str, seed: int, buckets: int = 10000) -> int:
    """Stable hash of (seed, smiles) into a bucket. Used for reproducible
    cross-run splits without depending on numpy's RNG seeding semantics."""
    h = hashlib.sha256(f"{seed}:{smiles}".encode()).hexdigest()
    return int(h[:8], 16) % buckets


def random_split_by_smiles(
    smiles_list: Iterable[str],
    seed: int = 0,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
) -> dict[str, set[str]]:
    """Partition unique canonical SMILES into train / val / test sets.

    Returns {'train': set, 'val': set, 'test': set}. Deterministic for a fixed
    seed across runs and across tasks (because we hash the SMILES, not its
    position).
    """
    if not np.isclose(train + val + test, 1.0):
        raise ValueError(f"split fractions must sum to 1.0, got {train+val+test}")

    unique = sorted(set(s for s in smiles_list if s))
    train_cut = int(round(train * 10000))
    val_cut = int(round((train + val) * 10000))

    out = {"train": set(), "val": set(), "test": set()}
    for s in unique:
        b = _stable_bucket(s, seed=seed, buckets=10000)
        if b < train_cut:
            out["train"].add(s)
        elif b < val_cut:
            out["val"].add(s)
        else:
            out["test"].add(s)
    log.info(
        "random split (seed=%d): train=%d val=%d test=%d (total=%d)",
        seed,
        len(out["train"]),
        len(out["val"]),
        len(out["test"]),
        len(unique),
    )
    return out


def assign_split(df: pd.DataFrame, splits: dict[str, set[str]], smiles_col: str = "smiles_canonical") -> pd.DataFrame:
    """Add a 'split' column to df based on the per-SMILES partition."""
    def _which(s: str) -> str:
        if s in splits["train"]:
            return "train"
        if s in splits["val"]:
            return "val"
        if s in splits["test"]:
            return "test"
        return "unassigned"

    out = df.copy()
    out["split"] = out[smiles_col].map(_which)
    return out


def cluster_aware_split(*args, **kwargs):
    """Phase 2 placeholder. Tanimoto/Morgan-FP cluster-aware split lands in Phase 2."""
    raise NotImplementedError("cluster-aware split is implemented in Phase 2")
