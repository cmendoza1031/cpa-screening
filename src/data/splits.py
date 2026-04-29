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


def kfold_split_by_smiles(
    smiles_list: Iterable[str],
    k: int = 5,
    seed: int = 0,
) -> list[tuple[set[str], set[str]]]:
    """Compound-level k-fold CV. Returns list of (train_smiles, test_smiles) sets.

    Deterministic: a given (seed, k) on the same canonical SMILES list always
    produces the same fold assignment. Reproducible across runs without
    relying on numpy RNG state ordering.
    """
    if k < 2:
        raise ValueError(f"k must be >=2, got {k}")
    unique = sorted(set(s for s in smiles_list if s))
    n = len(unique)
    if n == 0:
        return []
    # Stable hash-based fold assignment on (seed, smiles)
    folds: list[list[str]] = [[] for _ in range(k)]
    for s in unique:
        b = _stable_bucket(s, seed=seed, buckets=k)
        folds[b].append(s)
    out = []
    for fi in range(k):
        test_set = set(folds[fi])
        train_set = set(unique) - test_set
        out.append((train_set, test_set))
    log.info("k-fold split (seed=%d, k=%d): fold sizes = %s",
             seed, k, [len(f) for f in folds])
    return out


def loo_split_by_smiles(smiles_list: Iterable[str]) -> list[tuple[set[str], set[str]]]:
    """Leave-one-out CV. Each fold holds out exactly one compound.

    Order is sorted-canonical-SMILES so it's deterministic. Used for the
    RF-only secondary analysis on permeability where n=16.
    """
    unique = sorted(set(s for s in smiles_list if s))
    return [(set(unique) - {s}, {s}) for s in unique]


def _morgan_fp_array(smiles_list: list[str], radius: int = 2, n_bits: int = 2048) -> "np.ndarray":
    """Compute Morgan fingerprints for a list of canonical SMILES.

    Returns an (n, n_bits) numpy array of float32 0/1 values; rows for
    SMILES that fail RDKit parsing are zeros (rare since these have already
    been canonicalized upstream).
    """
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit import Chem
    from rdkit.DataStructs import ConvertToNumpyArray

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    arr = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        out = np.zeros(n_bits, dtype=np.float32)
        ConvertToNumpyArray(fp, out)
        arr[i] = out
    return arr


def _butina_clusters(
    smiles_list: list[str],
    threshold: float = 0.6,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[list[int]]:
    """Cluster compounds via Butina / Taylor algorithm on Morgan-FP Tanimoto.

    Two compounds are in the same cluster if their Tanimoto similarity
    exceeds `threshold`. Returns a list of clusters where each cluster is a
    list of compound indices (into the input smiles_list).

    Implementation note: RDKit's Butina cluster wants pairwise distance
    matrix as a 1-D condensed list. Distance = 1 - Tanimoto. We compute
    fingerprints once, then bulk-similarity to avoid Python-loop overhead.
    """
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit import Chem
    from rdkit.ML.Cluster import Butina

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            # Use empty fp so this compound is dissimilar to everything.
            fps.append(gen.GetFingerprint(Chem.MolFromSmiles("C")))
        else:
            fps.append(gen.GetFingerprint(mol))

    # Condensed pairwise distance list: [d(0,1), d(0,2), ..., d(n-2, n-1)]
    n = len(fps)
    dists: list[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)

    clusters = Butina.ClusterData(
        dists, n, distThresh=1.0 - threshold, isDistData=True,
    )
    return [list(c) for c in clusters]


def cluster_aware_kfold_split(
    smiles_list: Iterable[str],
    k: int = 5,
    threshold: float = 0.6,
    seed: int = 0,
) -> list[tuple[set[str], set[str]]]:
    """Cluster-aware k-fold split: assigns whole clusters to folds so no
    cluster spans train/test boundaries.

    Algorithm (greedy bin-packing): cluster via Butina, then assign each
    cluster to whichever fold currently has the fewest compounds. This
    produces approximately balanced folds while honoring the "no scaffold
    leakage" constraint.

    The seed shuffles the cluster ordering before greedy assignment so
    different seeds produce different (but still cluster-honoring) folds.
    """
    if k < 2:
        raise ValueError(f"k must be >=2, got {k}")
    unique = sorted(set(s for s in smiles_list if s))
    if not unique:
        return []

    clusters = _butina_clusters(unique, threshold=threshold)
    log.info(
        "cluster split: %d clusters at Tanimoto>%g (sizes: max=%d median=%d singletons=%d)",
        len(clusters),
        threshold,
        max((len(c) for c in clusters), default=0),
        sorted(len(c) for c in clusters)[len(clusters) // 2] if clusters else 0,
        sum(1 for c in clusters if len(c) == 1),
    )

    # Sort clusters by size descending; shuffle within size groups by seed
    rng = np.random.default_rng(seed)
    cluster_indices = list(range(len(clusters)))
    cluster_indices.sort(key=lambda i: -len(clusters[i]))
    rng.shuffle(cluster_indices)
    # Re-sort with shuffle preserved within ties: stable since equal sizes
    # got rng-randomized order
    cluster_indices.sort(key=lambda i: -len(clusters[i]))

    fold_smiles: list[set[str]] = [set() for _ in range(k)]
    for ci in cluster_indices:
        # Assign to currently smallest fold
        target = min(range(k), key=lambda f: len(fold_smiles[f]))
        for compound_idx in clusters[ci]:
            fold_smiles[target].add(unique[compound_idx])

    out = []
    all_smiles = set(unique)
    for fi in range(k):
        test_set = fold_smiles[fi]
        train_set = all_smiles - test_set
        out.append((train_set, test_set))
    log.info(
        "cluster-aware k-fold (seed=%d, k=%d, threshold=%g): fold sizes = %s",
        seed, k, threshold, [len(f) for f in fold_smiles],
    )
    return out


def cluster_aware_split(*args, **kwargs):
    """Backward-compat alias to cluster_aware_kfold_split."""
    return cluster_aware_kfold_split(*args, **kwargs)
