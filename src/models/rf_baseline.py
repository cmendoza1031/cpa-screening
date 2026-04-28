"""Random Forest baseline on RDKit features.

Co-equal baseline (not a strawman). One RandomForestRegressor per task, trained
only on rows with non-null label for that task. No imputation across tasks.

Features per molecule:
    Morgan fingerprint, radius=2, nBits=2048
    + 10 RDKit physicochemical descriptors
        MW, MolLogP, TPSA, NumHDonors, NumHAcceptors,
        NumRotatableBonds, NumAromaticRings, FractionCSP3,
        NumHeavyAtoms, RingCount

The descriptors complement the FP because RFs handle dense numerical features
well and the descriptors carry CPA-relevant chemistry (size, polarity, H-bond
capacity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..utils import canonical_smiles, get_logger

log = get_logger("models.rf")

DESCRIPTOR_NAMES = [
    "mw",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotb",
    "aromatic_rings",
    "frac_csp3",
    "heavy_atoms",
    "ring_count",
]
FP_NBITS = 2048
FP_RADIUS = 2


_MORGAN_GEN = None


def _morgan_gen():
    """Cache the MorganGenerator across calls (new RDKit API; replaces the
    deprecated AllChem.GetMorganFingerprintAsBitVect)."""
    global _MORGAN_GEN
    if _MORGAN_GEN is None:
        from rdkit.Chem import rdFingerprintGenerator

        _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(
            radius=FP_RADIUS, fpSize=FP_NBITS
        )
    return _MORGAN_GEN


def _featurize_one(smiles: str) -> Optional[np.ndarray]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
    except ImportError as e:
        raise ImportError("rdkit required: pip install rdkit") from e

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = _morgan_gen().GetFingerprint(mol)
    fp_arr = np.zeros(FP_NBITS, dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray

    ConvertToNumpyArray(fp, fp_arr)

    descs = np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            float(Lipinski.NumHDonors(mol)),
            float(Lipinski.NumHAcceptors(mol)),
            float(Lipinski.NumRotatableBonds(mol)),
            float(Lipinski.NumAromaticRings(mol)),
            float(Descriptors.FractionCSP3(mol)),
            float(mol.GetNumHeavyAtoms()),
            float(Lipinski.RingCount(mol)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([fp_arr, descs])


def featurize(smiles_list: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return (X, valid_smiles) skipping any that fail RDKit parsing."""
    X, valid = [], []
    for s in smiles_list:
        feats = _featurize_one(s)
        if feats is None:
            log.warning("rdkit could not parse %s; skipping", s)
            continue
        X.append(feats)
        valid.append(s)
    if not X:
        return np.zeros((0, FP_NBITS + len(DESCRIPTOR_NAMES)), dtype=np.float32), []
    return np.stack(X), valid


@dataclass
class RFConfig:
    n_estimators: int = 500
    max_features: str = "sqrt"
    min_samples_leaf: int = 1
    n_jobs: int = -1
    seed: int = 0


def _fit_rf(X_train, y_train, config: RFConfig):
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_features=config.max_features,
        min_samples_leaf=config.min_samples_leaf,
        n_jobs=config.n_jobs,
        random_state=config.seed,
    )
    model.fit(X_train, y_train)
    return model


def train_rf_kfold(
    long_df: pd.DataFrame,
    task: str,
    folds: list[tuple[set[str], set[str]]],
    config: Optional[RFConfig] = None,
) -> dict:
    """Train an RF per fold and produce out-of-fold predictions.

    folds: list of (train_smiles, test_smiles) sets. For 5-fold CV, list has
    5 entries; for LOO it has n entries.

    Returns a dict with:
        n_folds, smi_oof, y_oof, yhat_oof, per_fold_metrics, scheme_label
    """
    config = config or RFConfig()
    sub = long_df[long_df["task"] == task].dropna(subset=["value"])
    if sub.empty:
        log.warning("no data for task=%s; skipping CV", task)
        return {}

    smi_to_y = dict(zip(sub["smiles_canonical"], sub["value"]))

    smi_oof: list[str] = []
    y_oof: list[float] = []
    yhat_oof: list[float] = []
    per_fold = []

    for fi, (train_set, test_set) in enumerate(folds):
        train_smi = [s for s in train_set if s in smi_to_y]
        test_smi = [s for s in test_set if s in smi_to_y]
        if not train_smi or not test_smi:
            continue
        X_train, smi_train_kept = featurize(train_smi)
        X_test, smi_test_kept = featurize(test_smi)
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            continue
        y_train = np.array([smi_to_y[s] for s in smi_train_kept])
        y_test = np.array([smi_to_y[s] for s in smi_test_kept])
        model = _fit_rf(X_train, y_train, config)
        yhat = model.predict(X_test)
        smi_oof.extend(smi_test_kept)
        y_oof.extend(y_test.tolist())
        yhat_oof.extend(yhat.tolist())
        per_fold.append({
            "fold": fi,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        })
        log.info(
            "task=%s fold=%d  train=%d test=%d",
            task, fi, len(y_train), len(y_test),
        )

    return {
        "task": task,
        "scheme": "kfold" if len(folds) <= 10 else "loo",
        "n_folds": len(folds),
        "smi_oof": smi_oof,
        "y_oof": np.array(y_oof),
        "yhat_oof": np.array(yhat_oof),
        "per_fold_metrics": per_fold,
    }


def train_rf_per_task(
    long_df: pd.DataFrame,
    splits: dict[str, set[str]],
    config: Optional[RFConfig] = None,
) -> dict:
    """Train one RF regressor per task. Returns dict keyed by task with model
    and per-split predictions/labels.

    long_df columns: smiles_canonical, task, value, source
    splits: {'train', 'val', 'test'} -> set of canonical SMILES
    """
    from sklearn.ensemble import RandomForestRegressor

    config = config or RFConfig()
    out: dict = {}
    if long_df.empty:
        log.warning("long_df is empty, no RF training to do")
        return out

    for task, sub in long_df.groupby("task"):
        sub = sub.dropna(subset=["value"]).copy()
        if sub.empty:
            continue
        # split
        sub["split"] = sub["smiles_canonical"].map(
            lambda s: "train" if s in splits["train"]
            else "val" if s in splits["val"]
            else "test" if s in splits["test"]
            else "drop"
        )
        sub = sub[sub["split"] != "drop"]
        n_train = (sub["split"] == "train").sum()
        n_val = (sub["split"] == "val").sum()
        n_test = (sub["split"] == "test").sum()
        log.info(
            "task=%s  train=%d val=%d test=%d  value_range=[%g, %g]",
            task,
            n_train,
            n_val,
            n_test,
            sub["value"].min(),
            sub["value"].max(),
        )
        if n_train < 5:
            log.warning(
                "task=%s has only %d training rows; skipping (too few to fit RF)",
                task,
                n_train,
            )
            continue

        train_df = sub[sub["split"] == "train"]
        val_df = sub[sub["split"] == "val"]
        test_df = sub[sub["split"] == "test"]

        X_train, smi_train = featurize(train_df["smiles_canonical"].tolist())
        X_val, smi_val = featurize(val_df["smiles_canonical"].tolist())
        X_test, smi_test = featurize(test_df["smiles_canonical"].tolist())

        y_train = train_df.set_index("smiles_canonical").loc[smi_train, "value"].to_numpy()
        y_val = (
            val_df.set_index("smiles_canonical").loc[smi_val, "value"].to_numpy()
            if smi_val
            else np.zeros(0)
        )
        y_test = (
            test_df.set_index("smiles_canonical").loc[smi_test, "value"].to_numpy()
            if smi_test
            else np.zeros(0)
        )

        model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_features=config.max_features,
            min_samples_leaf=config.min_samples_leaf,
            n_jobs=config.n_jobs,
            random_state=config.seed,
        )
        model.fit(X_train, y_train)

        out[task] = {
            "model": model,
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "smi_train": smi_train,
            "smi_val": smi_val,
            "smi_test": smi_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "yhat_train": model.predict(X_train) if len(y_train) else np.zeros(0),
            "yhat_val": model.predict(X_val) if len(y_val) else np.zeros(0),
            "yhat_test": model.predict(X_test) if len(y_test) else np.zeros(0),
        }
    return out
