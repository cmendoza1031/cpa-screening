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


def featurize(
    smiles_list: list[str],
    concentrations: Optional[list[float]] = None,
) -> tuple[np.ndarray, list[str], Optional[list[float]]]:
    """Return (X, valid_smiles, valid_concentrations).

    If `concentrations` is provided (same length as smiles_list), each
    valid row gets its concentration appended as the last feature.
    Concentrations are kept aligned with valid_smiles (skipped along with
    any SMILES that RDKit can't parse). Returns valid_concentrations=None
    when no concentration arg was passed.
    """
    X, valid, valid_conc = [], [], []
    has_conc = concentrations is not None
    if has_conc and len(concentrations) != len(smiles_list):
        raise ValueError("concentrations length must match smiles_list length")
    for i, s in enumerate(smiles_list):
        feats = _featurize_one(s)
        if feats is None:
            log.warning("rdkit could not parse %s; skipping", s)
            continue
        if has_conc:
            c = float(concentrations[i]) if concentrations[i] is not None else 0.0
            feats = np.concatenate([feats, np.array([c], dtype=np.float32)])
            valid_conc.append(c)
        X.append(feats)
        valid.append(s)
    n_feat = FP_NBITS + len(DESCRIPTOR_NAMES) + (1 if has_conc else 0)
    if not X:
        return np.zeros((0, n_feat), dtype=np.float32), [], (valid_conc if has_conc else None)
    return np.stack(X), valid, (valid_conc if has_conc else None)


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

    For toxicity, each row keeps its (smiles, concentration_mol_kg, value)
    tuple; the model sees concentration as an extra feature so it can learn
    dose-response. For other tasks each compound has one value (the
    assay-fixed concentration is ignored as a constant feature).

    Returns a dict with:
        n_folds, smi_oof, y_oof, yhat_oof, per_fold_metrics, scheme_label
    """
    config = config or RFConfig()
    sub = long_df[long_df["task"] == task].dropna(subset=["value"]).copy()
    if sub.empty:
        log.warning("no data for task=%s; skipping CV", task)
        return {}

    use_conc = task == "toxicity" and "concentration_mol_kg" in sub.columns
    if use_conc:
        sub = sub.dropna(subset=["concentration_mol_kg"])
        # Per-row data: (smi, conc) -> y (toxicity is dose-response)
        # Use a list of (smi, conc, value) triples instead of a smi -> value dict
        rows = list(zip(
            sub["smiles_canonical"].tolist(),
            sub["concentration_mol_kg"].astype(float).tolist(),
            sub["value"].astype(float).tolist(),
        ))
    else:
        smi_to_y = dict(zip(sub["smiles_canonical"], sub["value"]))

    smi_oof: list[str] = []
    y_oof: list[float] = []
    yhat_oof: list[float] = []
    per_fold = []

    for fi, (train_set, test_set) in enumerate(folds):
        if use_conc:
            train_rows = [(s, c, y) for (s, c, y) in rows if s in train_set]
            test_rows = [(s, c, y) for (s, c, y) in rows if s in test_set]
            if not train_rows or not test_rows:
                continue
            train_smi = [r[0] for r in train_rows]
            train_conc = [r[1] for r in train_rows]
            train_y = [r[2] for r in train_rows]
            test_smi = [r[0] for r in test_rows]
            test_conc = [r[1] for r in test_rows]
            test_y = [r[2] for r in test_rows]
            X_train, smi_train_kept, conc_train_kept = featurize(train_smi, train_conc)
            X_test, smi_test_kept, conc_test_kept = featurize(test_smi, test_conc)
            # featurize drops bad SMILES; map back to retained y using the kept
            # smiles + concentrations (same order the input was passed in).
            keep_train = [(s, c) for (s, c) in zip(train_smi, train_conc)]
            train_kept_pairs = list(zip(smi_train_kept, conc_train_kept or []))
            y_train = np.array([
                train_y[keep_train.index((s, c))] for (s, c) in train_kept_pairs
            ])
            keep_test = [(s, c) for (s, c) in zip(test_smi, test_conc)]
            test_kept_pairs = list(zip(smi_test_kept, conc_test_kept or []))
            y_test = np.array([
                test_y[keep_test.index((s, c))] for (s, c) in test_kept_pairs
            ])
        else:
            train_smi = [s for s in train_set if s in smi_to_y]
            test_smi = [s for s in test_set if s in smi_to_y]
            if not train_smi or not test_smi:
                continue
            X_train, smi_train_kept, _ = featurize(train_smi)
            X_test, smi_test_kept, _ = featurize(test_smi)
            y_train = np.array([smi_to_y[s] for s in smi_train_kept])
            y_test = np.array([smi_to_y[s] for s in smi_test_kept])
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            continue
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
            "task=%s fold=%d  train=%d test=%d  (concentration-aware=%s)",
            task, fi, len(y_train), len(y_test), use_conc,
        )

    return {
        "task": task,
        "scheme": "kfold" if len(folds) <= 10 else "loo",
        "n_folds": len(folds),
        "smi_oof": smi_oof,
        "y_oof": np.array(y_oof),
        "yhat_oof": np.array(yhat_oof),
        "per_fold_metrics": per_fold,
        "concentration_aware": use_conc,
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

        X_train, smi_train, _ = featurize(train_df["smiles_canonical"].tolist())
        X_val, smi_val, _ = featurize(val_df["smiles_canonical"].tolist())
        X_test, smi_test, _ = featurize(test_df["smiles_canonical"].tolist())

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
