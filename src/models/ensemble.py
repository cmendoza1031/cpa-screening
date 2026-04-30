"""Deep ensemble + split-conformal calibration for both architectures.

Phase 2 deliverable. The eval pipeline:

    For each fold (5 random or cluster-aware folds):
      For each seed (5 seeds):
        Train a fresh RF / ChemBERTa on fold's train compounds.
        Predict on fold's test compounds.
      Aggregate per-compound predictions across the 5 seeds:
        mean = ensemble point estimate
        std  = epistemic uncertainty proxy

    -> Each compound has one (mean, std) prediction per task (its OOF tuple).

Conformal calibration (jackknife+ style on OOF residuals):
    For each task:
        residuals = |y_true - oof_mean| across all OOF predictions
        q95 = quantile(residuals, 0.95)
    Prediction interval at any new compound: pred +/- q95
    Empirical coverage check on the OOF set (sanity, target ~0.95).

For FDA IID candidate scoring we use a separate "full-data" ensemble:
    train 5 RF / ChemBERTa models on ALL training compounds (no held-out),
    apply the OOF-derived q95 as the calibration constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..utils import get_logger

log = get_logger("models.ensemble")

REG_TASKS = ["toxicity", "permeability", "iri"]


# -------------------- aggregation utilities --------------------------------


def aggregate_per_seed(
    per_seed_preds: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Collapse k seed-prediction dicts into one (mean, std) dict per task.

    per_seed_preds: list of {task: {smiles: pred}} dicts, length = n_seeds.
    Returns {task: {smiles: (mean, std)}}.
    """
    if not per_seed_preds:
        return {}
    tasks = list(per_seed_preds[0].keys())
    out: dict[str, dict[str, tuple[float, float]]] = {t: {} for t in tasks}
    for t in tasks:
        # Find the union of smiles predicted by any seed (should be the same
        # set for all seeds in a clean run, but be defensive)
        all_smi = set()
        for d in per_seed_preds:
            all_smi.update(d.get(t, {}).keys())
        for s in all_smi:
            preds = [d[t][s] for d in per_seed_preds if s in d.get(t, {})]
            if not preds:
                continue
            arr = np.array(preds, dtype=float)
            out[t][s] = (float(arr.mean()), float(arr.std(ddof=0)))
    return out


def conformal_q95(residuals: np.ndarray, alpha: float = 0.05) -> float:
    """Split-conformal quantile for a (1-alpha) prediction interval.

    Uses the (1 + 1/n) * (1 - alpha) quantile per Lei et al. 2018; for n
    moderately large this is essentially the empirical quantile, with a
    small finite-sample correction.
    """
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    n = len(residuals)
    if n == 0:
        return float("nan")
    q_level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(residuals, q_level))


def empirical_coverage(
    y_true: np.ndarray, y_mean: np.ndarray, q95: float,
) -> float:
    """Fraction of test points whose true value falls in [mean - q95, mean + q95]."""
    y_true = np.asarray(y_true, dtype=float)
    y_mean = np.asarray(y_mean, dtype=float)
    if not np.isfinite(q95) or len(y_true) == 0:
        return float("nan")
    inside = (y_true >= y_mean - q95) & (y_true <= y_mean + q95)
    return float(inside.mean())


# -------------------- RF ensemble ------------------------------------------


def train_rf_ensemble_kfold(
    long_df: pd.DataFrame,
    task: str,
    folds: list[tuple[set[str], set[str]]],
    n_seeds: int = 5,
    n_estimators: int = 500,
) -> dict:
    """Train n_seeds * len(folds) RFs; aggregate per-(smi,conc) (mean, std).

    For toxicity, the prediction unit is (smiles, concentration_mol_kg)
    because the same compound has multiple measurements at different
    concentrations and the model uses concentration as an input feature.
    For other tasks, prediction unit is just smiles.

    Returns:
        smi_oof: list of canonical SMILES (one per OOF prediction)
        conc_oof: list of concentrations or None (toxicity only)
        y_oof: ndarray of true values
        mean_oof: ndarray of ensemble mean predictions
        std_oof: ndarray of ensemble std predictions
        q95: float, conformal calibration constant
        coverage: float, empirical coverage of mean +/- q95 (~0.95 target)
    """
    from .rf_baseline import featurize, _fit_rf, RFConfig

    sub = long_df[long_df["task"] == task].dropna(subset=["value"]).copy()
    if sub.empty:
        return {}

    use_conc = task == "toxicity" and "concentration_mol_kg" in sub.columns
    if use_conc:
        sub = sub.dropna(subset=["concentration_mol_kg"])
        rows = list(zip(
            sub["smiles_canonical"].tolist(),
            sub["concentration_mol_kg"].astype(float).tolist(),
            sub["value"].astype(float).tolist(),
        ))
    else:
        smi_to_y = dict(zip(sub["smiles_canonical"], sub["value"]))

    # Per-prediction-unit accumulator: key = (smi, conc) for toxicity, smi otherwise
    pred_by_seed: dict = {}
    y_by_key: dict = {}

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
            train_kept_pairs = list(zip(smi_train_kept, conc_train_kept or []))
            keep_train = [(s, c) for (s, c) in zip(train_smi, train_conc)]
            y_train = np.array([
                train_y[keep_train.index((s, c))] for (s, c) in train_kept_pairs
            ])
            test_kept_pairs = list(zip(smi_test_kept, conc_test_kept or []))
            keep_test = [(s, c) for (s, c) in zip(test_smi, test_conc)]
            y_test = np.array([
                test_y[keep_test.index((s, c))] for (s, c) in test_kept_pairs
            ])
            keys_test = test_kept_pairs
        else:
            train_smi = [s for s in train_set if s in smi_to_y]
            test_smi = [s for s in test_set if s in smi_to_y]
            if not train_smi or not test_smi:
                continue
            X_train, smi_train_kept, _ = featurize(train_smi)
            X_test, smi_test_kept, _ = featurize(test_smi)
            y_train = np.array([smi_to_y[s] for s in smi_train_kept])
            y_test = np.array([smi_to_y[s] for s in smi_test_kept])
            keys_test = smi_test_kept
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            continue
        for seed in range(n_seeds):
            cfg = RFConfig(seed=seed, n_estimators=n_estimators)
            model = _fit_rf(X_train, y_train, cfg)
            yhat = model.predict(X_test)
            for k, yh, yt in zip(keys_test, yhat, y_test):
                pred_by_seed.setdefault(k, []).append(float(yh))
                y_by_key[k] = float(yt)
        log.info(
            "rf ensemble task=%s fold=%d  train=%d test=%d  seeds=%d  conc-aware=%s",
            task, fi, len(y_train), len(keys_test), n_seeds, use_conc,
        )

    if not pred_by_seed:
        return {}
    keys = sorted(pred_by_seed.keys(), key=lambda k: str(k))
    means = np.array([np.mean(pred_by_seed[k]) for k in keys])
    stds = np.array([np.std(pred_by_seed[k], ddof=0) for k in keys])
    y = np.array([y_by_key[k] for k in keys])

    if use_conc:
        smi_list = [k[0] for k in keys]
        conc_list = [k[1] for k in keys]
    else:
        smi_list = list(keys)
        conc_list = None

    residuals = np.abs(y - means)
    q95 = conformal_q95(residuals, alpha=0.05)
    coverage = empirical_coverage(y, means, q95)
    log.info("rf ensemble task=%s  n=%d q95=%.3g  coverage=%.3f", task, len(keys), q95, coverage)
    return {
        "task": task,
        "smi_oof": smi_list,
        "conc_oof": conc_list,
        "y_oof": y,
        "mean_oof": means,
        "std_oof": stds,
        "q95": q95,
        "coverage": coverage,
        "n_seeds": n_seeds,
        "concentration_aware": use_conc,
    }


# Reference concentration for cross-compound toxicity ranking. 6 mol/kg
# is the inflection point in Higgins Dec 2025 where formamide alone goes
# toxic but formamide+glycerol does not, and it's mid-range across the
# 3 / 6 / 12 mol/kg sweep so it's a reasonable common comparison point.
TOX_REFERENCE_CONC_MOL_KG = 6.0


def train_rf_ensemble_full(
    long_df: pd.DataFrame,
    n_seeds: int = 5,
    n_estimators: int = 500,
) -> dict[str, list]:
    """Train one full-data ensemble per task (no held-out), for FDA IID scoring.

    Returns {task: list of trained RF models, length n_seeds}. Toxicity
    models are trained on (compound, concentration) features so they can
    predict at TOX_REFERENCE_CONC_MOL_KG when scoring candidates.
    """
    from .rf_baseline import featurize, _fit_rf, RFConfig

    out: dict[str, list] = {}
    for task in REG_TASKS:
        sub = long_df[long_df["task"] == task].dropna(subset=["value"]).copy()
        if sub.empty:
            continue
        use_conc = task == "toxicity" and "concentration_mol_kg" in sub.columns
        if use_conc:
            sub = sub.dropna(subset=["concentration_mol_kg"])
            smi = sub["smiles_canonical"].tolist()
            conc = sub["concentration_mol_kg"].astype(float).tolist()
            X, smi_kept, conc_kept = featurize(smi, conc)
            keep_pairs = list(zip(smi_kept, conc_kept or []))
            in_pairs = list(zip(smi, conc))
            y_full = sub["value"].astype(float).tolist()
            y = np.array([y_full[in_pairs.index(p)] for p in keep_pairs])
        else:
            smi = sub["smiles_canonical"].tolist()
            X, smi_kept, _ = featurize(smi)
            smi_to_y = dict(zip(sub["smiles_canonical"], sub["value"]))
            y = np.array([smi_to_y[s] for s in smi_kept])
        if X.shape[0] == 0:
            continue
        models = []
        for seed in range(n_seeds):
            cfg = RFConfig(seed=seed, n_estimators=n_estimators)
            models.append(_fit_rf(X, y, cfg))
        out[task] = models
        log.info("rf full-data ensemble task=%s  n_train=%d  seeds=%d  conc-aware=%s",
                 task, len(y), n_seeds, use_conc)
    return out


def predict_rf_ensemble(
    models_per_task: dict[str, list],
    smiles_list: list[str],
    toxicity_conc_mol_kg: float = TOX_REFERENCE_CONC_MOL_KG,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Predict (mean, std) per (task, smiles) using a full-data ensemble.

    For toxicity, predictions are made at toxicity_conc_mol_kg so all
    candidates are compared on the same dose. The reference concentration
    is configurable; default is 6 mol/kg (mid-range, see Higgins Dec 2025).
    """
    from .rf_baseline import featurize

    out: dict[str, dict[str, tuple[float, float]]] = {t: {} for t in models_per_task}
    for task, models in models_per_task.items():
        # Reconstruct the feature shape the models were trained with by
        # checking their n_features_in_; toxicity has +1 column.
        n_feat = int(models[0].n_features_in_) if hasattr(models[0], "n_features_in_") else None
        use_conc = task == "toxicity"
        if use_conc:
            X, smi_kept, _ = featurize(
                smiles_list,
                [toxicity_conc_mol_kg] * len(smiles_list),
            )
        else:
            X, smi_kept, _ = featurize(smiles_list)
        if n_feat is not None and X.shape[1] != n_feat:
            log.warning(
                "task=%s expected %d features, got %d; predictions may be invalid",
                task, n_feat, X.shape[1],
            )
        if X.shape[0] == 0:
            continue
        per_seed_preds = np.stack([m.predict(X) for m in models])  # (n_seeds, n)
        means = per_seed_preds.mean(axis=0)
        stds = per_seed_preds.std(axis=0, ddof=0)
        for s, m, sd in zip(smi_kept, means, stds):
            out[task][s] = (float(m), float(sd))
    return out


# -------------------- ChemBERTa ensemble -----------------------------------


def train_chemberta_ensemble_kfold(
    wide: pd.DataFrame,
    folds: list[tuple[set[str], set[str]]],
    args,
    n_seeds: int = 5,
) -> dict[str, dict]:
    """5 seeds * len(folds) ChemBERTas. OOF-aggregate per (task, compound).

    Returns {task: {smi_oof, y_oof, mean_oof, std_oof, q95, coverage, n_seeds}}.
    """
    from .chemberta_lora import (
        ChemBertaConfig,
        _make_config,
        _predict_smiles,
        _train_one_fold,
    )
    import torch

    base_config = _make_config(args)

    pred_by_seed: dict[str, dict[str, list[float]]] = {t: {} for t in REG_TASKS}

    for fi, (train_set, test_set) in enumerate(folds):
        smi_train_full = sorted(train_set)
        n_val = max(1, len(smi_train_full) // 10)
        smi_val_inner = smi_train_full[:n_val]
        smi_train = smi_train_full[n_val:]
        smi_test = sorted(test_set)
        for seed in range(n_seeds):
            log.info("chemberta ensemble fold=%d seed=%d  train=%d test=%d",
                     fi, seed, len(smi_train), len(smi_test))
            cfg = ChemBertaConfig(**{**base_config.__dict__, "seed": seed})
            model, standardizer = _train_one_fold(
                wide, smi_train, smi_val_inner, cfg, fold_label=f"f{fi}s{seed}",
            )
            preds = _predict_smiles(model, wide, standardizer, smi_test, cfg)
            for t in REG_TASKS:
                for s, p in preds[t].items():
                    pred_by_seed[t].setdefault(s, []).append(p)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    out: dict[str, dict] = {}
    for t in REG_TASKS:
        # Build (key -> y_true) lookup. For toxicity the key is "smi@conc"
        # because we predict at every measured concentration; for IRI /
        # permeability the key is the bare smiles.
        sub = wide.dropna(subset=[t])
        if t == "toxicity":
            key_to_y = {
                f"{r['smiles_canonical']}@{float(r['concentration_mol_kg']):g}":
                    float(r[t])
                for _, r in sub.iterrows()
            }
        else:
            key_to_y = {r["smiles_canonical"]: float(r[t]) for _, r in sub.iterrows()}
        keys = sorted(k for k in pred_by_seed[t].keys() if k in key_to_y)
        if not keys:
            continue
        means = np.array([np.mean(pred_by_seed[t][k]) for k in keys])
        stds = np.array([np.std(pred_by_seed[t][k], ddof=0) for k in keys])
        y = np.array([key_to_y[k] for k in keys])
        residuals = np.abs(y - means)
        q95 = conformal_q95(residuals, alpha=0.05)
        coverage = empirical_coverage(y, means, q95)
        out[t] = {
            "task": t,
            "smi_oof": keys,  # for toxicity these are "smi@conc" strings
            "y_oof": y,
            "mean_oof": means,
            "std_oof": stds,
            "q95": q95,
            "coverage": coverage,
            "n_seeds": n_seeds,
        }
        log.info("chemberta ensemble task=%s  n=%d q95=%.3g coverage=%.3f",
                 t, len(keys), q95, coverage)
    return out


def train_chemberta_ensemble_full(
    wide: pd.DataFrame,
    args,
    n_seeds: int = 5,
):
    """Full-data ensemble: n_seeds ChemBERTa models trained on all CPA labels.

    Splits 90/10 internally for early-stopping. Returns list of (model,
    standardizer) tuples for FDA IID prediction.
    """
    from .chemberta_lora import (
        ChemBertaConfig,
        _make_config,
        _train_one_fold,
    )
    import torch

    base_config = _make_config(args)

    smi_all = sorted(wide["smiles_canonical"].unique())
    n_val = max(1, len(smi_all) // 10)
    rng = np.random.default_rng(0)  # fixed split across seeds
    perm = rng.permutation(smi_all)
    smi_val_inner = sorted(perm[:n_val].tolist())
    smi_train = sorted(perm[n_val:].tolist())

    out = []
    for seed in range(n_seeds):
        log.info("chemberta full-data ensemble seed=%d  train=%d val=%d",
                 seed, len(smi_train), len(smi_val_inner))
        cfg = ChemBertaConfig(**{**base_config.__dict__, "seed": seed})
        model, standardizer = _train_one_fold(
            wide, smi_train, smi_val_inner, cfg, fold_label=f"full-seed{seed}",
        )
        out.append((model, standardizer, cfg))
    return out


def predict_chemberta_ensemble(
    ensemble: list,
    wide: pd.DataFrame,
    smiles_list: list[str],
    toxicity_conc_mol_kg: float = TOX_REFERENCE_CONC_MOL_KG,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Predict (mean, std) per (task, smiles) using a ChemBERTa full-data
    ensemble. For FDA scoring we want one prediction per task per
    candidate compound, so we predict at task-specific reference
    concentrations: TOX_REFERENCE_CONC_MOL_KG for toxicity, the assay
    defaults for IRI / permeability.

    `ensemble` is a list of (model, standardizer, cfg) tuples produced by
    train_chemberta_ensemble_full().
    """
    from .chemberta_lora import (
        _predict_at_conditions,
        REG_TASKS as _REG,
        IRI_REFERENCE_CONC_MOL_KG,
        PERM_REFERENCE_CONC_MOL_KG,
    )

    task_conc = {
        "toxicity": float(toxicity_conc_mol_kg),
        "permeability": PERM_REFERENCE_CONC_MOL_KG,
        "iri": IRI_REFERENCE_CONC_MOL_KG,
    }

    pred_by_seed: dict[str, dict[str, list[float]]] = {t: {} for t in _REG}
    for model, standardizer, cfg in ensemble:
        for t in _REG:
            cond_rows = [(s, task_conc[t]) for s in smiles_list]
            seed_preds = _predict_at_conditions(model, standardizer, cond_rows, cfg)
            for s, p in zip(smiles_list, seed_preds[t]):
                pred_by_seed[t].setdefault(s, []).append(float(p))

    out: dict[str, dict[str, tuple[float, float]]] = {t: {} for t in _REG}
    for t in _REG:
        for s, vals in pred_by_seed[t].items():
            arr = np.array(vals, dtype=float)
            out[t][s] = (float(arr.mean()), float(arr.std(ddof=0)))
    return out
