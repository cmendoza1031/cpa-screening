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
    """Train n_seeds * len(folds) RFs; aggregate per-compound (mean, std).

    Returns:
        smi_oof: list of canonical SMILES (one per OOF prediction)
        y_oof: ndarray of true values
        mean_oof: ndarray of ensemble mean predictions
        std_oof: ndarray of ensemble std predictions
        q95: float, conformal calibration constant
        coverage: float, empirical coverage of mean +/- q95 (~0.95 target)
    """
    from .rf_baseline import featurize, _fit_rf, RFConfig

    sub = long_df[long_df["task"] == task].dropna(subset=["value"])
    if sub.empty:
        return {}
    smi_to_y = dict(zip(sub["smiles_canonical"], sub["value"]))

    # Per-compound predictions across seeds: smi -> list of preds (one per seed)
    pred_by_seed: dict[str, list[float]] = {}
    y_by_smi: dict[str, float] = {}

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
        for seed in range(n_seeds):
            cfg = RFConfig(seed=seed, n_estimators=n_estimators)
            model = _fit_rf(X_train, y_train, cfg)
            yhat = model.predict(X_test)
            for s, yh in zip(smi_test_kept, yhat):
                pred_by_seed.setdefault(s, []).append(float(yh))
                y_by_smi[s] = float(smi_to_y[s])
        log.info("rf ensemble task=%s fold=%d  train=%d test=%d  seeds=%d",
                 task, fi, len(y_train), len(smi_test_kept), n_seeds)

    smi_list = sorted(pred_by_seed.keys())
    means = np.array([np.mean(pred_by_seed[s]) for s in smi_list])
    stds = np.array([np.std(pred_by_seed[s], ddof=0) for s in smi_list])
    y = np.array([y_by_smi[s] for s in smi_list])

    residuals = np.abs(y - means)
    q95 = conformal_q95(residuals, alpha=0.05)
    coverage = empirical_coverage(y, means, q95)
    log.info("rf ensemble task=%s  n=%d q95=%.3g  coverage=%.3f", task, len(smi_list), q95, coverage)
    return {
        "task": task,
        "smi_oof": smi_list,
        "y_oof": y,
        "mean_oof": means,
        "std_oof": stds,
        "q95": q95,
        "coverage": coverage,
        "n_seeds": n_seeds,
    }


def train_rf_ensemble_full(
    long_df: pd.DataFrame,
    n_seeds: int = 5,
    n_estimators: int = 500,
) -> dict[str, list]:
    """Train one full-data ensemble per task (no held-out), for FDA IID scoring.

    Returns {task: list of trained RF models, length n_seeds}.
    """
    from .rf_baseline import featurize, _fit_rf, RFConfig

    out: dict[str, list] = {}
    for task in REG_TASKS:
        sub = long_df[long_df["task"] == task].dropna(subset=["value"])
        if sub.empty:
            continue
        smi = sub["smiles_canonical"].tolist()
        X, smi_kept = featurize(smi)
        if X.shape[0] == 0:
            continue
        smi_to_y = dict(zip(sub["smiles_canonical"], sub["value"]))
        y = np.array([smi_to_y[s] for s in smi_kept])
        models = []
        for seed in range(n_seeds):
            cfg = RFConfig(seed=seed, n_estimators=n_estimators)
            models.append(_fit_rf(X, y, cfg))
        out[task] = models
        log.info("rf full-data ensemble task=%s  n_train=%d  seeds=%d",
                 task, len(y), n_seeds)
    return out


def predict_rf_ensemble(
    models_per_task: dict[str, list],
    smiles_list: list[str],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Predict (mean, std) per (task, smiles) using a full-data ensemble."""
    from .rf_baseline import featurize

    X, smi_kept = featurize(smiles_list)
    out: dict[str, dict[str, tuple[float, float]]] = {t: {} for t in models_per_task}
    if X.shape[0] == 0:
        return out
    for task, models in models_per_task.items():
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
        # Get true labels from wide for compounds where we have predictions
        sub = wide.dropna(subset=[t])
        smi_to_y = dict(zip(sub["smiles_canonical"], sub[t]))
        smi_with_pred = sorted(s for s in pred_by_seed[t].keys() if s in smi_to_y)
        if not smi_with_pred:
            continue
        means = np.array([np.mean(pred_by_seed[t][s]) for s in smi_with_pred])
        stds = np.array([np.std(pred_by_seed[t][s], ddof=0) for s in smi_with_pred])
        y = np.array([smi_to_y[s] for s in smi_with_pred])
        residuals = np.abs(y - means)
        q95 = conformal_q95(residuals, alpha=0.05)
        coverage = empirical_coverage(y, means, q95)
        out[t] = {
            "task": t,
            "smi_oof": smi_with_pred,
            "y_oof": y,
            "mean_oof": means,
            "std_oof": stds,
            "q95": q95,
            "coverage": coverage,
            "n_seeds": n_seeds,
        }
        log.info("chemberta ensemble task=%s  n=%d q95=%.3g coverage=%.3f",
                 t, len(smi_with_pred), q95, coverage)
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
) -> dict[str, dict[str, tuple[float, float]]]:
    """Predict (mean, std) per (task, smiles) using a ChemBERTa full-data ensemble.

    `ensemble` is a list of (model, standardizer, cfg) tuples produced by
    train_chemberta_ensemble_full().
    """
    from .chemberta_lora import _predict_smiles, REG_TASKS as _REG

    pred_by_seed: dict[str, dict[str, list[float]]] = {t: {} for t in _REG}
    for model, standardizer, cfg in ensemble:
        preds = _predict_smiles(model, wide, standardizer, smiles_list, cfg)
        for t in _REG:
            for s, p in preds[t].items():
                pred_by_seed[t].setdefault(s, []).append(p)

    out: dict[str, dict[str, tuple[float, float]]] = {t: {} for t in _REG}
    for t in _REG:
        for s, vals in pred_by_seed[t].items():
            arr = np.array(vals, dtype=float)
            out[t][s] = (float(arr.mean()), float(arr.std(ddof=0)))
    return out
