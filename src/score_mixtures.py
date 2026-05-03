"""Mixture analysis + scoring driver.

Two outputs:

1. **Additive baseline evaluation** on the 16 known Higgins binary
   mixtures. Compares four combination rules (max / mean / sum_then_cap /
   weighted_max) on Spearman / MAE / R^2 and counts neutralization misses.
   Per-row residuals are saved so the formamide+glycerol case is visible.

2. **FDA mixture pair scoring**. Enumerates all binary pairs from the v2
   FDA candidate pool, scores each with the additive-baseline composite,
   and outputs Pareto-ranked top-K mixture pairs.

Run from the repo root:

    python -m src.score_mixtures \
        --architecture rf \
        --rule max \
        --conc-total 6.0 \
        --top-k 20

The architecture flag selects which single-compound ensemble drives the
predictions. RF is the recommended default (toxicity Spearman 0.64
cluster-ensemble, the strongest single-compound model in v2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import build_dataset
from .data.fda_iid import load_fda_iid
from .models.ensemble import (
    TOX_REFERENCE_CONC_MOL_KG,
    predict_rf_ensemble,
    train_rf_ensemble_full,
)
from .models.mixture import (
    SinglePrediction,
    evaluate_additive_baseline,
    load_mixtures,
    score_fda_mixture_pairs,
    train_pair_encoder_loo,
    MIX_RESULTS_DIR,
)
from .utils import RESULTS_DIR, get_logger

log = get_logger("score_mixtures")


def _build_rf_predictors(long_df: pd.DataFrame, n_seeds: int):
    """Train the v2 RF ensemble on (compound, concentration) data and
    return three callables (smiles, conc_mol_kg) -> SinglePrediction for
    toxicity / permeability / IRI."""
    log.info("training v2 RF ensemble (n_seeds=%d)", n_seeds)
    models_per_task = train_rf_ensemble_full(long_df, n_seeds=n_seeds)
    if not models_per_task:
        raise RuntimeError("train_rf_ensemble_full returned empty; check long_df")

    def _make(task: str):
        def predictor(smi: str, conc: float) -> SinglePrediction:
            preds = predict_rf_ensemble(
                {task: models_per_task[task]},
                [smi],
                toxicity_conc_mol_kg=conc,  # only used for toxicity
            )
            if smi not in preds.get(task, {}):
                # SMILES failed to featurize; fall back to flat prediction
                return SinglePrediction(mean=float("nan"), std=float("inf"))
            mean, std = preds[task][smi]
            return SinglePrediction(mean=float(mean), std=float(std))
        return predictor

    return _make("toxicity"), _make("permeability"), _make("iri")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score CPA mixtures")
    p.add_argument("--architecture", choices=["rf"], default="rf",
                   help="single-compound ensemble to use for the additive "
                        "baseline. Only RF is wired here; ChemBERTa would "
                        "need a per-compound concentration-aware predictor.")
    p.add_argument("--n-seeds", type=int, default=5,
                   help="ensemble size for the single-compound model")
    p.add_argument(
        "--rule", choices=["max", "mean", "sum_then_cap", "weighted_max"],
        default="max",
        help="combination rule used in the FDA pair scoring composite. "
             "Evaluation against ground-truth mixtures runs all four rules.",
    )
    p.add_argument(
        "--conc-total", type=float, default=6.0,
        help="total mixture concentration (mol/kg) for FDA pair scoring. "
             "Each compound enters at half. Default 6.0 is the lower of the "
             "two regimes Higgins Dec 2025 covers.",
    )
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--no-fda", action="store_true",
        help="skip FDA pair scoring (just run the 16-mixture additive eval)",
    )
    p.add_argument(
        "--max-fda-compounds", type=int, default=None,
        help="cap the FDA candidate pool size for fast iteration. Default uses "
             "the full v2-filtered pool (~140); 50 -> 1225 pairs runs in <1 min.",
    )
    return p.parse_args()


def _format_dashed_cas(cas) -> str:
    """FDA IID stores CAS as bare digits; format as standard dashed CAS."""
    if pd.isna(cas):
        return ""
    s = str(int(cas))
    if len(s) < 5:
        return s
    return f"{s[:-3]}-{s[-3:-1]}-{s[-1]}"


def main() -> None:
    args = parse_args()

    log.info("loading dataset")
    long_df, _ = build_dataset()

    log.info("training single-compound predictor ensemble")
    predict_tox, predict_perm, predict_iri = _build_rf_predictors(long_df, args.n_seeds)

    # 1. Additive baseline on Higgins mixtures
    mixtures = load_mixtures()
    if mixtures.empty:
        log.warning(
            "no Higgins mixture data found at data/processed/higgins_mixtures.parquet; "
            "skipping additive-baseline evaluation"
        )
        eval_df, summary = pd.DataFrame(), {}
    else:
        log.info("evaluating additive baseline on %d Higgins mixtures", len(mixtures))
        eval_df, summary = evaluate_additive_baseline(mixtures, predict_tox)

    MIX_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not eval_df.empty:
        eval_df.to_csv(MIX_RESULTS_DIR / "additive_baseline_predictions.csv", index=False)
        log.info("wrote %s", MIX_RESULTS_DIR / "additive_baseline_predictions.csv")

    # PairEncoder LOO-CV against the same mixtures: does a learned
    # interaction term beat the additive baseline at this sample size?
    if not mixtures.empty:
        pe = train_pair_encoder_loo(mixtures)
        if pe:
            pe["per_row"].to_csv(
                MIX_RESULTS_DIR / "pair_encoder_predictions.csv", index=False,
            )
            with open(MIX_RESULTS_DIR / "pair_encoder_metrics.json", "w") as f:
                json.dump({"n": pe["n"], "metrics": pe["metrics"]}, f, indent=2)
            log.info(
                "wrote %s and %s",
                MIX_RESULTS_DIR / "pair_encoder_predictions.csv",
                MIX_RESULTS_DIR / "pair_encoder_metrics.json",
            )

    if summary:
        # Save summary as JSON for later inspection
        with open(MIX_RESULTS_DIR / "additive_baseline_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        log.info("\nAdditive baseline summary (%d known mixtures):", len(mixtures))
        log.info("rule              n   Spearman    MAE   RMSE     R^2  n_neut_misses")
        log.info("------------------------------------------------------------------")
        for rule, m in summary.items():
            log.info(
                "%-14s %4d   %+.3f   %5.2f  %5.2f  %+.3f      %d",
                rule, m["n"], m["spearman"], m["mae"], m["rmse"], m["r2"],
                m["n_neutralization_misses"],
            )

        # Print the formamide+glycerol cases by name for the highlight reel
        if not eval_df.empty:
            interesting = eval_df[
                eval_df["name_b"].str.contains("formamide", case=False, na=False)
                | eval_df["name_a"].str.contains("formamide", case=False, na=False)
            ]
            if not interesting.empty:
                log.info("\nNeutralization case study (formamide-containing rows):")
                cols = ["name_a", "name_b", "conc_total", "viability_4c",
                        "v_pred_max", "v_pred_mean", "v_pred_sum_then_cap"]
                # Some rules may not have been computed; subset to what's there
                cols = [c for c in cols if c in interesting.columns]
                log.info("\n%s\n", interesting[cols].to_string(index=False))

    # 2. FDA pair scoring
    if args.no_fda:
        return

    log.info("loading FDA candidate pool")
    fda = load_fda_iid()
    if args.max_fda_compounds is not None:
        fda = fda.head(args.max_fda_compounds).copy()
    log.info("scoring binary pairs from %d FDA candidates", len(fda))

    top_k_df, all_df = score_fda_mixture_pairs(
        fda,
        predictor_tox=predict_tox,
        predictor_perm=predict_perm,
        predictor_iri=predict_iri,
        conc_total_mol_kg=args.conc_total,
        rule=args.rule,
        top_k=args.top_k,
    )

    if not top_k_df.empty:
        top_k_df.to_csv(MIX_RESULTS_DIR / f"top{args.top_k}_pairs.csv", index=False)
        all_df.to_csv(MIX_RESULTS_DIR / "all_pairs_scored.csv", index=False)
        log.info(
            "wrote %s (%d rows) and %s (%d rows)",
            MIX_RESULTS_DIR / f"top{args.top_k}_pairs.csv", len(top_k_df),
            MIX_RESULTS_DIR / "all_pairs_scored.csv", len(all_df),
        )
        log.info("\nTop-%d FDA mixture pairs (rule=%s, total=%g mol/kg):",
                 args.top_k, args.rule, args.conc_total)
        cols = ["rank", "name_a", "name_b", "tox_pred", "perm_pred", "iri_pred",
                "composite_score"]
        log.info("\n%s\n", top_k_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
