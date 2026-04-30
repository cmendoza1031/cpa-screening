"""Training entrypoint.

Eval scheme (see RATIONALE.md for the full reasoning):
    iri (n=303)         -> 70/15/15 train/val/test, seed-keyed on SMILES
    permeability (n=16) -> 5-fold CV (consistent with ChemBERTa)
                           + RF-only LOO secondary analysis
    toxicity (n=22)     -> 5-fold CV

Examples:
    python -m src.train --model rf --seed 0
    python -m src.train --model chemberta --lora-rank 8 --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import (
    build_dataset,
    cluster_aware_kfold_split,
    kfold_split_by_smiles,
    loo_split_by_smiles,
    random_split_by_smiles,
)
from .eval import parity_plot, regression_metrics, summarize_runs
from .models.rf_baseline import RFConfig, train_rf_kfold, train_rf_per_task
from .utils import RESULTS_DIR, get_logger, seed_everything

log = get_logger("train")

SUMMARY_PATH = RESULTS_DIR / "summary.json"
TABLE_PATH = RESULTS_DIR / "results_table.csv"

# Eval scheme constants
IRI_TRAIN = 0.70
IRI_VAL = 0.15
IRI_TEST = 0.15
KFOLD_K = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CPA models")
    p.add_argument("--model", choices=["rf", "chemberta"], default="rf")
    p.add_argument("--task", default="all", help="all | toxicity | permeability | iri")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-mode", choices=["random", "cluster"], default="random",
                   help="random or Tanimoto-cluster-aware k-fold (Phase 2)")
    p.add_argument("--cluster-threshold", type=float, default=0.6,
                   help="Tanimoto threshold for Butina clustering when --split-mode=cluster")
    p.add_argument("--n-seeds", type=int, default=1,
                   help="ensemble size; >1 runs Phase 2 deep-ensemble eval")
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    # ChemBERTa-only:
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    # 1e-4 (spec) -> 5e-5 -> 2e-5 -> 5e-5. The 2e-5 setting was over-conservative:
    # combined with clip_grad_value(0.5) the max per-element update was 1e-5,
    # too small to fine-tune meaningfully in 30 epochs (train_loss only fell
    # ~10% per fold). Now that training is stable (0 NaN skips with the
    # nan_to_num + value-clip safety nets), 5e-5 lets the model actually learn.
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument(
        "--tox21-aux", action="store_true",
        help="ChemBERTa: enable the Tox21 auxiliary classification head. "
             "Adds a 12-class BCE loss on the shared encoder, weighted at 0.1, "
             "to give the encoder a broader toxicity-relevant signal during "
             "fine-tuning. Pulls the Tox21 CSV from the deepchem GitHub mirror "
             "(no DeepChem package needed).",
    )
    p.add_argument(
        "--tox21-aux-weight", type=float, default=0.1,
        help="weight on the Tox21 aux BCE loss (default 0.1; total = cpa_huber + w * aux_bce)",
    )
    p.add_argument(
        "--no-tox21-aux", action="store_true",
        help="(Legacy alias; tox21 aux head is off by default. Use --tox21-aux to enable.)",
    )
    p.add_argument(
        "--cv",
        action="store_true",
        help="ChemBERTa: train 5 models on a 5-fold CV split for apples-to-apples "
             "comparison with the RF baseline on the small Higgins tasks. Default "
             "off uses a single 70/15/15 train/val/test split.",
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="number of CV folds when --cv is set (default 5)",
    )
    p.add_argument("--smoke", action="store_true",
                   help="2-epoch smoke run on subset for local sanity")
    return p.parse_args()


def filter_task(long_df: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "all":
        return long_df
    return long_df[long_df["task"] == task].copy()


def _print_audit_summary(audit: dict) -> None:
    print(f"\nDataset audit: {audit['unique_compounds']} unique compounds, "
          f"{audit['n_long_rows']} (compound, task) labels")
    for task, stats in audit.get("per_task", {}).items():
        print(f"  {task:14s} n={stats['n']:5d}  range=[{stats['value_min']:.3g}, "
              f"{stats['value_max']:.3g}]  mean={stats['value_mean']:.3g}")


def _append_results_table(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    if TABLE_PATH.exists():
        prev = pd.read_csv(TABLE_PATH)
        combined = pd.concat([prev, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=[c for c in ("model", "task", "split", "scheme") if c in combined.columns],
            keep="last",
        )
        combined.to_csv(TABLE_PATH, index=False)
    else:
        df.to_csv(TABLE_PATH, index=False)
    log.info("wrote %s (%d rows)", TABLE_PATH, len(df))


def run_rf(args: argparse.Namespace) -> dict:
    """RF baseline using the per-task eval scheme.

    iri: random 70/15/15.
    permeability: 5-fold CV + LOO secondary.
    toxicity: 5-fold CV.
    """
    long_df, audit = build_dataset()
    _print_audit_summary(audit)
    long_df = filter_task(long_df, args.task)
    if long_df.empty:
        raise RuntimeError(
            f"no data available for task={args.task}; check data audit"
        )

    config = RFConfig(seed=args.seed)
    rows: list[dict] = []
    summary_metrics: dict = {}

    # ---- IRI: 70/15/15 train/val/test (canonical) + 5-fold CV (for ChemBERTa
    # comparison since ChemBERTa --cv runs all tasks under 5-fold CV) -------
    iri_df = long_df[long_df["task"] == "iri"]
    if not iri_df.empty:
        # Primary: 70/15/15 (n=303 supports it; this is the canonical IRI eval)
        splits = random_split_by_smiles(
            iri_df["smiles_canonical"].unique(),
            seed=args.seed,
            train=IRI_TRAIN, val=IRI_VAL, test=IRI_TEST,
        )
        rf_results = train_rf_per_task(iri_df, splits, config=config)
        for task, art in rf_results.items():
            for split in ("train", "val", "test"):
                m = regression_metrics(art[f"y_{split}"], art[f"yhat_{split}"])
                rows.append({
                    "model": f"rf_seed{args.seed}",
                    "task": task,
                    "scheme": "70/15/15",
                    "split": split,
                    **m,
                })
            for split in ("val", "test"):
                parity_plot(
                    art[f"y_{split}"], art[f"yhat_{split}"],
                    task=task, split=split,
                    model_tag=f"rf_seed{args.seed}",
                )
        summary_metrics["iri_70_15_15"] = {
            "scheme": "70/15/15",
            "n_train": rf_results["iri"]["n_train"],
            "n_val": rf_results["iri"]["n_val"],
            "n_test": rf_results["iri"]["n_test"],
        }
        # Secondary: 5-fold CV (for direct comparison with ChemBERTa --cv)
        folds = kfold_split_by_smiles(
            iri_df["smiles_canonical"].unique(), k=KFOLD_K, seed=args.seed,
        )
        cv = train_rf_kfold(iri_df, "iri", folds, config=config)
        m = regression_metrics(cv["y_oof"], cv["yhat_oof"])
        rows.append({
            "model": f"rf_seed{args.seed}",
            "task": "iri",
            "scheme": f"{KFOLD_K}-fold-CV",
            "split": "oof",
            **m,
        })
        parity_plot(cv["y_oof"], cv["yhat_oof"], task="iri",
                    split=f"oof_{KFOLD_K}fold",
                    model_tag=f"rf_seed{args.seed}")
        summary_metrics["iri_kfold"] = {
            "scheme": f"{KFOLD_K}-fold-CV",
            "n_oof": int(len(cv["y_oof"])),
        }

    # ---- Toxicity: 5-fold CV --------------------------------------------
    tox_df = long_df[long_df["task"] == "toxicity"]
    if not tox_df.empty:
        folds = kfold_split_by_smiles(
            tox_df["smiles_canonical"].unique(), k=KFOLD_K, seed=args.seed,
        )
        cv = train_rf_kfold(tox_df, "toxicity", folds, config=config)
        m = regression_metrics(cv["y_oof"], cv["yhat_oof"])
        rows.append({
            "model": f"rf_seed{args.seed}",
            "task": "toxicity",
            "scheme": f"{KFOLD_K}-fold-CV",
            "split": "oof",
            **m,
        })
        parity_plot(cv["y_oof"], cv["yhat_oof"], task="toxicity",
                    split=f"oof_{KFOLD_K}fold", model_tag=f"rf_seed{args.seed}")
        summary_metrics["toxicity"] = {
            "scheme": f"{KFOLD_K}-fold-CV",
            "n_oof": int(len(cv["y_oof"])),
            "per_fold": cv["per_fold_metrics"],
        }

    # ---- Permeability: 5-fold CV + LOO secondary -----------------------
    perm_df = long_df[long_df["task"] == "permeability"]
    if not perm_df.empty:
        # 5-fold (primary, comparable with ChemBERTa)
        folds = kfold_split_by_smiles(
            perm_df["smiles_canonical"].unique(), k=KFOLD_K, seed=args.seed,
        )
        cv = train_rf_kfold(perm_df, "permeability", folds, config=config)
        m = regression_metrics(cv["y_oof"], cv["yhat_oof"])
        rows.append({
            "model": f"rf_seed{args.seed}",
            "task": "permeability",
            "scheme": f"{KFOLD_K}-fold-CV",
            "split": "oof",
            **m,
        })
        parity_plot(cv["y_oof"], cv["yhat_oof"], task="permeability",
                    split=f"oof_{KFOLD_K}fold", model_tag=f"rf_seed{args.seed}")

        # LOO (secondary, RF-only sanity check)
        loo = loo_split_by_smiles(perm_df["smiles_canonical"].unique())
        cv_loo = train_rf_kfold(perm_df, "permeability", loo, config=config)
        m_loo = regression_metrics(cv_loo["y_oof"], cv_loo["yhat_oof"])
        rows.append({
            "model": f"rf_seed{args.seed}",
            "task": "permeability",
            "scheme": "LOO-CV",
            "split": "oof",
            **m_loo,
        })
        parity_plot(cv_loo["y_oof"], cv_loo["yhat_oof"], task="permeability",
                    split="oof_loo", model_tag=f"rf_seed{args.seed}")
        summary_metrics["permeability"] = {
            "primary_scheme": f"{KFOLD_K}-fold-CV",
            "n_oof_kfold": int(len(cv["y_oof"])),
            "n_oof_loo": int(len(cv_loo["y_oof"])),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": "rf",
        "seed": args.seed,
        "split_mode": args.split_mode,
        "task_filter": args.task,
        "audit": audit,
        "metrics": rows,
        "task_eval": summary_metrics,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    log.info("wrote %s", SUMMARY_PATH)

    _append_results_table(rows)

    print("\n========== RESULTS ==========")
    print(pd.DataFrame(rows).to_string(index=False))
    print("=============================\n")
    return summary


def run_chemberta(args: argparse.Namespace) -> dict:
    from .models.chemberta_lora import train_chemberta_multitask

    long_df, audit = build_dataset()
    _print_audit_summary(audit)

    if args.n_seeds > 1:
        rows = run_chemberta_ensemble(long_df, args)
    else:
        rows = train_chemberta_multitask(long_df, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": "chemberta",
        "seed": args.seed,
        "lora_rank": args.lora_rank,
        "n_seeds": args.n_seeds,
        "split_mode": args.split_mode,
        "audit": audit,
        "metrics": rows,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    _append_results_table(rows)

    print("\n========== RESULTS ==========")
    print(pd.DataFrame(rows).to_string(index=False))
    print("=============================\n")
    return summary


def _build_folds(args, smiles: list[str], k: int):
    if args.split_mode == "cluster":
        return cluster_aware_kfold_split(
            smiles, k=k, threshold=args.cluster_threshold, seed=args.seed,
        )
    return kfold_split_by_smiles(smiles, k=k, seed=args.seed)


def run_chemberta_ensemble(long_df: pd.DataFrame, args) -> list[dict]:
    """Phase 2 deep-ensemble eval for ChemBERTa.

    For each fold (5 folds), train n_seeds fresh models, aggregate per-
    compound predictions across seeds (mean = point estimate, std =
    epistemic uncertainty). Conformal-calibrate q95 on OOF residuals,
    report empirical coverage as a sanity check on the calibration.
    """
    from .models.chemberta_lora import _wide_targets, REG_TASKS as _REG
    from .models.ensemble import train_chemberta_ensemble_kfold

    wide = _wide_targets(long_df)
    folds = _build_folds(args, wide["smiles_canonical"].unique().tolist(), k=args.cv_folds)
    log.info(
        "ChemBERTa ensemble: %s split, %d folds, %d seeds = %d trainings",
        args.split_mode, args.cv_folds, args.n_seeds, args.cv_folds * args.n_seeds,
    )
    results = train_chemberta_ensemble_kfold(wide, folds, args, n_seeds=args.n_seeds)

    rows: list[dict] = []
    model_tag = f"chemberta_r{args.lora_rank}_{args.split_mode}_n{args.n_seeds}"
    scheme = f"{args.split_mode}-{args.cv_folds}fold-CV-ensemble{args.n_seeds}"
    for t, art in results.items():
        m = regression_metrics(art["y_oof"], art["mean_oof"])
        rows.append({
            "model": model_tag,
            "task": t,
            "scheme": scheme,
            "split": "oof",
            **m,
            "q95": art["q95"],
            "coverage_95": art["coverage"],
        })
        parity_plot(
            art["y_oof"], art["mean_oof"], task=t, split=f"oof_{scheme}",
            model_tag=model_tag,
        )
    return rows


def run_rf_ensemble(args: argparse.Namespace) -> dict:
    """Phase 2 deep-ensemble eval for RF (parallel to ChemBERTa)."""
    from .models.ensemble import train_rf_ensemble_kfold

    long_df, audit = build_dataset()
    _print_audit_summary(audit)

    rows: list[dict] = []
    summary_metrics: dict = {}
    model_tag = f"rf_{args.split_mode}_n{args.n_seeds}"
    scheme = f"{args.split_mode}-{args.cv_folds}fold-CV-ensemble{args.n_seeds}"

    for task in ["iri", "toxicity", "permeability"]:
        sub = long_df[long_df["task"] == task]
        if sub.empty:
            continue
        folds = _build_folds(args, sub["smiles_canonical"].unique().tolist(), k=args.cv_folds)
        art = train_rf_ensemble_kfold(long_df, task, folds, n_seeds=args.n_seeds)
        if not art:
            continue
        m = regression_metrics(art["y_oof"], art["mean_oof"])
        rows.append({
            "model": model_tag,
            "task": task,
            "scheme": scheme,
            "split": "oof",
            **m,
            "q95": art["q95"],
            "coverage_95": art["coverage"],
        })
        parity_plot(
            art["y_oof"], art["mean_oof"], task=task, split=f"oof_{scheme}",
            model_tag=model_tag,
        )
        summary_metrics[task] = {
            "scheme": scheme,
            "n_oof": int(len(art["y_oof"])),
            "q95": art["q95"],
            "coverage_95": art["coverage"],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "rf_ensemble",
        "n_seeds": args.n_seeds,
        "split_mode": args.split_mode,
        "audit": audit,
        "metrics": rows,
        "task_eval": summary_metrics,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    _append_results_table(rows)
    print("\n========== RESULTS ==========")
    print(pd.DataFrame(rows).to_string(index=False))
    print("=============================\n")
    return summary


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    log.info("config: %s", vars(args))

    if args.model == "rf":
        if args.n_seeds > 1:
            run_rf_ensemble(args)
        else:
            run_rf(args)
    elif args.model == "chemberta":
        run_chemberta(args)


if __name__ == "__main__":
    main()
