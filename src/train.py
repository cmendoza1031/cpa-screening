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
    p.add_argument("--split-mode", choices=["random"], default="random",
                   help="random for v1; cluster-aware lands in Phase 2")
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    # ChemBERTa-only:
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--no-tox21-aux", action="store_true",
                   help="disable Tox21 auxiliary head (faster, no DeepChem dep)")
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

    # ---- IRI: 70/15/15 train/val/test ------------------------------------
    iri_df = long_df[long_df["task"] == "iri"]
    if not iri_df.empty:
        splits = random_split_by_smiles(
            iri_df["smiles_canonical"].unique(),
            seed=args.seed,
            train=IRI_TRAIN, val=IRI_VAL, test=IRI_TEST,
        )
        # train_rf_per_task can take only one task by filtering input
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
        summary_metrics["iri"] = {"scheme": "70/15/15", "n_train": rf_results["iri"]["n_train"],
                                  "n_val": rf_results["iri"]["n_val"],
                                  "n_test": rf_results["iri"]["n_test"]}

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

    rows = train_chemberta_multitask(long_df, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": "chemberta",
        "seed": args.seed,
        "lora_rank": args.lora_rank,
        "audit": audit,
        "metrics": rows,
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
        run_rf(args)
    elif args.model == "chemberta":
        run_chemberta(args)


if __name__ == "__main__":
    main()
