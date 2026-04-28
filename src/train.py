"""Training entrypoint.

Phase 1 Day 1: --model rf only.
Phase 1 Day 2 will add --model chemberta with --lora_rank.

Examples:
    python -m src.train --model rf --seed 0
    python -m src.train --model rf --seed 0 --task toxicity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import build_dataset, random_split_by_smiles
from .eval import parity_plot, summarize_runs
from .models.rf_baseline import RFConfig, train_rf_per_task
from .utils import RESULTS_DIR, get_logger, seed_everything

log = get_logger("train")

SUMMARY_PATH = RESULTS_DIR / "summary.json"
TABLE_PATH = RESULTS_DIR / "results_table.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CPA models")
    p.add_argument("--model", choices=["rf", "chemberta"], default="rf")
    p.add_argument("--task", default="all", help="all | toxicity | permeability | iri")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-mode", choices=["random"], default="random",
                   help="random for v1; cluster-aware lands in Phase 2")
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    # ChemBERTa-only (used Phase 1 Day 2):
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    return p.parse_args()


def filter_task(long_df: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "all":
        return long_df
    return long_df[long_df["task"] == task].copy()


def run_rf(args: argparse.Namespace) -> dict:
    long_df, audit = build_dataset()
    print(f"\nDataset audit: {audit['unique_compounds']} unique compounds, "
          f"{audit['n_long_rows']} (compound, task) labels")
    for task, stats in audit.get("per_task", {}).items():
        print(f"  {task:14s} n={stats['n']:5d}  range=[{stats['value_min']:.3g}, "
              f"{stats['value_max']:.3g}]  mean={stats['value_mean']:.3g}")
    long_df = filter_task(long_df, args.task)
    if long_df.empty:
        raise RuntimeError(
            f"no data available for task={args.task}; check data audit / template files"
        )

    splits = random_split_by_smiles(long_df["smiles_canonical"].unique(), seed=args.seed)
    rf_results = train_rf_per_task(
        long_df, splits, config=RFConfig(seed=args.seed)
    )

    rows = summarize_runs(rf_results, model_tag=f"rf_seed{args.seed}")
    metrics_df = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for task, art in rf_results.items():
        for split in ("val", "test"):
            parity_plot(
                art[f"y_{split}"],
                art[f"yhat_{split}"],
                task=task,
                split=split,
                model_tag=f"rf_seed{args.seed}",
            )

    summary = {
        "model": "rf",
        "seed": args.seed,
        "split_mode": args.split_mode,
        "task_filter": args.task,
        "audit": audit,
        "metrics": rows,
        "split_sizes": {k: len(v) for k, v in splits.items()},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    log.info("wrote %s", SUMMARY_PATH)

    if TABLE_PATH.exists():
        prev = pd.read_csv(TABLE_PATH)
        combined = pd.concat([prev, metrics_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["model", "task", "split"], keep="last"
        )
        combined.to_csv(TABLE_PATH, index=False)
    else:
        metrics_df.to_csv(TABLE_PATH, index=False)
    log.info("wrote %s (%d rows)", TABLE_PATH, len(metrics_df))

    print("\n========== RESULTS ==========")
    print(metrics_df.to_string(index=False))
    print("=============================\n")

    return summary


def run_chemberta(args: argparse.Namespace) -> dict:
    raise NotImplementedError(
        "ChemBERTa training arrives in Phase 1 Day 2 (after PAUSE POINT 1)."
    )


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
