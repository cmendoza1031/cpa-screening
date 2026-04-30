"""Tox21 auxiliary head weight sweep for ChemBERTa toxicity.

The v2 cluster-ensemble result with Tox21 aux head active (weight=0.1) was
ChemBERTa toxicity Spearman 0.181, vs v1's no-aux 0.217. The 0.04 drop is
within noise at n=50 (SE ~0.13), but the original hypothesis was the aux
head would HELP not hurt. This script settles the question by running a
single-seed 5-fold CV at six weights:

    {0.0, 0.05, 0.1, 0.2, 0.5, 1.0}

Single-seed (not 5-seed cluster ensemble) because we're hyperparameter-
sweeping; the within-sweep variance dominates seed-to-seed at n=50, so
adding 5 seeds doesn't change the conclusion shape and would 5x the
runtime. Final reporting in the README still uses the cluster-ensemble
numbers from the chosen weight.

Cost: 6 weights * 5 folds = 30 ChemBERTa trainings. ~10 min on Blackwell,
~20-30 min on T4/L4 (G4).

Output: writes results/sweeps/tox21_aux_sweep.csv with columns
{weight, task, n, mae, rmse, r2, spearman}, plus a summary print.

Usage:

    python -m src.sweep_tox21_aux

The Colab cell wraps this and tabulates the toxicity row across weights.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .utils import RESULTS_DIR, get_logger

log = get_logger("sweep_tox21_aux")

SWEEP_DIR = RESULTS_DIR / "sweeps"
OUTPUT_CSV = SWEEP_DIR / "tox21_aux_sweep.csv"
RESULTS_TABLE = RESULTS_DIR / "results_table.csv"
SUMMARY_JSON = RESULTS_DIR / "summary.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tox21 aux weight sweep")
    p.add_argument(
        "--weights", type=str,
        default="0.0,0.05,0.1,0.2,0.5,1.0",
        help="comma-separated list of aux weights to sweep",
    )
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    return p.parse_args()


def _run_one_weight(weight: float, args: argparse.Namespace) -> pd.DataFrame:
    """Run python -m src.train with the chosen aux weight; return the
    rows it appends to results_table.csv."""
    # snapshot the table so we can diff after this training run
    before_rows = pd.read_csv(RESULTS_TABLE) if RESULTS_TABLE.exists() else pd.DataFrame()
    n_before = len(before_rows)

    cmd = [
        sys.executable, "-m", "src.train",
        "--model", "chemberta",
        "--lora-rank", str(args.lora_rank),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--cv", "--cv-folds", str(args.cv_folds),
    ]
    if weight > 0.0:
        cmd += ["--tox21-aux", "--tox21-aux-weight", str(weight)]

    log.info("running aux_weight=%s: %s", weight, " ".join(cmd))
    subprocess.run(cmd, check=True)

    after_rows = pd.read_csv(RESULTS_TABLE)
    new_rows = after_rows.iloc[n_before:].copy()
    new_rows["aux_weight"] = weight
    return new_rows


def main() -> None:
    args = parse_args()
    weights = [float(w.strip()) for w in args.weights.split(",")]
    log.info("sweeping Tox21 aux weights: %s", weights)

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for w in weights:
        rows = _run_one_weight(w, args)
        all_runs.append(rows)

    df = pd.concat(all_runs, ignore_index=True)
    df.to_csv(OUTPUT_CSV, index=False)
    log.info("wrote %s (%d rows)", OUTPUT_CSV, len(df))

    # Pretty summary: toxicity Spearman by weight
    tox_rows = df[(df["task"] == "toxicity")].copy()
    if not tox_rows.empty:
        summary = tox_rows.groupby("aux_weight")[["spearman", "mae", "r2"]].mean().reset_index()
        summary = summary.sort_values("aux_weight")
        print("\n=== Tox21 aux weight sweep: ChemBERTa toxicity 5-fold CV OOF ===")
        print(summary.round(3).to_string(index=False))

        best_idx = summary["spearman"].idxmax()
        best = summary.loc[best_idx]
        print(
            f"\nBest aux_weight: {best['aux_weight']:.2f} "
            f"(toxicity Spearman = {best['spearman']:.3f})"
        )
        # Compare to no-aux baseline
        no_aux = summary[summary["aux_weight"] == 0.0]
        if not no_aux.empty:
            print(
                f"vs no-aux baseline ({no_aux.iloc[0]['spearman']:.3f}): "
                f"{best['spearman'] - no_aux.iloc[0]['spearman']:+.3f}"
            )


if __name__ == "__main__":
    main()
