"""Query-by-committee disagreement scoring across RF and ChemBERTa.

The 5-seed deep ensembles in `src/models/ensemble.py` give epistemic
uncertainty WITHIN each architecture (variance across seeds of the same
model class). They cannot capture the kind of uncertainty that comes from
**model-class disagreement**. A Random Forest on Morgan fingerprints and
a ChemBERTa-LoRA encoder will encode molecular similarity differently,
and where they disagree on a candidate's predicted toxicity, neither one
is necessarily right.

This module reads `results/candidates/all_scored.csv`, which already
contains both architectures' mean predictions side-by-side (the score
columns are emitted by `src/score_candidates.py` when `--architecture
both` is passed), computes disagreement scores per candidate per task,
and ranks candidates by total disagreement.

The output (`results/candidates/top_disagreement.csv`) is a different
top-K list from the composite-score top-20: it surfaces compounds where
the two architectures disagree most, which is exactly what an
active-learning loop should test next. If both models agree a compound
is good (or bad), there's nothing to learn from screening it. If they
disagree by 20 percentage points on toxicity, screening that compound
resolves the disagreement and constrains both models for the next batch.

Run:

    python -m src.qbc --top-k 20

This expects `all_scored.csv` to have been produced by:

    python -m src.score_candidates --architecture both --n-seeds 5 --top-k 20

The interaction term to optimize for in active learning is the per-task
disagreement; we report toxicity-disagreement, permeability-disagreement,
iri-disagreement, plus the L2 norm across the three. The L2 ranking is
the default 'next to test' list.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import RESULTS_DIR, get_logger

log = get_logger("qbc")

ALL_SCORED_PATH = RESULTS_DIR / "candidates" / "all_scored.csv"
DISAGREEMENT_PATH = RESULTS_DIR / "candidates" / "top_disagreement.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Query-by-committee: rank FDA candidates by RF/ChemBERTa disagreement"
    )
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--all-scored", type=Path, default=ALL_SCORED_PATH)
    p.add_argument("--output", type=Path, default=DISAGREEMENT_PATH)
    p.add_argument(
        "--standardize",
        action="store_true",
        default=True,
        help="z-score each task before computing disagreement so the three "
             "tasks contribute on a comparable scale (default: on)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all_scored.exists():
        raise FileNotFoundError(
            f"{args.all_scored} not found. Run "
            f"`python -m src.score_candidates --architecture both --n-seeds 5 "
            f"--top-k 20` first to populate per-architecture predictions."
        )

    df = pd.read_csv(args.all_scored)
    log.info("loaded %d scored candidates from %s", len(df), args.all_scored)

    # Expect both rf_* and chemberta_* prediction columns to exist when
    # --architecture both was used. If only one architecture's columns are
    # present, fail loudly.
    required = []
    for arch in ("rf", "chemberta"):
        for task in ("toxicity", "permeability", "iri"):
            required.append(f"{arch}_{task}_mean")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{args.all_scored} is missing per-architecture prediction columns "
            f"(missing: {missing[:6]}...). Re-run score_candidates with "
            f"--architecture both."
        )

    # Per-task disagreement = |rf_mean - chemberta_mean|.
    for task in ("toxicity", "permeability", "iri"):
        df[f"disagreement_{task}"] = (
            df[f"rf_{task}_mean"] - df[f"chemberta_{task}_mean"]
        ).abs()

    if args.standardize:
        # z-score the per-task disagreements so they're on a comparable scale
        # before combining. Toxicity is in mortality % (range ~30-65 in v3),
        # permeability is in P_CPA × 1e-3 (range ~14-30), IRI is in %MGS
        # (range ~30-80). Without standardization, the L2 score is dominated
        # by whichever task has the largest absolute spread (toxicity).
        for task in ("toxicity", "permeability", "iri"):
            v = df[f"disagreement_{task}"]
            mu = float(v.mean())
            sd = float(v.std()) or 1.0
            df[f"disagreement_{task}_z"] = (v - mu) / sd
        z_cols = [f"disagreement_{t}_z" for t in ("toxicity", "permeability", "iri")]
        df["disagreement_l2"] = np.sqrt((df[z_cols] ** 2).sum(axis=1))
    else:
        cols = [f"disagreement_{t}" for t in ("toxicity", "permeability", "iri")]
        df["disagreement_l2"] = np.sqrt((df[cols] ** 2).sum(axis=1))

    df = df.sort_values("disagreement_l2", ascending=False).reset_index(drop=True)
    df.insert(0, "disagreement_rank", range(1, len(df) + 1))
    log.info(
        "disagreement L2 stats: mean=%.2f, std=%.2f, max=%.2f",
        df["disagreement_l2"].mean(), df["disagreement_l2"].std(),
        df["disagreement_l2"].max(),
    )

    top = df.head(args.top_k).copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(args.output, index=False)
    log.info("wrote %s (%d rows)", args.output, len(top))

    # Pretty summary
    cols_to_show = [
        "disagreement_rank", "ingredient_name",
        "rf_toxicity_mean", "chemberta_toxicity_mean", "disagreement_toxicity",
        "rf_permeability_mean", "chemberta_permeability_mean", "disagreement_permeability",
        "rf_iri_mean", "chemberta_iri_mean", "disagreement_iri",
        "disagreement_l2",
    ]
    cols_to_show = [c for c in cols_to_show if c in top.columns]
    print(f"\nTop-{args.top_k} FDA candidates by RF/ChemBERTa disagreement (L2 over z-scored per-task gaps):\n")
    print(top[cols_to_show].round(2).to_string(index=False))
    print()
    print("These are the candidates an active-learning loop should test next:")
    print("the two architectures disagree most about them, so a single wet-lab")
    print("measurement maximally constrains both models.")


if __name__ == "__main__":
    main()
