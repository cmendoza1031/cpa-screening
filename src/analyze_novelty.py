"""Top-K candidate ranking restricted to compounds NOT in the training set.

Most of the v2 single-compound top-20 is amino acids that appear in the
DOLMEN training set (phenylalanine, tryptophan, histidine, arginine,
valine, isoleucine). The model's score on those is partially memorization,
not generalization. For a wet-lab reviewer the more interesting question
is: which compounds does the model rank highly that it has NEVER seen
in training? Those are the genuine virtual-screen recommendations.

This script:
1. Loads `results/candidates/all_scored.csv` (the full ranked FDA pool)
2. Loads the training SMILES from `data/processed/long.parquet`
3. Filters out any candidate whose canonical SMILES appears in training
4. Re-ranks the survivors by composite score
5. Writes `results/candidates/top20_novel.csv`

Run it after `python -m src.score_candidates` has populated all_scored.csv:

    python -m src.analyze_novelty --top-k 20

The novel-only top-20 is the right list to actually send to the wet lab,
because the in-training compounds are model self-consistency, not
discoveries. Both lists are useful for different purposes; we keep both.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import build_dataset
from .utils import RESULTS_DIR, get_logger

log = get_logger("analyze_novelty")

ALL_SCORED_PATH = RESULTS_DIR / "candidates" / "all_scored.csv"
NOVEL_OUTPUT_PATH = RESULTS_DIR / "candidates" / "top20_novel.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter FDA candidates to novel-only")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--all-scored", type=Path, default=ALL_SCORED_PATH,
                   help="path to all_scored.csv produced by score_candidates")
    p.add_argument("--output", type=Path, default=NOVEL_OUTPUT_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.all_scored.exists():
        raise FileNotFoundError(
            f"{args.all_scored} not found. Run `python -m src.score_candidates "
            f"--architecture both --n-seeds 5 --top-k 20` first."
        )

    log.info("loading scored candidates from %s", args.all_scored)
    scored = pd.read_csv(args.all_scored)
    log.info("scored candidates: %d", len(scored))

    log.info("loading training SMILES from build_dataset")
    long_df, _ = build_dataset()
    train_smiles = set(long_df["smiles_canonical"].dropna().unique())
    log.info("unique training SMILES across all tasks: %d", len(train_smiles))

    overlap_mask = scored["smiles_canonical"].isin(train_smiles)
    n_overlap = int(overlap_mask.sum())
    log.info(
        "candidate overlap with training set: %d / %d (%.1f%%)",
        n_overlap, len(scored), 100.0 * n_overlap / max(len(scored), 1),
    )

    novel = scored[~overlap_mask].copy()
    if "composite_score" in novel.columns:
        novel = novel.sort_values("composite_score", ascending=False)
    novel = novel.reset_index(drop=True)
    novel.insert(0, "novel_rank", range(1, len(novel) + 1))
    log.info("novel candidates after filtering: %d", len(novel))

    top = novel.head(args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(args.output, index=False)
    log.info("wrote %s (%d rows)", args.output, len(top))

    # Print a compact summary
    display_cols = [
        c for c in [
            "novel_rank", "ingredient_name", "cas",
            "toxicity_mean", "toxicity_std",
            "permeability_mean", "permeability_std",
            "iri_mean", "iri_std",
            "composite_score",
        ] if c in top.columns
    ]
    print(f"\nTop-{args.top_k} novel FDA candidates (NOT in DOLMEN/Higgins training):\n")
    print(top[display_cols].round(3).to_string(index=False))
    print()

    # Also tell the user which top-20 entries from the original scoring were
    # filtered out as in-training. Useful for the README narrative.
    if "composite_score" in scored.columns:
        original_top20 = scored.sort_values("composite_score", ascending=False).head(20)
        in_train = original_top20[original_top20["smiles_canonical"].isin(train_smiles)]
        if not in_train.empty:
            print(f"Compounds dropped from the original top-20 because they were in training:")
            for _, r in in_train.iterrows():
                print(f"  - {r.get('ingredient_name', r['smiles_canonical'])}")


if __name__ == "__main__":
    main()
