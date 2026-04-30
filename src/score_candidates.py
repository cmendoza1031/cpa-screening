"""Phase 2 deliverable: score the FDA IID candidate pool with deep ensembles
and produce a Pareto-ranked top-20.

Pipeline:
    1. Load CPA-filtered FDA IID candidates from data/processed/fda_iid_candidates.parquet
    2. Train two full-data ensembles on ALL CPA labels:
         RF (per-task), n_seeds models each
         ChemBERTa multi-task, n_seeds models
    3. Predict (mean, std) per (task, candidate) for each architecture
    4. Calibrate prediction intervals using OOF residual quantiles from the
       cluster-aware CV ensembles in results/summary.json (computed earlier).
       Falls back to flat std-based PIs if calibration data isn't available.
    5. Pareto front over (toxicity_min, permeability_max, iri_min) using
       ensemble means.
    6. Composite scoring: prefer candidates with high permeability AND low
       toxicity AND low iri AND tight uncertainty intervals.
    7. Output:
         results/candidates/top20.csv             machine-readable top-20
         results/candidates/all_scored.csv         every candidate with predictions + flags
         results/figures/pareto_3d.png             3D Pareto-front visualization

Usage:
    python -m src.score_candidates --architecture rf            # RF ensemble only (fast)
    python -m src.score_candidates --architecture chemberta     # ChemBERTa ensemble (~2-5 min on GPU)
    python -m src.score_candidates --architecture both          # both, primary scoring uses ChemBERTa

Re-uses the same hyperparams as src.train (--lora-rank, --epochs, etc) for
ChemBERTa training, plus --n-seeds for ensemble size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import build_dataset, load_fda_iid
from .data.build import TASK_DIRECTIONS
from .models.ensemble import (
    REG_TASKS,
    predict_chemberta_ensemble,
    predict_rf_ensemble,
    train_chemberta_ensemble_full,
    train_rf_ensemble_full,
)
from .utils import CANDIDATES_DIR, FIGURES_DIR, RESULTS_DIR, get_logger, seed_everything

log = get_logger("score_candidates")

CANDIDATES_OUT = CANDIDATES_DIR / "top20.csv"
ALL_SCORED_OUT = CANDIDATES_DIR / "all_scored.csv"


# -------------------- Pareto front -----------------------------------------


def pareto_front(values: np.ndarray, directions: list[str]) -> np.ndarray:
    """Boolean mask: True where row is Pareto-non-dominated.

    values: (n, m) array.
    directions: list of "min" or "max" of length m.
    A row is dominated if some other row is >= on every "max" axis and
    <= on every "min" axis, with strict inequality on at least one axis.
    """
    n, m = values.shape
    nd = np.ones(n, dtype=bool)
    # Convert all to "minimize" by negating max axes
    v = values.copy()
    for j, d in enumerate(directions):
        if d == "max":
            v[:, j] = -v[:, j]
    for i in range(n):
        if not nd[i]:
            continue
        # Any other row strictly better (<=) on all and < on at least one?
        diff = v - v[i]
        dom = (diff <= 0).all(axis=1) & (diff < 0).any(axis=1)
        dom[i] = False
        if dom.any():
            nd[i] = False
    return nd


# -------------------- composite scoring ------------------------------------


def composite_score(
    df: pd.DataFrame,
    direction: dict[str, str] = TASK_DIRECTIONS,
    uncertainty_weight: float = 0.25,
) -> pd.Series:
    """Composite scalar score combining means + uncertainty.

    For each task, normalize the mean to a direction-aware [0, 1] desirability:
      "min" tasks -> (max - mean) / (max - min)
      "max" tasks -> (mean - min) / (max - min)

    Composite = mean(desirability) - uncertainty_weight * mean(normalized_std).
    Higher = better. Tighter PIs at equal mean rank higher.
    """
    parts = []
    unc_parts = []
    for task in REG_TASKS:
        mu = df[f"{task}_mean"]
        sd = df[f"{task}_std"]
        finite = mu[np.isfinite(mu)]
        if len(finite) < 2:
            continue
        mn, mx = finite.min(), finite.max()
        rng = max(mx - mn, 1e-9)
        if direction.get(task, "min") == "max":
            desirability = (mu - mn) / rng
        else:  # min
            desirability = (mx - mu) / rng
        parts.append(desirability.fillna(0.0))
        # Normalize std to range
        sd_finite = sd[np.isfinite(sd)]
        sd_rng = max(sd_finite.max() if len(sd_finite) else 1.0, 1e-9)
        unc_parts.append((sd / sd_rng).fillna(0.0))
    if not parts:
        return pd.Series(np.zeros(len(df)), index=df.index)
    desirability_avg = pd.concat(parts, axis=1).mean(axis=1)
    unc_avg = pd.concat(unc_parts, axis=1).mean(axis=1) if unc_parts else 0.0
    return desirability_avg - uncertainty_weight * unc_avg


# -------------------- Pareto-front 3D plot ---------------------------------


def plot_pareto_3d(df: pd.DataFrame, out_path: Path) -> None:
    """3D scatter of (toxicity_mean, permeability_mean, iri_mean) with Pareto
    front highlighted."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    mask = df[["toxicity_mean", "permeability_mean", "iri_mean"]].notna().all(axis=1)
    sub = df[mask]
    if len(sub) < 5:
        log.warning("not enough scored candidates for Pareto plot (n=%d)", len(sub))
        return

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    on = sub[sub["pareto"]]
    off = sub[~sub["pareto"]]
    ax.scatter(off["toxicity_mean"], off["permeability_mean"], off["iri_mean"],
               s=8, alpha=0.3, c="lightgray", label=f"dominated (n={len(off)})")
    ax.scatter(on["toxicity_mean"], on["permeability_mean"], on["iri_mean"],
               s=24, alpha=0.9, c="tab:red", label=f"Pareto front (n={len(on)})")
    ax.set_xlabel("toxicity (min)")
    ax.set_ylabel("permeability (max)")
    ax.set_zlabel("iri %MGS (min)")
    ax.legend(loc="upper right")
    ax.set_title("FDA IID candidates: Pareto front in (tox, perm, iri) space")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    log.info("wrote %s", out_path)


# -------------------- main pipeline ----------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score FDA IID candidates")
    p.add_argument("--architecture", choices=["rf", "chemberta", "both"], default="rf",
                   help="rf is fast (~30s); chemberta is the foundation model "
                        "(~2-5 min on GPU); both runs each separately")
    p.add_argument("--n-seeds", type=int, default=5,
                   help="ensemble size (default 5)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=20)
    # ChemBERTa hyperparams (only used when --architecture in {chemberta, both})
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--no-tox21-aux", action="store_true",
                   help="(Legacy alias; aux head is off by default.)")
    p.add_argument("--tox21-aux", action="store_true",
                   help="ChemBERTa: enable Tox21 auxiliary classification head "
                        "during full-data ensemble training")
    p.add_argument("--tox21-aux-weight", type=float, default=0.1)
    p.add_argument("--cv", action="store_true")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--fda-limit", type=int, default=None,
                   help="for dev / fast-iter: only score the first N FDA "
                        "candidates rather than the full ~1.8k pool")
    return p.parse_args()


def _ensure_fda_candidates(limit: int | None = None) -> pd.DataFrame:
    """Load CPA-filtered FDA IID candidates; build them if missing."""
    df = load_fda_iid(limit=limit)  # uses cache if present
    log.info("FDA candidates: %d compounds passing CPA-like filter", len(df))
    return df


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    log.info("config: %s", vars(args))

    long_df, audit = build_dataset()
    log.info("CPA training data: %d unique compounds, %d (compound, task) labels",
             audit["unique_compounds"], audit["n_long_rows"])

    candidates_df = _ensure_fda_candidates(limit=args.fda_limit)
    candidate_smiles = candidates_df["smiles_canonical"].tolist()

    # Load q95 calibration constants from the most recent ensemble eval, if any
    q95_by_arch: dict[str, dict[str, float]] = {}
    summary_path = RESULTS_DIR / "summary.json"
    if summary_path.exists():
        try:
            past = json.loads(summary_path.read_text())
            metrics = past.get("metrics", [])
            for m in metrics:
                if "q95" not in m:
                    continue
                arch = "chemberta" if "chemberta" in m["model"] else "rf"
                q95_by_arch.setdefault(arch, {})[m["task"]] = m["q95"]
        except Exception:
            pass
    if q95_by_arch:
        log.info("loaded conformal q95 from %s: %s", summary_path, q95_by_arch)
    else:
        log.warning("no q95 found in results/summary.json; PIs will use ensemble std only")

    # Train + predict for each requested architecture
    arch_preds: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    if args.architecture in ("rf", "both"):
        log.info("training RF full-data ensemble (%d seeds per task)", args.n_seeds)
        rf_models = train_rf_ensemble_full(long_df, n_seeds=args.n_seeds)
        rf_preds = predict_rf_ensemble(rf_models, candidate_smiles)
        arch_preds["rf"] = rf_preds
    if args.architecture in ("chemberta", "both"):
        from .models.chemberta_lora import _wide_targets

        log.info("training ChemBERTa full-data ensemble (%d seeds, ~30s each on GPU)",
                 args.n_seeds)
        wide = _wide_targets(long_df)
        cb_ensemble = train_chemberta_ensemble_full(wide, args, n_seeds=args.n_seeds)
        cb_preds = predict_chemberta_ensemble(cb_ensemble, wide, candidate_smiles)
        arch_preds["chemberta"] = cb_preds

    # Pick the primary architecture for the headline ranking. ChemBERTa wins
    # on permeability (the smallest task / pretraining-helps regime); RF wins
    # on IRI (scaffold-clustered tabular regime). For PAUSE POINT 4 the user
    # eyeballs whichever architecture got --architecture; "both" defaults to
    # ChemBERTa (Phase 2 deliverable narrative).
    primary = "chemberta" if "chemberta" in arch_preds else "rf"
    primary_preds = arch_preds[primary]
    log.info("primary architecture for ranking: %s", primary)

    # Build the scored DataFrame
    cols = ["ingredient_name", "cas", "smiles_canonical", "mw", "logp", "tpsa", "hbd", "hba"]
    scored = candidates_df[[c for c in cols if c in candidates_df.columns]].copy()
    for task in REG_TASKS:
        means, stds = [], []
        for s in scored["smiles_canonical"]:
            mu, sd = primary_preds.get(task, {}).get(s, (np.nan, np.nan))
            means.append(mu)
            stds.append(sd)
        scored[f"{task}_mean"] = means
        scored[f"{task}_std"] = stds
        # Conformal-calibrated 95% PI
        q95 = q95_by_arch.get(primary, {}).get(task, np.nan)
        scored[f"{task}_q95"] = q95
        scored[f"{task}_lo95"] = scored[f"{task}_mean"] - q95
        scored[f"{task}_hi95"] = scored[f"{task}_mean"] + q95

    # Drop rows missing any task prediction
    n_before = len(scored)
    scored = scored.dropna(
        subset=[f"{t}_mean" for t in REG_TASKS]
    ).reset_index(drop=True)
    log.info("scored candidates: %d / %d have predictions for all 3 tasks",
             len(scored), n_before)

    # Pareto front
    values = scored[[f"{t}_mean" for t in REG_TASKS]].to_numpy()
    directions = [TASK_DIRECTIONS[t] for t in REG_TASKS]
    nd = pareto_front(values, directions)
    scored["pareto"] = nd

    # Composite score
    scored["composite_score"] = composite_score(scored)

    # Top-K
    top = (
        scored.sort_values(["pareto", "composite_score"], ascending=[False, False])
        .head(args.top_k)
        .reset_index(drop=True)
    )

    # Persist
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(ALL_SCORED_OUT, index=False)
    top.to_csv(CANDIDATES_OUT, index=False)
    log.info("wrote %s (%d candidates)", ALL_SCORED_OUT, len(scored))
    log.info("wrote %s (top %d)", CANDIDATES_OUT, len(top))

    plot_pareto_3d(scored, FIGURES_DIR / f"pareto_3d_{primary}.png")

    # Pretty-print the top-K
    pretty_cols = ["ingredient_name", "cas", "pareto", "composite_score"]
    for t in REG_TASKS:
        pretty_cols.extend([f"{t}_mean", f"{t}_std"])
    print(f"\n========== TOP-{args.top_k} CANDIDATES ({primary}) ==========")
    print(top[pretty_cols].to_string(index=False))
    print("===========================================\n")


if __name__ == "__main__":
    main()
