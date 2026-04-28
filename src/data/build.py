"""Consolidation: load all sources -> long-format dataset + audit report.

Long format columns:
    smiles_canonical, task, value, source

Tasks:
    toxicity     -- mortality at 4 C, defined as (100 - viability_4c).
                    Lower is better. Sources: higgins_jan2025, higgins_dec2025.
    permeability -- effective CPA permeability at 4 C (raw paper units).
                    Higher is better. Source: higgins_jan2025.
    iri          -- ice recrystallization inhibition, %MGS from splat assay.
                    Lower is better. Source: dolmen_amino, dolmen_glyco2.

Direction is documented here so train.py / score_candidates.py can know
which way to optimize each axis at Pareto time.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

from ..utils import PROCESSED_DIR, get_logger
from .dolmen import load_dolmen
from .higgins import load_higgins_dec2025, load_higgins_jan2025
from .pubchem import cache_stats

log = get_logger("data.build")

LONG_PATH = PROCESSED_DIR / "long.parquet"
AUDIT_PATH = PROCESSED_DIR / "audit.json"

# Optimization direction per task: "min" or "max".
TASK_DIRECTIONS = {
    "toxicity": "min",
    "permeability": "max",
    "iri": "min",
}


def _smiles_len_distribution(smiles: pd.Series) -> dict:
    if smiles.empty:
        return {"n": 0}
    lens = smiles.str.len()
    return {
        "n": int(lens.shape[0]),
        "min": int(lens.min()),
        "p25": float(lens.quantile(0.25)),
        "median": float(lens.median()),
        "p75": float(lens.quantile(0.75)),
        "max": int(lens.max()),
        "mean": float(lens.mean()),
    }


def build_dataset(
    force_reprocess: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Load all sources, emit a long-format dataframe + audit dict.

    Returns (long_df, audit_dict). Also writes data/processed/long.parquet
    and data/processed/audit.json.
    """
    if LONG_PATH.exists() and AUDIT_PATH.exists() and not force_reprocess:
        log.info("loading cached long dataset from %s", LONG_PATH)
        long_df = pd.read_parquet(LONG_PATH)
        audit = json.loads(AUDIT_PATH.read_text())
        return long_df, audit

    rows: list[pd.DataFrame] = []
    n_dropped: dict[str, int] = {}

    # DOLMEN -> IRI
    dolmen = load_dolmen()
    if not dolmen.empty:
        d = dolmen[["smiles_canonical", "mgs_percent", "source"]].copy()
        d = d.rename(columns={"mgs_percent": "value"})
        d["task"] = "iri"
        d_drop = d["value"].isna().sum()
        n_dropped["iri/dolmen"] = int(d_drop)
        rows.append(d.dropna(subset=["value"])[["smiles_canonical", "task", "value", "source"]])

    # Higgins Jan 2025 -> toxicity (4 C) + permeability (4 C)
    jan = load_higgins_jan2025()
    if not jan.empty:
        if "viability_4c" in jan.columns:
            tox = jan[["smiles_canonical", "viability_4c", "source"]].copy()
            tox["value"] = 100.0 - pd.to_numeric(tox["viability_4c"], errors="coerce")
            tox["task"] = "toxicity"
            n_dropped["toxicity/higgins_jan2025"] = int(tox["value"].isna().sum())
            rows.append(
                tox.dropna(subset=["value"])[["smiles_canonical", "task", "value", "source"]]
            )
        if "permeability_4c" in jan.columns:
            perm = jan[["smiles_canonical", "permeability_4c", "source"]].copy()
            perm["value"] = pd.to_numeric(perm["permeability_4c"], errors="coerce")
            perm["task"] = "permeability"
            n_dropped["permeability/higgins_jan2025"] = int(perm["value"].isna().sum())
            rows.append(
                perm.dropna(subset=["value"])[
                    ["smiles_canonical", "task", "value", "source"]
                ]
            )

    # Higgins Dec 2025 -> toxicity (4 C). Multiple concentration rows per
    # compound; we average within (smiles, source) for the regression target.
    dec = load_higgins_dec2025()
    if not dec.empty and "viability_4c" in dec.columns:
        tox = dec[["smiles_canonical", "viability_4c", "source"]].copy()
        tox["viability_4c"] = pd.to_numeric(tox["viability_4c"], errors="coerce")
        tox = tox.dropna(subset=["viability_4c"])
        if not tox.empty:
            agg = tox.groupby(["smiles_canonical", "source"], as_index=False)[
                "viability_4c"
            ].mean()
            agg["value"] = 100.0 - agg["viability_4c"]
            agg["task"] = "toxicity"
            rows.append(agg[["smiles_canonical", "task", "value", "source"]])

    if not rows:
        long_df = pd.DataFrame(columns=["smiles_canonical", "task", "value", "source"])
    else:
        long_df = pd.concat(rows, ignore_index=True)

    long_df = long_df.dropna(subset=["smiles_canonical", "value"]).copy()

    # Within (smiles, task), average across sources to get one label per pair.
    # Source column collapses to a comma-joined string for traceability.
    if not long_df.empty:
        agg = (
            long_df.groupby(["smiles_canonical", "task"], as_index=False)
            .agg(value=("value", "mean"), source=("source", lambda s: ",".join(sorted(set(s)))))
        )
        long_df = agg

    long_df.to_parquet(LONG_PATH, index=False)

    audit = _audit(long_df)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, default=str))
    log.info("audit written to %s", AUDIT_PATH)
    _print_audit(audit)
    return long_df, audit


def _audit(long_df: pd.DataFrame) -> dict:
    a: dict = {}
    a["pubchem_cache"] = cache_stats()
    a["task_directions"] = TASK_DIRECTIONS
    a["n_long_rows"] = int(len(long_df))
    if long_df.empty:
        a["unique_compounds"] = 0
        a["per_task"] = {}
        a["smiles_length"] = {}
        return a
    a["unique_compounds"] = int(long_df["smiles_canonical"].nunique())
    a["smiles_length"] = _smiles_len_distribution(
        long_df.drop_duplicates("smiles_canonical")["smiles_canonical"]
    )
    per_task = {}
    for task, sub in long_df.groupby("task"):
        per_task[task] = {
            "n": int(len(sub)),
            "n_unique_compounds": int(sub["smiles_canonical"].nunique()),
            "value_min": float(sub["value"].min()),
            "value_max": float(sub["value"].max()),
            "value_mean": float(sub["value"].mean()),
            "value_std": float(sub["value"].std(ddof=0)) if len(sub) > 1 else 0.0,
            "sources": sorted(set(",".join(sub["source"]).split(","))) if len(sub) else [],
        }
    a["per_task"] = per_task
    return a


def _print_audit(a: dict) -> None:
    print("\n========== DATA AUDIT ==========")
    print(f"Unique compounds: {a.get('unique_compounds', 0)}")
    print(f"Total (compound, task) labels: {a.get('n_long_rows', 0)}")
    sl = a.get("smiles_length", {})
    if sl.get("n", 0):
        print(
            f"SMILES length: n={sl['n']} median={sl['median']:.0f} "
            f"min={sl['min']} max={sl['max']}"
        )
    print(f"PubChem cache: {a.get('pubchem_cache', {})}")
    print("\nPer-task summary:")
    for task, stats in a.get("per_task", {}).items():
        direction = TASK_DIRECTIONS.get(task, "?")
        print(
            f"  {task:14s} ({direction})  n={stats['n']:5d} "
            f"compounds={stats['n_unique_compounds']:5d}  "
            f"value=[{stats['value_min']:.3g}, {stats['value_max']:.3g}] "
            f"mean={stats['value_mean']:.3g} std={stats['value_std']:.3g}"
        )
        print(f"    sources: {stats['sources']}")
    print("================================\n")


if __name__ == "__main__":
    long_df, audit = build_dataset(force_reprocess=True)
    print(f"\nlong_df shape: {long_df.shape}")
