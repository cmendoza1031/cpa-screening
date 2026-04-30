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

    Stale or empty caches are auto-invalidated. Specifically: if the cached
    long.parquet is empty, or the cached audit lacks the 'iri' task that
    DOLMEN should provide, we rebuild from sources. This guards against
    poisoned caches from a previous failed run.
    """
    if LONG_PATH.exists() and AUDIT_PATH.exists() and not force_reprocess:
        long_df = pd.read_parquet(LONG_PATH)
        audit = json.loads(AUDIT_PATH.read_text())
        cache_is_stale = (
            len(long_df) == 0
            or "iri" not in audit.get("per_task", {})
            or "concentration_mol_kg" not in long_df.columns
        )
        if cache_is_stale:
            log.warning(
                "cached long dataset at %s looks stale "
                "(rows=%d, tasks=%s); rebuilding from sources",
                LONG_PATH,
                len(long_df),
                list(audit.get("per_task", {}).keys()),
            )
        else:
            log.info("loading cached long dataset from %s (%d rows)", LONG_PATH, len(long_df))
            return long_df, audit

    rows: list[pd.DataFrame] = []
    n_dropped: dict[str, int] = {}

    # Default per-task assay concentrations (mol/kg). For IRI we use the
    # 20-22 mM splat assay concentration converted to mol/kg (water is
    # ~1 kg/L for dilute solutions, so 20 mmol/kg ~= 0.020 mol/kg). For
    # permeability the Higgins Jan 2025 assay uses ~1 osmol/kg total which
    # is roughly 0.5 mol/kg of CPA depending on van't Hoff factor. These
    # are placeholders; the values matter for the toxicity head only,
    # because toxicity is the only task with concentration variation in
    # training.
    DOLMEN_CONC_MOL_KG = 0.020
    HIGGINS_JAN_CONC_MOL_KG = 0.5

    # DOLMEN -> IRI
    dolmen = load_dolmen()
    if not dolmen.empty:
        d = dolmen[["smiles_canonical", "mgs_percent", "source"]].copy()
        d = d.rename(columns={"mgs_percent": "value"})
        d["task"] = "iri"
        d["concentration_mol_kg"] = DOLMEN_CONC_MOL_KG
        n_dropped["iri/dolmen"] = int(d["value"].isna().sum())
        rows.append(d.dropna(subset=["value"])[
            ["smiles_canonical", "task", "value", "concentration_mol_kg", "source"]
        ])

    # Higgins Jan 2025 -> permeability (4 C). Viability left NaN (see README).
    jan = load_higgins_jan2025()
    if not jan.empty:
        if "permeability_4c" in jan.columns:
            perm = jan[["smiles_canonical", "permeability_4c", "source"]].copy()
            perm["value"] = pd.to_numeric(perm["permeability_4c"], errors="coerce")
            perm["task"] = "permeability"
            perm["concentration_mol_kg"] = HIGGINS_JAN_CONC_MOL_KG
            n_dropped["permeability/higgins_jan2025"] = int(perm["value"].isna().sum())
            rows.append(perm.dropna(subset=["value"])[
                ["smiles_canonical", "task", "value", "concentration_mol_kg", "source"]
            ])

    # Higgins Dec 2025 -> toxicity (4 C). Each (compound, concentration)
    # measurement is its own row; concentration is the third axis the
    # toxicity head will use as an input feature.
    dec = load_higgins_dec2025()
    if not dec.empty and "viability_4c" in dec.columns:
        tox = dec[
            ["smiles_canonical", "viability_4c", "concentration_mol_kg", "source"]
        ].copy()
        tox["viability_4c"] = pd.to_numeric(tox["viability_4c"], errors="coerce")
        tox["concentration_mol_kg"] = pd.to_numeric(
            tox["concentration_mol_kg"], errors="coerce"
        )
        tox = tox.dropna(subset=["viability_4c", "concentration_mol_kg"])
        if not tox.empty:
            tox["value"] = 100.0 - tox["viability_4c"]
            tox["task"] = "toxicity"
            rows.append(tox[
                ["smiles_canonical", "task", "value", "concentration_mol_kg", "source"]
            ])

    if not rows:
        long_df = pd.DataFrame(columns=[
            "smiles_canonical", "task", "value", "concentration_mol_kg", "source"
        ])
    else:
        long_df = pd.concat(rows, ignore_index=True)

    long_df = long_df.dropna(subset=["smiles_canonical", "value"]).copy()

    # For toxicity we keep one row per (smiles, concentration) measurement
    # so the model sees dose-response. For other tasks we still collapse
    # duplicates across sources by mean.
    if not long_df.empty:
        non_tox = long_df[long_df["task"] != "toxicity"]
        tox_rows = long_df[long_df["task"] == "toxicity"]
        if not non_tox.empty:
            non_tox_agg = (
                non_tox.groupby(
                    ["smiles_canonical", "task", "concentration_mol_kg"],
                    as_index=False,
                )
                .agg(
                    value=("value", "mean"),
                    source=("source", lambda s: ",".join(sorted(set(s)))),
                )
            )
        else:
            non_tox_agg = non_tox
        # Toxicity: average across same (smiles, concentration) if duplicates;
        # this preserves dose-response across the 3 / 6 / 12 mol/kg axis.
        if not tox_rows.empty:
            tox_agg = (
                tox_rows.groupby(
                    ["smiles_canonical", "task", "concentration_mol_kg"],
                    as_index=False,
                )
                .agg(
                    value=("value", "mean"),
                    source=("source", lambda s: ",".join(sorted(set(s)))),
                )
            )
        else:
            tox_agg = tox_rows
        long_df = pd.concat([non_tox_agg, tox_agg], ignore_index=True)

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
