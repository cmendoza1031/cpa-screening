"""Higgins lab CPA toxicity + permeability datasets.

Two papers, both small-N hand-curated data:

Jan 2025 (Sci Rep, DOI: 10.1038/s41598-025-85509-x)
    27 compounds, viability + permeability at 4 C and 25 C.
    Numeric values live in the paper's Tables 1 / 2 / Fig 6 and the SI PDF.
    Not machine-readable from the publisher; we expect a hand-transcribed CSV.

Dec 2025 (Cryobiology, DOI: 10.1016/j.cryobiol.2025.105315)
    22 single CPAs + binary mixtures, viability at 4 C up to 12 mol/kg.
    Single-compound rows used for v1 training. Mixture rows are flagged
    is_mixture=True and dropped from v1, written separately for reference.

Behavior:
    * If the expected CSV is present at data/raw/ , parse it.
    * If missing, write a template CSV with the right columns, print clear
      instructions pointing at the source paper / Table / SI PDF, and exit
      cleanly (returns an empty DataFrame). The pipeline keeps going so you
      can train DOLMEN-only first, then drop in Higgins data later.

Expected CSV schemas (case-insensitive on column names):

    higgins_jan2025.csv
        compound_name        free-text name as it appears in Table 1
        smiles               (optional; if blank we look up via PubChem)
        permeability_4c      P_CPA at 4 C, units cm/s x 1e-7 (or whatever
                             unit is used in the paper -- be consistent)
        permeability_25c     same at 25 C
        viability_4c         viability percent at 4 C (0-100)
        viability_25c        viability percent at 25 C (0-100)

    higgins_dec2025.csv
        compound_name        primary compound
        compound_name_2      blank for singles, second compound for mixtures
        smiles               (optional; looked up if blank)
        smiles_2             (optional; only for mixtures)
        concentration_mol_kg
        viability_4c         viability percent at 4 C (0-100)
        is_mixture           True / False (auto-set if compound_name_2 is non-empty)

For Phase 1 v1 training we only consume single-compound rows (is_mixture==False)
and we standardize on Higgins toxicity = viability_4c (since 4 C is the
condition Until cares about for organ CPA equilibration).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..utils import PROCESSED_DIR, RAW_DIR, canonical_smiles, get_logger
from .pubchem import name_or_cas_to_smiles

log = get_logger("data.higgins")

JAN_RAW_PATH = RAW_DIR / "higgins_jan2025.csv"
DEC_RAW_PATH = RAW_DIR / "higgins_dec2025.csv"
JAN_PROCESSED = PROCESSED_DIR / "higgins_jan2025.parquet"
DEC_PROCESSED = PROCESSED_DIR / "higgins_dec2025.parquet"
DEC_MIXTURES = PROCESSED_DIR / "higgins_mixtures.parquet"

JAN_COLUMNS = [
    "compound_name",
    "smiles",
    "permeability_4c",
    "permeability_25c",
    "viability_4c",
    "viability_25c",
]
DEC_COLUMNS = [
    "compound_name",
    "compound_name_2",
    "smiles",
    "smiles_2",
    "concentration_mol_kg",
    "viability_4c",
    "is_mixture",
]

# Compound names extracted from the Jan 2025 Materials section so the user
# has a complete list to transcribe from. Not used as data, just as a hint.
JAN2025_COMPOUNDS = [
    "1,3-propanediol", "2-methoxyethanol", "diethylene glycol", "2,3-butanediol",
    "2-methyl-1,3-propanediol", "N-methylacetamide", "acetamide",
    "1,3-dihydroxyacetone", "glycerol", "DMSO", "propylene glycol",
    "ethylene glycol", "formamide", "triethylene glycol", "propionamide",
    "2-methyl-2,4-pentanediol", "diglyme", "tetraethylene glycol dimethyl ether",
    "dimethylacetamide", "triglyme", "triethanolamine", "ethanolamine", "pyridine",
    "N,N-dimethylethanolamine", "diethylene glycol monobutyl ether",
    "tetrahydrofurfuryl alcohol", "sucrose",
]


def _write_jan_template() -> None:
    rows = [{c: "" for c in JAN_COLUMNS} for _ in JAN2025_COMPOUNDS]
    for r, name in zip(rows, JAN2025_COMPOUNDS):
        r["compound_name"] = name
    pd.DataFrame(rows, columns=JAN_COLUMNS).to_csv(JAN_RAW_PATH, index=False)


def _write_dec_template() -> None:
    pd.DataFrame(columns=DEC_COLUMNS).to_csv(DEC_RAW_PATH, index=False)


def _resolve_smiles(name: Optional[str], existing: Optional[str]) -> Optional[str]:
    if existing and isinstance(existing, str) and existing.strip():
        canon = canonical_smiles(existing)
        if canon:
            return canon
        log.warning("higgins: provided SMILES for %s is invalid: %r", name, existing)
    if not name:
        return None
    raw = name_or_cas_to_smiles(name, kind="name")
    return canonical_smiles(raw) if raw else None


def load_higgins_jan2025(force_reprocess: bool = False) -> pd.DataFrame:
    """Single dataframe with one row per Higgins Jan 2025 compound.

    Columns: compound_name, smiles_canonical, permeability_4c,
    permeability_25c, viability_4c, viability_25c, source.
    """
    if JAN_PROCESSED.exists() and not force_reprocess:
        return pd.read_parquet(JAN_PROCESSED)

    if not JAN_RAW_PATH.exists():
        _write_jan_template()
        msg = f"""
[higgins_jan2025] No data file found.

A template has been written to:
    {JAN_RAW_PATH}

Please open it and fill in numeric values from:
    Paper: https://www.nature.com/articles/s41598-025-85509-x
    Tables 1 + 2 (permeability), Figure 6 (viability), SI PDF for the rest.

Required columns (leave SMILES blank if you want PubChem lookup by name):
    {JAN_COLUMNS}

Skipping Higgins Jan 2025 for now -- the rest of the pipeline will run
without it. Re-run after filling in the CSV.
"""
        log.warning(msg)
        return pd.DataFrame(
            columns=[
                "compound_name",
                "smiles_canonical",
                "permeability_4c",
                "permeability_25c",
                "viability_4c",
                "viability_25c",
                "source",
            ]
        )

    df = pd.read_csv(JAN_RAW_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in JAN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"higgins_jan2025.csv is missing required columns: {missing}; "
            f"found {df.columns.tolist()}"
        )

    df["smiles_canonical"] = [
        _resolve_smiles(n, s) for n, s in zip(df["compound_name"], df["smiles"])
    ]
    n_pre = len(df)
    df = df.dropna(subset=["smiles_canonical"]).copy()
    log.info(
        "higgins_jan2025: %d rows in CSV, %d resolved to canonical SMILES",
        n_pre,
        len(df),
    )

    for c in ["permeability_4c", "permeability_25c", "viability_4c", "viability_25c"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["source"] = "higgins_jan2025"
    out = df[
        [
            "compound_name",
            "smiles_canonical",
            "permeability_4c",
            "permeability_25c",
            "viability_4c",
            "viability_25c",
            "source",
        ]
    ]
    out.to_parquet(JAN_PROCESSED, index=False)
    return out


def load_higgins_dec2025(force_reprocess: bool = False) -> pd.DataFrame:
    """Higgins Dec 2025 single-compound rows. Mixtures saved separately.

    Returns DataFrame with: compound_name, smiles_canonical,
    concentration_mol_kg, viability_4c, source. Only is_mixture==False rows.
    """
    if DEC_PROCESSED.exists() and not force_reprocess:
        return pd.read_parquet(DEC_PROCESSED)

    if not DEC_RAW_PATH.exists():
        _write_dec_template()
        msg = f"""
[higgins_dec2025] No data file found.

A template has been written to:
    {DEC_RAW_PATH}

Please fill in from:
    Preprint:  https://www.biorxiv.org/content/10.1101/2025.05.07.652719v1
    Published: https://doi.org/10.1016/j.cryobiol.2025.105315

Required columns: {DEC_COLUMNS}
Set is_mixture=True for binary mixture rows; v1 training drops these but
saves them to {DEC_MIXTURES} for documentation.

Skipping Higgins Dec 2025 for now.
"""
        log.warning(msg)
        return pd.DataFrame(
            columns=[
                "compound_name",
                "smiles_canonical",
                "concentration_mol_kg",
                "viability_4c",
                "source",
            ]
        )

    df = pd.read_csv(DEC_RAW_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in DEC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"higgins_dec2025.csv missing columns: {missing}; found {df.columns.tolist()}"
        )

    df["compound_name_2"] = df["compound_name_2"].fillna("").astype(str)
    df["is_mixture"] = df["is_mixture"].fillna(False).astype(bool) | (
        df["compound_name_2"].str.strip() != ""
    )

    mixtures = df[df["is_mixture"]].copy()
    singles = df[~df["is_mixture"]].copy()

    if not mixtures.empty:
        mixtures["smiles_canonical"] = [
            _resolve_smiles(n, s)
            for n, s in zip(mixtures["compound_name"], mixtures["smiles"])
        ]
        mixtures["smiles_canonical_2"] = [
            _resolve_smiles(n, s)
            for n, s in zip(mixtures["compound_name_2"], mixtures["smiles_2"])
        ]
        mixtures.to_parquet(DEC_MIXTURES, index=False)
        log.info(
            "higgins_dec2025: %d mixture rows saved to %s (excluded from v1 training)",
            len(mixtures),
            DEC_MIXTURES,
        )

    singles["smiles_canonical"] = [
        _resolve_smiles(n, s) for n, s in zip(singles["compound_name"], singles["smiles"])
    ]
    n_pre = len(singles)
    singles = singles.dropna(subset=["smiles_canonical"]).copy()
    log.info(
        "higgins_dec2025: %d single-compound rows in CSV, %d resolved to SMILES",
        n_pre,
        len(singles),
    )

    singles["concentration_mol_kg"] = pd.to_numeric(
        singles["concentration_mol_kg"], errors="coerce"
    )
    singles["viability_4c"] = pd.to_numeric(singles["viability_4c"], errors="coerce")

    singles["source"] = "higgins_dec2025"
    out = singles[
        [
            "compound_name",
            "smiles_canonical",
            "concentration_mol_kg",
            "viability_4c",
            "source",
        ]
    ]
    out.to_parquet(DEC_PROCESSED, index=False)
    return out


if __name__ == "__main__":
    jan = load_higgins_jan2025()
    dec = load_higgins_dec2025()
    print(f"Jan 2025: {len(jan)} compounds")
    print(f"Dec 2025: {len(dec)} single-compound rows")
