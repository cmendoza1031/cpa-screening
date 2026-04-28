"""DOLMEN ice recrystallization inhibition dataset.

Source: https://github.com/gcsosso/DOLMEN
Paper:  Warren et al., Nat Commun (2024). DOI: 10.1038/s41467-024-52266-w

We use the Amino (63) + Glyco2 (223) datasets = 286 compounds, all already
SMILES-annotated. We download raw CSVs directly rather than git-cloning
(simpler, no subprocess dependency, files are small).

Columns in the upstream CSVs:
    Name, SMILES, Concentration (mM), Exp. % MGS, Exp. error, Set

Lower %MGS = stronger ice recrystallization inhibition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from ..utils import PROCESSED_DIR, RAW_DIR, canonical_smiles, get_logger

log = get_logger("data.dolmen")

DOLMEN_BASE = "https://raw.githubusercontent.com/gcsosso/DOLMEN/main/data/datasets"
FILES = {
    "amino": f"{DOLMEN_BASE}/amino.csv",
    "glyco2": f"{DOLMEN_BASE}/glyco2.csv",
}

DOLMEN_RAW_DIR = RAW_DIR / "dolmen"
PROCESSED_PATH = PROCESSED_DIR / "dolmen.parquet"


def _download(force: bool = False) -> dict[str, Path]:
    DOLMEN_RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in FILES.items():
        out = DOLMEN_RAW_DIR / f"{name}.csv"
        if out.exists() and not force:
            log.info("dolmen %s already cached at %s", name, out)
        else:
            log.info("downloading dolmen %s from %s", name, url)
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            out.write_text(r.text)
        paths[name] = out
    return paths


def _parse_one(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "SMILES" not in df.columns or "Exp. % MGS" not in df.columns:
        raise ValueError(
            f"unexpected DOLMEN columns in {path}: {df.columns.tolist()}"
        )
    df = df.rename(
        columns={
            "Name": "name",
            "SMILES": "smiles_raw",
            "Concentration (mM)": "concentration_mM",
            "Exp. % MGS": "mgs_percent",
            "Exp. error": "mgs_error",
            "Set": "upstream_split",
        }
    )
    # glyco2.csv lacks 'mgs_error' and 'upstream_split'; fill defaults.
    if "mgs_error" not in df.columns:
        df["mgs_error"] = pd.NA
    if "upstream_split" not in df.columns:
        df["upstream_split"] = "unknown"

    df["concentration_mM"] = pd.to_numeric(df["concentration_mM"], errors="coerce")
    df["mgs_percent"] = pd.to_numeric(df["mgs_percent"], errors="coerce")
    df["mgs_error"] = pd.to_numeric(df["mgs_error"], errors="coerce")

    df["smiles_canonical"] = df["smiles_raw"].astype(str).map(canonical_smiles)
    n_pre = len(df)
    df = df.dropna(subset=["smiles_canonical", "mgs_percent"]).copy()
    log.info("dolmen %s: parsed %d rows, %d valid after canonicalization", source, n_pre, len(df))
    df["source"] = source
    return df[
        [
            "name",
            "smiles_canonical",
            "smiles_raw",
            "concentration_mM",
            "mgs_percent",
            "mgs_error",
            "upstream_split",
            "source",
        ]
    ]


def load_dolmen(force_download: bool = False, force_reprocess: bool = False) -> pd.DataFrame:
    """Return a DataFrame with one row per DOLMEN compound.

    Caches to data/processed/dolmen.parquet; pass force_reprocess=True to rebuild.
    """
    if PROCESSED_PATH.exists() and not force_reprocess:
        log.info("loading cached dolmen parquet from %s", PROCESSED_PATH)
        return pd.read_parquet(PROCESSED_PATH)

    paths = _download(force=force_download)
    parts: list[pd.DataFrame] = []
    for source, path in paths.items():
        parts.append(_parse_one(path, source=f"dolmen_{source}"))
    df = pd.concat(parts, ignore_index=True)

    n_before = len(df)
    df = df.drop_duplicates(subset=["smiles_canonical"], keep="first").reset_index(drop=True)
    log.info(
        "dolmen total: %d rows, %d unique canonical SMILES (deduped within DOLMEN)",
        n_before,
        len(df),
    )

    df.to_parquet(PROCESSED_PATH, index=False)
    log.info("wrote %s (%d rows)", PROCESSED_PATH, len(df))
    return df


if __name__ == "__main__":
    df = load_dolmen()
    print(df.head())
    print(f"\nTotal: {len(df)} compounds")
    print(f"Source breakdown:\n{df['source'].value_counts()}")
    print(f"\n%MGS range: [{df['mgs_percent'].min():.2f}, {df['mgs_percent'].max():.2f}]")
