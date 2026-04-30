"""Tox21 broad cytotoxicity dataset (auxiliary task for ChemBERTa).

The ChemBERTa multi-task model has a 12-class binary classification head
that's trained jointly with the three CPA regression heads. Tox21 gives
the encoder ~7800 extra labeled compounds with toxicity-relevant labels
across 12 nuclear-receptor and stress-response assays. The aux loss is
weighted low (0.1) so it can't dominate the CPA gradient; the goal is
regularization / better encoder representations rather than direct
transfer.

Loader: direct CSV download from the DeepChem GitHub raw mirror. The
older code used `deepchem.molnet.load_tox21()` but the deepchem package
fails to install on Python 3.12 (no wheel as of April 2026), so we
download the same CSV that DeepChem ships, parse it ourselves, and
canonicalize SMILES with RDKit. The resulting parquet matches what
DeepChem would have given us.
"""

from __future__ import annotations

import gzip
import io
import urllib.request
from pathlib import Path

import pandas as pd

from ..utils import PROCESSED_DIR, RAW_DIR, canonical_smiles, get_logger

log = get_logger("data.tox21")

PROCESSED_PATH = PROCESSED_DIR / "tox21.parquet"
RAW_PATH = RAW_DIR / "tox21.csv.gz"

# Same URL DeepChem points at; this is the canonical MoleculeNet CSV.
TOX21_URL = "https://github.com/deepchem/deepchem/raw/master/datasets/tox21.csv.gz"

TOX21_TASKS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]


def _download() -> bytes:
    """Fetch the raw gzipped CSV. Cached on disk."""
    if RAW_PATH.exists():
        log.info("using cached tox21 raw at %s", RAW_PATH)
        return RAW_PATH.read_bytes()
    log.info("downloading tox21 from %s", TOX21_URL)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        TOX21_URL, headers={"User-Agent": "cpa-screening/1.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    RAW_PATH.write_bytes(data)
    log.info("wrote %s (%d bytes)", RAW_PATH, len(data))
    return data


def load_tox21(force_reprocess: bool = False) -> pd.DataFrame:
    """Return Tox21 as a long DataFrame.

    Columns: smiles_canonical, task (one of TOX21_TASKS), label (0/1).
    Missing labels in the original CSV (empty cells) are dropped, which
    matches DeepChem's behavior of using the per-task weight matrix to
    mask out unmeasured (compound, task) pairs.
    """
    if PROCESSED_PATH.exists() and not force_reprocess:
        cached = pd.read_parquet(PROCESSED_PATH)
        if len(cached) > 0:
            log.info("loading cached tox21 parquet from %s (%d rows)", PROCESSED_PATH, len(cached))
            return cached
        log.warning("cached tox21 parquet is empty; rebuilding")

    try:
        raw = _download()
    except Exception as e:
        log.error("tox21 download failed: %s", e)
        return pd.DataFrame(columns=["smiles_canonical", "task", "label"])

    text = gzip.decompress(raw).decode("utf-8")
    df = pd.read_csv(io.StringIO(text))
    log.info("tox21 raw: %d rows, columns=%s", len(df), df.columns.tolist())

    if "smiles" not in df.columns:
        log.error("tox21 CSV missing 'smiles' column; got %s", df.columns.tolist())
        return pd.DataFrame(columns=["smiles_canonical", "task", "label"])

    rows = []
    n_canon_failed = 0
    for _, row in df.iterrows():
        smi = row["smiles"]
        if not isinstance(smi, str):
            continue
        canon = canonical_smiles(smi)
        if canon is None:
            n_canon_failed += 1
            continue
        for t in TOX21_TASKS:
            v = row.get(t, "")
            # Empty cell or NaN (unmeasured); skip rather than impute
            if pd.isna(v) or v == "":
                continue
            try:
                label = int(float(v))
            except (TypeError, ValueError):
                continue
            rows.append({"smiles_canonical": canon, "task": t, "label": label})

    out = pd.DataFrame(rows)
    n_compounds = out["smiles_canonical"].nunique() if not out.empty else 0
    log.info(
        "tox21: %d (compound, task) labels across %d unique compounds, %d tasks "
        "(canonicalization dropped %d malformed SMILES)",
        len(out), n_compounds, out["task"].nunique() if not out.empty else 0,
        n_canon_failed,
    )

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED_PATH, index=False)
    log.info("wrote %s", PROCESSED_PATH)
    return out


if __name__ == "__main__":
    df = load_tox21()
    print(df.head())
    print(f"\nTotal rows: {len(df)}")
    if not df.empty:
        print(f"Unique compounds: {df['smiles_canonical'].nunique()}")
        print(f"Per-task positive rate:\n{df.groupby('task')['label'].mean()}")
