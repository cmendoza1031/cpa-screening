"""FDA Inactive Ingredients Database -> CPA-like candidate pool.

Source: https://www.fda.gov/drugs/drug-approvals-and-databases/inactive-ingredients-database-download
File:   January 2026 release at https://www.fda.gov/media/190589/download?attachment

Pipeline:
    1. Download the IID file (cached). It's a comma-delimited text file with
       a header row; common columns include 'Inactive Ingredient', 'CAS Number',
       'UNII', 'Route', 'Dosage Form', 'CAS Number', 'Maximum Potency'.
       Column naming varies subtly between releases, so we match defensively.
    2. Dedupe on (Inactive Ingredient + CAS) so multiple route/dose form rows
       don't cause repeated PubChem queries.
    3. Resolve CAS -> SMILES via PubChem (cached). Where CAS is missing or
       fails, fall back to ingredient name. Failures are logged + skipped.
    4. Filter to CPA-like physicochemistry:
         MW < 500 Da AND (HBD >= 1 OR HBA >= 2).
       Rationale: CPAs are small, polar, hydrogen-bond-capable solutes that
       can permeate cell membranes. The filter is intentionally permissive --
       we want a wide pool to score, not a tight pre-filter.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from ..utils import PROCESSED_DIR, RAW_DIR, canonical_smiles, get_logger
from .pubchem import name_or_cas_to_smiles

log = get_logger("data.fda_iid")

FDA_URL = "https://www.fda.gov/media/190589/download?attachment"
RAW_PATH = RAW_DIR / "fda_iid.txt"
PROCESSED_PATH = PROCESSED_DIR / "fda_iid_candidates.parquet"

CPA_MW_MAX = 500.0
CPA_HBD_MIN = 1
CPA_HBA_MIN = 2


def _download(force: bool = False) -> Path:
    if RAW_PATH.exists() and not force:
        log.info("fda_iid already cached at %s", RAW_PATH)
        return RAW_PATH
    log.info("downloading FDA IID from %s", FDA_URL)
    headers = {
        "User-Agent": "cpa-screening/0.1 (research; mailto:noreply@example.org)"
    }
    r = requests.get(FDA_URL, headers=headers, timeout=60)
    r.raise_for_status()
    RAW_PATH.write_bytes(r.content)
    log.info("wrote %s (%d bytes)", RAW_PATH, len(r.content))
    return RAW_PATH


def _read_iid(path: Path) -> pd.DataFrame:
    """Read the IID file. FDA ships it as comma-delimited or tab-delimited
    depending on the release; sniff the separator from the first line."""
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    first_line = text.splitlines()[0] if text else ""
    if "\t" in first_line:
        sep = "\t"
    elif ";" in first_line and first_line.count(";") > first_line.count(","):
        sep = ";"
    else:
        sep = ","
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    log.info(
        "fda_iid: parsed %d rows with separator %r and columns %s",
        len(df),
        sep,
        df.columns.tolist(),
    )
    return df


def _pick_column(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Return the first column name in df whose lowercased form matches one of
    the candidate substrings (case-insensitive contains)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for lc, original in cols_lower.items():
            if cand.lower() in lc:
                return original
    return None


def _compute_descriptors(smiles: str) -> Optional[dict]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
    except ImportError as e:
        raise ImportError("rdkit required: pip install rdkit") from e
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rotb": int(Lipinski.NumRotatableBonds(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
    }


def _passes_cpa_filter(d: dict) -> bool:
    return (
        d["mw"] < CPA_MW_MAX
        and (d["hbd"] >= CPA_HBD_MIN or d["hba"] >= CPA_HBA_MIN)
    )


def load_fda_iid(
    force_download: bool = False,
    force_reprocess: bool = False,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Return CPA-filtered FDA IID candidates with SMILES + descriptors.

    Columns:
        ingredient_name, cas, smiles_canonical, mw, logp, tpsa, hbd, hba, rotb, heavy_atoms.

    `limit` truncates the input list before PubChem lookup -- useful for smoke
    tests so the first run isn't slow on every dev iteration.
    """
    if PROCESSED_PATH.exists() and not force_reprocess and limit is None:
        log.info("loading cached fda_iid candidates from %s", PROCESSED_PATH)
        return pd.read_parquet(PROCESSED_PATH)

    path = _download(force=force_download)
    raw = _read_iid(path)

    name_col = _pick_column(raw, "Inactive Ingredient", "Ingredient")
    cas_col = _pick_column(raw, "CAS Number", "CAS_Number", "CAS")
    if name_col is None:
        raise ValueError(
            f"could not locate ingredient name column among {raw.columns.tolist()}"
        )

    work = raw[[c for c in [name_col, cas_col] if c is not None]].copy()
    work.columns = ["ingredient_name"] + (["cas"] if cas_col else [])
    if "cas" not in work.columns:
        work["cas"] = ""

    work["ingredient_name"] = work["ingredient_name"].astype(str).str.strip()
    work["cas"] = work["cas"].astype(str).str.strip().fillna("")
    work = work[work["ingredient_name"] != ""]
    work = work.drop_duplicates(subset=["ingredient_name", "cas"]).reset_index(drop=True)
    log.info("fda_iid: %d unique (name, cas) pairs after dedup", len(work))

    if limit is not None:
        work = work.head(limit).copy()
        log.info("fda_iid: limited to first %d entries for development", limit)

    smiles: list[Optional[str]] = []
    for i, row in work.iterrows():
        cas = row["cas"]
        name = row["ingredient_name"]
        s = None
        if cas and cas.lower() not in {"nan", "none", ""}:
            s = name_or_cas_to_smiles(cas, kind="cas")
        if s is None and name:
            s = name_or_cas_to_smiles(name, kind="name")
        smiles.append(s)
        if (i + 1) % 100 == 0:
            log.info("fda_iid: pubchem progress %d / %d (hits so far: %d)",
                     i + 1, len(work), sum(1 for x in smiles if x))

    work["smiles_canonical"] = smiles
    n_pre = len(work)
    work = work.dropna(subset=["smiles_canonical"]).copy()
    log.info(
        "fda_iid: %d / %d entries resolved to SMILES (%.1f%% hit rate)",
        len(work),
        n_pre,
        100.0 * len(work) / max(n_pre, 1),
    )

    work = work.drop_duplicates(subset=["smiles_canonical"]).reset_index(drop=True)
    log.info("fda_iid: %d unique molecules after SMILES dedup", len(work))

    descs = [_compute_descriptors(s) for s in work["smiles_canonical"]]
    work = work.assign(**{k: [d[k] if d else None for d in descs]
                          for k in ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "heavy_atoms"]})
    work = work.dropna(subset=["mw"]).copy()
    log.info("fda_iid: %d molecules with valid descriptors", len(work))

    pre_filter = len(work)
    mask = work.apply(
        lambda r: _passes_cpa_filter(
            {"mw": r["mw"], "hbd": r["hbd"], "hba": r["hba"]}
        ),
        axis=1,
    )
    work = work[mask].reset_index(drop=True)
    log.info(
        "fda_iid: %d / %d molecules pass CPA-like filter (MW<%g and (HBD>=%d or HBA>=%d))",
        len(work),
        pre_filter,
        CPA_MW_MAX,
        CPA_HBD_MIN,
        CPA_HBA_MIN,
    )

    if limit is None:
        work.to_parquet(PROCESSED_PATH, index=False)
        log.info("wrote %s (%d candidates)", PROCESSED_PATH, len(work))
    return work


if __name__ == "__main__":
    df = load_fda_iid()
    print(df.head())
    print(f"\nTotal candidates passing CPA-like filter: {len(df)}")
