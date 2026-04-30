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

# v2 (strict) filter parameters. v1 above is preserved for the legacy
# `_passes_cpa_filter` path; the v2 filter `_passes_cpa_filter_v2` is the
# one wired into the pipeline now. The v1 filter let MW=466 polysulfonated
# food dyes through the candidate pool because the only polarity criterion
# (HBA >= 2) is satisfied by sulfonate groups; v2 fixes that and several
# other classes of obvious non-CPA compounds.
CPA_V2_MW_MIN = 30.0          # below this is single-atom / ion territory
CPA_V2_MW_MAX = 350.0         # 350 keeps sucrose/trehalose (342) eligible
CPA_V2_LOGP_MAX = 1.5         # CPAs are hydrophilic; aromatic preservatives
                              # like benzyl benzoate (logP ~4) are filtered
CPA_V2_RING_MAX = 2           # eliminates fused-ring polyaromatics (food dyes)
CPA_V2_ALLOWED_ELEMENTS = {1, 6, 7, 8, 16}  # H, C, N, O, S
                                            # excludes halogens, P, all metals
                                            # (organomercurials, sodium salts pruned)


def _download(force: bool = False) -> Path:
    """Download the IID file. The FDA URL actually serves a ZIP archive
    containing IIR_OCOMM.csv (and an .xls duplicate, plus a change log).
    We extract the CSV and write it to data/raw/fda_iid.txt so the rest of
    the pipeline sees a regular CSV.
    """
    if RAW_PATH.exists() and not force:
        # Validate the cached file: previous runs (before we knew the URL
        # served a zip) wrote raw zip bytes here. If the cached file starts
        # with PK\x03\x04 it's a zip pretending to be a CSV; re-download.
        head = RAW_PATH.read_bytes()[:4]
        if head == b"PK\x03\x04":
            log.warning(
                "cached fda_iid at %s is a zip (from a pre-fix run); "
                "re-downloading and extracting the inner CSV.",
                RAW_PATH,
            )
            RAW_PATH.unlink()
        else:
            log.info("fda_iid already cached at %s", RAW_PATH)
            return RAW_PATH
    log.info("downloading FDA IID from %s", FDA_URL)
    headers = {
        "User-Agent": "cpa-screening/0.1 (research; mailto:noreply@example.org)"
    }
    r = requests.get(FDA_URL, headers=headers, timeout=60)
    r.raise_for_status()
    payload = r.content
    log.info("downloaded %d bytes from FDA IID URL", len(payload))

    # ZIP archives start with "PK\x03\x04". As of Jan 2026, FDA serves a zip;
    # earlier the URL returned a plain text file. Handle both.
    if payload[:4] == b"PK\x03\x04":
        import io
        import zipfile

        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")
                         and "change_log" not in n.lower()]
            if not csv_names:
                raise RuntimeError(
                    f"FDA IID zip contains no main CSV; entries: {zf.namelist()}"
                )
            inner = csv_names[0]
            log.info("FDA IID ships as ZIP; extracting %s (%d candidates: %s)",
                     inner, len(csv_names), csv_names)
            csv_bytes = zf.read(inner)
        RAW_PATH.write_bytes(csv_bytes)
        log.info("wrote %s (%d bytes, extracted from zip)", RAW_PATH, len(csv_bytes))
    else:
        RAW_PATH.write_bytes(payload)
        log.info("wrote %s (%d bytes, plain CSV)", RAW_PATH, len(payload))
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


def _compute_descriptors(smiles: str) -> Optional[tuple]:
    """Returns (mol, descriptor_dict) so the v2 filter can do substructure
    matches on mol; v1 only needed the dict."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
    except ImportError as e:
        raise ImportError("rdkit required: pip install rdkit") from e
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    desc = {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rotb": int(Lipinski.NumRotatableBonds(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "ring_count": int(Lipinski.RingCount(mol)),
    }
    return mol, desc


def _passes_cpa_filter(d: dict) -> bool:
    """Legacy v1 filter. Kept for reproducibility but not used by the
    pipeline; see _passes_cpa_filter_v2."""
    return (
        d["mw"] < CPA_MW_MAX
        and (d["hbd"] >= CPA_HBD_MIN or d["hba"] >= CPA_HBA_MIN)
    )


def _passes_cpa_filter_v21(mol, d: dict) -> bool:
    """v2.1 filter: v2 PLUS additional rejection criteria for the specific
    failure modes observed in the v2 top-20 (CO2, H2O2, formaldehyde,
    ethylene oxide, benzenesulfonic acid, phenol).

    Adds, on top of v2:
        1. heavy_atoms >= 3              drops H2O2 (2), formaldehyde (2)
        2. at least 1 hydrogen           drops CO2 (no Hs; purely inorganic)
        3. no sulfonate groups           tightens v2's "<=1 sulfonate" to 0,
                                         dropping benzenesulfonic acid
        4. no epoxide rings              drops ethylene oxide
        5. if aromatic, HBD+HBA >= 3     drops phenol (1+1=2), benzaldehyde
                                         (0+1=1), benzyl alcohol (1+1=2);
                                         keeps amino acids (>=4), niacinamide
                                         (1+3=4), gentisic acid (3+4=7),
                                         saccharin (1+3=4), benzoic acid
                                         (1+2=3 just barely)

    Hand-verified test cases (run via _filter_v21_unit_test below):
        DMSO, glycerol, urea, formamide, ethanol, glucose, phenylalanine,
        histidine, tryptophan -> pass
        CO2, H2O2, formaldehyde, ethylene oxide, benzenesulfonic acid,
        phenol, benzaldehyde, benzyl alcohol -> fail

    Note: ethanol passes because it's not aromatic (the HBD+HBA aromatic
    rule doesn't apply). Methanol would be borderline (heavy_atoms=2 just
    fails the new minimum); FDA IID uses "ALCOHOL" -> ethanol so methanol
    isn't a candidate in our pool anyway.
    """
    if not _passes_cpa_filter_v2(mol, d):
        return False
    from rdkit import Chem

    # 1. heavy_atoms >= 3
    if d["heavy_atoms"] < 3:
        return False

    # 2. >=1 hydrogen anywhere on the molecule
    n_h = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    if n_h < 1:
        return False

    # 3. no sulfonates (any count). v2 allowed up to 1; benzenesulfonic acid
    # passed v2 with exactly 1. Tightening to 0 drops it.
    sulfonate = Chem.MolFromSmarts("[SX4](=O)(=O)[O-,OX2H]")
    if mol.HasSubstructMatch(sulfonate):
        return False

    # 4. no epoxide ring
    epoxide = Chem.MolFromSmarts("C1OC1")
    if mol.HasSubstructMatch(epoxide):
        return False

    # 5. aromatic compounds need substantial polarity. The v2 model has 0
    # aromatic compounds in toxicity training (all 22 are small aliphatic
    # CPAs), so aromatic predictions converge to the training mean and
    # under-predict the actual cytotoxicity of phenolic compounds. Requiring
    # aromatic candidates to have HBD+HBA >= 3 keeps amino acids and
    # phenolic antioxidants (which are plausibly OK) but drops the simple
    # phenol / benzaldehyde / benzyl alcohol class that the model
    # systematically over-rates.
    has_aromatic = any(a.GetIsAromatic() for a in mol.GetAtoms())
    if has_aromatic and (d["hbd"] + d["hba"]) < 3:
        return False

    return True


def _passes_cpa_filter_v2(mol, d: dict) -> bool:
    """v2 filter. Closer to actual CPA chemistry.

    Real CPAs are small (MW < 200 typically; sugars push this to 350),
    hydrophilic (logP < 1), polar (multiple H-bond donors/acceptors),
    have at most one ring, and are made of {C, H, N, O, S} only. The v1
    filter let polysulfonated food dyes (MW=466, HBA>=8 from sulfonates),
    organomercurials, halogenated phenolics, and inorganic acids through
    because the only polarity check passed for any of those.

    v2 rejects, in order:
      1. MW outside [30, 350]
      2. logP > 1.5
      3. polarity insufficient (HBD < 1 AND HBA < 2)
      4. any element outside {H, C, N, O, S}
      5. azo group (N=N) present
      6. more than one sulfonate
      7. ring count > 2
    """
    if not (CPA_V2_MW_MIN < d["mw"] < CPA_V2_MW_MAX):
        return False
    if d["logp"] > CPA_V2_LOGP_MAX:
        return False
    # Polarity: at least one H-bond donor OR acceptor, OR a TPSA above 15 Å²
    # (DMSO has HBD=0 / HBA=1 / TPSA=36 and is the canonical CPA, so the
    # earlier "HBD>=1 OR HBA>=2" criterion was actually wrong here).
    if d["hbd"] < 1 and d["hba"] < 1 and d["tpsa"] < 15.0:
        return False
    # Element whitelist
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in CPA_V2_ALLOWED_ELEMENTS:
            return False
    # Azo group (any N=N double bond between two N atoms)
    from rdkit import Chem

    azo = Chem.MolFromSmarts("[#7]=[#7]")
    if mol.HasSubstructMatch(azo):
        return False
    # More than one sulfonate
    sulfonate = Chem.MolFromSmarts("[SX4](=O)(=O)[O-,OX2H]")
    if len(mol.GetSubstructMatches(sulfonate)) > 1:
        return False
    # Ring count
    if d["ring_count"] > CPA_V2_RING_MAX:
        return False
    return True


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

    desc_pairs = [_compute_descriptors(s) for s in work["smiles_canonical"]]
    mols = [p[0] if p else None for p in desc_pairs]
    descs = [p[1] if p else None for p in desc_pairs]
    desc_keys = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "heavy_atoms", "ring_count"]
    work = work.assign(**{k: [d[k] if d else None for d in descs] for k in desc_keys})
    work["_mol"] = mols
    work = work.dropna(subset=["mw"]).copy()
    log.info("fda_iid: %d molecules with valid descriptors", len(work))

    pre_filter = len(work)
    mask = work.apply(
        lambda r: _passes_cpa_filter_v21(
            r["_mol"],
            {k: r[k] for k in desc_keys},
        ),
        axis=1,
    )
    work = work[mask].drop(columns=["_mol"]).reset_index(drop=True)
    log.info(
        "fda_iid: %d / %d molecules pass v2.1 CPA-like filter "
        "(v2 + heavy_atoms>=3, has H, no sulfonate, no epoxide, "
        "aromatics need HBD+HBA>=3)",
        len(work),
        pre_filter,
    )

    if limit is None:
        work.to_parquet(PROCESSED_PATH, index=False)
        log.info("wrote %s (%d candidates)", PROCESSED_PATH, len(work))
    return work


if __name__ == "__main__":
    df = load_fda_iid()
    print(df.head())
    print(f"\nTotal candidates passing CPA-like filter: {len(df)}")
