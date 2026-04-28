"""PubChem name/CAS -> canonical SMILES with on-disk JSON cache.

Design notes:
- Single cache keyed by (kind, query_lower). Survives across runs.
- Failures are logged to data/.cache/pubchem_failures.txt, never fabricated.
- Polite: 0.2s sleep between live queries + exponential backoff on transient errors.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional

from ..utils import CACHE_DIR, canonical_smiles, get_logger

log = get_logger("data.pubchem")

CACHE_PATH = CACHE_DIR / "pubchem.json"
FAILURES_PATH = CACHE_DIR / "pubchem_failures.txt"

_MEMORY_CACHE: Optional[dict] = None


def _load_cache() -> dict:
    global _MEMORY_CACHE
    if _MEMORY_CACHE is not None:
        return _MEMORY_CACHE
    if CACHE_PATH.exists():
        try:
            _MEMORY_CACHE = json.loads(CACHE_PATH.read_text())
        except json.JSONDecodeError:
            log.warning("pubchem cache corrupted, starting fresh")
            _MEMORY_CACHE = {}
    else:
        _MEMORY_CACHE = {}
    return _MEMORY_CACHE


def _save_cache() -> None:
    cache = _load_cache()
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _log_failure(kind: str, query: str, reason: str) -> None:
    with FAILURES_PATH.open("a") as f:
        f.write(f"{kind}\t{query}\t{reason}\n")


def name_or_cas_to_smiles(
    query: str,
    kind: str = "name",
    save_every: int = 10,
    sleep: float = 0.2,
    max_retries: int = 3,
) -> Optional[str]:
    """Look up a single query against PubChem and return canonical SMILES.

    kind: 'name' or 'cas' (CAS is queried as a name; PubChem accepts both).
    Returns None on miss; the miss is logged to FAILURES_PATH.
    """
    if not query or not query.strip():
        return None
    key = f"{kind}::{query.strip().lower()}"
    cache = _load_cache()
    if key in cache:
        cached = cache[key]
        return cached if cached else None

    try:
        import pubchempy as pcp
    except ImportError as e:
        raise ImportError("pubchempy is required: pip install pubchempy") from e

    smiles: Optional[str] = None
    last_err: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            time.sleep(sleep)
            compounds = pcp.get_compounds(query.strip(), namespace="name")
            if not compounds:
                last_err = "no_match"
                break
            c = compounds[0]
            raw_smiles = (
                getattr(c, "isomeric_smiles", None)
                or getattr(c, "canonical_smiles", None)
            )
            if raw_smiles:
                smiles = canonical_smiles(raw_smiles)
                if smiles is None:
                    last_err = "rdkit_unparseable"
                break
            last_err = "no_smiles_field"
            break
        except pcp.PubChemHTTPError as e:
            last_err = f"http_{e}"
            time.sleep(min(2 ** attempt, 8))
        except Exception as e:  # network blips, JSON errors, etc.
            last_err = f"err_{type(e).__name__}"
            time.sleep(min(2 ** attempt, 8))

    cache[key] = smiles or ""
    if smiles is None:
        _log_failure(kind, query, last_err or "unknown")
        log.info("pubchem MISS [%s] %s -> %s", kind, query, last_err)
    else:
        log.debug("pubchem HIT [%s] %s -> %s", kind, query, smiles)

    if len(cache) % save_every == 0:
        _save_cache()
    return smiles


def batch_lookup(queries: Iterable[str], kind: str = "name") -> dict:
    """Look up many queries; returns {query: smiles_or_None} and persists cache."""
    out = {}
    queries = list(queries)
    for i, q in enumerate(queries):
        out[q] = name_or_cas_to_smiles(q, kind=kind)
        if (i + 1) % 25 == 0:
            log.info("pubchem progress: %d / %d", i + 1, len(queries))
            _save_cache()
    _save_cache()
    return out


def cache_stats() -> dict:
    cache = _load_cache()
    n_total = len(cache)
    n_hits = sum(1 for v in cache.values() if v)
    return {
        "n_total": n_total,
        "n_hits": n_hits,
        "n_misses": n_total - n_hits,
        "cache_path": str(CACHE_PATH),
        "failures_path": str(FAILURES_PATH),
    }
