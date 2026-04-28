"""Shared utilities: paths, seeding, canonical SMILES."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / ".cache"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CANDIDATES_DIR = RESULTS_DIR / "candidates"

for _d in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, FIGURES_DIR, CANDIDATES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def seed_everything(seed: int) -> None:
    """Seed Python, numpy, and torch (if available). Idempotent and torch-optional."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


_RDKIT_READY = False


def _ensure_rdkit() -> None:
    """Import RDKit lazily and fail LOUDLY if it isn't installed.

    Previous versions silently caught ImportError inside canonical_smiles(),
    which on a fresh Colab where rdkit failed to install made every SMILES
    look like 'unparseable' (0 valid) instead of 'rdkit missing'.
    """
    global _RDKIT_READY
    if _RDKIT_READY:
        return
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError as e:
        raise ImportError(
            "rdkit is required (pip install rdkit). On Colab, run "
            "'pip install rdkit' BEFORE running the data pipeline; if you "
            "see version conflicts with deepchem/torch, install the base "
            "requirements first (requirements.txt) and the deep-learning "
            "extras (requirements-deep.txt) only when you reach Phase 1 Day 2."
        ) from e
    _RDKIT_READY = True


def canonical_smiles(smiles: str) -> Optional[str]:
    """Return RDKit-canonical SMILES, or None if RDKit can't parse it.

    Raises ImportError if rdkit is missing (callers want this loud, not silent).
    Returns None only when rdkit is available and the SMILES truly fails to parse.
    """
    if smiles is None:
        return None
    s = smiles.strip()
    if not s:
        return None
    _ensure_rdkit()
    from rdkit import Chem

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)
