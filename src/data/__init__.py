"""Data layer: per-source loaders + consolidated build_dataset()."""

from .build import TASK_DIRECTIONS, build_dataset
from .dolmen import load_dolmen
from .fda_iid import load_fda_iid
from .higgins import load_higgins_dec2025, load_higgins_jan2025
from .pubchem import batch_lookup, cache_stats, name_or_cas_to_smiles
from .splits import assign_split, random_split_by_smiles
from .tox21 import load_tox21

__all__ = [
    "build_dataset",
    "TASK_DIRECTIONS",
    "load_dolmen",
    "load_higgins_jan2025",
    "load_higgins_dec2025",
    "load_tox21",
    "load_fda_iid",
    "name_or_cas_to_smiles",
    "batch_lookup",
    "cache_stats",
    "random_split_by_smiles",
    "assign_split",
]
