"""Tox21 broad cytotoxicity dataset (auxiliary task for ChemBERTa).

Source: MoleculeNet via DeepChem (deepchem.molnet.load_tox21).
~7800 compounds, 12 binary nuclear-receptor / stress-response assays.

In v1 of this project, Tox21 is *not* fed to the RF baseline (which is one
regressor per CPA task). It exists for the ChemBERTa+LoRA model where a
multi-task auxiliary classification head leverages this larger background
to improve toxicity-relevant representations.

We download it Day 1 anyway to fail fast on DeepChem install/network issues.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils import PROCESSED_DIR, RAW_DIR, canonical_smiles, get_logger

log = get_logger("data.tox21")

PROCESSED_PATH = PROCESSED_DIR / "tox21.parquet"

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


def load_tox21(force_reprocess: bool = False) -> pd.DataFrame:
    """Return Tox21 as a long DataFrame.

    Columns: smiles_canonical, task (one of TOX21_TASKS), label (0/1).
    Missing labels (NaN in the original) are dropped.
    """
    if PROCESSED_PATH.exists() and not force_reprocess:
        log.info("loading cached tox21 parquet from %s", PROCESSED_PATH)
        return pd.read_parquet(PROCESSED_PATH)

    try:
        import deepchem as dc
    except ImportError as e:
        log.error(
            "deepchem not installed; Tox21 unavailable. "
            "RF baseline does not need it; ChemBERTa auxiliary task will skip. "
            "Install with: pip install deepchem"
        )
        return pd.DataFrame(columns=["smiles_canonical", "task", "label"])

    log.info("loading Tox21 via DeepChem (this may download ~5 MB and take a moment)")
    try:
        tasks, datasets, transformers = dc.molnet.load_tox21(
            featurizer="Raw",
            data_dir=str(RAW_DIR / "tox21"),
            save_dir=str(RAW_DIR / "tox21" / "save"),
        )
    except Exception as e:
        log.error("Tox21 load failed: %s", e)
        return pd.DataFrame(columns=["smiles_canonical", "task", "label"])

    train_ds, valid_ds, test_ds = datasets
    smiles, ys, ws = [], [], []
    for ds in (train_ds, valid_ds, test_ds):
        for x, y, w, ids in ds.itersamples():
            smiles.append(ids)
            ys.append(y)
            ws.append(w)

    rows = []
    for s, y, w in zip(smiles, ys, ws):
        canon = canonical_smiles(str(s))
        if canon is None:
            continue
        for t, label, weight in zip(tasks, y, w):
            if weight <= 0:
                continue
            rows.append({"smiles_canonical": canon, "task": t, "label": int(label)})

    df = pd.DataFrame(rows)
    n_compounds = df["smiles_canonical"].nunique() if not df.empty else 0
    log.info(
        "tox21: %d (compound, task) labels across %d unique compounds, %d tasks",
        len(df),
        n_compounds,
        df["task"].nunique() if not df.empty else 0,
    )

    df.to_parquet(PROCESSED_PATH, index=False)
    return df


if __name__ == "__main__":
    df = load_tox21()
    print(df.head())
    print(f"\nTotal rows: {len(df)}")
    if not df.empty:
        print(f"Unique compounds: {df['smiles_canonical'].nunique()}")
        print(f"Per-task positive rate:\n{df.groupby('task')['label'].mean()}")
