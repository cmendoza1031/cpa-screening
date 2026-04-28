"""Run the data pipeline end-to-end:

    python -m src.data            # downloads, dedupes, builds long-format
    python -m src.data --force    # rebuilds everything from raw
"""

from __future__ import annotations

import argparse

from .build import build_dataset
from .fda_iid import load_fda_iid
from .tox21 import load_tox21


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the consolidated dataset")
    parser.add_argument("--force", action="store_true", help="rebuild from raw files")
    parser.add_argument(
        "--skip-fda",
        action="store_true",
        help="skip FDA IID download (slow first run; it is cached after)",
    )
    parser.add_argument(
        "--skip-tox21",
        action="store_true",
        help="skip Tox21 download (only needed for ChemBERTa aux task)",
    )
    parser.add_argument(
        "--fda-limit",
        type=int,
        default=None,
        help="for dev: only resolve first N FDA IID entries against PubChem",
    )
    args = parser.parse_args()

    long_df, audit = build_dataset(force_reprocess=args.force)
    print(f"\nlong_df: {long_df.shape}")

    if not args.skip_tox21:
        tox21 = load_tox21(force_reprocess=args.force)
        print(f"\ntox21: {tox21.shape if tox21 is not None else 'unavailable'}")

    if not args.skip_fda:
        fda = load_fda_iid(force_reprocess=args.force, limit=args.fda_limit)
        print(f"\nfda_iid candidates: {fda.shape}")


if __name__ == "__main__":
    main()
