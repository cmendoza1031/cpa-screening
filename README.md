# cpa-screening

ML pipeline for cryoprotective agent (CPA) discovery: predicts toxicity, membrane permeability, and ice recrystallization inhibition (IRI) from SMILES, then Pareto-ranks FDA-recognized excipients as actionable candidates.

> "Cryoprotectants are small molecules that help biological systems avoid ice formation during cooling… But these molecules are both protective and perilous." — Until Labs

The point: screen virtually before screening at the bench.

---

## Status: Phase 1 Day 1 (data + RF baseline)

Phase 1 Day 1 is complete. ChemBERTa+LoRA, cluster-aware splits, deep-ensemble uncertainty, and FDA IID Pareto ranking land in subsequent phases (see [plan](.cursor/plans/) and the **Roadmap** section below).

What runs today:

- DOLMEN ice-recrystallization data (Glyco2 + Amino, 286–303 compounds)
- Higgins Jan 2025 (toxicity + permeability at 4 °C / 25 °C, 27 compounds) — *requires hand-curated CSV from the published Tables / SI*
- Higgins Dec 2025 (toxicity at 4 °C, single compounds; binary mixtures flagged for v2) — same caveat
- Tox21 (auxiliary cytotoxicity background, used by ChemBERTa Day 2)
- FDA Inactive Ingredients Database → CPA-like candidate pool (MW < 500, polar, H-bond-capable)
- Random Forest baseline on Morgan FP + 10 RDKit descriptors, one regressor per task

## Quickstart

```bash
pip install -r requirements.txt

# Build consolidated dataset + audit (data/processed/audit.json):
python -m src.data

# Faster first run (skip slow PubChem CAS resolution for FDA IID):
python -m src.data --skip-fda --skip-tox21

# Train RF on all available tasks:
python -m src.train --model rf --seed 0
```

Run end-to-end on Colab: open `colab_runner.ipynb` and execute top-to-bottom.

## Data

| Source | Task(s) | n compounds | License | Acquisition |
|---|---|---|---|---|
| DOLMEN (Warren et al., Nat Commun 2024) | IRI (%MGS) | 80 + 223 = 303 | repo BSD | auto-downloaded raw CSVs |
| Higgins Jan 2025 (Sci Rep) | permeability @ 4°C / 25°C | 28 listed; 16 quantitative @ 4°C, 13 @ 25°C | CC-BY | exact values from [Table 1](https://www.nature.com/articles/s41598-025-85509-x/tables/1) |
| Higgins Dec 2025 (Cryobiology / bioRxiv) | toxicity @ 4°C, 3/6/12 mol/kg | 22 unique singles | per publisher | viability **read from Figures 2–4 bar charts**, precision ±5% |
| Tox21 (MoleculeNet) | aux. classification | ~7800 | open | DeepChem |
| FDA IID (Jan 2026) | candidate pool | ~1.8k entries | public | auto-downloaded |

Dedup is on RDKit-canonical SMILES. PubChem misses are logged to `data/.cache/pubchem_failures.txt`; we never fabricate SMILES.

### Data provenance, in detail

**Higgins Jan 2025.** Exact `permeability_4c` and `permeability_25c` values transcribed from Table 1 of the paper (units: $\bar{P}_{CPA} \times 10^{-3}$, s⁻¹). Compounds tagged "Fast" (above measurement ceiling) or "Toxic" (no permeability measurable) are kept in the CSV with NaN values so they're available for downstream filtering. **Viability is left NaN** because the paper publishes viability only as a scatter plot in Figure 6 with numeric labels keyed to a compound legend that is not exposed in the public HTML; the four 4°C-toxic compounds are known by name from the paper text but their exact viability percentages are not. Toxicity training data therefore comes entirely from the Dec 2025 paper.

**Higgins Dec 2025.** The published version (Cryobiology) and bioRxiv preprint do not include a numeric data table. Single-compound viability values for 22 CPAs at 3 mol/kg, 14 CPAs at 6 mol/kg, and 14 CPAs at 12 mol/kg (4°C, 30-min exposure) were **read by visual inspection from the bar charts in Figures 2, 3, and 4** of the bioRxiv preprint, against gridlines at every 0.2 viability. Reported precision is approximately **±5 percentage points**. Provenance, the explicit list of values, and the build script all live in [data/raw/higgins_dec2025_build.py](data/raw/higgins_dec2025_build.py) so values are auditable and easy to refresh if/when the authors publish raw data. The build also stores 16 binary mixtures (including the headline formamide/glycerol toxicity-neutralization at 12 mol/kg) in `data/processed/higgins_mixtures.parquet` for documentation; these are excluded from v1 model training.

**Concentration handling for v1.** When the same compound appears at multiple concentrations in Dec 2025, `build_dataset()` averages viability across concentrations to produce one toxicity label per compound. This is a deliberate v1 simplification — it loses concentration-dose information. Phase 2 will add concentration as an explicit feature.

## Method (Phase 1)

- **RF baseline**: Morgan fingerprint (radius 2, 2048 bits) + 10 physicochemical descriptors (MW, LogP, TPSA, HBD, HBA, RotB, aromatic rings, FractionCSP3, heavy atoms, ring count). One `RandomForestRegressor` per task, trained only on rows with non-null label for that task. No imputation across tasks.
- **ChemBERTa+LoRA (Day 2)**: `DeepChem/ChemBERTa-77M-MLM`, mean-pool, three regression heads + Tox21 auxiliary classification head. LoRA on q/k/v with rank ∈ {4, 8, 16}.
- **Splits (v1)**: deterministic random 70/15/15 keyed on canonical SMILES, hashed with the seed (so the same compound always lands in the same split across tasks). Cluster-aware splits (Tanimoto on Morgan FP, threshold 0.6) land in Phase 2 — gap between random and cluster-aware splits is the realistic generalization signal.

## Results (Phase 1 Day 1, random split, seed 0)

```bash
python -m src.train --model rf --seed 0
```

Numbers below are from a local CPU run. They are noisy and that's the point — Phase 1 Day 1 is the honest baseline that ChemBERTa+LoRA must beat. Toxicity and permeability tasks are tiny (n=22 and n=16 respectively); random splits land 0–2 compounds in test for those, which makes test R² and MAE essentially uninterpretable. Spearman on val is the more meaningful signal.

| Model | Task | Split | n | MAE | RMSE | R² | Spearman |
|---|---|---|---|---|---|---|---|
| RF (seed 0) | iri | train | 217 | 8.7 | 11.1 | 0.84 | 0.93 |
| RF (seed 0) | iri | val | 46 | 20.9 | 27.3 | 0.07 | 0.42 |
| RF (seed 0) | iri | test | 40 | 23.4 | 28.4 | 0.09 | 0.37 |
| RF (seed 0) | permeability | train | 12 | 5.8 | 6.7 | 0.87 | 0.96 |
| RF (seed 0) | permeability | val | 4 | 6.2 | 6.4 | 0.26 | 1.00 |
| RF (seed 0) | permeability | test | 0 | — | — | — | — |
| RF (seed 0) | toxicity | train | 14 | 5.3 | 6.4 | 0.91 | 0.99 |
| RF (seed 0) | toxicity | val | 6 | 25.9 | 28.9 | 0.35 | 0.81 |
| RF (seed 0) | toxicity | test | 2 | 22.6 | 25.4 | -1.87 | 1.00 |

Read: train metrics are strong everywhere because RF can fit Morgan fingerprints to small datasets easily; val/test gaps are large, especially for IRI (where the dataset is largest and the gap is therefore the most reliable). The 5-seed deep ensemble in Phase 2 will smooth the small-N noise and produce calibrated 95% prediction intervals.

## Repo layout

```
cpa-screening/
├── data/
│   ├── raw/                       # downloaded / hand-curated (gitignored)
│   ├── processed/                 # parquet caches + audit.json (gitignored)
│   └── .cache/                    # PubChem cache + failure log (gitignored)
├── results/
│   ├── figures/                   # parity plots (gitignored)
│   ├── candidates/                # Pareto top-20 (Phase 2; gitignored)
│   ├── summary.json
│   └── results_table.csv
├── src/
│   ├── data/
│   │   ├── pubchem.py             # cached name/CAS → SMILES
│   │   ├── dolmen.py              # IRI raw CSV download + parse
│   │   ├── higgins.py             # Jan + Dec 2025 loaders (singles only)
│   │   ├── tox21.py               # MoleculeNet via DeepChem
│   │   ├── fda_iid.py             # FDA IID download + CPA-like filter
│   │   ├── splits.py              # seeded random split (cluster-aware in Phase 2)
│   │   └── build.py               # consolidated long-format + audit
│   ├── models/
│   │   └── rf_baseline.py         # Morgan + descriptors + RF
│   ├── train.py                   # argparse entry; --model {rf,chemberta}
│   ├── eval.py                    # metrics + parity plots
│   └── utils.py                   # paths, seeding, canonical SMILES
├── colab_runner.ipynb
├── requirements.txt
└── README.md
```

## Roadmap

- **Phase 1 Day 2** — `src/models/chemberta_lora.py`: ChemBERTa-2 + LoRA, three regression heads + Tox21 auxiliary classification, multi-task MSE with task weights inversely proportional to dataset size. → PAUSE POINT 2.
- **Phase 2 Day 3** — cluster-aware splits, 5-seed deep ensembles, conformal calibration → 95% prediction intervals, target empirical coverage ~0.95.
- **Phase 2 Day 4** — FDA IID Pareto ranking with uncertainty-weighted composite score → top-20 actionable candidates.
- **Phase 3 Day 5** — final results + `DESIGN_DOC.md` (mixture modeling, MD-ML coupling, closed-loop active learning with the wet lab).

## Limitations

- Higgins data is small-N (~22–27 compounds per task) — RF / transformer test metrics will be noisy regardless of architecture, and reported confidence intervals matter as much as point estimates.
- v1 does not model binary mixtures. The Dec 2025 Higgins finding (formamide alone → 20% viability; +6 mol/kg glycerol → 97%) is a striking example of where the action lives. Mixture-aware architecture is sketched in `DESIGN_DOC.md` (Phase 3).
- No molecular dynamics features yet. MD-derived hydration / glass-transition descriptors would be a natural complement; this is also discussed in `DESIGN_DOC.md`.
- Predictions are not validated against any wet-lab experiment I ran — this repo is virtual screening only. Pareto top-20 candidates are *recommendations* for which FDA-recognized excipients to test.

## Citations

- DOLMEN: Warren et al., *Nat Commun* (2024). DOI: [10.1038/s41467-024-52266-w](https://doi.org/10.1038/s41467-024-52266-w)
- Higgins Jan 2025: Ahmadkhani et al., *Sci Rep* 15:1862 (2025). DOI: [10.1038/s41598-025-85509-x](https://doi.org/10.1038/s41598-025-85509-x)
- Higgins Dec 2025: Ahmadkhani et al., *Cryobiology* 121:105315 (2025). DOI: [10.1016/j.cryobiol.2025.105315](https://doi.org/10.1016/j.cryobiol.2025.105315)
- Tox21 / MoleculeNet: Wu et al. (2018). DOI: [10.1039/C7SC02664A](https://doi.org/10.1039/C7SC02664A)
- ChemBERTa-2: Ahmad et al. (2022). [arXiv:2209.01712](https://arxiv.org/abs/2209.01712)
