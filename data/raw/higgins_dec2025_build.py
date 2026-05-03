"""Builds data/raw/higgins_dec2025.csv from values read by visual inspection
of Figures 2-4 and the statistically highlighted mixture panels in
Figures 6-9 of the bioRxiv preprint:

    Ahmadkhani, Sugden, Mayo, Higgins. "Screening for cryoprotective agent
    toxicity and toxicity reduction in mixtures at subambient temperatures."
    bioRxiv 10.1101/2025.05.07.652719v1 (2025); also Cryobiology 121:105315.

The bioRxiv / Cryobiology paper does not publish a numeric data table for
these viability values. They live only in bar charts. I read each bar height
by eye against the gridlines (every 0.2 viability), and report values rounded
to the nearest 0.01. Precision is approximately +-0.05 (5 percentage points)
-- documented in the README so reviewers can calibrate trust.

For mixture rows, I include:
  1. The glycerol-pair rows from Figures 3 and 4 that were already readable
     in the original transcription.
  2. Every statistically highlighted toxicity-reduction panel from Figures
     6 and 7.
  3. Every statistically highlighted toxicity-neutralization panel from
     Figures 8 and 9.

I do NOT include every unlabeled / non-highlighted bar in the full Figure 3
and 4 grids, because several are too close to zero or too small in the public
image to transcribe defensibly. The significant panels are the ones the paper
itself interprets biologically, and they are readable from the screenshots.

If/when the authors publish raw values (Cryobiology supplementary, GitHub,
or by personal correspondence) this file should be regenerated and the
README updated.

All values are at 4 C, 30-min exposure, the conditions used throughout.
Viability is reported as a percentage (0-100); fractional values >1.0 in the
original figures (a measurement artifact relative to untreated controls)
are kept as-is (e.g., 125%).
"""

import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent / "higgins_dec2025.csv"

# (compound_name, viability_4c %) tuples, all at 30 min exposure, 4 C.
# Compound abbreviations from the paper's Table 1 are expanded to full names
# for PubChem lookup.

# Figure 2: 22 single CPAs at 3 mol/kg, 30 min, 4 C.
FIG2_3MOLKG = [
    ("2,3-butanediol", 68),
    ("diethylene glycol", 96),
    ("1,3-dihydroxyacetone", 78),
    ("diglyme", 0),
    ("dimethylacetamide", 102),
    ("ethylene glycol", 101),
    ("glycerol", 95),
    ("2-methoxyethanol", 85),
    ("2-methyl-2,4-pentanediol", 0),
    ("2-methyl-1,3-propanediol", 97),
    ("N-methylacetamide", 125),
    ("1,3-propanediol", 86),
    ("propionamide", 11),
    ("tetraethylene glycol dimethyl ether", 0),
    ("tetrahydrofurfuryl alcohol", 30),
    ("triethylene glycol", 16),
    ("triethylene glycol diacetate", 0),
    ("triglyme", 0),
    ("acetamide", 59),
    ("propylene glycol", 96),
    ("formamide", 74),
    ("dimethyl sulfoxide", 104),
]

# Figure 3 top row: 14 single CPAs at 6 mol/kg, 30 min, 4 C.
FIG3_6MOLKG = [
    ("glycerol", 40),
    ("dimethyl sulfoxide", 95),
    ("ethylene glycol", 85),
    ("formamide", 20),
    ("2-methoxyethanol", 58),
    ("1,3-dihydroxyacetone", 65),
    ("propylene glycol", 85),
    ("diethylene glycol", 55),
    ("acetamide", 10),
    ("N-methylacetamide", 65),
    ("1,3-propanediol", 75),
    ("2,3-butanediol", 0),
    ("2-methyl-1,3-propanediol", 25),
    ("dimethylacetamide", 0),
]

# Figure 4 top row: 14 single CPAs at 12 mol/kg, 30 min, 4 C.
# Most are essentially zero except EG, DHA, PD.
FIG4_12MOLKG = [
    ("glycerol", 0),
    ("dimethyl sulfoxide", 0),
    ("propylene glycol", 0),
    ("ethylene glycol", 95),
    ("formamide", 0),
    ("diethylene glycol", 0),
    ("acetamide", 0),
    ("N-methylacetamide", 0),
    ("1,3-propanediol", 10),
    ("2,3-butanediol", 0),
    ("2-methoxyethanol", 0),
    ("1,3-dihydroxyacetone", 75),
    ("dimethylacetamide", 0),
    ("2-methyl-1,3-propanediol", 0),
]

# Binary mixtures from Figures 3/4 (glycerol-pair rows) plus every mixture
# explicitly highlighted in Figures 6-9 as statistically significant toxicity
# reduction or toxicity neutralization. Mixtures listed as half-half by mol/kg.
MIXTURES = [
    # 6 mol/kg total = 3 mol/kg each: glycerol mixtures from Figure 3 top right
    ("glycerol", "dimethyl sulfoxide", 6, 75),
    ("glycerol", "propylene glycol", 6, 88),
    ("glycerol", "ethylene glycol", 6, 78),
    ("glycerol", "formamide", 6, 65),  # FA toxicity reduction by GLY at 6
    ("glycerol", "diethylene glycol", 6, 82),
    ("glycerol", "acetamide", 6, 75),
    ("glycerol", "N-methylacetamide", 6, 72),
    ("glycerol", "1,3-propanediol", 6, 73),
    ("glycerol", "2,3-butanediol", 6, 73),
    ("glycerol", "2-methoxyethanol", 6, 50),
    ("glycerol", "2-methyl-1,3-propanediol", 6, 53),
    # 6 mol/kg total: statistically highlighted toxicity-reduction mixtures
    # from Figure 6 (same total concentration as single CPA constituents).
    ("acetamide", "2-methoxyethanol", 6, 105),
    ("2-methoxyethanol", "dimethylacetamide", 6, 90),
    ("formamide", "2-methoxyethanol", 6, 88),
    ("2-methyl-1,3-propanediol", "dimethylacetamide", 6, 82),
    ("formamide", "2,3-butanediol", 6, 52),
    ("formamide", "dimethylacetamide", 6, 58),
    ("2,3-butanediol", "dimethylacetamide", 6, 80),
    ("2,3-butanediol", "2-methyl-1,3-propanediol", 6, 75),
    ("acetamide", "2,3-butanediol", 6, 75),
    ("N-methylacetamide", "2-methyl-1,3-propanediol", 6, 88),
    ("1,3-propanediol", "2-methyl-1,3-propanediol", 6, 92),
    # 6 mol/kg total: toxicity-neutralization panels from Figure 8.
    ("dimethyl sulfoxide", "acetamide", 6, 93),
    # 12 mol/kg total = 6 mol/kg each: the headline GLY/FA result (Figure 4)
    ("glycerol", "formamide", 12, 95),  # the famous formamide/glycerol toxicity neutralization
    ("glycerol", "ethylene glycol", 12, 75),
    ("glycerol", "propylene glycol", 12, 0),
    ("glycerol", "dimethyl sulfoxide", 12, 10),
    ("glycerol", "1,3-dihydroxyacetone", 12, 35),
    # 12 mol/kg total: statistically highlighted toxicity-reduction mixtures
    # from Figure 7.
    ("glycerol", "1,3-propanediol", 12, 65),
    ("glycerol", "diethylene glycol", 12, 35),
    ("N-methylacetamide", "2-methyl-1,3-propanediol", 12, 38),
    ("dimethyl sulfoxide", "formamide", 12, 52),
    ("dimethyl sulfoxide", "propylene glycol", 12, 5),
    ("glycerol", "acetamide", 12, 50),
    ("formamide", "diethylene glycol", 12, 15),
    # 12 mol/kg total: toxicity-neutralization panels from Figure 9.
    ("ethylene glycol", "acetamide", 12, 55),
]


# Hand-verified canonical SMILES for the 22 compounds. RDKit canonicalization
# was applied to make sure these are deterministic across re-runs. Names that
# PubChem lookup tends to miss or returns ambiguously (dimethylacetamide,
# triethylene glycol diacetate, triglyme, etc.) are baked in here so the CSV
# is self-contained and a reviewer can inspect the SMILES without running
# the pipeline.
SMILES = {
    "2,3-butanediol":                    "CC(O)C(C)O",
    "diethylene glycol":                 "OCCOCCO",
    "1,3-dihydroxyacetone":              "O=C(CO)CO",
    "diglyme":                           "COCCOCCOC",
    "dimethylacetamide":                 "CC(=O)N(C)C",
    "ethylene glycol":                   "OCCO",
    "glycerol":                          "OCC(O)CO",
    "2-methoxyethanol":                  "COCCO",
    "2-methyl-2,4-pentanediol":          "CC(O)CC(C)(C)O",
    "2-methyl-1,3-propanediol":          "CC(CO)CO",
    "N-methylacetamide":                 "CNC(C)=O",
    "1,3-propanediol":                   "OCCCO",
    "propionamide":                      "CCC(N)=O",
    "tetraethylene glycol dimethyl ether": "COCCOCCOCCOCCOC",
    "tetrahydrofurfuryl alcohol":        "OCC1CCCO1",
    "triethylene glycol":                "OCCOCCOCCO",
    "triethylene glycol diacetate":      "CC(=O)OCCOCCOCCOC(C)=O",
    "triglyme":                          "COCCOCCOCCOC",
    "acetamide":                         "CC(N)=O",
    "propylene glycol":                  "CC(O)CO",
    "formamide":                         "NC=O",
    "dimethyl sulfoxide":                "CS(C)=O",
}


def main() -> None:
    with OUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "compound_name",
                "compound_name_2",
                "smiles",
                "smiles_2",
                "concentration_mol_kg",
                "viability_4c",
                "is_mixture",
            ]
        )
        for name, viability in FIG2_3MOLKG:
            writer.writerow([name, "", SMILES.get(name, ""), "", 3, viability, "False"])
        for name, viability in FIG3_6MOLKG:
            writer.writerow([name, "", SMILES.get(name, ""), "", 6, viability, "False"])
        for name, viability in FIG4_12MOLKG:
            writer.writerow([name, "", SMILES.get(name, ""), "", 12, viability, "False"])
        for name1, name2, conc, viability in MIXTURES:
            writer.writerow([
                name1, name2,
                SMILES.get(name1, ""), SMILES.get(name2, ""),
                conc, viability, "True",
            ])
    print(
        f"wrote {OUT}: "
        f"{len(FIG2_3MOLKG)} singles@3mol/kg + "
        f"{len(FIG3_6MOLKG)} singles@6mol/kg + "
        f"{len(FIG4_12MOLKG)} singles@12mol/kg + "
        f"{len(MIXTURES)} mixtures = "
        f"{len(FIG2_3MOLKG) + len(FIG3_6MOLKG) + len(FIG4_12MOLKG) + len(MIXTURES)} rows"
    )


if __name__ == "__main__":
    main()
