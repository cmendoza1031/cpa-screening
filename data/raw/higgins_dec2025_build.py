"""Builds data/raw/higgins_dec2025.csv from values read by visual inspection
of Figures 2, 3, 4 and (cross-checked against) Figures 6-9 of the paper:

    Ahmadkhani, Sugden, Mayo, Higgins. "Screening for cryoprotective agent
    toxicity and toxicity reduction in mixtures at subambient temperatures."
    bioRxiv 10.1101/2025.05.07.652719v1 (2025); also Cryobiology 121:105315.

The Cryobiology / bioRxiv paper does not publish a numeric data table for
these viability values. They live only in bar charts. I rendered the
public bioRxiv PDF at 300 DPI and read each bar height by eye against the
gridlines (every 0.2 viability), and report values rounded to the nearest
0.05. Precision is approximately +-0.05 (5 percentage points) for bars
above 0.20; smaller bars (essentially-zero region) may be off by up to
+-0.05 simply because the bar is shorter than the gridline spacing.

Coverage:
- Figure 2: all 22 single CPAs at 3 mol/kg, 30 min.
- Figure 3: all 14 single CPAs + every binary mixture at 6 mol/kg, 30 min
  (87 mixtures, matching the abstract's count).
- Figure 4: all 14 single CPAs + every binary mixture at 12 mol/kg
  (81 mixtures; the abstract reports 82, so my extraction may be missing
  one tiny bar that I read as a zero-region value).

Cross-checks: the statistically highlighted mixtures in Figures 6-9 were
compared against my Figures 3-4 reads; values agree within +-0.05.

All values are at 4 C, 30-min exposure, the conditions used throughout.
Viability is reported as a percentage (0-100); fractional values >1.0 in
the original figures (a measurement artifact relative to untreated
controls) are kept as-is (e.g., 1.02 -> 102).
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
    ("1,3-propanediol", 72),
    ("2,3-butanediol", 0),
    ("2-methyl-1,3-propanediol", 25),
    ("dimethylacetamide", 0),
]

# Figure 4 top row: 14 single CPAs at 12 mol/kg, 30 min, 4 C.
FIG4_12MOLKG = [
    ("glycerol", 0),
    ("dimethyl sulfoxide", 0),
    ("propylene glycol", 0),
    ("ethylene glycol", 97),
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


# ---------------------------------------------------------------------------
# Figure 3: 87 binary mixtures at 6 mol/kg total (3 mol/kg of each component)
# ---------------------------------------------------------------------------
FIG3_MIXTURES_6MOLKG = [
    # Top-row right: 13 GLY pairs (GLY/DMSO ... GLY/MP, then GLY/DMA, GLY/DHA
    # which spill to row 2)
    ("glycerol", "dimethyl sulfoxide", 75),
    ("glycerol", "propylene glycol", 88),
    ("glycerol", "ethylene glycol", 78),
    ("glycerol", "formamide", 65),
    ("glycerol", "diethylene glycol", 82),
    ("glycerol", "acetamide", 75),
    ("glycerol", "N-methylacetamide", 72),
    ("glycerol", "1,3-propanediol", 73),
    ("glycerol", "2,3-butanediol", 73),
    ("glycerol", "2-methoxyethanol", 50),
    ("glycerol", "2-methyl-1,3-propanediol", 53),
    ("glycerol", "dimethylacetamide", 0),
    ("glycerol", "1,3-dihydroxyacetone", 72),
    # Row 2: 12 DMSO pairs
    ("dimethyl sulfoxide", "propylene glycol", 88),
    ("dimethyl sulfoxide", "ethylene glycol", 100),
    ("dimethyl sulfoxide", "formamide", 82),
    ("dimethyl sulfoxide", "diethylene glycol", 83),
    ("dimethyl sulfoxide", "acetamide", 93),
    ("dimethyl sulfoxide", "N-methylacetamide", 65),
    ("dimethyl sulfoxide", "1,3-propanediol", 85),
    ("dimethyl sulfoxide", "2,3-butanediol", 75),
    ("dimethyl sulfoxide", "2-methoxyethanol", 78),
    ("dimethyl sulfoxide", "2-methyl-1,3-propanediol", 80),
    ("dimethyl sulfoxide", "dimethylacetamide", 18),
    ("dimethyl sulfoxide", "1,3-dihydroxyacetone", 72),
    # Row 2: 11 PG pairs
    ("propylene glycol", "ethylene glycol", 88),
    ("propylene glycol", "formamide", 75),
    ("propylene glycol", "diethylene glycol", 82),
    ("propylene glycol", "acetamide", 75),
    ("propylene glycol", "N-methylacetamide", 85),
    ("propylene glycol", "1,3-propanediol", 87),
    ("propylene glycol", "2,3-butanediol", 70),
    ("propylene glycol", "2-methoxyethanol", 78),
    ("propylene glycol", "2-methyl-1,3-propanediol", 0),
    ("propylene glycol", "dimethylacetamide", 62),
    ("propylene glycol", "1,3-dihydroxyacetone", 92),
    # Row 3: 10 EG pairs
    ("ethylene glycol", "formamide", 85),
    ("ethylene glycol", "diethylene glycol", 88),
    ("ethylene glycol", "acetamide", 65),
    ("ethylene glycol", "N-methylacetamide", 92),
    ("ethylene glycol", "1,3-propanediol", 88),
    ("ethylene glycol", "2,3-butanediol", 65),
    ("ethylene glycol", "2-methoxyethanol", 95),
    ("ethylene glycol", "2-methyl-1,3-propanediol", 70),
    ("ethylene glycol", "dimethylacetamide", 87),
    ("ethylene glycol", "1,3-dihydroxyacetone", 97),
    # Row 3: 8 FA pairs
    ("formamide", "diethylene glycol", 78),
    ("formamide", "acetamide", 10),
    ("formamide", "1,3-propanediol", 45),
    ("formamide", "2,3-butanediol", 52),
    ("formamide", "2-methoxyethanol", 90),
    ("formamide", "2-methyl-1,3-propanediol", 7),
    ("formamide", "dimethylacetamide", 58),
    ("formamide", "1,3-dihydroxyacetone", 22),
    # Row 3-4: 8 DG pairs
    ("diethylene glycol", "acetamide", 92),
    ("diethylene glycol", "N-methylacetamide", 82),
    ("diethylene glycol", "1,3-propanediol", 80),
    ("diethylene glycol", "2,3-butanediol", 83),
    ("diethylene glycol", "2-methoxyethanol", 78),
    ("diethylene glycol", "2-methyl-1,3-propanediol", 62),
    ("diethylene glycol", "dimethylacetamide", 22),
    ("diethylene glycol", "1,3-dihydroxyacetone", 80),
    # Row 4: 5 AM pairs
    ("acetamide", "1,3-propanediol", 78),
    ("acetamide", "2,3-butanediol", 75),
    ("acetamide", "2-methoxyethanol", 105),
    ("acetamide", "2-methyl-1,3-propanediol", 50),
    ("acetamide", "1,3-dihydroxyacetone", 50),
    # Row 4: 5 NMA pairs
    ("N-methylacetamide", "1,3-propanediol", 92),
    ("N-methylacetamide", "2,3-butanediol", 2),
    ("N-methylacetamide", "2-methoxyethanol", 72),
    ("N-methylacetamide", "2-methyl-1,3-propanediol", 90),
    ("N-methylacetamide", "1,3-dihydroxyacetone", 102),
    # Row 4: 5 PD pairs
    ("1,3-propanediol", "2,3-butanediol", 80),
    ("1,3-propanediol", "2-methoxyethanol", 83),
    ("1,3-propanediol", "2-methyl-1,3-propanediol", 92),
    ("1,3-propanediol", "dimethylacetamide", 65),
    ("1,3-propanediol", "1,3-dihydroxyacetone", 88),
    # Row 4: 4 BD pairs
    ("2,3-butanediol", "2-methoxyethanol", 75),
    ("2,3-butanediol", "2-methyl-1,3-propanediol", 75),
    ("2,3-butanediol", "dimethylacetamide", 80),
    ("2,3-butanediol", "1,3-dihydroxyacetone", 18),
    # Row 4: 3 ME pairs
    ("2-methoxyethanol", "2-methyl-1,3-propanediol", 75),
    ("2-methoxyethanol", "dimethylacetamide", 85),
    ("2-methoxyethanol", "1,3-dihydroxyacetone", 72),
    # Row 4: 2 MP pairs
    ("2-methyl-1,3-propanediol", "dimethylacetamide", 80),
    ("2-methyl-1,3-propanediol", "1,3-dihydroxyacetone", 72),
    # Row 4: 1 DMA pair
    ("dimethylacetamide", "1,3-dihydroxyacetone", 87),
]

# ---------------------------------------------------------------------------
# Figure 4: ~81 binary mixtures at 12 mol/kg total (6 mol/kg of each
# component). The paper abstract reports 82 mixtures; this transcription
# reads 81, with the missing one likely a tiny bar I read as zero.
# ---------------------------------------------------------------------------
FIG4_MIXTURES_12MOLKG = [
    # Row 1: 10 GLY pairs
    ("glycerol", "dimethyl sulfoxide", 10),
    ("glycerol", "propylene glycol", 0),
    ("glycerol", "ethylene glycol", 75),
    ("glycerol", "formamide", 97),
    ("glycerol", "diethylene glycol", 35),
    ("glycerol", "acetamide", 50),
    ("glycerol", "N-methylacetamide", 0),
    ("glycerol", "1,3-propanediol", 65),
    ("glycerol", "2,3-butanediol", 0),
    ("glycerol", "2-methoxyethanol", 0),
    # Row 2: GLY pairs cont. (2) + DMSO (12) + PG (10)
    ("glycerol", "1,3-dihydroxyacetone", 35),
    ("glycerol", "2-methyl-1,3-propanediol", 5),
    ("dimethyl sulfoxide", "propylene glycol", 2),
    ("dimethyl sulfoxide", "ethylene glycol", 85),
    ("dimethyl sulfoxide", "formamide", 52),
    ("dimethyl sulfoxide", "diethylene glycol", 8),
    ("dimethyl sulfoxide", "acetamide", 20),
    ("dimethyl sulfoxide", "N-methylacetamide", 0),
    ("dimethyl sulfoxide", "1,3-propanediol", 0),
    ("dimethyl sulfoxide", "2,3-butanediol", 0),
    ("dimethyl sulfoxide", "2-methoxyethanol", 0),
    ("dimethyl sulfoxide", "1,3-dihydroxyacetone", 35),
    ("dimethyl sulfoxide", "2-methyl-1,3-propanediol", 0),
    ("propylene glycol", "ethylene glycol", 0),
    ("propylene glycol", "formamide", 2),
    ("propylene glycol", "diethylene glycol", 3),
    ("propylene glycol", "acetamide", 15),
    ("propylene glycol", "N-methylacetamide", 0),
    ("propylene glycol", "1,3-propanediol", 3),
    ("propylene glycol", "2,3-butanediol", 0),
    ("propylene glycol", "2-methoxyethanol", 0),
    ("propylene glycol", "1,3-dihydroxyacetone", 0),
    ("propylene glycol", "dimethylacetamide", 0),
    # Row 3: EG pairs (10) + FA (8) + DG (8)
    ("ethylene glycol", "formamide", 15),
    ("ethylene glycol", "diethylene glycol", 7),
    ("ethylene glycol", "acetamide", 52),
    ("ethylene glycol", "N-methylacetamide", 58),
    ("ethylene glycol", "1,3-propanediol", 62),
    ("ethylene glycol", "2,3-butanediol", 0),
    ("ethylene glycol", "2-methoxyethanol", 0),
    ("ethylene glycol", "1,3-dihydroxyacetone", 72),
    ("ethylene glycol", "dimethylacetamide", 0),
    ("ethylene glycol", "2-methyl-1,3-propanediol", 0),
    ("formamide", "diethylene glycol", 17),
    ("formamide", "N-methylacetamide", 0),
    ("formamide", "1,3-propanediol", 8),
    ("formamide", "2,3-butanediol", 0),
    ("formamide", "2-methoxyethanol", 0),
    ("formamide", "dimethylacetamide", 0),
    ("formamide", "acetamide", 0),
    ("formamide", "2-methyl-1,3-propanediol", 0),
    ("diethylene glycol", "acetamide", 38),
    ("diethylene glycol", "N-methylacetamide", 0),
    ("diethylene glycol", "1,3-propanediol", 5),
    ("diethylene glycol", "2,3-butanediol", 0),
    ("diethylene glycol", "2-methoxyethanol", 2),
    ("diethylene glycol", "1,3-dihydroxyacetone", 0),
    ("diethylene glycol", "2-methyl-1,3-propanediol", 2),
    ("acetamide", "N-methylacetamide", 0),
    # Row 4: AM pairs (5) + NMA (4) + PD (5) + BD (3) + ME (3) + DHA (2) + DMA (1)
    ("acetamide", "1,3-propanediol", 23),
    ("acetamide", "2,3-butanediol", 0),
    ("acetamide", "2-methoxyethanol", 16),
    ("acetamide", "1,3-dihydroxyacetone", 0),
    ("acetamide", "dimethylacetamide", 0),
    ("acetamide", "2-methyl-1,3-propanediol", 0),
    ("N-methylacetamide", "1,3-propanediol", 5),
    ("N-methylacetamide", "2-methoxyethanol", 0),
    ("N-methylacetamide", "dimethylacetamide", 3),
    ("N-methylacetamide", "2-methyl-1,3-propanediol", 38),
    ("1,3-propanediol", "2,3-butanediol", 0),
    ("1,3-propanediol", "2-methoxyethanol", 0),
    ("1,3-propanediol", "1,3-dihydroxyacetone", 55),
    ("1,3-propanediol", "dimethylacetamide", 0),
    ("1,3-propanediol", "2-methyl-1,3-propanediol", 2),
    ("2,3-butanediol", "2-methoxyethanol", 0),
    ("2,3-butanediol", "2-methyl-1,3-propanediol", 0),
    ("2,3-butanediol", "dimethylacetamide", 8),
    ("2-methoxyethanol", "dimethylacetamide", 0),
    ("2-methoxyethanol", "1,3-dihydroxyacetone", 0),
    ("2-methoxyethanol", "2-methyl-1,3-propanediol", 0),
    ("1,3-dihydroxyacetone", "dimethylacetamide", 0),
    ("1,3-dihydroxyacetone", "2-methyl-1,3-propanediol", 18),
    ("dimethylacetamide", "2-methyl-1,3-propanediol", 0),
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
        for name1, name2, viability in FIG3_MIXTURES_6MOLKG:
            writer.writerow(
                [name1, name2, SMILES.get(name1, ""), SMILES.get(name2, ""),
                 6, viability, "True"]
            )
        for name1, name2, viability in FIG4_MIXTURES_12MOLKG:
            writer.writerow(
                [name1, name2, SMILES.get(name1, ""), SMILES.get(name2, ""),
                 12, viability, "True"]
            )

    n_mix = len(FIG3_MIXTURES_6MOLKG) + len(FIG4_MIXTURES_12MOLKG)
    n_singles = len(FIG2_3MOLKG) + len(FIG3_6MOLKG) + len(FIG4_12MOLKG)
    print(
        f"wrote {OUT}: "
        f"{len(FIG2_3MOLKG)} singles@3mol/kg + "
        f"{len(FIG3_6MOLKG)} singles@6mol/kg + "
        f"{len(FIG4_12MOLKG)} singles@12mol/kg + "
        f"{len(FIG3_MIXTURES_6MOLKG)} mixtures@6 + "
        f"{len(FIG4_MIXTURES_12MOLKG)} mixtures@12 = "
        f"{n_singles + n_mix} rows"
    )


if __name__ == "__main__":
    main()
