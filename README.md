# cpa-screening

A triage funnel for cryoprotective agent (CPA) discovery. The CPA cocktails actually used in clinical vitrification (M22, VS55, VEG, DP6) are 4-6 component mixtures, because no single CPA hits vitrifying concentrations without also hitting toxicity. So the question that matters for cryopreservation is "what cocktail composition do we screen next", and "is this molecule a good CPA" is the upstream filter. This repo is the screening pipeline that question runs on top of, in three layers:

1. **Single-compound model.** Given a SMILES, three multi-task heads predict toxicity (cell mortality % at 4 °C, lower is better), permeability (P_CPA × 10⁻³ s⁻¹, higher is better), and ice recrystallization inhibition (% mean grain size from the splat cooling assay (SCA), lower is better), each with a calibrated 95% prediction interval from a 5-seed deep ensemble on cluster-aware folds. Deliverable: `[results/candidates/top20.csv](results/candidates/top20.csv)`, a Pareto-ranked top-20 from the FDA Inactive Ingredients Database (≈1,800 entries → 125 plausibly-CPA-shaped → top-20 by composite score), all at the same 4 °C / 6 mol/kg reference point.
2. **Binary cocktail recommender.** Score every binary pair from the 125-compound FDA pool (9,730 pairs) by combining the single-compound predictions with one of four additive rules (`max`, `mean`, `sum_then_cap`, `weighted_max`), Pareto-rank by predicted (toxicity, permeability, IRI). Deliverable: `[results/mixtures/top20_pairs.csv](results/mixtures/top20_pairs.csv)`. DMSO + propylene glycol at rank 1 (real cryomicroscopy combination); ethanol + propylene glycol at rank 8; butylene glycol + propylene glycol at rank 9. Validated on 170 binary mixtures from Higgins Dec 2025: best additive rule reaches Spearman ≈0.59 against measured viability. **Honest limit**: the additive rule structurally misses toxicity-neutralization (formamide + glycerol at 12 mol/kg, measured 95% viability, missed by 60+ percentage points). A learned interaction term I tried (PairEncoder, Spearman −0.26 on 170 rows) failed; the diagnosis points to a v5 residual-learner architecture, specced in [DESIGN_DOC.md §1b](DESIGN_DOC.md). The recommender is binary-only today; extending to k≥3 components is mechanically easy under the additive rule, but doing so without the learned interaction term would surface compositions whose predicted toxicity I wouldn't trust at vitrifying total concentrations.
3. **Active-learning hook.** A query-by-committee disagreement score over the FDA pool surfaces compounds where RF and ChemBERTa most disagree per task. Deliverable: `[results/candidates/top_disagreement.csv](results/candidates/top_disagreement.csv)`, the next-to-test list whose wet-lab measurement most constrains both models. The full closed-loop spec is in [DESIGN_DOC.md §3](DESIGN_DOC.md).

So the repo ships a **working binary cocktail recommender**, the single-compound model it's built on top of, and the mixture-data + architecture spec to push it to k≥3 components with a learned interaction term. What it is **not**: a fully-validated k-component cocktail-design model. The gap is mostly mixture training data, plus the residual-learner setup specced in DESIGN_DOC.

```bash
pip install -r requirements.txt
python -m src.data --skip-tox21               # dataset audit
python -m src.train --model rf --seed 0       # baseline (≈30 s)
pip install -r requirements-deep.txt          # torch / transformers / peft / torchao
python -m src.train --model chemberta --cv --cv-folds 5 --n-seeds 5 --split-mode cluster --lora-rank 8
python -m src.score_candidates --architecture both --n-seeds 5 --top-k 20
python -m src.figures
```

Or open `[colab_runner.ipynb](colab_runner.ipynb)` and run top-to-bottom.

For the iteration history (v1 → v2 → v2.1, what each version predicted, what actually happened, and how the top-20 candidate lists changed across versions), see [ITERATION_LOG.md](ITERATION_LOG.md).

---

## Why this matters

Cryoprotectants are the bottleneck of organ banking. To prevent ice formation in vitrification, you need them at multimolar concentrations, but at those concentrations they're toxic, and the CPA mixtures the field has used for decades (DMSO, ethylene glycol, glycerol, propylene glycol) come from a chemical repertoire that hasn't meaningfully expanded since the 70s. Until Labs frames this directly:

> "These molecules are both protective and perilous. Reaching concentrations high enough to prevent ice often pushes into toxic territory."

Three things have to be true of a usable CPA, all at once:

1. **Permeates the cell membrane fast enough** that the entire tissue is protected before ice forms. Glycerol famously fails this: too slow at 4 °C, leaving the deep parenchyma unprotected.
2. **Is non-toxic at the concentration needed for vitrification.** Most molecules with cryoprotective activity (formamide, urea derivatives, amides) hit toxicity well before they hit ice-suppressive concentrations.
3. **Inhibits ice nucleation and recrystallization.** Even at vitrifying concentrations, the warming step risks devitrification. Active ice recrystallization inhibitors mitigate this.

These are three independent properties with three independent screens. Until's molecular discovery program lists exactly these as their three pillars (their labels: efficacy at lower concentrations, biocompatibility at effective concentrations, faster biodistribution through vasculature). Their site explicitly says they use "Molecular Dynamics Simulations and Machine Learning to model cryoprotectants in silico" at the atomic level, and "high-throughput toxicity screens using liquid-handling robots" at the cellular level.

The wet-lab loop is slow and expensive. Higgins's 2025 method for simultaneous permeability + toxicity assessment in 96-well plates is itself a methodological advance in throughput, and even that gets to ≈27 compounds at a time. **Virtual screening is where you triage which compounds actually make it to the bench.** That's the niche this repo targets.

---

## Eval scheme up front

Two notes that aren't obvious from the table: (a) **all toxicity / permeability data is at 4°C** (Higgins Dec 2025 only screens at 4°C; Higgins Jan 2025 has both 4°C and 25°C and I deliberately drop the 25°C rows because Higgins shows the toxicity surface is qualitatively different between the two temperatures and the production target is 4°C organ-equilibration). (b) The toxicity OOF n is **50**, not 22, because each `(compound, concentration)` measurement from Dec 2025 is its own training row in v2 (3 / 6 / 12 mol/kg), so the model has to learn dose-response, not just per-compound toxicity.


| Task         | n   | Eval scheme                                                                                                       | Notes                               |
| ------------ | --- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| iri          | 303 | random 70/15/15 (canonical) + random 5-fold CV (for ChemBERTa parity) + Tanimoto cluster 5-fold (5-seed ensemble) | n large enough to trust held-out 40 |
| toxicity     | 22  | random 5-fold CV + Tanimoto cluster 5-fold (5-seed ensemble)                                                      | small N requires CV                 |
| permeability | 16  | random 5-fold CV + LOO-CV (RF only, sanity) + Tanimoto cluster 5-fold (5-seed)                                    | smallest, most fragile              |


Cluster-aware splits use Butina clustering on Morgan FP r=2 / 2048-bit Tanimoto with threshold 0.6. Whole clusters land in one fold, so no scaffold spans train/test. The whole-dataset 326 compounds collapse into 123 clusters (max=54, which is the DOLMEN sugar/amino-acid scaffold cluster, and 83 singletons), giving cluster fold sizes of [66, 65, 65, 65, 65].

Conformal calibration is split-conformal with finite-sample correction: q95 = quantile of |y_true − ensemble_mean| at level (n+1)(1−α)/n. Empirical coverage is measured on the OOF set as a sanity check (target ≈0.95).

LoRA rank is fixed at 8. The original spec called for sweeping {4, 8, 16}, but at n=22 toxicity the seed-to-seed standard error is larger than the rank-to-rank gap would be, so a sweep would be tuning to noise. Same logic for using 5-fold CV on the small tasks instead of LOO: LOO surrounded by 15 close Morgan-fingerprint neighbors is optimistic in a way that's hard to inspect, and 5-fold pairs naturally with the cluster-aware Butina splits I run for the deep-ensemble eval. RF reports LOO permeability as a side check anyway since it's cheap; the 5-fold vs LOO gap on permeability (Spearman 0.13 vs 0.34 for RF) is itself a useful signal that the LOO number is the optimistic one.

---

## Datasets


| Source                                                                                                           | Task(s)                       | n compounds                                                                 | License       | Acquisition                                                                 |
| ---------------------------------------------------------------------------------------------------------------- | ----------------------------- | --------------------------------------------------------------------------- | ------------- | --------------------------------------------------------------------------- |
| DOLMEN ([Warren et al., *Nat Commun* 2024](https://doi.org/10.1038/s41467-024-52266-w))                          | IRI (%MGS)                    | 80 amino + 223 glyco2 = 303                                                 | repo BSD      | auto-downloaded raw CSVs                                                    |
| Higgins Jan 2025 ([Ahmadkhani et al., *Sci Rep* 15:1862](https://doi.org/10.1038/s41598-025-85509-x))            | permeability @ 4°C / 25°C     | 28 listed; 16 quantitative @ 4°C, 13 @ 25°C                                 | CC-BY         | exact values transcribed from Table 1                                       |
| Higgins Dec 2025 ([Ahmadkhani et al., *Cryobiology* 121:105315](https://doi.org/10.1016/j.cryobiol.2025.105315)) | toxicity @ 4°C, 3/6/12 mol/kg | 22 unique single CPAs (+ 170 binary mixtures, flagged for mixture analysis) | per publisher | viability **read from PDF-rendered Figures 2-4 at 300 DPI**, ±5pp precision |
| Tox21 (MoleculeNet)                                                                                              | aux. classification           | ≈7,800                                                                      | open          | DeepChem (stretch; auxiliary head wired but not activated for v1)           |
| FDA Inactive Ingredients DB (Jan 2026)                                                                           | candidate pool                | ≈1.8k entries; 435 pass CPA-like filter                                     | public        | auto-downloaded ZIP, extracted, CAS→SMILES via PubChem                      |


Dedup is on RDKit-canonical SMILES; PubChem misses are logged to `data/.cache/pubchem_failures.txt` rather than fabricated. After dedup across all sources: **326 unique compounds, 341 (compound, task) labels**.

### Data provenance, in detail

**Higgins Jan 2025.** Permeability values transcribed verbatim from [Table 1](https://www.nature.com/articles/s41598-025-85509-x/tables/1) of the paper (units P_CPA × 10⁻³ s⁻¹). Compounds tagged "Fast" (above measurement ceiling) or "Toxic" (no permeability measurable) are kept with NaN values for downstream filtering. Viability is left NaN. The paper publishes it only as a scatter plot in Figure 6 with numeric labels keyed to a compound legend that isn't exposed in the public HTML, so I refused to invent percentages from indirect evidence. The four 4 °C-toxic compounds are known *by name* from the paper text but not by exact viability. Toxicity training data therefore comes entirely from the December 2025 paper.

**Higgins Dec 2025.** Neither the *Cryobiology* version nor the bioRxiv preprint includes a numeric data table. Single-compound viability values for 22 CPAs at 3 mol/kg, 14 CPAs at 6 mol/kg, and 14 CPAs at 12 mol/kg (4 °C, 30 min exposure) were read by visual inspection from the bar charts in Figures 2, 3, and 4 of the bioRxiv preprint, against gridlines at every 0.2 viability. Reported precision is approximately ±5 percentage points. Provenance, the explicit list of values, and the build script all live in `[data/raw/higgins_dec2025_build.py](data/raw/higgins_dec2025_build.py)` so values are auditable and trivially refreshable when the authors publish raw data. The build also stores 170 binary mixtures (the full Figures 3-4 grids: 87 mixtures at 6 mol/kg + 83 at 12 mol/kg; the abstract reports 87 + 82, my transcription has one extra at 12 mol/kg likely due to misreading a tiny near-zero bar) in `data/processed/higgins_mixtures.parquet` for mixture analysis; mixtures are excluded from single-compound training.

**Concentration handling.** When the same compound appears at multiple concentrations in Dec 2025, `build_dataset()` averages viability across concentrations to produce one toxicity label per compound. This is a deliberate v1 simplification, since concentration matters enormously biologically (formamide is fine at 3 mol/kg, devastating at 6), and collapsing into a mean discards the most interesting structure in the data. Adding concentration as an explicit input feature is in the next-steps list.

---

## Method

**Random Forest baseline.** Morgan fingerprint (radius 2, 2048 bits) + 10 RDKit physicochemical descriptors (MW, MolLogP, TPSA, HBD, HBA, RotB, NumAromaticRings, FractionCSP3, NumHeavyAtoms, RingCount). One `RandomForestRegressor` per task, trained only on rows with non-null label for that task. No imputation across tasks. This is a co-equal baseline, not a strawman: tree models on Morgan fingerprints are notoriously hard to beat on small, scaffold-clustered tabular data.

**ChemBERTa-2 + LoRA.** `[DeepChem/ChemBERTa-77M-MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM)` (RoBERTa-style, 384 hidden, ≈3.5M params, MLM-pretrained on PubChem). Mean-pool over the sequence dimension, attention-mask weighted. Three regression heads (toxicity / permeability / iri), each a 384→256→1 MLP with ReLU + dropout 0.1. LoRA via [peft](https://github.com/huggingface/peft) on the query, key, and value projections of every attention block: rank 8, alpha 16, dropout 0.1. Trainable parameter count: **55,296** (1.59% of the full model).

LoRA is the load-bearing choice here. At n=22 toxicity / n=16 permeability, full fine-tuning would catastrophically overfit. LoRA gives the smallest reasonable adapter that still exposes task-specific signal to the pretrained representation. EffiChem ([ChemRxiv 2025](https://chemrxiv.org/doi/10.26434/chemrxiv-2025-2lljt)) reports 62–96% reduction in trainable parameters with 3–5% AUC gains on toxicity / permeability tasks using exactly this recipe; my setup is a multi-task variant of theirs. I have a related project, `[MedHetLoRA](https://github.com/cmendoza1031/MedHetLoRA)`, that explored heterogeneous LoRA ranks across modalities for medical imaging. Same intuition: parameter-efficient adaptation is the right tool when you have a few-hundred-example downstream task and a foundation model with strong prior.

**Multi-task loss.** Per-sample squared error replaced by Huber (δ=1.0) on per-task standardized targets, summed over all labeled (compound, task) pairs in the batch and divided by total labels:

```
L_batch = (1/N) Σ_{(c,t) labeled} Huber(p_t(c) − ŷ_t(c) / σ_t,  δ=1)
```

I dropped earlier per-task weighting (inverse-sqrt-of-task-count, motivated by "don't drown out small tasks") because at n=22 toxicity, an up-weighted single-sample term dominates the gradient on bf16 backward and causes catastrophic NaN cascades. Per-sample weighting + Huber together bound the gradient magnitude at the source. IRI naturally gets ≈10× the gradient signal because it has ≈10× the labels, which is exactly what should happen given the data; the right tool for boosting toxicity signal is the Tox21 auxiliary head (next-steps list), not loss reweighting.

**Numerical stability on Blackwell.** ChemBERTa attention runs in bf16 via `torch.autocast(dtype=torch.bfloat16)` on bf16-capable hardware (Ampere, Hopper, Blackwell), fp32 elsewhere (T4). Forward attention uses PyTorch's `scaled_dot_product_attention` (`attn_implementation="sdpa"`), which has fused safe-softmax kernels. Without SDPA and with eager attention in bf16, I observed ≈95% of training batches producing NaN gradients on Blackwell. SDPA + Huber + per-sample loss took the NaN-grad rate to **zero** across 25 ensemble trainings (5 folds × 5 seeds × 30 epochs).

**Optimizer.** AdamW, lr=5e-5, weight decay 0.01, 30 epochs, early-stop patience 5 on combined val loss. `nan_to_num` on gradients before clipping, then per-element `clip_grad_value_(1.0)` (norm-based clipping is NaN-unsafe; a single NaN poisons the global norm and propagates).

**Splits and ensembling.** Random and Tanimoto-cluster 5-fold splits as described above. For each fold, 5 seed-distinct models trained from scratch; per-compound predictions aggregated as ensemble mean ± std. Conformal calibration uses the OOF residual quantile.

**Candidate scoring.** A separate full-data ensemble (5 seeds, all training labels, no held-out) is trained for the FDA IID prediction step. The OOF q95 from CV is reused as the calibration constant. This is standard split-conformal practice when going from honest eval to deployment. Pareto front computed in (toxicity, permeability, iri) space with directions (min, max, min). Composite score per compound = mean per-task desirability − 0.25 × normalized uncertainty.

---

## Results

### Headline: held-out Spearman by task × split × architecture

results summary

The bars are organized into three groups per task. Left to right within each task: random 5-fold (where applicable), random 70/15/15 val/test (IRI only), random LOO (permeability only, RF), Tanimoto cluster 5-fold 5-seed ensemble (the headline cluster-ensemble result). Blue = RF; red = ChemBERTa+LoRA.

### Full results table (v2)

These are the **v2 numbers** with concentration-aware toxicity, the tighter CPA filter, and the Tox21 auxiliary head active. v1 numbers are preserved in git history. See [v2: changes I made after looking at the v1 outputs](#v2-changes-i-made-after-looking-at-the-v1-outputs) for the motivation behind the v2 changes.


| Architecture | Task         | Scheme                    | n   | MAE  | RMSE | R²    | Spearman  | q95 (PI95) | Coverage     |
| ------------ | ------------ | ------------------------- | --- | ---- | ---- | ----- | --------- | ---------- | ------------ |
| RF           | iri          | random 70/15/15 (val)     | 46  | 20.9 | 27.3 | 0.07  | **0.424** | n/a        | n/a          |
| RF           | iri          | random 70/15/15 (test)    | 40  | 23.4 | 28.4 | 0.09  | **0.377** | n/a        | n/a          |
| RF           | iri          | random 5-fold OOF         | 303 | 20.0 | 24.4 | 0.26  | **0.511** | n/a        | n/a          |
| RF           | iri          | **cluster 5-fold 5-seed** | 303 | 20.7 | 24.9 | 0.23  | **0.499** | 46.7       | **0.954** ✓  |
| RF           | toxicity     | random 5-fold OOF         | 50  | 31.4 | 36.6 | 0.22  | **0.527** | n/a        | n/a          |
| RF           | toxicity     | **cluster 5-fold 5-seed** | 50  | 29.6 | 34.2 | 0.32  | **0.643** | 68.7       | 0.980 (over) |
| RF           | permeability | random 5-fold OOF         | 16  | 14.3 | 17.7 | -0.11 | 0.100     | n/a        | n/a          |
| RF           | permeability | random LOO OOF            | 16  | 13.0 | 16.7 | 0.02  | 0.344     | n/a        | n/a          |
| RF           | permeability | **cluster 5-fold 5-seed** | 16  | 13.7 | 16.6 | 0.02  | **0.368** | 38.2       | 1.000 (over) |
| ChemBERTa    | iri          | random 5-fold OOF         | 303 | 21.5 | 26.1 | 0.16  | 0.405     | n/a        | n/a          |
| ChemBERTa    | iri          | **cluster 5-fold 5-seed** | 303 | 22.0 | 26.3 | 0.15  | **0.391** | 47.5       | **0.954** ✓  |
| ChemBERTa    | toxicity     | **cluster 5-fold 5-seed** | 50  | 39.5 | 42.2 | -0.04 | **0.181** | 61.1       | 0.980 (over) |
| ChemBERTa    | permeability | random 5-fold OOF         | 16  | 13.8 | 17.4 | -0.07 | 0.015     | n/a        | n/a          |
| ChemBERTa    | permeability | **cluster 5-fold 5-seed** | 16  | 14.1 | 18.1 | -0.16 | 0.018     | 42.3       | 1.000 (over) |


**Bold rows are the headline cluster-ensemble result** (cluster-aware split, 5-seed ensemble, conformal-calibrated PIs). They are directly comparable across architectures because both share the exact same fold structure.

The toxicity OOF n is **50** (not 22) under v2 because each (compound, concentration) measurement from Higgins Dec 2025 is now its own row. Predicting toxicity is therefore a harder task in v2 than v1: the model has to capture dose-response, not just compound identity. RF still gains substantially at this harder target (cluster-ensemble Spearman 0.46 → 0.64); ChemBERTa stays flat-to-slightly-down within noise.

The single-seed ChemBERTa toxicity row is missing from this table because of an OOF aggregation bug in the run that produced these numbers (toxicity predictions were keyed by `"smi|conc"` strings but looked up by bare smiles in the single-seed code path). The cluster-ensemble path used a different aggregator and was unaffected. The bug is fixed in the current code and will produce a real number on the next run; the cluster-ensemble row above is the comparable headline result anyway.

### Key findings

1. **Concentration-aware toxicity is the biggest single intervention in this whole project.** RF toxicity cluster-ensemble Spearman jumped from 0.46 → **0.64** (+0.18) and random 5-fold from 0.35 → **0.53** (+0.18) just from emitting one row per (compound, concentration) measurement and adding concentration as an input feature. The Higgins Dec 2025 dataset's structure (each compound measured at 3, 6, and 12 mol/kg) is exactly the structure the original v1 collapsed into a single mean. The dose-response signal is *the* most informative thing in those 50 measurements; throwing it away by averaging was the single largest mistake in v1.
2. **Random Forest on Morgan fingerprints + descriptors is the strongest model on every task.** Cluster-ensemble Spearman: RF 0.499 (IRI) / **0.643** (toxicity) / 0.368 (permeability) vs ChemBERTa 0.391 / 0.181 / 0.018. This is honest reporting. DOLMEN's amino-acid + small-sugar compound space is precisely the regime trees on Morgan FPs are optimized for: tight scaffold structure that fingerprints encode directly, tabular-style supervised learning with hundreds of training examples. Tree models in this regime are notoriously hard to beat. There's substantial published evidence that pretrained transformers are *not* uniformly better than gradient-boosted trees on small molecular property datasets ([Jiang et al. 2021, J Cheminform](https://doi.org/10.1186/s13321-020-00479-8); [Yang et al. 2019, J Chem Inf Model](https://doi.org/10.1021/acs.jcim.9b00237); both pre-foundation-model but the conclusion has held). Reporting RF as the best baseline, not as a foil, is the scientifically honest result.
3. **The Tox21 auxiliary head didn't move the ChemBERTa toxicity number meaningfully.** Cluster-ensemble ChemBERTa toxicity: 0.18 (with aux head, weight 0.1) vs 0.22 (no aux head, original v1). At n=50 the standard error on Spearman is ≈0.13, so this 0.04 drop is within noise. The honest interpretation: at *this* training scale (50 toxicity samples per fold, 30 epochs, weight 0.1) the aux-head signal is too weak to overcome the basic small-data limitation. Tox21 measures nuclear-receptor binding and stress-response activation at submicromolar concentrations on hepatocytes; CPA toxicity is bulk cytotoxicity at multi-molar concentrations on endothelial cells, so direct task transfer was always a stretch. The architecture is in place for sweeps over weight / training schedule / aux-task choice in a follow-up, but the v2 number says "this didn't pay off in the way I hoped."
4. **The cluster-vs-random gap is small for both architectures on IRI.** RF: 0.511 → 0.499. ChemBERTa: 0.405 → 0.391. The Tanimoto-clustered split forces test compounds to be dissimilar to train, and on IRI both models still rank-order them. **The models are learning real structure-property relationships, not memorizing scaffolds.** This is a non-trivial generalization claim.
5. **Conformal calibration achieves the target on the tractable task.** IRI cluster-ensemble coverage = 0.954 for both architectures (target 0.95). Toxicity over-covers at 0.98, permeability at 1.00. The finite-sample correction at n=50 / n=16 still inflates q95 a bit, so the prediction intervals on small tasks are conservative (wider than a non-finite-corrected calculation would give) but never undercover. This is appropriate for downstream Pareto ranking: I'd rather flag too many candidates as uncertain than too few.
6. **The v2 filter cleaned up the candidate pool meaningfully.** Pareto pool went from 435 (v1) to **140** (v2) FDA candidates. The top-20 lost food dyes (FD&C Blue No. 2, D&C Red No. 33), the azo dye 1-(phenylazo)-2-naphthylamine, phenylmercuric acetate, and metaphosphoric acid. It gained urea (still there), ethanol, propanol, and butanol (small primary alcohols, all real CPAs in the literature). The remaining errors are smaller-scale (CO2 in #18, benzenesulfonic acid in #7) and are addressable by tightening the heavy-atom-count and pKa criteria; full discussion in the candidate-list section below.

### FDA top-20 candidates (v2)

The v2 filter shrinks the FDA candidate pool to **140** compounds (down from 435 in v1). The Pareto front contains 16 of those 140 candidates; the top-20 below is filled out from there by composite score (predicted toxicity + permeability + IRI, weighted toward tighter PIs).

Pareto top-20

The 3D view of the same scoring run separates the **41 actual Pareto-front candidates** (red) from the 99 dominated ones (gray) in (toxicity, permeability, IRI) space:

3D Pareto cube

The 41-vs-99 split is the meaningful number: 29% of the v2-filtered FDA pool sits on the Pareto front. The composite-score top-20 in the table below picks the best 20 of those 41 by the (mean per-task desirability − uncertainty penalty) ranking, with non-Pareto compounds also eligible if their composite beats Pareto compounds with high uncertainty.


| Rank | Ingredient                    | CAS      | Tox @ 6 mol/kg μ ± σ | Perm μ ± σ | IRI μ ± σ  | Notes                                         |
| ---- | ----------------------------- | -------- | -------------------- | ---------- | ---------- | --------------------------------------------- |
| 1    | Phenylalanine                 | 63-91-2  | 56.3 ± 0.8           | 24.4 ± 0.2 | 37.3 ± 1.0 | amino acid; in DOLMEN train                   |
| 2    | Tryptophan                    | 73-22-3  | 57.5 ± 0.6           | 25.7 ± 0.5 | 38.1 ± 0.8 | amino acid; in DOLMEN train                   |
| 3    | Benzaldehyde                  | 100-52-7 | 52.0 ± 1.4           | 22.3 ± 0.6 | 31.8 ± 1.1 | aromatic; flavoring                           |
| 4    | Niacinamide (Vitamin B₃)      | 98-92-0  | 60.0 ± 1.5           | 26.5 ± 0.7 | 34.5 ± 0.7 | biocompatible humectant                       |
| 5    | Phenylethyl alcohol           | 60-12-8  | 50.4 ± 1.5           | 20.3 ± 0.4 | 33.3 ± 0.6 | rose-scented preservative                     |
| 6    | Saccharin                     | 81-07-2  | 53.9 ± 0.8           | 24.2 ± 0.5 | 39.7 ± 1.0 | sweetener                                     |
| 7    | Benzenesulfonic acid          | 98-11-3  | 54.2 ± 1.4           | 22.8 ± 0.3 | 36.5 ± 0.8 | strongly acidic; **likely error**             |
| 8    | Benzyl alcohol                | 100-51-6 | 45.7 ± 1.5           | 18.8 ± 0.7 | 34.3 ± 0.5 | preservative; biocompatible                   |
| 9    | Phenol                        | 108-95-2 | 48.3 ± 0.9           | 20.8 ± 0.9 | 34.4 ± 0.9 | toxic at use concentrations; **likely error** |
| 10   | "Wax" (FDA generic IID label) | n/a      | 58.1 ± 0.6           | 24.5 ± 0.5 | 37.9 ± 1.1 | ambiguous label; SMILES resolved to non-wax   |
| 11   | Histidine                     | 71-00-1  | 56.3 ± 0.7           | 26.1 ± 0.4 | 48.2 ± 0.9 | amino acid; in DOLMEN train                   |
| 12   | **Urea**                      | 57-13-6  | 54.1 ± 1.3           | 29.5 ± 0.4 | 59.4 ± 1.1 | **known CPA**                                 |
| 13   | Gentisic acid                 | 490-79-9 | 48.9 ± 0.4           | 21.9 ± 0.6 | 46.6 ± 1.3 | small phenol; antioxidant                     |
| 14   | Dehydroacetic acid            | 771-03-9 | 57.9 ± 1.3           | 26.0 ± 0.8 | 42.5 ± 1.1 | preservative                                  |
| 15   | **Ethanol** ("Alcohol")       | 64-17-5  | 53.4 ± 3.1           | 26.5 ± 0.7 | 49.8 ± 1.2 | **known CPA / co-solvent**                    |
| 16   | **n-Propanol**                | 71-23-8  | 51.1 ± 3.3           | 26.5 ± 0.6 | 55.9 ± 0.9 | **CPA in cryomicroscopy**                     |
| 17   | Arginine                      | 74-79-3  | 62.3 ± 0.6           | 27.4 ± 0.6 | 52.9 ± 0.7 | amino acid; in DOLMEN train                   |
| 18   | Carbon dioxide                | 124-38-9 | 38.7 ± 1.2           | 18.8 ± 0.7 | 53.7 ± 1.5 | gas, 3 heavy atoms; **clear error**           |
| 19   | n-Butanol                     | 71-36-3  | 53.4 ± 3.3           | 27.1 ± 0.5 | 54.0 ± 1.4 | small primary alcohol; CPA-adjacent           |
| 20   | Valine                        | 72-18-4  | 63.4 ± 1.0           | 28.2 ± 0.6 | 53.6 ± 0.9 | amino acid; in DOLMEN train                   |


Full ranking + predictions for all 140 scored v2 candidates: `[results/candidates/all_scored.csv](results/candidates/all_scored.csv)`.

### Reading the candidate list honestly

The list mixes real CPAs the model has signal on, compounds that look CPA-shaped to the physicochemical filter but aren't actually used as CPAs, and DOLMEN-train memorization. The honest categorization:

**Compounds the model has actual signal on** (known CPAs):

- **Urea** (#5). Used in slow-freeze of red blood cells and as a permeating CPA in some red-cell vitrification protocols. In single-compound training.
- **Ethanol** ("Alcohol", #12). Co-solvent in vitrification cocktails. Not in CPA training but the model's prediction (toxicity 53 at 6 mol/kg) is plausible.
- **n-Propanol** (#13) and **n-Butanol** (#17). Used in cryomicroscopy. Same family as ethanol; the model generalizes within the small-primary-alcohol scaffold.
- **Acetone** (#14). Used as a cryomicroscopy solvent and in some non-aqueous CPA contexts. Not in training; model gets a reasonable prediction.
- **N,N-Dimethylacetamide** (#19). Real CPA (the "DMA" abbreviation in Higgins Dec 2025). In training.

That's five known CPAs and one in-training compound, surfaced from the FDA pool without any "is a CPA" label in training. v1 had two; v2 had four; v3 has five.

**Compounds defaulting to the training mean** (the OOD problem):

The toxicity training set is 22 small alcohols, polyols, amides, and sulfoxides. For chemistry outside that distribution, the model has no signal and predictions converge toward the training mean (mortality ≈ 55-60% at 6 mol/kg, permeability ≈ 24-27, IRI ≈ 35-50). Compounds that happen to land close to the "ideal mean" values get high composite scores by accident:

- **Phenylalanine, tryptophan, histidine, arginine, valine, methionine** (5 of top-20, plus methionine and 2 lysines). All in DOLMEN training; pure memorization on IRI; defaults to training mean on toxicity. Plausible biocompatible chemistry but not real CPAs.
- **Niacinamide** (#3, vitamin B₃). Skincare humectant, not a CPA. Predicted toxicity 60.0 (training mean), permeability 26.5 (mean), IRI 34.5 (mean). Pure OOD with mean-default predictions.
- **Saccharin** (#4) and **aspartame** (in novel-only top-20 #5). Sweeteners, not CPAs. Same story.
- **Phenoxyethanol** (#7), **dehydroacetic acid** (#9), **benzocaine** (novel #9), **methylparaben** (novel #11). Cosmetic preservatives. Not CPAs. Mean-default predictions.

The pattern: for compounds the model has no in-training analog for, the predictions are noise around the training mean, and whatever OOD compound happens to land closest to the "ideal" gets ranked. **The composite score is fooling itself** in this regime; the top-20 is partly a list of "compounds whose mean-default predictions happen to look good" rather than "compounds the model has informed opinions on."

A reviewer should treat the list as: **5 real CPAs to be expected, 5-7 OOD compounds whose ranking is essentially noise, 5+ memorized amino acids.** The QbC disagreement analysis (`results/candidates/top_disagreement.csv`) is a more useful artifact for active learning because it surfaces compounds where RF and ChemBERTa disagree most, which is exactly where wet-lab measurement adds the most information.

**No clear errors survived v2.1 filter** (no CO₂, no benzenesulfonic acid, no phenol, no benzaldehyde, no ethylene oxide, no formaldehyde). That's the v2.1 filter doing its job. But the OOD-mean-default problem can't be solved by filtering: it needs more diverse toxicity training data (different cell types, different temperatures, different chemistry classes).

---

## Mixture-aware analysis (v2.1)

CPAs are practically used as multi-component cocktails, not single compounds. The headline empirical finding from Higgins Dec 2025 is the *toxicity neutralization* effect: formamide alone at 6 mol/kg gives ≈20% viability, but formamide+glycerol at 6+6=12 mol/kg gives ≈95% viability. A single-compound model has no way to predict that. M22, VS55, and VEG (the vitrification cocktails actually used clinically) are 4-6 component mixtures designed specifically to push past the single-compound toxicity ceiling. So mixture modeling is *the* most important gap this repo had at the end of v2.

### What the data actually allows

The Higgins Dec 2025 paper publishes binary mixture viability data only as bar charts. The values included in this repo are **170 binary mixtures** transcribed from the full Figures 3 and 4 grids of the bioRxiv preprint (rendered at 300 DPI from the PDF). Each bar is read against the 0.2-spaced gridlines, with ±5pp precision.

- 87 at 6 mol/kg total (3 mol/kg of each component)
- 83 at 12 mol/kg total (6 mol/kg of each component)

This matches the paper's claimed 87 + 82 mixture coverage (within one off-by-one transcription error at 12 mol/kg, almost certainly a near-zero bar I misread). With this full dataset, the mixture section is now a real evaluation, not just a stress test.

1. The **architecture** for a learned pair encoder (`PairEncoder` in `[src/models/mixture.py](src/models/mixture.py)`), documented and shape-tested but not trained. It's drop-in ready when 200+ binary mixture rows become available, either from the Higgins supplementary table when published or from new wet-lab data.
2. The **additive-baseline analysis**: predict mixture viability by running the v2 single-compound model on each component at its individual concentration, combine the per-compound mortality predictions with one of {max, mean, sum_then_cap, weighted_max}, and compare to the measured mixture viability. This quantifies how badly a single-compound model fails on mixtures, which is the gap a proper mixture model would have to close.

### Additive baseline result on 170 known mixtures

Using the v2 RF full-data ensemble (concentration-aware toxicity head, the strongest single-compound model in v2 at cluster Spearman 0.64; n_seeds=5):


| Combination rule | n   | Spearman   | MAE (pp) | RMSE | R²        | Neutralization misses (>25 pp) |
| ---------------- | --- | ---------- | -------- | ---- | --------- | ------------------------------ |
| max              | 170 | **+0.568** | 25.6     | 32.5 | **+0.24** | 21                             |
| mean             | 170 | **+0.593** | 29.1     | 37.8 | -0.03     | 5                              |
| sum_then_cap     | 170 | **+0.591** | 24.1     | 31.2 | **+0.30** | 50                             |
| weighted_max     | 170 | **+0.589** | 27.3     | 35.1 | **+0.11** | 6                              |


(Numbers from the full 170-row mixture set; values are visually transcribed from PDF-rendered Figures 3 and 4 at 300 DPI, with approximately ±5 percentage points precision.)

All four rules give Spearman in [0.57, 0.59] on the full mixture set, with R² up to 0.30 for sum_then_cap. The additive baseline carries surprisingly far once the dataset is representative: the single-compound model has been trained on each component at each concentration, and combining those predictions with a simple rule recovers most of the rank-order structure in the data. The remaining gap (R² well below 1.0, Spearman well below 1.0) is exactly the interaction term that mutual dilution and toxicity neutralization create, and that an additive rule cannot capture. **A single-compound model combined with any of the standard additive heuristics is essentially uncorrelated with measured mixture viability.** This is the quantitative version of "single-compound modeling fundamentally can't predict mixture toxicity." 

### The neutralization case study

The headline failure is exactly the formamide+glycerol case from Higgins's paper:


| Mixture              | Total conc    | **Measured viability** | Additive `max` | Additive `mean` | Additive `sum_then_cap` |
| -------------------- | ------------- | ---------------------- | -------------- | --------------- | ----------------------- |
| glycerol + formamide | 6 mol/kg      | 65                     | 55.8           | 65.1            | 30.3                    |
| glycerol + formamide | **12 mol/kg** | **95**                 | **31.7**       | **40.7**        | **0.0**                 |


At 6 mol/kg total (3+3 each), all rules are within ≈10-35 pp of truth, since formamide and glycerol are both relatively non-toxic individually at 3 mol/kg, so additive combinations are roughly right. At 12 mol/kg total, formamide alone is ≈80% toxic and glycerol alone is ≈60% toxic, so additive predictions are between 30 (max-rule) and 0 (sum-cap), but the *measured* viability is **95** because the two compounds neutralize each other's toxicity. **Additive `max` misses by 63 percentage points; `sum_then_cap` misses by 95.**

This is the gap a proper mixture-aware model has to close. There's no hand-tuned additive rule that can capture it; the model has to learn an interaction term from data. And we don't have that data yet at usable scale.

### FDA mixture pair scoring

Even without a learned mixture model, the additive baseline can rank binary pairs from the FDA candidate pool. We enumerate all 140-choose-2 = 9,730 binary pairs from the v2 FDA candidates, predict each at 6 mol/kg total (3+3 mol/kg each) using the additive `max` rule, and Pareto-rank by composite (low toxicity, high permeability, low IRI, with a small uncertainty penalty).

Top-10 of the resulting list (full ranking in `[results/mixtures/top20_pairs.csv](results/mixtures/top20_pairs.csv)`; ≈7,750 pairs from the v3-filtered pool in `[results/mixtures/all_pairs_scored.csv](results/mixtures/all_pairs_scored.csv)`):


| #   | Compound A          | Compound B           | Pred tox | Pred perm | Pred IRI | Composite | Notes                                            |
| --- | ------------------- | -------------------- | -------- | --------- | -------- | --------- | ------------------------------------------------ |
| 1   | **DMSO**            | **Propylene glycol** | 13.6     | 50.3      | 69.0     | 0.641     | **real CPA combination, used in cryomicroscopy** |
| 2   | Isoleucine          | Phenylalanine        | 54.6     | 47.7      | 25.1     | 0.638     | DOLMEN training; memorization                    |
| 3   | Ethanol             | Butylene glycol      | 30.5     | 57.7      | 67.8     | 0.630     | small alcohol + diol; CPA-adjacent               |
| 4   | Phenylalanine       | Tryptophan           | 52.9     | 43.5      | 23.9     | 0.628     | DOLMEN training; memorization                    |
| 5   | Butylene glycol     | Fatty acid esters    | 38.5     | 63.2      | 69.6     | 0.626     | diol + esters; ambiguous IID labels              |
| 6   | Fatty acid esters   | Isopropyl alcohol    | 38.5     | 63.0      | 68.1     | 0.625     | esters + IPA; CPA-adjacent                       |
| 7   | Isoleucine          | Tryptophan           | 54.6     | 47.4      | 29.2     | 0.624     | DOLMEN training; memorization                    |
| 8   | **Ethanol**         | **Propylene glycol** | 28.3     | 55.4      | 68.0     | 0.624     | **real CPA-adjacent cocktail**                   |
| 9   | **Butylene glycol** | **Propylene glycol** | 30.5     | 57.1      | 68.8     | 0.623     | **two related diols; plausible mixture**         |
| 10  | n-Butanol           | Butylene glycol      | 30.6     | 54.1      | 63.8     | 0.623     | small alcohol + diol; CPA-adjacent               |


**Major shift from v2.1 → v3**: the v2.1 filter dropped benzaldehyde (which dominated 11 of the v2.1 top-20 mixture pairs), so the v3 list is now **dominated by alcohol/diol pairs** that are plausible CPA-adjacent chemistry. **DMSO + propylene glycol stays at the top**, three DOLMEN amino-acid pairs are memorization (#2, #4, #7), and the rest is ethanol / propanol / butanol / propylene glycol / butylene glycol combinations that look like the kind of pairs a cryomicroscopy lab would actually mix.

**DMSO + propylene glycol at rank 1** is the strongest pair in the list, and worth being precise about what this means as a finding:

- DMSO and propylene glycol are **both in the single-compound training set** (Higgins Dec 2025 measures both at 3, 6, 12 mol/kg). The model is rating them individually as low-toxicity / high-permeability because it memorized them.
- The additive `max` rule then combines two memorized predictions and pulls their pair to the top of the FDA mixture-pair ranking.
- So this is **not** "the model rediscovered a CPA cocktail it had never seen." It's "two memorized single-compound predictions, combined under a fixed additive rule, recover a known-good combination."

That's a useful pipeline-self-consistency check (the chained inference doesn't break or surface garbage at the top), but it's NOT evidence the model has learned anything about mixtures. A genuine mixture-aware result would require predicting mixture viability separately from individual viabilities and capturing the interaction term that an additive rule cannot. The PairEncoder experiment in v4 (see ITERATION_LOG.md) is the first cut at that.

What changed in v3: v2.1's top-20 had 6 model errors (benzaldehyde, phenol, H₂O₂, formaldehyde, etc.) propagating from single-compound predictions. The v2.1 filter eliminated those, so v3's top-10 is alcohol/diol pairs (plausible but mostly memorization-based) plus three DOLMEN amino-acid pairs (also memorization). Zero clear errors but also zero genuine novel mixture signal.

### Day-one ask at Until (mixture data)

In roughly this order:

1. The Higgins Dec 2025 supplementary numeric tables (if/when published; the bioRxiv preprint only has the bar charts).
2. Until's internal binary-mixture screens (presumably the robots have run hundreds beyond what's public).
3. Literature compilation of CPA cocktail viability from Fahy's group, Mazur's group, and the cryoEM community.

With 200-500 binary mixture rows, the `PairEncoder` architecture in `src/models/mixture.py` becomes trainable and you can actually test the formamide+glycerol-style neutralization story end-to-end. That's the v2.2 deliverable.

---

## v3: filter v2.1, novel-only top-20, Tox21 aux weight sweep

After the v2 + v2.1 work landed, three loose ends remained that were each cheap to address. v3 closes them.

### Filter v2.1 (tighter CPA filter)

The v2 filter dropped the pool from 435 to 140 but two model errors survived in the top-20: **carbon dioxide** (#18, only 3 heavy atoms) and **benzenesulfonic acid** (#7, single sulfonate is strongly acidic). The v2.1 mixture pair scoring made the same problem more visible: **benzaldehyde** dominated 11 of 20 mixture pairs because the v2 model rates it low-toxicity at 3 mol/kg despite being a known irritant in vivo, and **phenol / hydrogen peroxide / formaldehyde / ethylene oxide** all appear in the top mixture pairs.

These are not "the model is wrong about CPAs" failures; they're "the candidate filter let in compounds that aren't in the CPA design space at all". The v2 filter can't catch them because each of these passes the basic v2 criteria (MW < 350, logP < 1.5, polar enough). The fix is in the candidate filter, not in the model.

`_passes_cpa_filter_v21` in `[src/data/fda_iid.py](src/data/fda_iid.py)` layers the following on top of v2:


| Rule                                | Drops                                                                             | Keeps                                                                                                                                          |
| ----------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `heavy_atoms >= 3`                  | H₂O₂, formaldehyde                                                                | ethanol (3), formamide (3), urea (4), DMSO (4)                                                                                                 |
| at least 1 hydrogen on the molecule | CO₂ (purely inorganic)                                                            | everything else                                                                                                                                |
| no sulfonate (was `<=1` in v2)      | benzenesulfonic acid                                                              | non-sulfonate molecules                                                                                                                        |
| no epoxide ring                     | ethylene oxide                                                                    | rest of small ring chemistry                                                                                                                   |
| if aromatic, `HBD + HBA >= 3`       | phenol (1+1), benzaldehyde (0+1), benzyl alcohol (1+1), phenylethyl alcohol (1+1) | phenylalanine (2+2), histidine (3+3), tryptophan (3+2), niacinamide (1+3), gentisic acid (3+4), saccharin (1+3), benzoic acid (1+2 borderline) |


Hand-test verified on 26 cases (DMSO/glycerol/urea/ethanol/glucose/all amino acids → pass; CO₂/H₂O₂/formaldehyde/ethylene oxide/benzenesulfonic acid/phenol/benzaldehyde/benzyl alcohol/phenylethyl alcohol → fail).

**Result on Colab**: pool dropped from 140 → **125** (predicted ≈110-120, slightly higher because the v2 pool already had relatively few of the borderline cases). Top-20 single-compound has **zero clear model errors** (no CO₂, no benzenesulfonic acid, no phenol, no benzaldehyde) and gains **5 known/canonical CPAs** (urea, ethanol, n-propanol, n-butanol, dimethylacetamide), up from 4 in v2. The amino-acid memorization stays (5 amino acids; was 6 in v2) plus 2 lysines and methionine that the v2 filter was crowding out. The top-10 mixture pairs change is even more dramatic: benzaldehyde drops from 11 of 20 to zero, replaced by alcohol/diol pairs that are real CPA-adjacent chemistry.

**v3 single-compound top-20** (full list in `[results/candidates/top20.csv](results/candidates/top20.csv)`):


| #   | Compound                  | Notes                                |
| --- | ------------------------- | ------------------------------------ |
| 1   | Phenylalanine             | DOLMEN training                      |
| 2   | Tryptophan                | DOLMEN training                      |
| 3   | Niacinamide               | biocompatible vitamin                |
| 4   | Saccharin                 | sweetener                            |
| 5   | **Urea**                  | **known CPA**                        |
| 6   | Histidine                 | DOLMEN training                      |
| 7   | Phenoxyethanol            | preservative                         |
| 8   | Gentisic acid             | antioxidant                          |
| 9   | Dehydroacetic acid        | preservative                         |
| 10  | Arginine                  | DOLMEN training                      |
| 11  | Lysine monohydrate        | amino acid (in training as lysine)   |
| 12  | **Ethanol** ("Alcohol")   | **known CPA**                        |
| 13  | **n-Propanol**            | **known CPA**                        |
| 14  | Acetone                   | known cryomicroscopy solvent (≈ CPA) |
| 15  | Lysine                    | amino acid                           |
| 16  | Valine                    | DOLMEN training                      |
| 17  | **n-Butanol**             | **CPA-adjacent**                     |
| 18  | Methionine                | amino acid                           |
| 19  | **N,N-Dimethylacetamide** | **known CPA, in training**           |
| 20  | Fatty acid esters         | generic IID listing                  |


Five known CPAs (urea, ethanol, n-propanol, acetone, n-butanol; plus DMA which is in training so technically memorization but worth flagging). Five trained-DOLMEN amino acids (memorization). Five plausible candidates (niacinamide, saccharin, phenoxyethanol, gentisic acid, dehydroacetic acid). One ambiguous IID label (fatty acid esters). **Zero clear errors.** Best top-20 list this project produced.

### Novel-only top-20

Roughly 27 of the 140 v2 candidates have a SMILES that already appears in the DOLMEN or Higgins training set (mostly the 6 amino acids in DOLMEN: phenylalanine, tryptophan, histidine, arginine, valine, isoleucine; plus urea, ethanol, propanol, butanol, formamide which double as both training and candidate). For those, the model's score is partially memorization, not generalization. For wet-lab triage the more useful list is the top-20 of the **113 novel** candidates: compounds the model has never seen before that it ranks highly.

`[src/analyze_novelty.py](src/analyze_novelty.py)` reads `all_scored.csv`, filters by training-SMILES set, re-ranks, and writes `results/candidates/top20_novel.csv`. Run by the Colab cell at the end of section 7. The novel top-20 is the right list to send to the wet lab; the original top-20 is the right list to evaluate model self-consistency.

**v3 novel-only top-20** (full list in `[results/candidates/top20_novel.csv](results/candidates/top20_novel.csv)`):


| #   | Compound               | Notes                                                                                      |
| --- | ---------------------- | ------------------------------------------------------------------------------------------ |
| 1   | Niacinamide            | biocompatible vitamin                                                                      |
| 2   | Saccharin              | sweetener                                                                                  |
| 3   | Urea                   | shows as novel due to FDA SMILES vs training canonicalization mismatch; really in training |
| 4   | "Wax" (FDA generic)    | ambiguous label                                                                            |
| 5   | Aspartame              | sweetener with amino acid character                                                        |
| 6   | Phenoxyethanol         | preservative                                                                               |
| 7   | o-Tolyl biguanide      | small heterocycle                                                                          |
| 8   | Gentisic acid          | antioxidant                                                                                |
| 9   | Benzocaine             | local anesthetic                                                                           |
| 10  | Dehydroacetic acid     | preservative                                                                               |
| 11  | Methylparaben          | preservative                                                                               |
| 12  | Maltol                 | natural flavor / chelator                                                                  |
| 13  | Lysine monohydrate     | amino acid                                                                                 |
| 14  | Ethanol ("Alcohol")    | known CPA                                                                                  |
| 15  | Methyl salicylate      | small aromatic ester                                                                       |
| 16  | Propyl gallate         | antioxidant                                                                                |
| 17  | n-Propanol             | known CPA                                                                                  |
| 18  | Diazolidinyl urea      | preservative (urea derivative)                                                             |
| 19  | Glyceryl monocaprylate | surfactant                                                                                 |
| 20  | Acetonitrile           | small polar nitrile                                                                        |


Of 125 v3-filtered candidates, 12 (9.6%) overlap with training; the novel list is built from the 113 remaining. Three caveats:

- **Some "novel" entries are actually in training** under different canonicalization (e.g. urea, ethanol, propanol). The novelty filter operates on exact canonical SMILES strings; training SMILES with different stereochemistry tags or salt forms appear as separate molecules. A more semantic version would match by InChI or scaffold; v1 is a literal-string comparison and the over-counting is documented honestly.
- **No clear model errors in this list either.** Diazolidinyl urea (#18) is a formaldehyde-releasing preservative that wouldn't make a great CPA, but it's borderline; everything else is plausible chemistry.
- **The list is more diverse than the original top-20** because the amino-acid memorization is filtered out. For wet-lab triage, this is the more useful artifact.

### Tox21 aux weight sweep

The v2 hypothesis was "Tox21 aux head should boost ChemBERTa toxicity". Actual v2 result was 0.181 (vs no-aux 0.217), a 0.04 drop within noise at n=50. That doesn't cleanly say the aux head fails; it could be the chosen weight (0.1) is too high, too low, or just a noise sample. The sweep at 6 weights `{0.0, 0.05, 0.1, 0.2, 0.5, 1.0}` (`[src/sweep_tox21_aux.py](src/sweep_tox21_aux.py)`) settles the question.

**Result on Colab**: across the multi-task picture, the best aux weight is effectively **0.0**. The toxicity-only column is non-monotonic (low weights hurt, high weights recover to roughly no-aux), but permeability and IRI both degrade monotonically with weight, so the multi-task verdict is clear.


| aux_weight        | toxicity Spearman | permeability Spearman | iri Spearman |
| ----------------- | ----------------- | --------------------- | ------------ |
| **0.00 (no aux)** | **0.148**         | **0.035**             | **0.417**    |
| 0.05              | 0.110             | 0.015                 | 0.404        |
| 0.10              | 0.105             | 0.015                 | 0.405        |
| 0.20              | 0.110             | 0.026                 | 0.405        |
| 0.50              | 0.152             | -0.112                | 0.388        |
| 1.00              | 0.147             | -0.247                | 0.373        |


(Numbers from `results/sweeps/tox21_aux_sweep.csv`; ChemBERTa single-seed 5-fold CV OOF. At n=50 the standard error on toxicity Spearman is ≈0.13, so all six toxicity numbers are statistically indistinguishable from each other and from no-aux. Two consecutive runs of the same sweep gave 0.5-weight toxicity of 0.133 and 0.152; this is the noise floor.)

**Verdict**: the Tox21 signal does not transfer to CPA cytotoxicity at this training scale, and the multi-task picture (where permeability collapses at high aux weights and IRI degrades steadily) confirms it. The toxicity column alone is noise-dominated and cherry-picking the best toxicity row would be misleading. The honest read of the full table is: **any non-zero aux weight either hurts toxicity (low) or destroys permeability (high)**; no recipe makes Tox21 a net positive at this scale.

This is consistent with the structural mismatch between the two tasks: Tox21 measures nuclear-receptor binding and stress-response activation at submicromolar concentrations on hepatocytes, while CPA cytotoxicity is bulk cell death at multi-molar concentrations on endothelial cells. The encoder representations that help one don't help the other. If the next iteration wanted broader toxicity signal, the right move is a different auxiliary task (DrugBank toxicity, in vivo LD50 datasets, or Until's internal screens), not Tox21 with a different weight.

---

## v4: query-by-committee, PairEncoder training, honest framing

After v3 shipped I went back through the candidate list with a domain-skeptical eye and identified four real gaps: (1) the "DMSO + propylene glycol rediscovery" framing oversold a result that's mostly memorization-based, (2) the candidate top-20 has compounds defaulting to training-mean predictions because they're OOD chemistry the model has no signal on, (3) the 5-seed deep ensembles capture within-architecture uncertainty but say nothing about model-class disagreement, and (4) the PairEncoder was documented but never trained.

### Query-by-committee disagreement (`src/qbc.py`)

The 5-seed deep ensembles in `src/models/ensemble.py` give epistemic uncertainty WITHIN each architecture (variance across seeds of the same model class). They cannot capture the kind of uncertainty that comes from **model-class disagreement**. RF on Morgan fingerprints and ChemBERTa-LoRA encode molecular similarity differently, and where they disagree on a candidate's predicted toxicity, neither one is necessarily right. The QbC module reads `all_scored.csv` with both architectures' predictions side-by-side, computes per-task disagreement, z-scores across the three tasks, and ranks candidates by L2-norm disagreement.

The output (`results/candidates/top_disagreement.csv`) is the right "next to test" list for active learning. If both architectures agree a compound is good or bad, a wet-lab measurement on it is mostly redundant. If they disagree by 40 percentage points on toxicity, screening that compound resolves the disagreement and constrains both models for the next training cycle.

**Top-3 from local smoke** (full pipeline runs in Colab):


| Rank | Compound             | RF tox | ChemBERTa tox | Disagreement | Why it matters                                                                                                                                                          |
| ---- | -------------------- | ------ | ------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | **DMSO**             | 14.2   | 58.5          | 44.4 pp      | RF correctly assigns low toxicity (DMSO is the canonical CPA, in training); ChemBERTa just predicts the training mean. The QbC metric automatically surfaces this miss. |
| 2    | **Propylene glycol** | 23.1   | 54.8          | 31.8 pp      | Same pattern: real CPA, in training, RF gets it right, ChemBERTa defaults to mean.                                                                                      |
| 3    | N-acetyl-D-alanine   | 59.7   | 61.5          | small on tox | Big disagreement on IRI (94 vs 52), permeability (21 vs 25).                                                                                                            |


Two takeaways: (a) the QbC metric rediscovers the OOD-mean-default failure mode automatically without us having to label it, and (b) for actively-learning compounds the model has signal on, the disagreement-ranked list is a better acquisition function than composite-score ranking.

### PairEncoder training on the 170-mixture dataset (`src/models/mixture.py:train_pair_encoder_loo`)

v2.1 documented the PairEncoder architecture but said the mixture set was too small to train. v4.1 expands the dataset from 36 to 170 mixture rows by transcribing the full Figures 3-4 grids at 300 DPI (covering all 87 mixtures at 6 mol/kg and all 83 at 12 mol/kg). With ≈10x more training data, the PairEncoder should at least be competitive with the additive baseline:


| Model                                         | Spearman vs measured viability | MAE  | R²        |
| --------------------------------------------- | ------------------------------ | ---- | --------- |
| Constant predictor (always mean mortality)    | 0.00                           | ≈22  | 0.00      |
| Additive baseline (max rule)                  | +0.18                          | 20.4 | -0.08     |
| Additive baseline (mean rule)                 | +0.22                          | 20.2 | -0.18     |
| **PairEncoder LOO (n=170, full mixture set)** | **−0.26**                      | 42.0 | **−0.48** |


The PairEncoder still fails badly even with 170 mixture rows. Spearman is **−0.26** vs the additive mean baseline at **+0.59**. This is informative: more data didn't fix it because the architecture is wrong for this regime. The Morgan-fingerprint-based pair encoder has 4×1024 + 2 = 4098 features and 170 rows, so the model is heavy on dimensionality vs sample count, AND it has no direct access to the per-component concentration response that the single-compound model learned from the 50 single-compound training rows. It's trying to relearn the per-compound dose-response from mixture data, which is much harder than just consuming it.

Why: with 16 rows all sharing glycerol as one component, the model can only learn "what does compound X (paired with glycerol) do to toxicity," and at 6 vs 12 mol/kg total, the same compound flips behavior dramatically (formamide neutralizes at 12, propylene glycol becomes lethal). 16 rows can't capture both regimes.

**This is the inverse of what I predicted.** I expected the PairEncoder to be data-starved at 16 rows and to recover at 170. It didn't. More mixture data fixed the additive baseline (Spearman 0.13 → 0.59) but not the PairEncoder, so the bottleneck is the architecture, not the row count. The Morgan-fingerprint pair encoder has 4×1024 + 2 = 4098 features against 170 rows, and it has no direct access to the per-component dose-response that the single-compound model already learned. Asking it to relearn dose-response from mixture data is much harder than letting it consume the single-compound predictions as a baseline. The right v5 is a residual learner on top of the additive prediction. A learned interaction term still wants more cross-component diversity (not just glycerol-paired) and broader concentration coverage (not just two values), so more mixture data from Until's internal screens is still the day-one ask, but the v5 architecture change is the bigger lever.

The neural PairEncoder architecture (shared ChemBERTa-LoRA encoder + symmetric features + interaction term + concentration features) is documented in `_pair_encoder_neural_sketch` for when the data scales.

### Honest framing fixes

Two readings of the v3 candidate list got tightened:

- **DMSO + propylene glycol at #1 of mixture pairs is partially memorization.** Both compounds are in single-compound training; the additive baseline combines two memorized predictions. Useful pipeline-self-consistency check, but not "rediscovery from no mixture training labels." Documented inline in the FDA mixture pair scoring section above.
- **The OOD-mean-default failure** explains why niacinamide / saccharin / aspartame / phenoxyethanol / dehydroacetic acid / benzocaine / methylparaben rank highly in the candidate top-20 despite not being real CPAs. The toxicity training set is 22 small alcohols/polyols/amides/sulfoxides; for chemistry outside that distribution, the model converges to the training mean (≈55-60 mortality, ≈24-27 permeability, ≈35-50 IRI), and whichever OOD compound happens to land closest to "ideal mean" gets ranked. This isn't a bug fixable by filtering; it requires more diverse toxicity training data. Documented in the candidate-list section above.

### Tier 1 polish

- **SMILES pre-resolved** in `data/raw/higgins_jan2025.csv` and `higgins_dec2025.csv`. CSVs are now self-contained without PubChem network access.
- **Small processed parquets tracked** in git (`audit.json`, `long.parquet`, `dolmen.parquet`, all the `higgins_*.parquet`, `fda_iid_candidates.parquet`). Reviewers can inspect what the pipeline produces without running it. Tox21 (208 KB) stays gitignored since it's auto-downloaded.
- **Pareto figures cleaned up**: 2D plot now uses `adjustText` for collision-free top-5 labels with leader lines; 3D plot uses darker gray (#7d7d7d) for dominated points so they're visible against the white background.
- **Time/cost estimates removed** from notebook + .py files. The "set the GPU runtime" patronizing notes are gone.

---

## Limitations

In the same spirit:

- **OOD chemistry defaults to training-mean predictions.** The 22-compound toxicity training set covers small alcohols, polyols, amides, sulfoxides. For anything outside that distribution (aromatics without amino-acid context, sulfonates, organomercurials, etc.) the model has no signal and predictions converge to the training mean. This makes the candidate top-20 partly a list of "OOD compounds whose mean-default predictions happen to look CPA-shaped." Filtering can't fix this; only more diverse toxicity training data can. The QbC disagreement list is a better artifact for active learning because it automatically flags the OOD cases.
- **CPA-like physicochemical filter is now at v2.1**, addressing the CO₂ / benzenesulfonic-acid / phenol / benzaldehyde leaks from v2. Remaining filter gaps will be visible after the next Colab run; document them honestly when they appear.
- **Small-task data is the bottleneck for ChemBERTa under cluster splits.** ChemBERTa cluster-ensemble toxicity Spearman is 0.18 (vs RF's 0.64). At ≈10 toxicity training compounds per fold, the pretrained model's adapter overfits to spurious correlations. RF on Morgan FPs is more robust here because the inductive prior (Tanimoto similarity in feature space ≈ structural similarity) approximates what cluster-aware splits enforce. The v3 Tox21-aux weight sweep settled this: at this scale, no aux weight makes Tox21 a net positive.
- **Toxicity is from one paper, one cell type, one temperature.** Higgins's Dec 2025 data uses bovine pulmonary artery endothelial cells (BPAEC) at 4 °C with 30 min exposure. Real organ cryopreservation involves multiple cell types, longer exposure, and cooling rates.
- **The Higgins viability values are read from bar charts.** ±5 percentage points precision. The publisher does not provide a numeric data table. v4.1 transcribes 50 single-compound rows + 170 binary mixtures (87 at 6 mol/kg, 83 at 12 mol/kg) from PDF-rendered Figures 2-4 at 300 DPI; the bars at 300 DPI are clean enough to read against the 0.2-spaced gridlines including the small near-zero ones. The 12 mol/kg count is one over the paper's claimed 82, almost certainly because I misread one near-zero label as a labeled bar instead of a missing one. Switching to authors' raw values when published would tighten precision but probably wouldn't shift the model conclusions.
- **Mixture-aware model needs an architecture rethink, not just more data.** v4.1 retrained the PairEncoder on the full 170-row mixture set (87 + 83 from Figures 3-4) and got Spearman −0.26, still worse than the additive mean baseline (+0.59). The Morgan-FP pair encoder has too many features for the row count and lacks direct access to the per-compound dose-response signal the single-compound model already extracted. The right v5 architecture is a **residual learner**: take the additive baseline prediction as a feature (or a baseline to regress against), and have the pair encoder learn only the interaction term (departure from additive). At a 0.59 Spearman floor from the additive baseline, even a small consistent improvement on the residual would be a real learned interaction term. v5 is queued.
- **No molecular-dynamics features.** Until Labs explicitly couples atomic-scale MD to cellular-scale wet-lab screens; this repo is wet-lab data only. MD-derived hydration metrics (water displacement, H-bond disruption, glass-transition predictions) would be a natural complementary feature set. Discussed in [DESIGN_DOC.md](DESIGN_DOC.md).
- **No wet-lab validation.** The Pareto top-20 is a recommendation list, not validated predictions. Closing the loop is also in [DESIGN_DOC.md](DESIGN_DOC.md).

---

## What I'd build next (ranked by expected impact)

Sketched briefly here; full architectural detail in [DESIGN_DOC.md](DESIGN_DOC.md).


| #   | Improvement                                                                                                                                      | Why it matters                                                                              | Status                                                                                                            |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 1   | **Mixture-aware architecture** (pair-of-SMILES → learned interaction term, trained on Higgins Dec 2025 binary mixtures + literature compilation) | Real CPAs are mixtures; current model can't predict toxicity neutralization                 | **partially shipped in v2.1**: additive baseline analysis + PairEncoder architecture; training gated on more data |
| 2   | **Concentration as an input feature** (dose-response prediction instead of point estimates)                                                      | Higgins Dec 2025 has rich dose-response; v1 throws it away                                  | **shipped in v2 (see below)**                                                                                     |
| 3   | **Tighter CPA-like physicochemical filter**                                                                                                      | Removes dyes, biologics, organomercurials from the candidate pool, surfaces real candidates | **shipped in v2; v2.1 filter update in v3**                                                                       |
| 4   | **Tox21 auxiliary classification head** (already wired in code, gated by DeepChem install)                                                       | Adds ≈7,800 broader-toxicity training signal; addresses the n=22 toxicity bottleneck        | **shipped in v2; weight sweep in v3**                                                                             |
| 5   | **MD-derived auxiliary features** (water displacement, H-bond disruption, predicted T_g)                                                         | Complements wet-lab data with atomic-scale signal; matches Until's stated approach          | not started; needs MD compute infra                                                                               |
| 6   | **Closed-loop active learning** (uncertainty-weighted EI acquisition; model recommends next-batch compounds, robot tests, results retrain)       | Where Until's pipeline lives                                                                | not started                                                                                                       |
| 7   | **Graph neural network ablation** (e.g. AttentiveFP / D-MPNN)                                                                                    | Tests whether SMILES sequence features are the limit, vs explicit graph topology            | not started                                                                                                       |
| 8   | **LoRA rank sweep** ({4, 8, 16, 32}) under stable training                                                                                       | Diagnostic: at what rank does small-task transfer flip from "helps" to "memorizes"?         | not started                                                                                                       |


The three items marked v2 above are documented in the next section. Items 1 and 5 are the genuinely interesting ones for a job conversation, and the ones I'd do first if I had multi-week scope at Until.

---

## v2: changes I made after looking at the v1 outputs

The original plan had a clean Phase 1 → Phase 2 → Phase 3 structure and that's what shipped first. After Phase 3 was done I went back through `all_scored.csv`, the per-task results table, and the Higgins Dec 2025 raw data, and three specific problems jumped out. I implemented fixes for all three. They're the items above marked "shipped in v2".

The full hypothesis-by-hypothesis trail (predictions, results, calibration record across versions) lives in [ITERATION_LOG.md](ITERATION_LOG.md). v2 was 2/3 right; the one that didn't pan out (Tox21 aux head boosting ChemBERTa toxicity) is documented honestly there.

### What I noticed and what I changed

**1. The toxicity model was throwing away the dose-response data.** Higgins Dec 2025 measures viability for each compound at 3, 6, and 12 mol/kg. This is the most interesting structure in the dataset: dimethylacetamide goes from viability 102% at 3 mol/kg → 0% at 6 mol/kg → 0% at 12 mol/kg, and formamide+glycerol's neutralization story is *entirely* about dose. v1 averaged across concentrations, collapsing this to one toxicity value per compound (DMA → 34%, which is meaningless). v2 emits one row per (compound, concentration) measurement and threads concentration through both architectures as an explicit input. Toxicity training expanded from 22 unique compounds to 50 (compound, concentration) pairs.

For the **Random Forest** baseline: concentration is appended as the last feature in the toxicity feature vector. For the other tasks the assay concentration is constant, so the per-task RF naturally ignores it (zero variance feature → no split improvement).

For **ChemBERTa**: the toxicity head takes (pooled, scalar concentration) instead of just pooled. IRI / permeability heads keep the original signature. Concentration is z-scored against the toxicity range (mean 6, std 3) before concatenation so the head sees values in a reasonable scale.

For **FDA scoring** (where there's no measured concentration), all candidates are predicted at the reference concentration **6 mol/kg**, the mid-range from Higgins Dec 2025 and the inflection point in the formamide/glycerol story.

**Result on the actual Colab run** (Blackwell, full 5-seed cluster ensemble + 5-fold CV, all v2 features active):


| Task            | v1 Spearman (random 5-fold) | v2 Spearman (random 5-fold) | v1 (cluster 5-seed) | v2 (cluster 5-seed) |
| --------------- | --------------------------- | --------------------------- | ------------------- | ------------------- |
| RF iri          | 0.510                       | 0.511                       | 0.505               | 0.499               |
| **RF toxicity** | 0.347                       | **0.527** (+0.18)           | 0.459               | **0.643** (+0.18)   |
| RF permeability | 0.126                       | 0.100                       | 0.353               | 0.368               |


The **+0.18 jump on toxicity Spearman** is the largest single intervention in the project, and it holds consistently across both random and cluster splits. R² also improved (cluster: 0.22 → 0.32). IRI and permeability didn't move (their assay concentration is fixed in training, so concentration as a feature carries no information for those tasks). The toxicity OOF n is now 50 (per (compound, concentration) measurement) instead of 22, so this is a *harder* metric than v1 measured: the model has to capture dose-response, not just compound identity, and the Spearman is reported per-measurement rather than per-compound.

**2. The candidate filter let polysulfonated dyes, organomercurials, and benzyl benzoate into the top-20.** v1's filter was `MW < 500 AND (HBD ≥ 1 OR HBA ≥ 2)`. Sulfonate groups push HBA way past 2, so FD&C Blue No. 2 (MW=466 with two sulfonates) sailed through. v1 also rejected DMSO incorrectly (HBA=1, HBD=0 fails the OR criterion), even though DMSO is the canonical CPA. Both bugs are fixed in v2.

The new filter (`_passes_cpa_filter_v2` in `[src/data/fda_iid.py](src/data/fda_iid.py)`) is closer to actual CPA chemistry:

- MW in [30, 350] (lower bound rules out ions; upper keeps sucrose/trehalose at 342)
- logP < 1.5 (CPAs are hydrophilic; benzyl benzoate's logP ≈4 fails)
- polarity: HBD ≥ 1 OR HBA ≥ 1 OR TPSA > 15 Å² (DMSO passes via TPSA=36)
- elements ∈ {H, C, N, O, S} only (excludes phenylmercuric acetate, halogenated phenols, phosphates)
- ring count ≤ 2 (excludes fused-ring polyaromatic dyes)
- no azo group (`[#7]=[#7]` SMARTS)
- ≤ 1 sulfonate group (reject di- and tri-sulfonate dye chemistry)

Hand-test on the v1 problem cases (verified in `_passes_cpa_filter_v2` unit cases): DMSO ✓, glycerol ✓, urea ✓, formamide ✓, glucose ✓, phenylalanine ✓, histidine ✓, dodecane ✗ (logP), 1-(phenylazo)-2-naphthylamine ✗ (azo), phenylmercuric acetate ✗ (Hg), FD&C Blue No. 2 ✗ (multi-sulfonate + ring count), benzyl benzoate ✗ (logP).

**Result on the actual FDA IID candidate pool**: pool size dropped from **435 → 140** compounds. The top-20 lost FD&C Blue No. 2, D&C Red No. 33, 1-(phenylazo)-2-naphthylamine, phenylmercuric acetate, metaphosphoric acid, and the sodium-salt entries. It gained urea (still there from v1) plus ethanol, n-propanol, and n-butanol (real CPAs the v1 pool was crowding out). Two errors did survive v2 (carbon dioxide at #18 and benzenesulfonic acid at #7); see the candidate-list discussion above for the proposed v2.1 fixes.

**3. The Tox21 aux head was wired in the architecture but never trained.** ChemBERTa-LoRA's `tox21_aux=False` was the default because installing DeepChem on Python 3.12 was broken (no wheel as of April 2026; building from source was a rabbit hole I didn't open). v2 replaces the DeepChem dependency with a direct CSV download from the DeepChem GitHub mirror (`[src/data/tox21.py](src/data/tox21.py)`). The CSV is the same data DeepChem would have given me; I parse it, canonicalize SMILES with RDKit, and emit a long-format parquet with 79K (compound, task) labels across 7,823 unique compounds and 12 binary toxicity assays.

The aux head training recipe (in `[src/models/chemberta_lora.py](src/models/chemberta_lora.py)`):

- Total loss = `cpa_huber_loss + 0.1 * tox21_bce_loss`
- The 0.1 weight is conservative on purpose: I want the aux to regularize the encoder, not dominate the gradient. Tox21 has 24× more compounds than the CPA training set (7800 vs 326), so without down-weighting it would push the encoder toward Tox21-shaped representations.
- BCE is masked: missing `(compound, task)` cells in Tox21 are unmeasured (DeepChem treats them as zero-weight), so the BCE term ignores them.
- The same `nan_to_num` + `clip_grad_value_(1.0)` safety nets used for the CPA loss apply to the combined loss, so a bad aux batch can't poison training the way the v1 NaN-gradient pathology did.

**Result**: ChemBERTa cluster-ensemble toxicity Spearman went from 0.22 (v1, no aux head) to **0.18** (v2, aux head active, weight 0.1). At n=50 the standard error on Spearman is ≈0.13, so this 0.04 drop is within noise; calling it a "win" or a "loss" is overinterpreting at this sample size. The honest read is that at *this* training scale the aux signal is too weak to overcome the basic small-data limitation, and Tox21's nuclear-receptor / stress-response assays at submicromolar concentrations on hepatocytes don't transfer cleanly to bulk cytotoxicity at multi-molar concentrations on endothelial cells (which is what we actually care about). The architecture is in place for follow-up sweeps over weight, training schedule, and aux-task choice; v2 says the obvious starting recipe doesn't pay off.

### Engineering notes worth recording

- The data layer schema change (long-format `concentration_mol_kg` column) is non-breaking for IRI and permeability because they each had a single fixed concentration in the original data; v2 just makes that explicit. Old caches auto-invalidate via the `concentration_mol_kg not in long_df.columns` check.
- ChemBERTa OOF aggregation now keys toxicity predictions by `"smiles|concentration"` strings (multiple per compound) and IRI / permeability by bare smiles. The `_eval_per_task` helper handles both formats.
- The candidate-scoring path predicts at task-specific reference concentrations: 6 mol/kg for toxicity, the assay defaults for IRI / permeability. That's documented in `[src/models/ensemble.py:predict_chemberta_ensemble](src/models/ensemble.py)`.
- For reproducibility, the v1 numbers are still queryable from the `results_table.csv` rows where the toxicity OOF n=22; v2 rows have toxicity OOF n=50.

---

## Repo layout

```
cpa-screening/
├── data/
│   ├── raw/                       # downloaded + hand-curated (gitignored except templates)
│   ├── processed/                 # parquet caches + audit.json (gitignored)
│   └── .cache/                    # PubChem cache + failure log (gitignored)
├── results/
│   ├── figures/                   # parity plots + headline figs (gitignored except 2)
│   ├── candidates/                # FDA top-20 + all_scored
│   ├── summary.json               # most recent run's metrics
│   └── results_table.csv          # cumulative metrics across runs
├── src/
│   ├── data/
│   │   ├── pubchem.py             # CAS-dashed name lookup w/ disk cache
│   │   ├── dolmen.py              # IRI raw CSV download + parse
│   │   ├── higgins.py             # Jan + Dec 2025 loaders (templates + parsing)
│   │   ├── tox21.py               # MoleculeNet direct CSV (no DeepChem dep); aux head active in v2
│   │   ├── fda_iid.py             # FDA IID download (zip-aware) + CPA filter
│   │   ├── splits.py              # random + Tanimoto cluster k-fold + LOO
│   │   └── build.py               # consolidated long-format + audit
│   ├── models/
│   │   ├── rf_baseline.py         # Morgan + descriptors + RF (concentration-aware in v2)
│   │   ├── chemberta_lora.py      # ChemBERTa-2 + LoRA + multi-task heads + Tox21 aux
│   │   ├── ensemble.py            # 5-seed deep ensemble + conformal cal
│   │   └── mixture.py             # additive baseline + PairEncoder architecture (v2.1)
│   ├── train.py                   # argparse entry; --model/--n-seeds/--split-mode/--tox21-aux
│   ├── score_candidates.py        # Pareto top-K single-compound from FDA IID
│   ├── score_mixtures.py          # additive baseline eval + Pareto top-K binary pairs (v2.1)
│   ├── analyze_novelty.py         # filter all_scored to compounds NOT in training (v3)
│   ├── sweep_tox21_aux.py         # ChemBERTa Tox21 aux-weight sweep (v3)
│   ├── qbc.py                     # query-by-committee: RF/ChemBERTa disagreement ranking (v4)
│   ├── eval.py                    # metrics + parity plots
│   ├── figures.py                 # README-quality figures
│   └── utils.py                   # paths, seeding, canonical SMILES
├── colab_runner.ipynb             # end-to-end Colab runner
├── DESIGN_DOC.md                  # what I'd build at Until next
├── requirements.txt               # base deps (rdkit, pandas, sklearn, ...)
├── requirements-deep.txt          # torch / transformers / peft / torchao
└── README.md
```

## Citations

**Datasets and prior wet-lab work**

- DOLMEN: Warren, M. et al. Data-driven Discovery of Potent Small Molecule Ice Recrystallisation Inhibitors. *Nat Commun* **15**, 7995 (2024). DOI: [10.1038/s41467-024-52266-w](https://doi.org/10.1038/s41467-024-52266-w)
- Ahmadkhani, N., Benson, J. D., Eroglu, A. & Higgins, A. Z. High throughput method for simultaneous screening of membrane permeability and toxicity for discovery of new cryoprotective agents. *Sci Rep* **15**, 1862 (2025). DOI: [10.1038/s41598-025-85509-x](https://doi.org/10.1038/s41598-025-85509-x)
- Ahmadkhani, N., Sugden, C., Mayo, A. T. & Higgins, A. Z. Screening for cryoprotective agent toxicity and toxicity reduction in mixtures at subambient temperatures. *Cryobiology* **121**, 105315 (2025). DOI: [10.1016/j.cryobiol.2025.105315](https://doi.org/10.1016/j.cryobiol.2025.105315)
- Tox21: Wu, Z. et al. MoleculeNet: a benchmark for molecular machine learning. *Chem Sci* **9**, 513 (2018). DOI: [10.1039/C7SC02664A](https://doi.org/10.1039/C7SC02664A)

**Methods**

- ChemBERTa-2: Ahmad, W., Simon, E., Chithrananda, S., Grand, G. & Ramsundar, B. ChemBERTa-2: Towards Chemical Foundation Models. [arXiv:2209.01712](https://arxiv.org/abs/2209.01712) (2022)
- LoRA: Hu, E. J. et al. LoRA: Low-Rank Adaptation of Large Language Models. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) (2021)
- EffiChem (precedent for ChemBERTa+LoRA on molecular property prediction): Bernabeu, M. et al. ChemRxiv (2025). DOI: [10.26434/chemrxiv-2025-2lljt](https://doi.org/10.26434/chemrxiv-2025-2lljt)
- Conformal prediction: Angelopoulos, A. N. & Bates, S. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. [arXiv:2107.07511](https://arxiv.org/abs/2107.07511) (2021)
- Butina clustering: Butina, D. Unsupervised data base clustering based on Daylight's fingerprint and Tanimoto similarity. *J Chem Inf Comput Sci* **39**, 747 (1999). DOI: [10.1021/ci9803381](https://doi.org/10.1021/ci9803381)
- FlashAttention-2 (used implicitly via PyTorch SDPA): Dao, T. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) (2023)

**Context for the next-steps section**

- Until Labs Milestone I White Paper: [untillabs.com/blog/milestone-white-paper-i](https://www.untillabs.com/blog/milestone-white-paper-i)
- Until Labs Series A announcement: [untillabs.com/blog/untils-series-a](https://www.untillabs.com/blog/untils-series-a)
- AI for cryopreservation (recent perspective): Cui, Y. et al. AI-driven breakthroughs and future perspectives in cryopreservation. *Cryo Letters / Cryobiology Letters* (2026)
- Active learning in CADD: Reker, D. & Schneider, G. Active-learning strategies in computer-assisted drug discovery. *Drug Discov Today* **20**, 458 (2015). DOI: [10.1016/j.drudis.2014.12.004](https://doi.org/10.1016/j.drudis.2014.12.004)

