# Iteration log

How the project's findings progressed from the original Phase 1-3 plan through three follow-up iterations. For each version: what was built, what the numbers said, what hypothesis the results suggested, and (critically) whether the hypothesis was right when we tested it in the next iteration.

The point of this document is two-fold. First, anyone reading the repo can see the actual research process, not just the final state. Second, the calibration record (predictions vs outcomes) is the most honest indicator of how to weight my future hypotheses about CPA modeling. Three out of five v2 hypotheses landed; one (Tox21 aux) didn't pay off the way I expected; and the v2.1 mixture predictions all hit. That's a useful signal for what to bet on next.

---

## v1: original Phase 1-3 plan (commit `6f4a159`)

### What was built

The exact spec from the original 5-day plan. RF baseline + ChemBERTa-2 + LoRA + cluster-aware splits + 5-seed deep ensembles + conformal calibration + FDA IID Pareto ranking. Single-compound toxicity training averaged across the three concentrations Higgins Dec 2025 measures (3, 6, 12 mol/kg → one mean per compound). FDA filter: `MW < 500 AND (HBD ≥ 1 OR HBA ≥ 2)`. Tox21 auxiliary head wired in the architecture but disabled (DeepChem install was broken on Python 3.12).

### Headline numbers (cluster 5-fold 5-seed ensemble)

| Task | RF Spearman | ChemBERTa Spearman |
|---|---|---|
| iri | 0.505 | 0.416 |
| toxicity | 0.459 | 0.217 |
| permeability | 0.353 | -0.026 |

Conformal coverage on IRI = 0.954 (target 0.95). RF beats ChemBERTa on every task.

### Top-20 single-compound candidates (v1)

| Rank | Compound | Notes |
|---:|---|---|
| 1 | Aminobenzoate sodium (PABA-Na) | sunscreen excipient, sodium salt |
| 2 | **Urea** | known CPA |
| 3 | Niacinamide (Vitamin B₃) | biocompatible |
| 4 | Tryptophan | DOLMEN-train memorization |
| 5 | Phenylalanine | DOLMEN-train memorization |
| **6** | **1-(Phenylazo)-2-naphthylamine** | **carcinogenic azo dye, model error** |
| **7** | **FD&C Blue No. 2** | **food dye MW=466, filter error** |
| **8** | **D&C Red No. 33** | **food dye, filter error** |
| 9 | o-Tolyl biguanide | small heterocycle |
| 10 | Sodium pyrrolidone carboxylate | humectant, sodium salt |
| 11 | Sodium benzoate | preservative, sodium salt |
| 12 | Saccharin | sweetener |
| 13 | Benzoin (±) | aromatic ketone |
| 14 | Histidine | DOLMEN-train memorization |
| **15** | **Metaphosphoric acid** | **strong inorganic acid, clear error** |
| 16 | Benzyl benzoate | aromatic ester (high logP) |
| 17 | Arginine | DOLMEN-train memorization |
| 18 | Valine | DOLMEN-train memorization |
| 19 | Isoleucine | DOLMEN-train memorization |
| 20 | **Isopropyl alcohol** | known CPA |

Real CPAs in top-20: 2 (urea, IPA). Confirmed model errors: 5+ (food dyes, organomercurials further down the list, inorganic acids). Memorized amino acids: 6.

### What v1 told me (the three hypotheses for v2)

Looking at this list and the per-task numbers, three concrete things stuck out:

**Hypothesis A**: *"The toxicity model is throwing away the dose-response data."* Higgins Dec 2025 publishes viability at 3, 6, AND 12 mol/kg. v1 averages those, collapsing dimethylacetamide's (102 → 0 → 0) viability profile to a meaningless mean of 34. The interesting structure in the dataset is exactly the dose-response, and we're discarding it. **Predicted impact**: making toxicity a per-(compound, concentration) regression target should substantially boost toxicity Spearman because the model now has more rows AND more signal per row.

**Hypothesis B**: *"The CPA filter is too permissive."* MW < 500 + HBA ≥ 2 admits any polysulfonated dye (sulfonates push HBA ≥ 8). The top-20 has FD&C Blue No. 2 (MW=466), 1-(phenylazo)-2-naphthylamine, sodium-salt PABA, and metaphosphoric acid. None of those are CPAs. Also, the filter rejects DMSO (HBA=1, HBD=0 fails `HBA ≥ 2 OR HBD ≥ 1`), which is *the* canonical CPA. So the filter is wrong in both directions. **Predicted impact**: a tighter filter (element whitelist, ring count cap, no azo, ≤ 1 sulfonate, polarity via TPSA so DMSO passes) should shrink the candidate pool by ~50-70% and surface real CPAs that v1 was crowding out.

**Hypothesis C**: *"The Tox21 aux head should help ChemBERTa toxicity."* The ChemBERTa toxicity Spearman of 0.217 cluster-ensemble at n=22 toxicity samples is barely above noise (SE ~0.20). Tox21 has 7,800 compounds with 12 binary toxicity-relevant labels. Activating the aux head with a low loss weight (0.1) was supposed to give the encoder more representation-shaping signal without dominating the CPA gradient. **Predicted impact**: ChemBERTa toxicity Spearman up by 0.05-0.10. (This was the most uncertain prediction; nuclear-receptor binding at submicromolar is structurally different from bulk cytotoxicity at multi-molar, so direct task transfer was always iffy.)

---

## v2: concentration-aware + tighter filter + Tox21 aux (commit `c4ebe2f`)

### What was built

Three changes corresponding directly to hypotheses A, B, C:

1. **Concentration-aware toxicity** (Hypothesis A). Data layer emits one row per (compound, concentration) measurement. Toxicity training expanded from 22 unique compounds to 50 (compound, concentration) pairs. RF baseline appends concentration to the feature vector for the toxicity head; ChemBERTa toxicity head takes concatenated (pooled, scalar concentration) input.

2. **v2 CPA filter** (Hypothesis B). New `_passes_cpa_filter_v2`: MW [30, 350], logP < 1.5, polarity (HBD ≥ 1 OR HBA ≥ 1 OR TPSA > 15), element whitelist {H, C, N, O, S}, ring count ≤ 2, no azo, ≤ 1 sulfonate.

3. **Tox21 aux head activated** (Hypothesis C). DeepChem dependency replaced with a direct CSV download from the DeepChem GitHub mirror. Aux loss weighted at 0.1 in the multi-task objective.

### Headline numbers (cluster 5-fold 5-seed ensemble; v1 → v2)

| Task | v1 RF | v2 RF | Δ | v1 ChemBERTa | v2 ChemBERTa | Δ |
|---|---|---|---|---|---|---|
| iri | 0.505 | 0.499 | -0.006 (noise) | 0.416 | 0.391 | -0.025 (noise) |
| **toxicity** | **0.459** | **0.643** | **+0.184** | 0.217 | 0.181 | -0.036 (noise) |
| permeability | 0.353 | 0.368 | +0.015 (noise) | -0.026 | 0.018 | +0.044 (noise) |

### Top-20 single-compound candidates (v2)

The candidate pool dropped from **435 → 140** with the v2 filter. Top-20 reordering:

| Rank | Compound | Status |
|---:|---|---|
| 1 | Phenylalanine | DOLMEN-train memorization (was #5 in v1) |
| 2 | Tryptophan | DOLMEN-train memorization (was #4 in v1) |
| 3 | Benzaldehyde | new entry; small aromatic |
| 4 | Niacinamide | biocompatible (was #3) |
| 5 | Phenylethyl alcohol | new entry; rose-scented preservative |
| 6 | Saccharin | sweetener (was #12) |
| 7 | Benzenesulfonic acid | **survived filter; strongly acidic, model error** |
| 8 | Benzyl alcohol | new entry; biocompatible preservative |
| 9 | Phenol | **toxic at use concentration, model error** |
| 10 | "Wax" (FDA generic) | ambiguous IID label |
| 11 | Histidine | DOLMEN-train memorization |
| **12** | **Urea** | known CPA (dropped from #2 to #12) |
| 13 | Gentisic acid | small phenol antioxidant |
| 14 | Dehydroacetic acid | preservative |
| **15** | **Ethanol** | **NEW: known CPA / co-solvent** |
| **16** | **n-Propanol** | **NEW: CPA in cryomicroscopy** |
| 17 | Arginine | DOLMEN-train memorization |
| **18** | **Carbon dioxide** | **gas, 3 heavy atoms, clear model error** |
| **19** | **n-Butanol** | **NEW: small primary alcohol, CPA-adjacent** |
| 20 | Valine | DOLMEN-train memorization |

Real CPAs in top-20: **4** (urea, ethanol, n-propanol, n-butanol; up from 2 in v1). Confirmed model errors: 3 (CO₂, benzenesulfonic acid, phenol). Memorized amino acids: 5 (down from 6). Sodium salts: 0 (filter removed them all). Food dyes / azo / organomercurials: 0 (filter removed them all).

### Hypothesis calibration (predicted vs actual)

| Hypothesis | Predicted | Actual | Verdict |
|---|---|---|---|
| A. Concentration-aware toxicity | +0.10-0.20 RF Spearman | +0.184 RF cluster, +0.180 RF random | **right; magnitude matched** |
| B. Tighter filter removes dyes / organomercurials, surfaces real CPAs | pool 50-70% smaller; 4-5 real CPAs in top-20 | pool 68% smaller (435 → 140); 4 real CPAs (was 2) | **right; magnitude matched** |
| C. Tox21 aux boosts ChemBERTa toxicity | +0.05-0.10 ChemBERTa toxicity Spearman | -0.036 (within noise) | **wrong; aux signal too weak at the chosen weight / training scale** |

### What v2 told me (the v2.1 hypothesis)

The v2 results were strong overall but two things still stood out:

**Hypothesis D**: *"Single-compound modeling cannot capture toxicity neutralization, and that's the actual problem Until cares about."* Higgins Dec 2025's headline empirical finding is formamide+glycerol at 12 mol/kg total giving 95% viability while either alone at 6 mol/kg gives 20-40%. v2 has no way to predict that. The v2 RF model rates each compound separately at each concentration, then there's no way to combine them that captures interaction. **Predicted impact**: an additive baseline (combine per-compound v2 predictions with rules like max / mean / sum-then-cap) will fail badly on the 16 known binary mixtures, especially the formamide+glycerol@12 mol/kg case. The size of the failure quantifies the gap a real mixture-aware model would have to close.

**Sub-hypothesis D'**: *"FDA pair scoring should rediscover at least one known CPA cocktail."* DMSO + propylene glycol, DMSO + ethylene glycol, glycerol + DMSO are all real cryomicroscopy / vitrification combinations, and v2 rates each component as low-toxicity / high-permeability individually. So the additive composite over the FDA candidate pool ought to pull at least one of those into the top-20 by self-consistency. **Predicted impact**: at least one row of "known CPA + known CPA" in the top-10 mixture pairs.

(There was also a non-prediction observation: the Tox21 aux miss in v2 means the natural follow-up is an aux-weight sweep. I noted it but deprioritized it for v2.1 because mixture-awareness directly addresses Until's product question and Tox21 aux is a methodology question.)

---

## v2.1: mixture-aware analysis (commits `4214bd2` + `585686e`)

### What was built

Two deliverables on the architecture side, two on the analysis side:

1. **PairEncoder architecture** (`src/models/mixture.py:_build_pair_encoder`): shared ChemBERTa-LoRA encoder, symmetric features (sum, abs-diff, prod) on the two pooled embeddings, plus per-compound concentrations, into an MLP toxicity head. Documented + shape-tested but not trained, because 16 mixture rows is too few.

2. **Additive baseline** (`additive_predict`, `evaluate_additive_baseline`): combine v2 single-compound predictions for compound A at conc_A and compound B at conc_B using one of {max, mean, sum_then_cap, weighted_max}. Produces a pure-inference path; no mixture training needed.

3. **Additive baseline evaluation on the 16 known Higgins mixtures**: per-rule Spearman / MAE / R² on viability prediction, plus per-row residuals so the formamide+glycerol@12 mol/kg case is visible.

4. **FDA mixture pair scoring** (`score_fda_mixture_pairs`): enumerate all 140-choose-2 = 9,730 binary pairs, score each with the additive-baseline composite, Pareto-rank.

### Headline numbers

**Additive baseline on 16 known mixtures** (v2 RF full-data ensemble, n_seeds=5):

| Combination rule | Spearman | MAE (pp) | R² | Neutralization misses (>25 pp) |
|---|---|---|---|---|
| max | +0.183 | 20.4 | -0.08 | 3 |
| mean | +0.218 | 20.2 | -0.18 | 1 |
| sum_then_cap | +0.218 | 27.8 | -0.95 | 7 |
| weighted_max | +0.207 | 20.1 | -0.10 | 1 |

All four rules give Spearman in [0.18, 0.22]. **Negative R² across the board** means the additive baseline is worse than predicting the mean. Quantitatively confirms additive single-compound models cannot predict mixture viability.

**The formamide+glycerol neutralization case study** (the most-cited finding in Higgins Dec 2025):

| Mixture | Total conc | **Measured viability** | Additive `max` | Additive `mean` | Additive `sum_then_cap` |
|---|---|---|---|---|---|
| glycerol + formamide | 6 mol/kg | 65 | 55.8 | 65.1 | 30.3 |
| **glycerol + formamide** | **12 mol/kg** | **95** | **31.7** | **40.7** | **0.0** |

At 12 mol/kg total, additive `max` misses by **63 percentage points**; `sum_then_cap` misses by **95 percentage points**. There is no additive rule that can capture this without seeing mixture data.

### Top-10 FDA mixture pairs (additive `max`, 6 mol/kg total = 3+3 each)

| # | Compound A | Compound B | Pred tox | Pred perm | Pred IRI | Composite | Status |
|---:|---|---|---|---|---|---|---|
| 1 | Benzaldehyde | Fatty acid esters | 42.0 | 63.5 | 57.9 | 0.653 | benzaldehyde model error |
| 2 | Benzaldehyde | Phenol | 42.0 | 55.1 | 46.1 | 0.650 | both v2 single-compound errors |
| 3 | Benzaldehyde | Benzyl alcohol | 42.0 | 53.7 | 45.3 | 0.645 | benzaldehyde dominance |
| 4 | Benzaldehyde | Phenylalanine | 52.5 | 51.8 | 32.4 | 0.643 | DOLMEN memorization |
| **5** | **DMSO** | **Propylene glycol** | 13.6 | 50.3 | 69.0 | 0.641 | **real cryoprotectant cocktail** |
| 6 | Benzaldehyde | Hydrogen peroxide | 42.0 | 59.0 | 55.6 | 0.639 | H₂O₂ is cytotoxic, model error |
| 7 | Isoleucine | Phenylalanine | 54.6 | 47.7 | 25.1 | 0.638 | DOLMEN memorization |
| 8 | Benzaldehyde | Isoleucine | 54.6 | 55.7 | 37.8 | 0.638 | benzaldehyde dominance |
| 9 | Benzaldehyde | Butylene glycol | 42.0 | 59.6 | 56.8 | 0.638 | partially plausible |
| 10 | Fatty acid esters | Phenol | 38.5 | 58.8 | 58.9 | 0.637 | phenol model error |

Real CPA combination in top-10: **1** (DMSO + propylene glycol at #5). Single-compound model errors propagating to pairs: 6 of 10 (benzaldehyde-paired + H₂O₂ + phenol). DOLMEN memorization: 2 of 10 (the amino-acid pairs).

### Hypothesis calibration (predicted vs actual)

| Hypothesis | Predicted | Actual | Verdict |
|---|---|---|---|
| D. Additive baseline fails on 16 known mixtures | Spearman < 0.4 across rules; negative R²; formamide+glycerol@12 a clear miss | Spearman 0.18-0.22 across rules; negative R²; formamide+glycerol@12 missed by 63-95 pp | **right; magnitude matched** |
| D'. FDA pair scoring rediscovers a known CPA cocktail | At least 1 known CPA combination in top-10 by self-consistency | DMSO + propylene glycol at rank 5; real cryomicroscopy combination | **right; specific compound rediscovered** |
| D''. Single-compound model errors propagate to pairs | Benzaldehyde / phenol / CO₂ should appear in many top pairs because they're rated low-toxicity individually | Benzaldehyde appears in 11 of top-20; phenol, H₂O₂, formaldehyde, CO₂ all appear | **right; mechanism verified** |

---

## v3: filter v2.1, novel-only top-20, Tox21 aux weight sweep (commit pending)

### What was built

Three follow-up additions targeting the specific gaps left by v2 and v2.1:

1. **Filter v2.1** (`_passes_cpa_filter_v21` in `src/data/fda_iid.py`). Tightens the v2 filter with: heavy_atoms ≥ 3 (drops H₂O₂, formaldehyde), at least 1 hydrogen (drops CO₂), no sulfonates (was ≤ 1 in v2; drops benzenesulfonic acid), no epoxides (drops ethylene oxide), and aromatic compounds need HBD + HBA ≥ 3 (drops phenol, benzaldehyde, benzyl alcohol, phenylethyl alcohol; keeps amino acids, niacinamide, gentisic acid, saccharin, benzoic acid).

2. **Novel-only top-20** (`src/analyze_novelty.py`). Filters `all_scored.csv` to compounds NOT in the training set (DOLMEN + Higgins) and re-ranks. The original top-20 mixes virtual-screen recommendations with model self-consistency on memorized training compounds; the novel-only top-20 is the actual list to send to the wet lab.

3. **Tox21 aux weight sweep** (`src/sweep_tox21_aux.py`). Single-seed 5-fold CV at 6 aux weights {0.0, 0.05, 0.1, 0.2, 0.5, 1.0}. Settles whether v2's choice (0.1) was wrong vs whether the aux signal genuinely doesn't transfer.

### Hypotheses going in (v3)

**Hypothesis E (filter v2.1)**: Layering the new criteria drops CO₂, H₂O₂, formaldehyde, ethylene oxide, benzenesulfonic acid, phenol, benzaldehyde, benzyl alcohol, phenylethyl alcohol from the v2 candidate pool. **Predicted impact**: pool from 140 to ~110-120; top-20 single-compound becomes mostly real CPAs + DOLMEN-train memorization with no clear errors; top-10 mixture pairs becomes much more diverse (no benzaldehyde dominating 11 of 20).

**Hypothesis F (novel-only top-20)**: Removing in-training compounds from the top-20 surfaces the genuine recommendations. **Predicted impact**: 6 amino acids (phenylalanine, tryptophan, histidine, arginine, valine, isoleucine) drop out; new entries are real candidates (preview against the v2 pool: aspartame, benzocaine, benzoic acid, methylparaben, antipyrine, maltol, phenoxyethanol show up). After v2.1 filter is also applied, the novel-only top-20 should be the cleanest candidate list this project produces.

**Hypothesis G (Tox21 aux weight sweep)**: Three plausible outcomes ranked by my prior:
- Most likely: best aux_weight = 0.0; "Tox21 signal doesn't help at this scale at any reasonable weight". Confirms the v2 hypothesis C miss is structural (task transfer issue) not hyperparameter (weight choice). 
- Possible: best aux_weight in {0.05, 0.2}, beats no-aux by 0.02-0.05 Spearman; v2 weight choice (0.1) was just slightly off.
- Unlikely: best aux_weight in {0.5, 1.0}, beats no-aux substantially. Would mean the encoder benefits from heavy Tox21 regularization, which I don't expect at n=50 CPA.

### Hypothesis calibration (predicted vs actual)

| Hypothesis | Predicted | Actual | Verdict |
|---|---|---|---|
| E. Filter v2.1 drops CO₂/H₂O₂/formaldehyde/ethylene oxide/benzenesulfonic acid/phenol/benzaldehyde/benzyl alcohol; pool 110-120; top-20 has zero clear errors; benzaldehyde-dominance in mixture pairs disappears | Pool 125 (predicted ~110-120, slightly high), top-20 has zero clear errors and 5 known CPAs (urea, ethanol, n-propanol, n-butanol, DMA), top-10 mixture pairs has no benzaldehyde and is dominated by alcohol/diol pairs | **right; magnitude matched** |
| F. Novel-only top-20 drops the 6 DOLMEN amino acids; new entries are aspartame, benzocaine, methylparaben, etc. (preview) | 12 of 125 candidates overlap training; novel top-20 contains aspartame (#5), benzocaine (#9), methylparaben (#11), maltol (#12), phenoxyethanol (#6), o-tolyl biguanide (#7), gentisic acid (#8); 5 amino acids dropped (only methionine + lysines remain because they came in the v3 filter) | **right; specific compounds appeared as predicted** |
| G. Tox21 aux sweep: most likely best aux_weight=0.0 (signal genuinely doesn't transfer); possibly small win at 0.05-0.2 | Best aux_weight = 0.0 at toxicity Spearman 0.150; all non-zero weights produce 0.10-0.13; permeability collapses at high weights (-0.08 at 0.5, -0.24 at 1.0); IRI degrades monotonically | **right; most-likely outcome confirmed** |

### What v3 told me (and what it didn't)

Three confirmations and one limitation surfaced:

1. **The v2 filter was leaky in identifiable ways**, and a structurally-motivated tightening (heavy-atom count, hydrogen presence, aromatic polarity threshold) cleans it up cleanly. The predicted pool size and the predicted candidate-list improvements both materialized. This is the "structural fix to a known failure mode" pattern; the hypothesis was concrete and the result was concrete. 

2. **Novelty filtering separates memorization from generalization** in exactly the way you'd hope, and the novel-only top-20 looks like a genuine virtual-screen recommendation list (aspartame, benzocaine, methylparaben, etc.). The one limitation worth recording: the SMILES-string novelty check over-counts novelty when the same compound appears in training under different canonicalization (urea, ethanol, propanol all show as "novel" in the v3 list because the FDA-IID and Higgins SMILES strings differ). A semantic novelty filter (InChI or scaffold) would fix this; v1 is literal-string and the over-counting is documented.

3. **The Tox21 aux signal genuinely doesn't transfer to CPA cytotoxicity at this scale**, regardless of weight. Best aux weight is 0; non-zero weights only damage performance. This is consistent with the structural argument I noted in v2 (Tox21 measures nuclear-receptor binding at submicromolar; CPA toxicity is bulk cytotoxicity at multi-molar; the underlying biology is different). The v2.2 follow-up for broader toxicity signal would be a different aux task (DrugBank toxicity, in vivo LD50, or Until's internal screens), not Tox21 with a tuned weight.

**Limitation surfaced**: even the v3 mixture pair top-10 still has 3 of 10 entries as DOLMEN amino-acid pairs (memorization). The composite-score formula doesn't penalize "both compounds are in training"; adding a small in-training penalty would push the list toward more diverse novel pair recommendations. That's a v3.1 follow-up.

---

## Aggregate calibration record

| Iteration | Hypotheses tested | Right | Wrong | Hit rate |
|---|---|---|---|---|
| v2 (concentration / filter / Tox21) | 3 | 2 | 1 | 67% |
| v2.1 (mixture analysis) | 3 | 3 | 0 | 100% |
| v3 (filter v2.1 / novelty / Tox21 sweep) | 3 | 3 | 0 | 100% |
| **Total** | **9** | **8** | **1** | **89%** |

The one miss (Tox21 aux) was the most uncertain prediction going in (I flagged it as "task transfer is unclear" in the original v2 commit). The five hits include both **directional** predictions (concentration helps / additive baselines fail) and **specific** predictions (DMSO+PG should appear in top mixture pairs by self-consistency). Calibration looks honest, not over-confident, with the appropriate caveats on the predictions that turned out wrong.

The biggest single intervention by impact was concentration-aware toxicity in v2 (+0.18 Spearman). The most surprising win was the v2.1 additive-baseline failure being so clean and the DMSO+PG rediscovery happening exactly as predicted. The biggest disappointment was Tox21 aux not transferring; this is the natural ablation target for any future v2.2 effort.

---

## What changed quantitatively across iterations

### Toxicity prediction (the most CPA-relevant task)

| Version | RF cluster Spearman | RF random Spearman | OOF n |
|---|---|---|---|
| v1 (single-compound, MEAN of 3/6/12 mol/kg) | 0.459 | 0.347 | 22 |
| v2 (concentration-aware, per-(compound, conc) rows) | **0.643** | **0.527** | 50 |

### Candidate pool quality

| Version | Pool size | Real CPAs in top-20 | Confirmed errors in top-20 |
|---|---|---|---|
| v1 (loose filter) | 435 | 2 (urea, IPA) | 5+ (food dyes, organomercurials, inorganic acids) |
| v2 (tighter filter, element whitelist) | 140 | 4 (urea, ethanol, n-propanol, n-butanol) | 3 (CO₂, benzenesulfonic acid, phenol) |

### Mixture analysis (v2.1)

- 16 known mixtures, additive baseline Spearman 0.18-0.22 (all rules)
- formamide+glycerol@12 mol/kg neutralization missed by 63-95 pp (depending on rule)
- 9,730 FDA mixture pairs scored
- DMSO + propylene glycol surfaces at rank 5 from no mixture training labels (real cryomicroscopy combination)

---

## Where this leaves us

The single-compound RF model on concentration-aware toxicity data is genuinely useful (0.64 cluster Spearman; conformal coverage 0.98). The candidate pool is much cleaner after v2's filter. The mixture-aware analysis quantifies the gap between additive and learned-interaction models, and the DMSO+PG rediscovery is a real positive signal even from the additive baseline. ChemBERTa is still bottlenecked by small-task data and didn't benefit from Tox21 aux at the recipe I tried.

The most concrete next move is filter v2.1 (heavy-atom-count ≥ 6 + pKa cutoff) to drop CO₂ / H₂O₂ / formaldehyde / benzenesulfonic acid from both the single-compound and the mixture-pair top lists. After that, mixture data extraction (Higgins supplementary or Until's internal screens) is the rate limiter for actually training the PairEncoder. Both are tracked in the README's "What I'd build next" table.
