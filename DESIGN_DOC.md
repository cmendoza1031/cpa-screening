# What I'd build at Until next

This is the design doc that goes alongside the working repo at [github.com/cmendoza1031/cpa-screening](https://github.com/cmendoza1031/cpa-screening). The repo is a one-week proof-of-concept on three public small-molecule CPA datasets; this doc is what I'd extend it into if I joined Until's molecular-discovery program.

I've organized it as six things, in order of what I think matters most:

1. The mixture-aware model (where I think the next real improvement lives)
1b. The cocktail recommender pipeline (how today's binary additive scorer becomes a real k-component composition recommender)
2. MD–ML coupling (how the wet-lab data and the atomic-scale simulations talk to each other)
3. The closed loop with the lab (the discovery engine the Series A blog mentioned)
4. What I'd want from Until on day one (data + clarity-of-target)
5. Things I'd explicitly *not* do first (to bound scope honestly)

---

## 1. The mixture problem is the lead

Until's Series A blog and Higgins's Dec 2025 paper both end up saying the same thing in different language: the clinically useful CPA work happens at the cocktail level, not the single-compound level. Higgins's headline empirical finding makes this concrete:

> 6 mol/kg formamide alone → ≈20% viability.
> 6 mol/kg formamide + 6 mol/kg glycerol (12 mol/kg total) → ≈97% viability.

A single-compound predictor cannot output the second number from the first. It can tell you formamide is toxic at 6 mol/kg and glycerol is mostly OK at 6 mol/kg; it cannot tell you their combination is fine. That gap is the entire reason CPAs are practically usable: the field has been mixing things since Fahy's "toxicity neutralization" work in the 80s, and the vitrification solutions used clinically (M22 has 6 components, VS55 has 4, VEG has 3) are designed precisely to push past the single-compound toxicity ceiling.

**Architecture sketch.** Two SMILES inputs (or N for general mixtures), per-component encoder (the ChemBERTa+LoRA stack from this repo, weights tied between positions), and a learned interaction term:

```
        SMILES A ───► ChemBERTa+LoRA (shared) ───► h_A ∈ ℝ^d
        SMILES B ───► ChemBERTa+LoRA (shared) ───► h_B ∈ ℝ^d
                                                        │
                              h_mix = MLP([h_A, h_B, h_A * h_B, h_A − h_B, c_A, c_B, c_A·c_B])
                                                        │
                                            ┌─────────┐ │ ┌─────────┐ ┌────────┐
                                            │ tox 4°C │◄┼─┤ perm 4°C│ │ iri    │
                                            └─────────┘ │ └─────────┘ └────────┘
```

The interaction term `h_A * h_B` (Hadamard) and `h_A − h_B` are the standard building blocks; concentration enters explicitly as `c_A`, `c_B`, and the cross term `c_A · c_B`. Permutation-invariance is enforced by always taking the mean of `f(A, B)` and `f(B, A)`, or by sorting components by canonical SMILES before encoding.

**Training data.** Higgins Dec 2025 ships with 87 binary mixtures at 6 mol/kg and 82 at 12 mol/kg. That's the core. I'd compile additional binary-mixture viability data from:

- Fahy's older VS55 / VEG / VS41A toxicity tables (pre-1995, available as scanned tables in *Cryobiology*)
- Best 2015 review on CPA toxicity ([10.1016/j.cryobiol.2015.05.005](https://doi.org/10.1016/j.cryobiol.2015.05.005)) and citations therein
- Internal Until Labs binary-mixture screens, if available. This is where the day-one ask matters (see §4)

**Active-learning fit.** The mixture model is a much better fit for active learning than the single-compound one: the design space is combinatorial (`N choose 2` for binary, `N choose k` for k-component) and far too large to brute-force, so uncertainty-weighted acquisition has compounding value. v1 single-compound AL (the spec's "stretch" goal) selects compounds; mixture AL selects *compositions*, which is what an actual CPA cocktail discovery loop would do.

**v1 limitations to be upfront about.** Mixture toxicity often shows non-monotonic concentration response (formamide+glycerol becomes more toxic again past ≈14 mol/kg total). The model needs enough concentration coverage to capture that, and Higgins Dec 2025 only goes to 12 mol/kg, which is right at the edge. Without higher-concentration data, the model won't know where toxicity returns. This is a data-gap not a method-gap.

---

## 1b. The cocktail recommender pipeline

§1 specifies the model. This section specifies the **product**: how the model becomes a system that takes Until's candidate library and weekly returns a ranked list of k-component CPA cocktail compositions to wet-test, with calibrated uncertainty on each. Today's repo ships v0 of this pipeline (binary additive). v1, v2, v3 are concrete extensions, each with a clear "what changes / what's needed / what it unlocks."

**v0 (today, in the repo).** Binary additive cocktail recommender. Input: a candidate library (currently the 125-compound v3-filtered FDA pool). For every (A, B) pair at a fixed total concentration (currently 6 mol/kg = 3+3), predict toxicity / permeability / IRI by combining single-compound predictions under one of {`max`, `mean`, `sum_then_cap`, `weighted_max`}, Pareto-rank by composite score with conformal PIs propagated from the single-compound layer. Calibration on Higgins's 170 binary mixtures: best rule reaches Spearman ≈0.59. Limit: structurally cannot predict toxicity neutralization; misses formamide+glycerol@12 mol/kg by 60+ pp.

**v1: residual learner on top of additive.** The PairEncoder failure on 170 mixture rows (Spearman −0.26) said the model can't relearn dose-response from scratch. The fix is to take the additive prediction as a baseline (or a feature) and have the learned model predict only the **residual**: `y_true − y_additive`. Architecture:

```
SMILES_A, SMILES_B, c_A, c_B
        │
        ├──► additive_baseline(...)  ────────► y_add  ┐
        │                                              │
        └──► PairEncoder (§1 architecture) ──► δ      ├──► y_pred = y_add + δ
                                                       │
                                       supervised on (y_true − y_add) residual
```

Why this beats the bare PairEncoder: at additive Spearman 0.59 the residual signal is already what the model sees, so the same 170 rows give it a much easier learning target (predict departures from additive, not viability from scratch). Even modest skill on the residual recovers everything additive captures plus the interaction term. Data ask: same 170-row dataset is enough to test whether this works; if it does, the immediate next ask is a few hundred more binary mixtures from Until's internal screens to push residual Spearman past 0.3.

**v2: concentration optimization within a fixed composition.** v0 / v1 score binary pairs at a fixed concentration grid (3+3, 6+6, etc.). Real cocktail design needs the concentration ratio that hits vitrifying total concentration with minimum predicted toxicity. For a binary (A, B) at total concentration C_total, optimize `(c_A, c_B = C_total − c_A)` over `c_A ∈ [0, C_total]` to minimize predicted toxicity subject to permeability ≥ threshold. This is a 1D bounded optimization per pair (Bayesian optimization or just grid search at 0.5 mol/kg resolution). For ternary it's 2D, quaternary 3D, scales fine to k = 6 with BO.

**v3: k-component cocktail search.** The combinatorics: 125-choose-2 = 7,750 (enumerate everything), 125-choose-3 = 317k (still enumerable on a laptop), 125-choose-4 = 9.7M, 125-choose-5 = 234M (need search). For k ≥ 4, the right tool is **beam search seeded by the v0/v1 binary top-K**: take the top-100 binary cocktails from v1, expand each by greedily adding the one component that most improves the composite score under v1's scoring (using the additive rule extended to k components for the components the residual learner hasn't seen as part of larger mixtures), prune to the new top-100, repeat to k = 5 or 6. Each beam-search step costs `O(beam_width × library_size)` model evaluations, which for 100 × 125 = 12,500 evaluations per step is comfortably tractable. Calibration of the k-component additive rule degrades vs. binary because more pair interactions accumulate, so v3 outputs need the v1 residual learner active to be trustworthy.

**Acquisition over cocktail compositions, not just compounds.** §3's closed loop talks about per-compound EHVI. Once the recommender is k-component, the acquisition function operates over **compositions**, not single compounds. The natural extension: at each iteration, pick the cocktail composition with the highest expected hypervolume improvement over the current Pareto front of (toxicity, permeability, IRI), with composition-level uncertainty propagated from the single-compound PIs through the additive (or residual) rule. This is what closes the loop on the actual product question: "what composition do we wet-test next."

**Pipeline output, all the way through.** A weekly artifact that looks like:

| Rank | Composition | c (mol/kg per component) | Pred tox (95% PI) | Pred perm (95% PI) | Pred IRI (95% PI) | Composite | EHVI vs current Pareto front |
|---|---|---|---|---|---|---|---|
| 1 | DMSO + EG + glycerol | 4 / 3 / 2 (9 total) | 18 ± 8 | 47 ± 10 | 52 ± 12 | 0.71 | 0.18 |
| ... | ... | ... | ... | ... | ... | ... | ... |

That's the deliverable Until's wet-lab team consumes each week. Cocktail compositions, not single compounds, ranked with calibrated uncertainty and an explicit information-gain score.

**Where the v0 → v3 path lives in the repo.** v0 is shipped (`src/score_mixtures.py`). v1 is a single new file (`src/models/residual_pair_encoder.py`) that wraps the existing single-compound ensemble + the additive baseline. v2 is a small inner-loop optimizer on top of either. v3 is a beam-search wrapper on top of v2. Each is a 1-2 day change behind the previous one, so the entire pipeline is realistic in a 4-6 week sprint at Until once mixture data is in hand.

---

## 2. MD–ML coupling

Until's atomic-scale work runs MD on candidate CPAs in water and characterizes how they "move, bond, and interact with water" (their language from the public site). This is fundamentally complementary to the wet-lab modeling the repo currently does, and there's a clean way to fuse them.

**Two coupling modes**, in order of how much I'd build first:

**(a) MD-derived features as auxiliary inputs.** For each candidate compound, compute on a 100 ps water-box MD trajectory:

- **Hydration shell metrics**: number of water hydrogen bonds per CPA, residence time of bound waters, radial distribution function characteristic distances
- **Water displacement**: how many bulk waters per CPA at the relevant concentration
- **Local diffusion coefficient** of CPA in water (correlates with membrane permeability; surrogate for the harder D_membrane question)
- **Predicted glass-transition contribution** (T_g additivity terms; there's recent ML+MD work for polymer T_g [Mondal et al. 2026, *ACS Appl Polym Mater](https://doi.org/10.1021/acsapm.5c04524)* that's portable to CPAs at the chemistry level)

These get concatenated to the ChemBERTa pooled embedding before the task heads. For a wet-lab dataset of ≈300 compounds, MD on each is 300 × ≈100 ps = ~hour-scale on a GPU cluster, completely tractable. The model now has both "what does this molecule look like at the chemistry level" (from SMILES) and "how does this molecule actually behave in water" (from MD).

**(b) MD outputs as auxiliary regression targets.** Train the encoder to also predict MD-derived properties (T_g contribution, hydration metric) using a separate auxiliary head. This is the fast-surrogate-for-MD framing. At deployment time, the model gives you predicted MD properties from SMILES alone, no MD run needed. It's the same setup as the Tox21 auxiliary head in this repo, just with MD-derived continuous targets instead of bioassay binary labels.

Mode (a) makes the property predictions better. Mode (b) makes future predictions cheaper. Both are useful; I'd build (a) first because the MD features give the encoder grounded physics that pretraining-on-PubChem doesn't have.

**The thing this enables.** Right now in the repo, when the model sees an out-of-distribution compound (a sulfonated polyaromatic dye, say), its toxicity prediction collapses to the training mean. It has no way to ground the prediction in physics. MD features give it a way: the dye has a distinctive hydration shell signature, a different diffusion behavior, a different T_g contribution. Even with no bioassay supervision in this regime, the MD-derived features tell the model "this is structurally different from training" in a way that's *causally relevant* to CPA function rather than an arbitrary chemical-similarity score.

---

## 3. Closed loop with the wet lab

The Series A post mentions Until's "discovery engine"; this is what I think the closed loop looks like in practice.

**The loop:**

```
┌─────────────────────────────────────────────────────────────────┐
│  model has trained predictions + uncertainty for all FDA IID +  │
│  bespoke synthesis catalogs (ChemSpider, eMolecules, ZINC)      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────┐
        │  acquisition function: rank candidates by │
        │  expected improvement under uncertainty   │
        │  weighted by Pareto rank, MW, cost,       │
        │  in-stock-at-supplier flag                │
        └──────────────────────────────────────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────┐
        │  select N=24-96 compounds for the next   │
        │  weekly liquid-handling batch             │
        └──────────────────────────────────────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────┐
        │  Higgins-style 96-well plate: simultaneous│
        │  permeability + viability assay at 4°C    │
        │  Tg measurement on a subset (DSC)         │
        └──────────────────────────────────────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────┐
        │  add new measurements to training set;   │
        │  retrain ensemble; recompute predictions  │
        │  on the candidate pool                    │
        └──────────────────────────────────────────┘
                                │
                                └──────► loop
```

**Acquisition function specifics.** I'd start with **uncertainty-weighted Expected Improvement (EI-σ)** because the regime (small training set, multiple correlated objectives, calibrated uncertainty already in the pipeline from conformal calibration) is exactly what this acquisition function is designed for. Reker & Schneider's 2015 review of active learning in CADD ([10.1016/j.drudis.2014.12.004](https://doi.org/10.1016/j.drudis.2014.12.004)) walks through the trade-offs cleanly.

A useful refinement: **Pareto-aware acquisition.** Most AL formulations are single-objective; here we have three correlated objectives. The right move is to use *expected hypervolume improvement* (EHVI) over the (toxicity, permeability, IRI) Pareto front. EHVI rewards picking compounds that move the front in any direction (including ones where the model is uncertain whether they're a Pareto win)which matches the discovery framing better than per-task EI.

**Two diagnostics I'd run on every loop iteration:**

1. **Coverage of the conformal PIs in the new batch.** Across the 24-96 newly-tested compounds, fraction of true values inside predicted PI95 should be ≈0.95. Drift below 0.85 means the model is becoming overconfident as data accumulates; drift above 0.99 means it's becoming pessimistic. Both are actionable signals about whether to re-tune calibration or retrain from scratch.
2. **Top-K rank correlation across iterations.** If between iterations the top-20 candidates list is shuffling around heavily, we're still in the high-variance regime where model identity is poorly determined; if it's stable, we've found the structural signal and AL is just refining locally. This is also a calibration check on whether the ensemble is contributing useful spread or just running 5 nearly-identical models.

**Compute.** Per-iteration retrain on the current pipeline is fast enough that the model is comfortably faster than the wet-lab cycle (which on liquid-handling robots is days per batch, not minutes). Weekly batches of 24-96 compounds feeding back into a weekly retrain are straightforwardly tractable on a single GPU.

**Where I think the loop converges.** With single-compound AL on the FDA IID + commercial libraries pool, you'd converge to a Pareto frontier of ≈20-100 compounds with strong predictions in 6-12 weeks. The interesting AL is the mixture variant from §1: the design space is `1500 choose 2 ≈ 10^6` binary mixtures, so brute force is impossible, and AL is the only tractable path. That's where Until gets disproportionate value out of the model.

---

## 4. What I'd want from Until on day one

Three things, in order:

1. **Internal screening data.** The Higgins public datasets are 22-303 compounds across one cell type and one temperature regime. Until presumably has more screens (different cell types, longer exposure, multiple temperatures, internal candidate molecules). Three weeks with that data would do more than three months on the public sets. It's the data scale that's bottlenecking the model's small-task learning, not the architecture.
2. **MD outputs from the atomic-scale program.** Even a pilot of 50-200 compounds with MD-computed hydration / diffusion / T_g properties would let me ship MD-ML coupling Mode (a) from §2 in the first month. Without it, I can synthesize MD on candidates myself but it's slow and the trajectories aren't ground-truth-comparable to the team's.
3. **A clear definition of the production decision the model is supporting.** This sounds banal but matters a lot for what to optimize. Three plausible production targets:
  - **Pareto ranking** (current setup): give wet-lab a top-20 for the next batch
  - **Single-objective optimization**: e.g. "find any compound with P_CPA > X *and* viability at 6 mol/kg > Y"; this is a discrete-feasibility problem, very different optimization
  - **Mixture composition**: find the M-component cocktail with the best (toxicity, vitrification temp, perfusion time) trade-off; this is vector optimization with combinatorial design space

The model architecture, the AL acquisition function, and the calibration choices all change depending on which of these the wet-lab team actually wants. I'd want to know the answer before committing to a v2.

---

## 5. Things I'd explicitly not do first

Bounding scope is part of the job. In the first 3-6 months I would *not*:

- **Pretrain a chemistry foundation model from scratch.** ChemBERTa, MolFormer, MolBERT, ChemGPT all exist; pretraining is hundreds of GPU-hours and the marginal gain on the Until-relevant downstream tasks is small. Adapt with LoRA, don't compete with the foundation-model labs.
- **Build a generative model for de novo CPA design.** It's seductive (RFdiffusion-for-small-molecules, or SMILES VAEs / discrete diffusion) but the chemistry constraint at Until is "molecule is small, polar, biocompatible, synthesizable, in-stock at a supplier", a heavily constrained design space that's served much better by enumerating and ranking than by sampling de novo. Once we've exhausted the FDA IID + commercial libraries (≈10-100k compounds), generative becomes interesting.
- **Compete with the protein-design groups working on engineered ice-binding proteins.** The iTHR designs from the Baker / Sosso labs ([PNAS 2025](https://www.pnas.org/doi/10.1073/pnas.2514871122)) are exciting work on a parallel problem, but they target ice-binding *proteins*, which are too large to permeate cells, useful for extracellular IRI but not for vasculature perfusion. Until's small-molecule angle is structurally different and the protein-design machinery doesn't transfer.
- **Build my own MD pipeline.** Until's atomic-scale program presumably has a working stack already. I'd consume their outputs as features rather than rebuilding.

---

## What the current repo gives you to build on

Three concrete things from the v4.1 build that any cocktail-aware extension would inherit:

- A clean multi-task dataset of 326 unique CPAs spanning toxicity, permeability, and IRI, with a 170-row binary mixture set and a 125-compound FDA candidate pool, all dedup'd on canonical SMILES with per-value provenance (including explicit notes where viability had to be read from bar charts because the authors didn't publish a numeric table).
- A stable ChemBERTa-2 + LoRA multi-task encoder on Blackwell (bf16 + SDPA + Huber + per-sample loss + `nan_to_num` + value clipping; 0 NaN-grad batches across 25 ensemble trainings). The same encoder slots into the §1 PairEncoder by design.
- 5-seed deep ensembles with split-conformal calibration (empirical coverage 0.954 on IRI), so the prediction intervals shipped with the candidate top-20 are real intervals, not error bars. The same calibration constants carry over to the cocktail-ranking layer.

Together with the §1-§3 directions above, that's enough scaffolding to start the cocktail-aware build on day one.