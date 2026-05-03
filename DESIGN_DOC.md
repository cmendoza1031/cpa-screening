# What I'd build at Until next

This goes alongside the working repo at [github.com/cmendoza1031/cpa-screening](https://github.com/cmendoza1031/cpa-screening). The repo is a one-week proof-of-concept on three public CPA datasets. This is what I'd extend it into.

Four things, in priority order:

1. **Mixture-aware model** — make the cocktail recommender predict toxicity neutralization
2. **MD–ML coupling** — give the model atomic-scale physics, not just SMILES
3. **Closed loop with the wet lab** — model picks the next batch, results retrain it
4. **Day-one asks** — the data and the production target I'd want defined

---

## 1. Mixture model

The repo's biggest gap. Single-compound predictions can tell you formamide is toxic at 6 mol/kg and glycerol is fine, but they cannot tell you the **mixture is 95% viable** at 12 mol/kg total (the headline finding in Higgins Dec 2025). Today's binary cocktail recommender uses an additive rule that hits Spearman ≈0.59 on 170 known mixtures but structurally misses neutralization by 60+ percentage points.

**The fix** is a **residual learner**: take the additive prediction as a baseline, train a learned model to predict only the **departure from additive** (`y_true − y_additive`). I tried a bare PairEncoder (predict mixture viability from scratch) and it failed at Spearman −0.26 on 170 rows. The residual framing should work because the same data gives a much easier learning target.

Once the binary residual learner is calibrated, extend to **k-component cocktails** by enumerating binary pairs, then beam-searching to k = 3, 4, 5. The acquisition function for the closed loop in §3 then ranks cocktail **compositions**, not single compounds. That is the actual product question.

## 2. MD–ML coupling

Until uses MD on candidate CPAs to model how they "move, bond, and interact with water." A clean integration is to use **MD-derived features as auxiliary inputs** to the ChemBERTa encoder: hydration shell metrics, water displacement, local diffusion coefficient, predicted glass-transition contribution. Each of those is a number you can compute from a 100-ps water-box trajectory and concatenate to the pooled SMILES embedding before the task heads.

Why it matters: when the current model sees out-of-distribution chemistry (a sulfonated dye, say), its toxicity prediction collapses to the training mean because it has no physics to ground itself in. MD features fix that. The model gets both "what does this molecule look like" (SMILES) and "how does it actually behave in water" (MD).

A second, cheaper mode is to train the encoder to **predict** MD properties from SMILES (auxiliary regression head). At deployment that gives you fast surrogate-MD without running the simulation. Mode-1 (features) makes predictions better; mode-2 (targets) makes them cheaper. I'd build features first.

## 3. Closed loop with the wet lab

The discovery engine: model has predictions + uncertainty for the candidate library. Acquisition function (start with **expected hypervolume improvement** over the (toxicity, permeability, IRI) Pareto front, since the regime is multi-objective with calibrated uncertainty already in the pipeline) picks the next cocktail compositions for liquid-handling batch. Wet-lab measurements feed back into the training set; the ensemble retrains in minutes; the candidate ranking updates.

Two diagnostics on every loop iteration: (a) **conformal coverage** in the new batch (target ≈0.95; drift below 0.85 means overconfident), and (b) **top-K rank stability** between iterations (heavy churn = high-variance regime; stable = found the structural signal). Both are concrete signals about whether to recalibrate or retrain from scratch.

---

## What carries forward from the current repo

- A clean multi-task dataset of 326 CPAs (toxicity, permeability, IRI) with provenance, plus 170 binary mixtures and a 125-compound FDA candidate pool.
- A stable ChemBERTa-2 + LoRA multi-task encoder on Blackwell. Slots directly into the mixture model in §1.
- 5-seed deep ensembles with split-conformal calibration (empirical coverage 0.954 on IRI). The same calibration constants carry into the cocktail layer.

