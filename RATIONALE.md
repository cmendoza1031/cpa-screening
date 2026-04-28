# Eval-scheme rationale

This document records the reasoning behind the cross-validation choices in
this project. It exists because (a) ML papers in low-data biology routinely
get the eval wrong, and (b) the difference between "0.42 Spearman is real
signal" and "0.42 Spearman is fold-leakage" is exactly the difference between
useful and useless.

## Final eval scheme

| Task | n | Scheme | Folds (≈ test size) | Architecture |
|---|---|---|---|---|
| iri | 303 | 70/15/15 single split, seed-keyed on SMILES | val 46, test 40 | RF + ChemBERTa, same split |
| permeability | 16 | 5-fold CV, compound-level | ~3 test per fold | RF + ChemBERTa, same folds |
| toxicity | 22 | 5-fold CV, compound-level | ~4 test per fold | RF + ChemBERTa, same folds |
| permeability (secondary) | 16 | LOO-CV | 1 test per fold | RF only, sanity check |

For ChemBERTa multi-task, a single global compound-level 5-fold structure is
applied: each held-out fold removes that compound's labels from *every* task
the compound appears in. This is more conservative than per-task CV (which
allows multi-task compounds to leak through other tasks' labels) and matches
what we'd want at deployment time.

## Why 5-fold and not LOO for the small tasks

The most natural reaction to n=22 toxicity is "use LOO, it gives the most
training data per fold". I went with 5-fold instead, for three reasons:

1. **Compute. ** Each ChemBERTa+LoRA fold is a fine-tune. LOO permeability
   = 16 fine-tunes. With 5-seed ensembling that's 80 multi-task transformer
   trainings just for permeability. Colab Pro can't host that in our budget,
   and the marginal information at fold 16 vs fold 5 doesn't pay for it.

2. **Fold leakage. ** The DOLMEN datasets are dense in scaffold space —
   amino acids and disaccharides cluster tightly under Morgan fingerprints.
   With 1 held-out compound surrounded by 15 close neighbors, LOO is
   optimistic in a way that's hard to inspect. 5-fold spreads the leakage
   across more held-out compounds and pairs naturally with the cluster-aware
   split work coming in Phase 2.

3. **Apples-to-apples. ** RF and ChemBERTa need the same fold structure to
   be comparable. 5-fold works for both. LOO would force RF-only or break
   the comparison.

RF additionally reports LOO permeability as a *secondary* analysis (it's
free for RF since training is fast). The gap between 5-fold and LOO numbers
is itself useful information — if they diverge a lot, the model is sensitive
to specific neighbor pairs and we should worry.

## Why iri stays on 70/15/15

n=303 is enough to trust the standard split. With 40 test compounds and a
reasonable Spearman, the standard error is ~0.13 — small enough to commit
to a single number. CV at this n complicates the eval pipeline and pads the
results table without adding information. We stay simple where we can.

## On the rank sweep {4, 8, 16}

The original spec calls for sweeping LoRA rank ∈ {4, 8, 16}. At Phase 2 with
5-seed ensembles, that's 5 seeds × 3 ranks × 5 folds = 75 ChemBERTa trainings
just on the small tasks. Not feasible on Colab in our time budget.

More importantly: at n=22 toxicity, the seed-to-seed standard error of any
metric is bigger than the rank-to-rank gap is likely to be. Picking "best
rank" by CV metric on this scale is a false signal — we'd be tuning to
noise.

**Decision:** commit to LoRA rank 8 (the middle option) for both the Day 2
single-seed sanity check and the Phase 2 5-seed ensemble. Skip the sweep.
If you want a rank comparison, it's a side study at single-seed only and
should be reported with explicit "within seed noise" caveats.

## On the toxicity-task definition

`build_dataset()` averages viability across the 3 / 6 / 12 mol/kg measurements
in Higgins Dec 2025 to produce one toxicity label per compound. This is a
deliberate v1 simplification — concentration matters a lot biologically
(formamide is fine at 3 mol/kg, devastating at 6, untestable at 12) and
collapsing it into a mean discards the most interesting structure in the
data. Phase 2 should add concentration as an explicit input feature so the
model produces dose-response predictions, not point estimates.

## What this is NOT

This eval is a baseline on three small public datasets, not a wet-lab
validation. The claim of the project is "model rank-orders compounds well
enough to prioritize a top-20 list for screening", not "model predicts CPA
toxicity to within 5%". The Pareto-ranking output is a recommendation,
not a guarantee. The DESIGN_DOC.md (Phase 3) makes this honest in front
of any reviewer.
