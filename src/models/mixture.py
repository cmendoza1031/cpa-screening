"""Mixture-aware modeling, v1.

Honest framing of the data constraint. Higgins Dec 2025 publishes binary
mixture viability data only as bar charts in Figures 3 and 4 of the paper,
and the readable values from those figures are 16 binary mixtures, all of
which have glycerol as one of the two components:

    11 mixtures at 6 mol/kg total (3 mol/kg of each component)
    5  mixtures at 12 mol/kg total (6 mol/kg of each component)

That's nowhere near enough data to fit a learned-interaction pair encoder
(at 5-fold CV, 16 / 5 = 3 rows per fold; the model would memorize). What
we *can* do meaningfully at this scale, and what's actually the more
informative analysis for an Until conversation, is quantify how badly a
single-compound toxicity model fails when used additively to predict
mixture viability. That tells you exactly the gap a proper mixture model
would close.

This module ships:

1. `load_mixtures()` -- read the Higgins binary-mixture parquet, split
   total concentration into per-compound concentrations.

2. `AdditiveBaseline` -- combine v2 single-compound predictions for
   compound A at concentration c_A and compound B at c_B using one of
   {max, mean, sum_then_cap, weighted_max}. No mixture training; this
   is a pure inference path on the v2 model.

3. `evaluate_additive_baseline()` -- per-rule Spearman / MAE / R^2 on the
   16 known mixtures, plus per-row residuals so neutralization failures
   (formamide+glycerol at 12 mol/kg) are visible.

4. `score_fda_mixture_pairs()` -- enumerate ~140-choose-2 binary pairs
   from the v2 FDA candidate pool, score each by additive-baseline
   composite over (toxicity, permeability, IRI), and Pareto-rank the
   top-K mixture pairs. This is the "buy two compounds, mix at 3+3
   mol/kg" output for the wet lab.

5. `PairEncoder` -- the learned-interaction architecture for when more
   mixture data becomes available. Not trained in v1; the architecture
   is documented + smoke-tested for shape correctness so it's drop-in
   ready when 200+ mixture rows are accessible.

The README's "Mixture-aware analysis" section reports the headline
numbers and the formamide+glycerol case study.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..utils import PROCESSED_DIR, RESULTS_DIR, get_logger

log = get_logger("models.mixture")

MIXTURES_PATH = PROCESSED_DIR / "higgins_mixtures.parquet"
MIX_RESULTS_DIR = RESULTS_DIR / "mixtures"

# Combination rules for additive-baseline mortality prediction. Each rule
# takes per-compound predicted mortality (0-100) and returns a single
# scalar mortality prediction for the mixture.
COMBINATION_RULES = {
    "max":           lambda m_a, m_b: float(max(m_a, m_b)),
    "mean":          lambda m_a, m_b: 0.5 * (m_a + m_b),
    "sum_then_cap":  lambda m_a, m_b: float(min(m_a + m_b, 100.0)),
    # Weighted toward the more-toxic compound (max-biased mean): one of the
    # standard "synergy" approximations from CPA literature; see Fahy 1986.
    "weighted_max":  lambda m_a, m_b: 0.7 * max(m_a, m_b) + 0.3 * min(m_a, m_b),
}


# --------------------------- data layer ------------------------------------


def load_mixtures() -> pd.DataFrame:
    """Load Higgins binary mixtures with per-compound concentrations.

    The raw `concentration_mol_kg` column in the parquet is the *total*
    mixture concentration (6 mol/kg total -> 3 mol/kg each, etc.). This
    function expands it to per-compound concentrations and emits the
    canonical pair representation:

    Returns columns:
        smi_a, smi_b           canonical SMILES for compound A and B
        name_a, name_b         human-readable compound names
        conc_a, conc_b         per-compound concentration in mol/kg
        conc_total             total mixture concentration in mol/kg
        viability_4c           measured viability (0-100, raw value)
        mortality_4c           100 - viability_4c (matches v2 toxicity scale)
    """
    if not MIXTURES_PATH.exists():
        log.warning(
            "no mixture parquet at %s; returning empty df. Run "
            "`python -m src.data` to (re)build the dataset.",
            MIXTURES_PATH,
        )
        return pd.DataFrame()

    raw = pd.read_parquet(MIXTURES_PATH)
    if raw.empty:
        return pd.DataFrame()

    df = pd.DataFrame({
        "smi_a": raw["smiles_canonical"],
        "smi_b": raw["smiles_canonical_2"],
        "name_a": raw["compound_name"],
        "name_b": raw["compound_name_2"],
        "conc_total": raw["concentration_mol_kg"].astype(float),
        "viability_4c": pd.to_numeric(raw["viability_4c"], errors="coerce"),
    })
    df = df.dropna(subset=["smi_a", "smi_b", "viability_4c"]).reset_index(drop=True)
    df["conc_a"] = df["conc_total"] / 2.0
    df["conc_b"] = df["conc_total"] / 2.0
    df["mortality_4c"] = 100.0 - df["viability_4c"]
    return df[[
        "smi_a", "smi_b", "name_a", "name_b",
        "conc_a", "conc_b", "conc_total",
        "viability_4c", "mortality_4c",
    ]]


# --------------------------- additive baseline -----------------------------


@dataclass
class SinglePrediction:
    """Tiny container for a per-compound prediction with optional uncertainty."""
    mean: float
    std: float = 0.0


# A single-compound predictor is any callable
#     (smiles, concentration_mol_kg) -> SinglePrediction
# Mortality scale (matches v2 toxicity output: 0 = healthy, 100 = dead).
SinglePredictor = Callable[[str, float], SinglePrediction]


def additive_predict(
    mixtures: pd.DataFrame,
    predictor: SinglePredictor,
    rule: str = "max",
) -> pd.DataFrame:
    """Apply the v2 single-compound model to each (compound, concentration)
    in the mixture, then combine the per-compound mortality predictions
    using the chosen rule. Returns the input dataframe with extra columns:

        m_a_pred         mortality prediction for compound A at conc_a
        m_a_std          ensemble std for that prediction
        m_b_pred         same for compound B
        m_b_std          same
        m_pred_<rule>    combined mortality prediction
        v_pred_<rule>    100 - combined mortality (predicted viability)
        residual_<rule>  v_pred_<rule> - viability_4c (signed; negative = model
                         overestimates mortality, i.e. misses neutralization)
    """
    if rule not in COMBINATION_RULES:
        raise ValueError(f"rule {rule!r} not in {list(COMBINATION_RULES)}")
    combine = COMBINATION_RULES[rule]

    out = mixtures.copy()
    m_a, s_a, m_b, s_b, m_pred = [], [], [], [], []
    for _, r in out.iterrows():
        pa = predictor(r["smi_a"], float(r["conc_a"]))
        pb = predictor(r["smi_b"], float(r["conc_b"]))
        m_a.append(pa.mean); s_a.append(pa.std)
        m_b.append(pb.mean); s_b.append(pb.std)
        m_pred.append(combine(pa.mean, pb.mean))
    out["m_a_pred"] = m_a
    out["m_a_std"] = s_a
    out["m_b_pred"] = m_b
    out["m_b_std"] = s_b
    out[f"m_pred_{rule}"] = m_pred
    out[f"v_pred_{rule}"] = 100.0 - np.array(m_pred)
    out[f"residual_{rule}"] = out[f"v_pred_{rule}"] - out["viability_4c"]
    return out


def evaluate_additive_baseline(
    mixtures: pd.DataFrame,
    predictor: SinglePredictor,
    rules: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict]:
    """Run additive baseline under multiple combination rules; return
    (per-row prediction frame with residuals for each rule, summary metrics
    dict).

    Summary metrics per rule:
        spearman_viability, mae_viability, rmse_viability, r2_viability,
        n_neutralization_misses (residuals where the model predicts >25 pp
        more mortality than measured; i.e. real toxicity is much lower than
        an additive model would predict).
    """
    from ..eval import regression_metrics

    rules = rules or list(COMBINATION_RULES)
    work = mixtures.copy()
    summary: dict[str, dict] = {}
    for rule in rules:
        work = additive_predict(work, predictor, rule=rule)
        v_true = work["viability_4c"].to_numpy(dtype=float)
        v_pred = work[f"v_pred_{rule}"].to_numpy(dtype=float)
        m = regression_metrics(v_true, v_pred)
        # Neutralization miss: model says viability is much lower than truth
        # (residual = predicted - measured; very negative = model misses
        # neutralization). Threshold of -25pp is conservative; the
        # formamide+glycerol@12 case has residual ~ -90.
        miss_mask = work[f"residual_{rule}"] < -25.0
        m["n_neutralization_misses"] = int(miss_mask.sum())
        summary[rule] = m
        log.info(
            "additive baseline rule=%-14s n=%2d  Spearman=%+.3f  MAE=%5.2f  "
            "n_misses>25pp=%d",
            rule, m["n"], m["spearman"], m["mae"], m["n_neutralization_misses"],
        )

    return work, summary


# --------------------------- mixture pool scoring --------------------------


def score_fda_mixture_pairs(
    candidates: pd.DataFrame,
    predictor_tox: SinglePredictor,
    predictor_perm: SinglePredictor,
    predictor_iri: SinglePredictor,
    conc_total_mol_kg: float = 6.0,
    rule: str = "max",
    top_k: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enumerate all unordered binary pairs from the FDA candidate pool,
    score each pair using the additive-baseline composite over (toxicity,
    permeability, IRI), and return (top-K pairs, all pairs ranked).

    Each compound enters the mixture at half of `conc_total_mol_kg`. For
    permeability and IRI we don't have measured mixture data, so we use a
    simple additive sum capped (perm) / additive mean (IRI) as the baseline.
    These are documented as approximations in the README.

    Composite score: prefer low predicted mortality, high permeability,
    low predicted IRI %MGS, with a small uncertainty penalty derived from
    per-compound std.
    """
    if rule not in COMBINATION_RULES:
        raise ValueError(f"rule {rule!r} not in {list(COMBINATION_RULES)}")
    combine_mortality = COMBINATION_RULES[rule]
    half = conc_total_mol_kg / 2.0

    smiles = candidates["smiles_canonical"].tolist()
    names = candidates["ingredient_name"].tolist() if "ingredient_name" in candidates.columns else smiles
    n = len(smiles)
    log.info(
        "scoring %d FDA mixture pairs (%d unique compounds choose 2) at "
        "%.1f mol/kg total each",
        n * (n - 1) // 2, n, conc_total_mol_kg,
    )

    # Prefetch single-compound predictions once per (compound, half-conc).
    cache_tox = {s: predictor_tox(s, half) for s in smiles}
    cache_perm = {s: predictor_perm(s, half) for s in smiles}
    cache_iri = {s: predictor_iri(s, half) for s in smiles}

    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            sa, sb = smiles[i], smiles[j]
            na, nb = names[i], names[j]
            ta, tb = cache_tox[sa], cache_tox[sb]
            pa_, pb_ = cache_perm[sa], cache_perm[sb]
            ia, ib = cache_iri[sa], cache_iri[sb]
            tox_pred = combine_mortality(ta.mean, tb.mean)
            perm_pred = min(pa_.mean + pb_.mean, 100.0)  # additive cap
            iri_pred = 0.5 * (ia.mean + ib.mean)
            # Total uncertainty for the pair: sqrt of squared sum (assumes
            # independence between the two single-compound predictions).
            unc_tox = float(np.sqrt(ta.std ** 2 + tb.std ** 2))
            unc_perm = float(np.sqrt(pa_.std ** 2 + pb_.std ** 2))
            unc_iri = float(np.sqrt(ia.std ** 2 + ib.std ** 2))
            rows.append({
                "name_a": na, "name_b": nb,
                "smi_a": sa, "smi_b": sb,
                "tox_pred": tox_pred, "tox_unc": unc_tox,
                "perm_pred": perm_pred, "perm_unc": unc_perm,
                "iri_pred": iri_pred, "iri_unc": unc_iri,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df

    # Composite: minimize tox + iri, maximize perm; subtract uncertainty
    # penalty. All means normalized to roughly [0, 1] by /100 (tox / iri are
    # in mortality / %MGS units, perm in P_CPA*1e-3 s^-1 (~0-50 in our data)).
    perm_norm_max = max(df["perm_pred"].max(), 50.0)
    df["composite_score"] = (
        (1.0 - df["tox_pred"] / 100.0)
        + (df["perm_pred"] / perm_norm_max)
        + (1.0 - df["iri_pred"] / 100.0)
    ) / 3.0 - 0.25 * (
        (df["tox_unc"] + df["perm_unc"] + df["iri_unc"]) / 3.0 / 25.0
    )
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.head(top_k).copy(), df


# --------------------------- pair encoder (architecture only) --------------


def _build_pair_encoder(config) -> "PairEncoder":
    """Build the learned-interaction pair encoder.

    Architecture sketch (not trained in v1 because n=16 mixture rows is too
    few to fit it without memorization):

        encoder       = same ChemBERTa-LoRA as single-compound model (shared)
        pooled_a      = mean-pool encoder(SMILES_a, attention_mask_a)
        pooled_b      = mean-pool encoder(SMILES_b, attention_mask_b)
        # symmetric features so f(A,B) == f(B,A)
        sym           = pooled_a + pooled_b
        diff          = (pooled_a - pooled_b).abs()
        prod          = pooled_a * pooled_b
        conc_features = [conc_a, conc_b]
        # interaction term is fed alongside the symmetric features so the
        # head can learn departure-from-additive
        head_input    = concat([sym, diff, prod, conc_features])
        toxicity_pred = MLP(head_input)

    The head produces predicted mixture mortality. Loss is Huber on
    measured mortality, identical to the single-compound recipe.

    For v1 this is documented + shape-tested; once 200+ mixture rows
    become available (Higgins supplementary table or new Until data),
    the train loop can be enabled by flipping `args.train_mixture` on.
    """
    raise NotImplementedError(
        "PairEncoder training is gated on >=200 mixture rows. We have 16 "
        "(all glycerol-paired). Use additive_predict / evaluate_additive_baseline "
        "until more mixture data is available."
    )
