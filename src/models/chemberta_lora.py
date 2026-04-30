"""ChemBERTa-2 + LoRA multi-task model for CPA prediction.

Phase 1 Day 2 — single-seed, single-rank (=8) sanity-check version.
The Phase 2 version layers on 5-seed deep ensembles, conformal calibration,
cluster-aware splits, and (optionally) the Tox21 auxiliary classification
head. This module exposes the architecture for both phases; Day 2 just
calls it once with --no-tox21-aux to keep the dep surface small.

Architecture
------------
    Input: canonicalized SMILES strings.
    Tokenizer: DeepChem/ChemBERTa-77M-MLM.
    Encoder: same model, with LoRA (rank 8 default) on the query / key / value
        projections of every attention block. peft handles the adapter wiring.
    Pooling: mean over sequence dim, attention-mask-weighted.
    Heads: a small MLP per task (Linear -> ReLU -> Dropout -> Linear -> 1-d).
        Three regression heads (toxicity, permeability, iri).
        One optional auxiliary multi-label classification head for Tox21
        (12 outputs), used in Phase 2 to provide a stronger gradient signal
        to the toxicity-relevant representations.

Loss
----
    Per-task standardized MSE for each regression head, averaged over rows
    that have a non-null label for that task. Per-task weight inversely
    proportional to (sqrt of) labeled-row count so the small Higgins tasks
    aren't drowned out by 303 IRI rows. Tox21 BCE loss when --tox21-aux is on.

Eval (Day 2 single-seed)
------------------------
    Reuses the RF random 70/15/15 split keyed on canonical SMILES so a given
    compound lands in the same split across tasks (no leakage between heads).
    Reports MAE / RMSE / R^2 / Spearman per task on val + test.
    Phase 2 will add 5-fold CV for the small tasks to match the RF baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..utils import RESULTS_DIR, get_logger, seed_everything

log = get_logger("models.chemberta")

CHEMBERTA_MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"

# Per-task regression heads
REG_TASKS = ["toxicity", "permeability", "iri"]


@dataclass
class ChemBertaConfig:
    model_name: str = CHEMBERTA_MODEL_NAME
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    head_hidden: int = 256
    head_dropout: float = 0.1
    max_length: int = 128
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.01
    patience: int = 5
    seed: int = 0
    tox21_aux: bool = False
    # Auxiliary loss weight. Total loss = cpa_huber + tox21_aux_weight * tox21_bce.
    # Kept low so the aux head provides regularization to the encoder without
    # dominating the CPA-task gradients (the aux task has 7800 compounds vs
    # ~330 for the CPA tasks). Sensitivity to this value should be checked
    # in a follow-up sweep; 0.1 is a reasonable starting point.
    tox21_aux_weight: float = 0.1
    smoke: bool = False  # 2-epoch tiny-subset run for local CI / sanity


# -------------------- Dataset / collator -----------------------------------


# Per-task default concentrations for FDA scoring + IRI/permeability rows.
# These match what build_dataset() uses; values are mol/kg.
TOX_REFERENCE_CONC_MOL_KG = 6.0   # mid-range from Higgins Dec 2025 (3/6/12)
IRI_REFERENCE_CONC_MOL_KG = 0.020  # DOLMEN splat assay (~20 mM)
PERM_REFERENCE_CONC_MOL_KG = 0.5   # Higgins Jan 2025 permeability assay
DEFAULT_CONC_MOL_KG = {
    "toxicity": TOX_REFERENCE_CONC_MOL_KG,
    "permeability": PERM_REFERENCE_CONC_MOL_KG,
    "iri": IRI_REFERENCE_CONC_MOL_KG,
}

# Concentration normalization. The toxicity range is 3-12 mol/kg with
# mean ~6; assay-fixed concentrations for the other tasks are 0.020 and
# 0.5 mol/kg. We z-score the toxicity range so the toxicity head sees
# values in a reasonable scale; the IRI/permeability rows look like very
# negative concentrations to the toxicity head, but their toxicity labels
# are NaN so the head's output for those rows is never used in loss.
CONC_MEAN = 6.0
CONC_STD = 3.0


def _wide_targets(long_df: pd.DataFrame) -> pd.DataFrame:
    """Build a row-per-(smiles, concentration) table for ChemBERTa training.

    Each row is one (compound, concentration) condition with the per-task
    label that was measured under that condition. Toxicity labels at
    different concentrations live in different rows so the toxicity head
    can see dose-response. IRI and permeability assays each have a single
    fixed concentration and produce one row per compound at that fixed
    concentration. The same compound can appear in multiple rows if it
    has measurements across multiple datasets / concentrations.

    Columns: smiles_canonical, concentration_mol_kg, toxicity, permeability, iri
    """
    sub = long_df[long_df["task"].isin(REG_TASKS)].copy()
    if "concentration_mol_kg" not in sub.columns:
        # Backward-compat: if loading an old long.parquet without the
        # concentration column, fall back to per-task default conc.
        sub["concentration_mol_kg"] = sub["task"].map(DEFAULT_CONC_MOL_KG)
    sub = sub.dropna(subset=["concentration_mol_kg"])
    wide = (
        sub.pivot_table(
            index=["smiles_canonical", "concentration_mol_kg"],
            columns="task",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    for t in REG_TASKS:
        if t not in wide.columns:
            wide[t] = np.nan
    return wide[["smiles_canonical", "concentration_mol_kg"] + REG_TASKS]


# -------------------- Model -----------------------------------------------


def _build_chemberta(config: ChemBertaConfig):
    """Load HF ChemBERTa + tokenizer, wrap encoder with LoRA. Returns
    (tokenizer, encoder_with_lora, hidden_size)."""
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "ChemBERTa needs `transformers` and `peft`. Install via "
            "`pip install -r requirements-deep.txt`."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # Force PyTorch's scaled_dot_product_attention (SDPA / FlashAttention-2 on
    # supported GPUs). Critical for Blackwell + bf16 stability: eager attention
    # in bf16 is prone to softmax underflow that produces NaN gradients in
    # backward; SDPA has built-in numerical stabilization (max-subtract + scale
    # + safe-softmax) and on Blackwell uses fused kernels that don't expose
    # those failure modes. transformers >=4.36 supports this for RoBERTa.
    try:
        base = AutoModel.from_pretrained(
            config.model_name, attn_implementation="sdpa",
        )
    except (TypeError, ValueError) as e:
        log.warning("attn_implementation=sdpa unavailable (%s); falling back to default", e)
        base = AutoModel.from_pretrained(config.model_name)
    hidden_size = base.config.hidden_size

    lora_cfg = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["query", "key", "value"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    encoder = get_peft_model(base, lora_cfg)
    encoder.print_trainable_parameters()
    return tokenizer, encoder, hidden_size


def _build_model(config: ChemBertaConfig):
    """Compose the multi-task architecture."""
    import torch
    from torch import nn

    tokenizer, encoder, hidden = _build_chemberta(config)

    class _Head(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, x):
            return self.net(x)

    class ChemBertaMultiTask(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.tokenizer = tokenizer
            # Toxicity head takes pooled embedding + scalar concentration.
            # IRI / permeability heads take pooled only (their assays use
            # a single fixed concentration so the input would be a constant
            # and the head would learn to ignore it).
            self.tox_head = _Head(hidden + 1, 1, config.head_hidden, config.head_dropout)
            self.heads = nn.ModuleDict({
                "permeability": _Head(hidden, 1, config.head_hidden, config.head_dropout),
                "iri": _Head(hidden, 1, config.head_hidden, config.head_dropout),
            })
            if config.tox21_aux:
                self.aux_head = _Head(hidden, 12, config.head_hidden, config.head_dropout)
            else:
                self.aux_head = None

        def encode(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            last = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return pooled

        def forward(self, input_ids, attention_mask, concentration):
            pooled = self.encode(input_ids, attention_mask)
            # Standardize concentration before concatenation so the head sees
            # values near zero-mean unit-variance over the toxicity range.
            conc_norm = (concentration.float() - CONC_MEAN) / CONC_STD
            pooled_with_conc = torch.cat(
                [pooled.float(), conc_norm.unsqueeze(-1)], dim=-1,
            )
            preds = {
                "toxicity": self.tox_head(pooled_with_conc).squeeze(-1),
                "permeability": self.heads["permeability"](pooled).squeeze(-1),
                "iri": self.heads["iri"](pooled).squeeze(-1),
            }
            if self.aux_head is not None:
                preds["tox21_aux"] = self.aux_head(pooled)
            return preds

    return ChemBertaMultiTask().to(_device())


def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _amp_dtype():
    """bf16 on hardware that supports it (A100, H100, Blackwell), else None
    (no autocast, fall back to fp32). bf16 has fp32's dynamic range so we
    don't need a GradScaler -- it just works in-place of fp32 for the
    forward + backward.

    Why this matters: ChemBERTa attention on Blackwell + fp32 is producing
    NaN gradients in ~95% of training steps (the user's previous run).
    bf16 attention is the standard cure -- Blackwell is designed for it,
    and the attention softmax + matmul kernels are more numerically robust
    in bf16 there than fp32 is.
    """
    import torch

    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported"):
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
    return None


# -------------------- Train / eval ----------------------------------------


@dataclass
class TaskStandardizer:
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> None:
        for t in REG_TASKS:
            vals = df[t].dropna().values
            if len(vals) == 0:
                self.means[t] = 0.0
                self.stds[t] = 1.0
            else:
                self.means[t] = float(vals.mean())
                self.stds[t] = float(vals.std()) or 1.0

    def transform(self, t: str, y: np.ndarray) -> np.ndarray:
        return (y - self.means[t]) / self.stds[t]

    def inverse(self, t: str, y_std: np.ndarray) -> np.ndarray:
        return y_std * self.stds[t] + self.means[t]


def _make_tox21_batches(
    tox21_long: pd.DataFrame,
    tokenizer,
    config: ChemBertaConfig,
    seed: int,
):
    """Build a DataLoader of (input_ids, mask, 12-task-labels, 12-task-mask)
    over the Tox21 dataset. We pivot the long-format Tox21 table to one row
    per compound with 12 binary columns; each (compound, task) pair has a
    label or NaN if unmeasured (mask=False).
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from .chemberta_lora import TOX21_TASK_NAMES  # forward ref (defined below)

    if tox21_long.empty:
        return None

    pivot = (
        tox21_long.pivot_table(
            index="smiles_canonical",
            columns="task",
            values="label",
            aggfunc="mean",
        )
        .reset_index()
    )
    for t in TOX21_TASK_NAMES:
        if t not in pivot.columns:
            pivot[t] = float("nan")
    pivot = pivot[["smiles_canonical"] + TOX21_TASK_NAMES].reset_index(drop=True)

    class _DS(Dataset):
        def __len__(self):
            return len(pivot)

        def __getitem__(self, i):
            row = pivot.iloc[i]
            s = row["smiles_canonical"]
            labels = [
                float(row[t]) if not pd.isna(row[t]) else 0.0 for t in TOX21_TASK_NAMES
            ]
            mask = [
                bool(not pd.isna(row[t])) for t in TOX21_TASK_NAMES
            ]
            return s, labels, mask

    def _collate(batch):
        smis, labels, masks = zip(*batch)
        toks = tokenizer(
            list(smis),
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )
        y_t = torch.tensor(list(labels), dtype=torch.float32)  # (B, 12)
        m_t = torch.tensor(list(masks), dtype=torch.bool)      # (B, 12)
        return toks, y_t, m_t

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        _DS(),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_collate,
        generator=g,
        num_workers=0,
    )


# Tox21 task names; populated from src.data.tox21 if available
try:
    from ..data.tox21 import TOX21_TASKS as TOX21_TASK_NAMES  # noqa: E402
except Exception:
    TOX21_TASK_NAMES = []


def _make_batches(
    rows: pd.DataFrame,
    tokenizer,
    config: ChemBertaConfig,
    shuffle: bool,
    seed: int,
):
    """Build a DataLoader from a (smiles, concentration, task-labels) wide table.

    Each row carries one (smiles, concentration_mol_kg) condition with whatever
    task labels are populated for that condition. The collator emits SMILES
    tokenization, a concentration tensor, plus per-task target+mask tensors.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    rows = rows.reset_index(drop=True)

    class _DS(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            row = rows.iloc[i]
            s = row["smiles_canonical"]
            c = float(row["concentration_mol_kg"])
            y = {t: float(row[t]) if not pd.isna(row[t]) else float("nan") for t in REG_TASKS}
            mask = {t: not pd.isna(row[t]) for t in REG_TASKS}
            return s, c, y, mask

    def _collate(batch):
        smis, concs, ys, masks = zip(*batch)
        toks = tokenizer(
            list(smis),
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )
        conc_tensor = torch.tensor(list(concs), dtype=torch.float32)
        y_tensors, m_tensors = {}, {}
        for t in REG_TASKS:
            y_tensors[t] = torch.tensor([y[t] for y in ys], dtype=torch.float32)
            m_tensors[t] = torch.tensor([m[t] for m in masks], dtype=torch.bool)
        return toks, conc_tensor, y_tensors, m_tensors

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        _DS(),
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=_collate,
        generator=g if shuffle else None,
        num_workers=0,
    )


def _task_weights(train_df: pd.DataFrame) -> dict[str, float]:
    """All tasks weighted equally per labeled sample.

    Earlier we used inverse-sqrt-of-task-count weighting to "compensate" for
    small tasks having few labels. On Blackwell+bf16 this turned out to be
    actively harmful: when a batch has 1-2 toxicity-labeled compounds and
    those predictions are off (common at init), the up-weighted toxicity
    loss dominates and produces gradient spikes that overflow bf16 backward.
    Result: ~80% of training batches got NaN-skipped.

    With uniform per-sample weighting, the loss reduces to
        total_squared_error / total_labeled_samples
    which is bounded near 1.0 on standardized targets at init and shrinks
    smoothly during training. Phase 2's Tox21 aux classification head is
    the right tool to give toxicity more signal -- not loss reweighting.
    """
    return {t: 1.0 for t in REG_TASKS}


def _epoch(
    model,
    loader,
    standardizer: TaskStandardizer,
    weights: dict[str, float],
    optimizer=None,
    tox21_loader=None,
    tox21_aux_weight: float = 0.0,
):
    import torch
    from contextlib import nullcontext

    is_train = optimizer is not None
    model.train(is_train)
    device = _device()
    amp = _amp_dtype()  # bf16 on supported GPUs, None on T4/CPU
    total_loss = 0.0
    n_batches = 0
    yhats = {t: [] for t in REG_TASKS}
    ys = {t: [] for t in REG_TASKS}
    # Aux loss is only computed during training. We cycle through the Tox21
    # dataloader as an infinite iterator so each CPA batch gets paired with
    # a Tox21 batch. The aux head exists only when config.tox21_aux=True
    # at model build time.
    use_aux = (
        is_train and tox21_loader is not None and tox21_aux_weight > 0.0
        and getattr(model, "aux_head", None) is not None
    )
    aux_iter = iter(tox21_loader) if use_aux else None
    aux_running_loss = 0.0
    aux_n = 0

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for toks, conc_t, y_t, m_t in loader:
            toks = {k: v.to(device) for k, v in toks.items()}
            conc_t = conc_t.to(device)

            # Autocast forward in bf16 where supported. Loss + standardization
            # arithmetic stay in fp32 (cast preds back) for numerical stability
            # in the small-batch multi-task setting.
            ac_ctx = (
                torch.autocast(device_type=device.type, dtype=amp)
                if amp is not None
                else nullcontext()
            )
            with ac_ctx:
                preds = model(**toks, concentration=conc_t)

            # Huber loss instead of MSE: gradient w.r.t. prediction is bounded
            # in [-1, 1] (delta=1.0). MSE has unbounded gradient on outliers
            # (any (p - y)^2 backward gives 2(p-y), which is huge for big errors
            # and overflows bf16 backward). Huber clips smoothly at delta and
            # behaves like MSE inside [-delta, delta].
            #
            # Aggregate as total Huber loss across all labeled (compound, task)
            # pairs / total labeled count. Bounded near 0.5 at init on
            # standardized targets; shrinks smoothly. No per-task weighting so
            # a single outlier in a small-task batch can't dominate.
            total_hl = torch.tensor(0.0, device=device, dtype=torch.float32)
            total_n = 0
            for t in REG_TASKS:
                m = m_t[t].to(device)
                if not m.any():
                    continue
                y = y_t[t].to(device)
                y_std = (y - standardizer.means[t]) / standardizer.stds[t]
                p = preds[t].float()  # bf16 -> fp32 for loss
                # Huber on the masked subset, with delta=1.0
                err = (p - y_std)[m]
                abs_err = err.abs()
                huber = torch.where(
                    abs_err < 1.0,
                    0.5 * err * err,
                    abs_err - 0.5,
                )
                total_hl = total_hl + huber.sum()
                total_n += int(m.sum().item())

                p_unstd = (p.detach() * standardizer.stds[t] + standardizer.means[t]).cpu().numpy()
                y_np = y.detach().cpu().numpy()
                m_np = m.cpu().numpy()
                yhats[t].extend(p_unstd[m_np].tolist())
                ys[t].extend(y_np[m_np].tolist())

            n_terms = total_n
            batch_loss = total_hl / max(total_n, 1)

            if n_terms == 0:
                continue

            # Tox21 auxiliary BCE loss. Run a separate forward on a Tox21
            # batch through the encoder + aux head, compute masked BCE, add
            # to total loss with low weight. Encoder is shared so the aux
            # gradient regularizes the same representation the CPA heads use.
            aux_loss_value = None
            if use_aux:
                try:
                    aux_batch = next(aux_iter)
                except StopIteration:
                    aux_iter = iter(tox21_loader)
                    aux_batch = next(aux_iter)
                aux_toks, aux_y, aux_m = aux_batch
                aux_toks = {k: v.to(device) for k, v in aux_toks.items()}
                aux_y = aux_y.to(device)
                aux_m = aux_m.to(device)
                aux_conc = torch.zeros(
                    aux_y.shape[0], dtype=torch.float32, device=device,
                )
                with ac_ctx:
                    aux_preds = model(**aux_toks, concentration=aux_conc)
                logits = aux_preds.get("tox21_aux")
                if logits is not None:
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits.float(), aux_y, reduction="none",
                    )
                    masked_bce = bce[aux_m]
                    if masked_bce.numel() > 0 and torch.isfinite(masked_bce).all():
                        aux_loss_value = masked_bce.mean()

            if is_train:
                step_loss = batch_loss
                if aux_loss_value is not None:
                    step_loss = step_loss + tox21_aux_weight * aux_loss_value
                    aux_running_loss += float(aux_loss_value.detach().cpu().item())
                    aux_n += 1
                if not torch.isfinite(step_loss):
                    log.warning(
                        "non-finite step loss (cpa=%s, aux=%s); skipping step",
                        float(batch_loss.detach()) if torch.isfinite(batch_loss) else "nan",
                        float(aux_loss_value.detach()) if aux_loss_value is not None and torch.isfinite(aux_loss_value) else "n/a",
                    )
                    optimizer.zero_grad()
                    continue
                optimizer.zero_grad()
                step_loss.backward()
                # Per-element value clipping. Unlike clip_grad_norm_ (which is
                # NaN-unsafe -- a single NaN in a grad tensor poisons the norm,
                # and norm-divided-by-NaN turns every grad NaN), clip_grad_value_
                # clamps each element to +/- max_value, naturally handling NaN
                # by setting it to the clamp limit. We also explicitly nan_to_num
                # before clipping for hard guarantees.
                for p in model.parameters():
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.5, neginf=-0.5)
                torch.nn.utils.clip_grad_value_(
                    [p for p in model.parameters() if p.requires_grad],
                    clip_value=1.0,
                )
                optimizer.step()
            total_loss += float(batch_loss.detach().cpu().item())
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    if use_aux and aux_n > 0:
        avg_aux = aux_running_loss / aux_n
        log.debug("epoch aux BCE loss avg=%.4f over %d batches", avg_aux, aux_n)
    return avg_loss, ys, yhats


def _train_one_fold(
    wide: pd.DataFrame,
    smi_train: list[str],
    smi_val: list[str],
    config: ChemBertaConfig,
    fold_label: str,
):
    """Train a fresh ChemBERTa multi-task model on smi_train; track val_loss
    on smi_val for early-stopping. Returns (model, standardizer) at the
    best-val checkpoint.
    """
    import torch
    from torch.optim import AdamW
    from ..eval import regression_metrics

    train_df = wide[wide["smiles_canonical"].isin(smi_train)]
    val_df = wide[wide["smiles_canonical"].isin(smi_val)]
    standardizer = TaskStandardizer()
    standardizer.fit(train_df)
    weights = _task_weights(train_df)
    log.info(
        "[%s] standardizer means=%s stds=%s",
        fold_label, standardizer.means, standardizer.stds,
    )
    log.info(
        "[%s] train rows=%d (unique smi=%d), val rows=%d (unique smi=%d)",
        fold_label, len(train_df), train_df["smiles_canonical"].nunique(),
        len(val_df), val_df["smiles_canonical"].nunique(),
    )
    log.info("[%s] task weights: %s", fold_label, weights)

    log.info("[%s] loading ChemBERTa-2 + LoRA rank=%d", fold_label, config.lora_rank)
    model = _build_model(config)
    train_loader = _make_batches(train_df, model.tokenizer, config, shuffle=True, seed=config.seed)
    val_loader = _make_batches(val_df, model.tokenizer, config, shuffle=False, seed=config.seed)

    # Tox21 aux loader (only used when config.tox21_aux=True). The loader
    # cycles indefinitely during _epoch so each CPA training step gets a
    # paired Tox21 batch. We load once per fold to amortize the parquet read.
    tox21_loader = None
    if config.tox21_aux:
        from ..data.tox21 import load_tox21
        tox21_long = load_tox21()
        if not tox21_long.empty:
            tox21_loader = _make_tox21_batches(
                tox21_long, model.tokenizer, config, seed=config.seed,
            )
            log.info(
                "[%s] tox21 aux head active: %d (compound, task) labels, weight=%.2f",
                fold_label, len(tox21_long), config.tox21_aux_weight,
            )
        else:
            log.warning(
                "[%s] tox21_aux=True but tox21 dataset empty/unavailable; "
                "running CPA-only", fold_label,
            )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_state = None
    for epoch in range(1, config.epochs + 1):
        train_loss, _, _ = _epoch(
            model, train_loader, standardizer, weights, optimizer,
            tox21_loader=tox21_loader,
            tox21_aux_weight=config.tox21_aux_weight,
        )
        val_loss, ys, yhats = _epoch(model, val_loader, standardizer, weights, optimizer=None)
        per_task = {t: regression_metrics(np.array(ys[t]), np.array(yhats[t])) for t in REG_TASKS}
        log.info(
            "[%s] epoch %d  train_loss=%.4f  val_loss=%.4f  "
            "tox MAE=%.3g rho=%.3g  perm MAE=%.3g rho=%.3g  iri MAE=%.3g rho=%.3g",
            fold_label, epoch, train_loss, val_loss,
            per_task["toxicity"]["mae"], per_task["toxicity"]["spearman"],
            per_task["permeability"]["mae"], per_task["permeability"]["spearman"],
            per_task["iri"]["mae"], per_task["iri"]["spearman"],
        )
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                log.info("[%s] early-stop at epoch %d (best=%d, val=%.4f)",
                         fold_label, epoch, best_epoch, best_val)
                break

    if best_state is None:
        log.error(
            "[%s] no epoch reduced val loss; every val batch produced NaN. "
            "Returning the final state for inspection.", fold_label,
        )
    else:
        model.load_state_dict({k: v.to(_device()) for k, v in best_state.items()})
    return model, standardizer


def _predict_at_conditions(
    model,
    standardizer: TaskStandardizer,
    rows: list[tuple[str, float]],
    config: ChemBertaConfig,
) -> dict[str, list[float]]:
    """Run the trained model on a list of (smiles, concentration) tuples
    and return {task: list-of-predictions} aligned to the input order.
    Predictions are in the original (unstandardized) target scale.
    """
    import torch
    from contextlib import nullcontext

    out: dict[str, list[float]] = {t: [] for t in REG_TASKS}
    if not rows:
        return out
    device = _device()
    amp = _amp_dtype()
    model.eval()
    bs = config.batch_size
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            batch = rows[i : i + bs]
            batch_smi = [r[0] for r in batch]
            batch_conc = [r[1] for r in batch]
            toks = model.tokenizer(
                batch_smi,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            conc_t = torch.tensor(batch_conc, dtype=torch.float32, device=device)
            ac_ctx = (
                torch.autocast(device_type=device.type, dtype=amp)
                if amp is not None else nullcontext()
            )
            with ac_ctx:
                preds = model(**toks, concentration=conc_t)
            for t in REG_TASKS:
                p_unstd = (
                    preds[t].float() * standardizer.stds[t] + standardizer.means[t]
                ).cpu().numpy()
                out[t].extend(p_unstd.tolist())
    return out


def _predict_smiles(
    model,
    wide: pd.DataFrame,
    standardizer: TaskStandardizer,
    smi_list: list[str],
    config: ChemBertaConfig,
) -> dict[str, dict]:
    """Predict per-task OOF values for the given smiles. Returns
    {task: {smiles: pred}} with toxicity predicted at the actual measured
    concentration when available (so the toxicity head sees its dose) and
    at TOX_REFERENCE_CONC_MOL_KG otherwise. IRI / permeability are predicted
    at their assay default concentration; the head doesn't actually use it
    so the choice is cosmetic.

    For CV OOF aggregation we want predictions at the measured conditions,
    so we pull rows from `wide` where they match the smi_list and have a
    label for each task.
    """
    out: dict[str, dict[str, float]] = {t: {} for t in REG_TASKS}
    rows_by_smi = wide[wide["smiles_canonical"].isin(smi_list)]
    for t in REG_TASKS:
        sub = rows_by_smi.dropna(subset=[t])
        if sub.empty:
            continue
        cond_rows = list(zip(
            sub["smiles_canonical"].tolist(),
            sub["concentration_mol_kg"].astype(float).tolist(),
        ))
        preds = _predict_at_conditions(model, standardizer, cond_rows, config)
        # Key by (smi, conc) for toxicity (multiple per compound), smi for others.
        if t == "toxicity":
            for (s, c), v in zip(cond_rows, preds[t]):
                out[t][f"{s}@{c:g}"] = float(v)
        else:
            for (s, _c), v in zip(cond_rows, preds[t]):
                out[t][s] = float(v)
    return out


def _make_config(args) -> ChemBertaConfig:
    return ChemBertaConfig(
        lora_rank=args.lora_rank,
        epochs=args.epochs if not args.smoke else 2,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        max_length=args.max_length,
        seed=args.seed,
        tox21_aux=getattr(args, "tox21_aux", False),
        tox21_aux_weight=getattr(args, "tox21_aux_weight", 0.1),
        smoke=args.smoke,
    )


def _eval_per_task(
    smi_to_pred: dict[str, dict[str, float]],
    wide: pd.DataFrame,
    smi_subset: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build per-task (y_true, y_pred) arrays from the predictions dict.

    For toxicity, prediction keys are "smiles@concentration" (multiple
    measurements per compound); for IRI / permeability, keys are smiles.
    Returns aligned (y, yhat) arrays per task, dropping rows where the
    prediction wasn't computed.
    """
    out = {}
    for t in REG_TASKS:
        sub = wide[wide["smiles_canonical"].isin(smi_subset)].dropna(subset=[t])
        ys, yhats = [], []
        for _, row in sub.iterrows():
            s = row["smiles_canonical"]
            c = float(row["concentration_mol_kg"])
            key = f"{s}@{c:g}" if t == "toxicity" else s
            if key in smi_to_pred[t]:
                ys.append(float(row[t]))
                yhats.append(smi_to_pred[t][key])
        out[t] = (np.array(ys), np.array(yhats))
    return out


def train_chemberta_multitask(long_df: pd.DataFrame, args) -> list[dict]:
    """Public entry. Dispatches to single-split or k-fold CV based on args.cv."""
    seed_everything(args.seed)
    wide = _wide_targets(long_df)
    config = _make_config(args)

    if args.cv:
        return _train_chemberta_kfold(wide, args, config)
    return _train_chemberta_single_split(wide, args, config)


def _train_chemberta_single_split(
    wide: pd.DataFrame, args, config: ChemBertaConfig,
) -> list[dict]:
    from ..data.splits import random_split_by_smiles
    from ..eval import parity_plot, regression_metrics

    splits = random_split_by_smiles(
        wide["smiles_canonical"].unique(), seed=config.seed,
        train=0.70, val=0.15, test=0.15,
    )
    smi_train = sorted([s for s in wide["smiles_canonical"] if s in splits["train"]])
    smi_val = sorted([s for s in wide["smiles_canonical"] if s in splits["val"]])
    smi_test = sorted([s for s in wide["smiles_canonical"] if s in splits["test"]])
    log.info(
        "ChemBERTa 70/15/15 split sizes: train=%d val=%d test=%d",
        len(smi_train), len(smi_val), len(smi_test),
    )

    model, standardizer = _train_one_fold(wide, smi_train, smi_val, config, fold_label="single")

    rows: list[dict] = []
    model_tag = f"chemberta_r{config.lora_rank}_seed{config.seed}"
    for split_name, smi_subset in (("val", smi_val), ("test", smi_test)):
        preds = _predict_smiles(model, wide, standardizer, smi_subset, config)
        per_task = _eval_per_task(preds, wide, smi_subset)
        for t in REG_TASKS:
            y, yh = per_task[t]
            m = regression_metrics(y, yh)
            rows.append({
                "model": model_tag,
                "task": t,
                "scheme": "70/15/15",
                "split": split_name,
                **m,
            })
            parity_plot(y, yh, task=t, split=split_name, model_tag=model_tag)
    return rows


def _train_chemberta_kfold(
    wide: pd.DataFrame, args, config: ChemBertaConfig,
) -> list[dict]:
    """5-fold CV: train k fresh ChemBERTas, each holding out one fold globally
    across all compounds. Aggregate OOF predictions for per-task metrics.
    """
    from ..data.splits import kfold_split_by_smiles
    from ..eval import parity_plot, regression_metrics

    k = args.cv_folds
    folds = kfold_split_by_smiles(
        wide["smiles_canonical"].unique(), k=k, seed=config.seed,
    )
    log.info("ChemBERTa %d-fold CV: fold sizes = %s", k, [len(test) for _, test in folds])

    # OOF predictions: smiles -> {task: pred}
    oof: dict[str, dict[str, float]] = {t: {} for t in REG_TASKS}
    for fi, (train_set, test_set) in enumerate(folds):
        smi_train_full = sorted(train_set)
        # Use a tiny inner val carve-out for early stopping (10% of train fold).
        # The inner val isn't task-specific; it's for loss-curve early stop.
        n_val = max(1, len(smi_train_full) // 10)
        smi_val_inner = smi_train_full[:n_val]
        smi_train = smi_train_full[n_val:]
        smi_test = sorted(test_set)
        log.info(
            "ChemBERTa fold %d/%d: train=%d (inner_val=%d) test=%d",
            fi + 1, k, len(smi_train), len(smi_val_inner), len(smi_test),
        )
        model, standardizer = _train_one_fold(
            wide, smi_train, smi_val_inner, config, fold_label=f"fold {fi}"
        )
        preds = _predict_smiles(model, wide, standardizer, smi_test, config)
        for t in REG_TASKS:
            for s, p in preds[t].items():
                oof[t][s] = p
        # Free GPU memory between folds
        import torch
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows: list[dict] = []
    model_tag = f"chemberta_r{config.lora_rank}_seed{config.seed}"
    scheme = f"{k}-fold-CV"
    for t in REG_TASKS:
        sub = wide.dropna(subset=[t])
        ys, yhats = [], []
        for _, row in sub.iterrows():
            s = row["smiles_canonical"]
            c = float(row["concentration_mol_kg"])
            # Toxicity OOF preds are keyed by "smi@conc" because the same
            # compound is predicted at multiple measured concentrations;
            # IRI / permeability are keyed by bare smiles.
            key = f"{s}@{c:g}" if t == "toxicity" else s
            if key in oof[t]:
                ys.append(float(row[t]))
                yhats.append(oof[t][key])
        y_arr, yh_arr = np.array(ys), np.array(yhats)
        m = regression_metrics(y_arr, yh_arr)
        rows.append({
            "model": model_tag,
            "task": t,
            "scheme": scheme,
            "split": "oof",
            **m,
        })
        parity_plot(y_arr, yh_arr, task=t, split=f"oof_{k}fold",
                    model_tag=model_tag)
    return rows
