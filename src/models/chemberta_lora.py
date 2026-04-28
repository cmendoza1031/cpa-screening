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
    smoke: bool = False  # 2-epoch tiny-subset run for local CI / sanity


# -------------------- Dataset / collator -----------------------------------


def _wide_targets(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot (smiles, task, value) long-format into wide per-compound table.

    Columns: smiles_canonical, toxicity, permeability, iri (any of these
    may be NaN for a given compound).
    """
    wide = (
        long_df[long_df["task"].isin(REG_TASKS)]
        .pivot_table(
            index="smiles_canonical",
            columns="task",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    for t in REG_TASKS:
        if t not in wide.columns:
            wide[t] = np.nan
    return wide[["smiles_canonical"] + REG_TASKS]


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
            self.heads = nn.ModuleDict(
                {t: _Head(hidden, 1, config.head_hidden, config.head_dropout) for t in REG_TASKS}
            )
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

        def forward(self, input_ids, attention_mask):
            pooled = self.encode(input_ids, attention_mask)
            preds = {t: self.heads[t](pooled).squeeze(-1) for t in REG_TASKS}
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


def _make_batches(
    smiles: list[str],
    targets: pd.DataFrame,
    tokenizer,
    config: ChemBertaConfig,
    shuffle: bool,
    seed: int,
):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class _DS(Dataset):
        def __init__(self):
            self.smiles = smiles
            self.targets = targets

        def __len__(self):
            return len(self.smiles)

        def __getitem__(self, i):
            s = self.smiles[i]
            row = self.targets[self.targets["smiles_canonical"] == s].iloc[0]
            y = {t: float(row[t]) if not pd.isna(row[t]) else float("nan") for t in REG_TASKS}
            mask = {t: not pd.isna(row[t]) for t in REG_TASKS}
            return s, y, mask

    def _collate(batch):
        smis, ys, masks = zip(*batch)
        toks = tokenizer(
            list(smis),
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )
        y_tensors, m_tensors = {}, {}
        for t in REG_TASKS:
            y_tensors[t] = torch.tensor([y[t] for y in ys], dtype=torch.float32)
            m_tensors[t] = torch.tensor([m[t] for m in masks], dtype=torch.bool)
        return toks, y_tensors, m_tensors

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
    """Inverse-sqrt-of-count weighting so small tasks aren't drowned out."""
    counts = {t: int(train_df[t].notna().sum()) for t in REG_TASKS}
    raw = {t: 1.0 / math.sqrt(max(counts[t], 1)) for t in REG_TASKS}
    s = sum(raw.values()) or 1.0
    return {t: raw[t] / s * len(REG_TASKS) for t in REG_TASKS}


def _epoch(
    model,
    loader,
    standardizer: TaskStandardizer,
    weights: dict[str, float],
    optimizer=None,
):
    import torch

    is_train = optimizer is not None
    model.train(is_train)
    device = _device()
    total_loss = 0.0
    n_batches = 0
    per_task_se = {t: 0.0 for t in REG_TASKS}
    per_task_n = {t: 0 for t in REG_TASKS}
    yhats = {t: [] for t in REG_TASKS}
    ys = {t: [] for t in REG_TASKS}

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for toks, y_t, m_t in loader:
            toks = {k: v.to(device) for k, v in toks.items()}
            preds = model(**toks)

            batch_loss = torch.tensor(0.0, device=device)
            n_terms = 0
            for t in REG_TASKS:
                m = m_t[t].to(device)
                if not m.any():
                    continue
                y = y_t[t].to(device)
                y_std = (y - standardizer.means[t]) / standardizer.stds[t]
                p = preds[t]
                loss_t = ((p - y_std) ** 2)[m].mean()
                batch_loss = batch_loss + weights[t] * loss_t
                n_terms += 1

                p_unstd = (p.detach() * standardizer.stds[t] + standardizer.means[t]).cpu().numpy()
                y_np = y.detach().cpu().numpy()
                m_np = m.cpu().numpy()
                yhats[t].extend(p_unstd[m_np].tolist())
                ys[t].extend(y_np[m_np].tolist())
                per_task_se[t] += float(((p_unstd[m_np] - y_np[m_np]) ** 2).sum())
                per_task_n[t] += int(m_np.sum())

            if n_terms == 0:
                continue
            if is_train:
                if not torch.isfinite(batch_loss):
                    log.warning("non-finite batch loss (%s); skipping step", batch_loss.item())
                    optimizer.zero_grad()
                    continue
                optimizer.zero_grad()
                batch_loss.backward()
                # Skip the optimizer step if any grad is non-finite. Training
                # transformers fine-tunes on tiny batches occasionally produces
                # NaN grads via softmax saturation in attention; clip_grad_norm_
                # itself is NaN-unsafe (norm becomes NaN, clipped grads become
                # NaN, optimizer pushes params to NaN).
                bad_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters() if p.requires_grad
                )
                if bad_grad:
                    log.warning("non-finite gradients; skipping step")
                    optimizer.zero_grad()
                    continue
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
            total_loss += float(batch_loss.detach().cpu().item())
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, ys, yhats


def train_chemberta_multitask(long_df: pd.DataFrame, args) -> list[dict]:
    """Train one multi-task ChemBERTa+LoRA model and return metrics rows."""
    import torch
    from torch.optim import AdamW

    from ..data.splits import random_split_by_smiles
    from ..eval import parity_plot, regression_metrics

    config = ChemBertaConfig(
        lora_rank=args.lora_rank,
        epochs=args.epochs if not args.smoke else 2,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        max_length=args.max_length,
        seed=args.seed,
        tox21_aux=False,  # Day 2: off; Phase 2 turns this on
        smoke=args.smoke,
    )
    seed_everything(config.seed)

    wide = _wide_targets(long_df)
    # Smoke = full data (so per-task label coverage is realistic), 2 epochs.
    # Subsampling produces degenerate batches (e.g. all 16 batch rows IRI-only)
    # that exercise edge cases more than the model itself.

    splits = random_split_by_smiles(
        wide["smiles_canonical"].unique(), seed=config.seed,
        train=0.70, val=0.15, test=0.15,
    )
    smi_train = sorted([s for s in wide["smiles_canonical"] if s in splits["train"]])
    smi_val = sorted([s for s in wide["smiles_canonical"] if s in splits["val"]])
    smi_test = sorted([s for s in wide["smiles_canonical"] if s in splits["test"]])
    log.info(
        "ChemBERTa split sizes: train=%d val=%d test=%d",
        len(smi_train), len(smi_val), len(smi_test),
    )

    train_df = wide[wide["smiles_canonical"].isin(smi_train)]
    standardizer = TaskStandardizer()
    standardizer.fit(train_df)
    log.info("standardizer means=%s stds=%s", standardizer.means, standardizer.stds)

    weights = _task_weights(train_df)
    log.info("task weights: %s", weights)

    log.info("loading ChemBERTa-2 + LoRA rank=%d", config.lora_rank)
    model = _build_model(config)
    train_loader = _make_batches(smi_train, wide, model.tokenizer, config, shuffle=True, seed=config.seed)
    val_loader = _make_batches(smi_val, wide, model.tokenizer, config, shuffle=False, seed=config.seed)
    test_loader = _make_batches(smi_test, wide, model.tokenizer, config, shuffle=False, seed=config.seed)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    for epoch in range(1, config.epochs + 1):
        train_loss, _, _ = _epoch(model, train_loader, standardizer, weights, optimizer)
        val_loss, ys, yhats = _epoch(model, val_loader, standardizer, weights, optimizer=None)
        per_task = {t: regression_metrics(np.array(ys[t]), np.array(yhats[t])) for t in REG_TASKS}
        log.info(
            "epoch %d  train_loss=%.4f  val_loss=%.4f  "
            "tox MAE=%.3g rho=%.3g  perm MAE=%.3g rho=%.3g  iri MAE=%.3g rho=%.3g",
            epoch, train_loss, val_loss,
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
                log.info("early-stop at epoch %d (best=%d, val=%.4f)", epoch, best_epoch, best_val)
                break

    if best_epoch < 0:
        log.error(
            "no epoch reduced val loss below the initial sentinel; this means "
            "every val batch produced NaN. Returning whatever the final state "
            "is (likely useless) so we can still inspect the failure mode."
        )
    else:
        model.load_state_dict({k: v.to(_device()) for k, v in best_state.items()})

    rows: list[dict] = []
    model_tag = f"chemberta_r{config.lora_rank}_seed{config.seed}"
    for split_name, loader in (("val", val_loader), ("test", test_loader)):
        _, ys, yhats = _epoch(model, loader, standardizer, weights, optimizer=None)
        for t in REG_TASKS:
            y = np.array(ys[t])
            yh = np.array(yhats[t])
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
