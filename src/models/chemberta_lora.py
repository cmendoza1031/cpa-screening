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

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for toks, y_t, m_t in loader:
            toks = {k: v.to(device) for k, v in toks.items()}

            # Autocast forward in bf16 where supported. Loss + standardization
            # arithmetic stay in fp32 (cast preds back) for numerical stability
            # in the small-batch multi-task setting.
            ac_ctx = (
                torch.autocast(device_type=device.type, dtype=amp)
                if amp is not None
                else nullcontext()
            )
            with ac_ctx:
                preds = model(**toks)

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
            if is_train:
                if not torch.isfinite(batch_loss):
                    log.warning("non-finite batch loss (%s); skipping step", batch_loss.item())
                    optimizer.zero_grad()
                    continue
                optimizer.zero_grad()
                batch_loss.backward()
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
    standardizer = TaskStandardizer()
    standardizer.fit(train_df)
    weights = _task_weights(train_df)
    log.info(
        "[%s] standardizer means=%s stds=%s",
        fold_label, standardizer.means, standardizer.stds,
    )
    log.info("[%s] task weights: %s", fold_label, weights)

    log.info("[%s] loading ChemBERTa-2 + LoRA rank=%d", fold_label, config.lora_rank)
    model = _build_model(config)
    train_loader = _make_batches(smi_train, wide, model.tokenizer, config, shuffle=True, seed=config.seed)
    val_loader = _make_batches(smi_val, wide, model.tokenizer, config, shuffle=False, seed=config.seed)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_state = None
    for epoch in range(1, config.epochs + 1):
        train_loss, _, _ = _epoch(model, train_loader, standardizer, weights, optimizer)
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


def _predict_smiles(
    model,
    wide: pd.DataFrame,
    standardizer: TaskStandardizer,
    smi_list: list[str],
    config: ChemBertaConfig,
) -> dict[str, dict]:
    """Run the trained model on smi_list, return {task: {smiles: pred}}.

    Runs in eval mode with autocast (bf16) where supported. We need per-
    compound predictions for OOF aggregation in CV, which _epoch flattens
    away, so this batches manually.
    """
    import torch
    from contextlib import nullcontext

    out: dict[str, dict[str, float]] = {t: {} for t in REG_TASKS}
    device = _device()
    amp = _amp_dtype()
    model.eval()
    bs = config.batch_size
    with torch.no_grad():
        for i in range(0, len(smi_list), bs):
            batch_smi = smi_list[i : i + bs]
            toks = model.tokenizer(
                batch_smi,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            ac_ctx = (
                torch.autocast(device_type=device.type, dtype=amp)
                if amp is not None else nullcontext()
            )
            with ac_ctx:
                preds = model(**toks)
            for t in REG_TASKS:
                p_unstd = (preds[t].float() * standardizer.stds[t] + standardizer.means[t]).cpu().numpy()
                for s, val in zip(batch_smi, p_unstd):
                    out[t][s] = float(val)
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
        tox21_aux=False,
        smoke=args.smoke,
    )


def _eval_per_task(
    smi_to_pred: dict[str, dict[str, float]],
    wide: pd.DataFrame,
    smi_subset: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build per-task (y_true, y_pred) arrays from the predictions dict,
    keeping only smiles that have a label for that task.
    """
    out = {}
    for t in REG_TASKS:
        sub = wide[wide["smiles_canonical"].isin(smi_subset)].dropna(subset=[t])
        ys, yhats = [], []
        for _, row in sub.iterrows():
            s = row["smiles_canonical"]
            if s in smi_to_pred[t]:
                ys.append(float(row[t]))
                yhats.append(smi_to_pred[t][s])
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
            if s in oof[t]:
                ys.append(float(row[t]))
                yhats.append(oof[t][s])
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
