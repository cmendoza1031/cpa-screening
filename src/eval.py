"""Metrics + parity plots for the multi-task regression problem."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

# Force a non-interactive matplotlib backend before any pyplot import so this
# works in headless environments (Colab, CI, sandbox).
os.environ.setdefault("MPLBACKEND", "Agg")

from .utils import FIGURES_DIR, get_logger

log = get_logger("eval")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, R^2, Spearman. Robust to empty input, NaN predictions
    (from a freshly-initialized network mid-training), and zero-variance.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"),
                "r2": float("nan"), "spearman": float("nan")}
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return {"n": int(y_true.size), "mae": float("nan"), "rmse": float("nan"),
                "r2": float("nan"), "spearman": float("nan")}
    yt, yp = y_true[finite], y_pred[finite]
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    if n_finite >= 2 and np.std(yt) > 0:
        r2 = float(r2_score(yt, yp))
        rho_res = spearmanr(yt, yp)
        rho = float(rho_res.statistic if hasattr(rho_res, "statistic") else rho_res[0])
    else:
        r2 = float("nan")
        rho = float("nan")
    return {"n": int(y_true.size), "mae": mae, "rmse": rmse, "r2": r2, "spearman": rho}


def parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
    split: str,
    model_tag: str,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Save a parity plot (predicted vs measured). Returns the file path."""
    if y_true.size == 0:
        return None
    out_dir = out_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(y_true, y_pred, alpha=0.7, s=20)

    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.5)

    metrics = regression_metrics(y_true, y_pred)
    ax.set_xlabel(f"measured {task}")
    ax.set_ylabel(f"predicted {task}")
    ax.set_title(
        f"{model_tag} | {task} | {split}\n"
        f"n={metrics['n']}  MAE={metrics['mae']:.3g}  R2={metrics['r2']:.3f}"
    )
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    path = out_dir / f"parity_{model_tag}_{task}_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("wrote %s", path)
    return path


def summarize_runs(rf_results: dict, model_tag: str) -> list[dict]:
    """Flatten the RF results dict into a list of one row per (task, split)."""
    rows = []
    for task, art in rf_results.items():
        for split in ("train", "val", "test"):
            y = art[f"y_{split}"]
            yh = art[f"yhat_{split}"]
            m = regression_metrics(y, yh)
            rows.append(
                {
                    "model": model_tag,
                    "task": task,
                    "split": split,
                    **m,
                }
            )
    return rows
