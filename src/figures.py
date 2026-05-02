"""Generate publication figures for the README from results_table.csv +
results/candidates/top20.csv. Run after Phase 2 to refresh the README plots:

    python -m src.figures
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

from .utils import FIGURES_DIR, RESULTS_DIR, get_logger

log = get_logger("figures")

TABLE_PATH = RESULTS_DIR / "results_table.csv"
TOP20_PATH = RESULTS_DIR / "candidates" / "top20.csv"


def _spearman_summary_plot() -> Path:
    """Grouped bar chart: Spearman per (task, scheme), grouped by architecture.

    Pulls every row from results_table.csv that has a non-null spearman.
    Filters to OOF / test rows for fair comparison (skips train rows).
    """
    df = pd.read_csv(TABLE_PATH)
    df = df[df["spearman"].notna()].copy()
    # Keep only "comparable" rows: oof, test, or val. Drop train.
    df = df[df["split"].isin(["oof", "test", "val"])]

    # Architecture label
    def arch(row):
        if "rf" in row["model"]:
            return "RF"
        if "chemberta" in row["model"]:
            return "ChemBERTa+LoRA"
        return row["model"]
    df["arch"] = df.apply(arch, axis=1)

    # Friendly scheme label
    def scheme_label(row):
        s, sp = row["scheme"], row["split"]
        if "ensemble" in s:
            return f"{s.split('-')[0]} 5-fold (5-seed)"
        if "5-fold" in s:
            return "random 5-fold"
        if "LOO" in s:
            return "random LOO"
        if "70/15/15" in s:
            return f"random 70/15/15 ({sp})"
        return s
    df["scheme_label"] = df.apply(scheme_label, axis=1)

    # Per-task subplot
    tasks = ["iri", "toxicity", "permeability"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    archs = ["RF", "ChemBERTa+LoRA"]
    arch_colors = {"RF": "#2c7fb8", "ChemBERTa+LoRA": "#e34a33"}

    for ax, task in zip(axes, tasks):
        sub = df[df["task"] == task].sort_values(["scheme_label", "arch"])
        scheme_order = sorted(sub["scheme_label"].unique())
        n_groups = len(scheme_order)
        bar_w = 0.38
        x = np.arange(n_groups)
        for i, a in enumerate(archs):
            vals = []
            for s in scheme_order:
                row = sub[(sub["arch"] == a) & (sub["scheme_label"] == s)]
                vals.append(row["spearman"].iloc[0] if not row.empty else np.nan)
            offset = (i - 0.5) * bar_w
            bars = ax.bar(x + offset, vals, width=bar_w * 0.95,
                          color=arch_colors[a], edgecolor="black", linewidth=0.4,
                          label=a if ax is axes[0] else None)
            # Annotate values
            for xi, v in zip(x, vals):
                if np.isfinite(v):
                    ax.text(xi + offset, v + 0.02 if v >= 0 else v - 0.05,
                            f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top",
                            fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(scheme_order, rotation=20, ha="right", fontsize=8)
        ns = sub.drop_duplicates("scheme_label")["n"].iloc[0] if not sub.empty else "n/a"
        ax.set_title(f"{task}  (n={int(ns) if isinstance(ns, (int, np.integer)) else ns})", fontsize=11)
        ax.axhline(0, color="black", lw=0.6)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_ylim(-0.1, 0.7)

    axes[0].set_ylabel("Spearman ρ (held-out)")
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Held-out Spearman by task × split × architecture", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "spearman_summary.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)
    return out


def _pareto_2d_top20_plot() -> Path | None:
    """2D Pareto: predicted toxicity (x) vs predicted permeability (y),
    color = predicted IRI. Top-20 candidates highlighted; top-5 labeled
    with hand-tuned offsets to avoid collisions with the marker circles.
    """
    if not TOP20_PATH.exists():
        log.warning("no %s; skipping pareto plot", TOP20_PATH)
        return None

    all_path = RESULTS_DIR / "candidates" / "all_scored.csv"
    if not all_path.exists():
        log.warning("no %s; skipping", all_path)
        return None

    all_df = pd.read_csv(all_path)
    top = pd.read_csv(TOP20_PATH)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    # Background: all candidates, colored by predicted IRI
    sc = ax.scatter(
        all_df["toxicity_mean"],
        all_df["permeability_mean"],
        c=all_df["iri_mean"],
        s=22, alpha=0.65, cmap="viridis_r",
        edgecolors="white", linewidths=0.4,
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("predicted IRI %MGS  (lower = stronger inhibition)", fontsize=10)

    # Top-20 highlighted as red rings
    ax.scatter(
        top["toxicity_mean"], top["permeability_mean"],
        s=180, facecolors="none", edgecolors="#d62728", linewidths=1.8, zorder=4,
        label=f"top-{len(top)} (Pareto + composite)",
    )

    # Label top-5 with adjustText so labels don't collide with markers or
    # each other. The top-5 candidates cluster tightly in (toxicity,
    # permeability) space, so any fixed-offset scheme stacks labels on top
    # of each other. adjustText runs an iterative force-based layout that
    # repels labels from points, axes, and other labels, then draws leader
    # lines to the original markers.
    from adjustText import adjust_text

    n_label = min(5, len(top))
    texts = []
    for _, row in top.head(n_label).iterrows():
        t = ax.text(
            row["toxicity_mean"], row["permeability_mean"],
            row["ingredient_name"][:24].title(),
            fontsize=9, color="#8b0000", weight="medium",
            bbox=dict(
                boxstyle="round,pad=0.25", fc="white", ec="#8b0000",
                lw=0.7, alpha=0.95,
            ),
        )
        texts.append(t)
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="#8b0000", lw=0.8, alpha=0.75),
        expand=(2.0, 2.0),
        force_text=(1.2, 1.4),
        force_static=(1.0, 1.2),
        only_move={"points": "xy", "text": "xy", "objects": "xy"},
    )

    ax.set_xlabel("predicted toxicity  (mortality % at 6 mol/kg, 4 °C; lower = better)", fontsize=10)
    ax.set_ylabel("predicted permeability  (P_CPA × 10⁻³ s⁻¹; higher = better)", fontsize=10)
    ax.set_title(
        f"FDA IID candidates scored by ChemBERTa ensemble (n={len(all_df)})\n"
        f"red rings: top-{len(top)} by Pareto + composite score",
        fontsize=11,
    )
    ax.legend(loc="lower right", frameon=True, fontsize=9, framealpha=0.95)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "pareto_2d_top20.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)
    return out


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if TABLE_PATH.exists():
        _spearman_summary_plot()
    if TOP20_PATH.exists():
        _pareto_2d_top20_plot()


if __name__ == "__main__":
    main()
