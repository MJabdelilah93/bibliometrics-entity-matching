"""figures.py -- Manuscript-ready figures for the ablation stage (E6).

Produces:

  Main figure (paper body):
    figure_ablation_main.{png/pdf/svg}
    -- Grouped bar chart: F1_match +/- 95% CI for each variant in A2 / A4 / A5,
      grouped by task (AND left, AIN right).

  K-sensitivity figure (supplement):
    figure_ablation_k_sensitivity.{png/pdf/svg}
    -- Line plot: blocking recall at K (x-axis = K) per task.

  Threshold-sweep heat map (supplement, one per task):
    figure_ablation_sweep_{task}.{png/pdf/svg}
    -- Heat map of F1_match over the (t_match x m_signals) grid.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# Colour palette (consistent with E5 evaluation figures)
_TASK_COLORS = {"and": "#2166ac", "ain": "#d6604d"}
_VARIANT_MARKERS = {
    "bem_full":           ("o",  "#333333"),
    "c5_direct":          ("s",  "#4dac26"),
    "c5_conf":            ("D",  "#b8e186"),
    "remove_coauthor":    ("^",  "#f4a582"),
    "remove_affiliation": ("v",  "#d6604d"),
    "raw_affil_only":     ("^",  "#92c5de"),
    "remove_author_link": ("v",  "#4393c3"),
    "single_prompt":      ("*",  "#e08214"),
}
_FLOOR_LINE = 0.90


def build_ablation_figures(
    results: dict[str, Any],
    output_dir: Path,
    tasks: list[str],
    dpi: int = 300,
    fmt: str = "png",
) -> dict[str, Path]:
    """Build and save all ablation figures.

    Args:
        results: dict from runner.run_ablations().
        output_dir: Destination directory.
        tasks: Task names in display order.
        dpi: Figure resolution.
        fmt: File format ('png', 'pdf', or 'svg').

    Returns:
        Dict of {label: path} for every figure written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # --- Main ablation figure ---
    main_rows = _collect_main_rows(results, tasks)
    if main_rows:
        path = _plot_main(main_rows, output_dir, tasks, dpi, fmt)
        if path:
            written["figure_ablation_main"] = path

    # --- K-sensitivity figure ---
    a3 = results.get("a3_k_sensitivity", {})
    if a3:
        path = _plot_k_sensitivity(a3, output_dir, tasks, dpi, fmt)
        if path:
            written["figure_ablation_k_sensitivity"] = path

    # --- Threshold-sweep heat maps ---
    a6 = results.get("a6_threshold_sweep", {})
    for task, sweep_df in a6.items():
        if isinstance(sweep_df, pd.DataFrame) and not sweep_df.empty:
            path = _plot_threshold_sweep(sweep_df, task, output_dir, dpi, fmt)
            if path:
                written[f"figure_ablation_sweep_{task}"] = path

    return written


# ---------------------------------------------------------------------------
# Main ablation bar chart
# ---------------------------------------------------------------------------

def _collect_main_rows(results: dict, tasks: list[str]) -> list[dict]:
    rows = []
    for key in ("a2_no_guards", "a1_single_prompt",
                "a4_missing_fields_and", "a5_missing_fields_ain"):
        data = results.get(key, {})
        if not data:
            continue
        for _, variant_list in data.items():
            if not isinstance(variant_list, list):
                continue
            for entry in variant_list:
                if not isinstance(entry, dict):
                    continue
                t = entry.get("task", "")
                if t not in tasks:
                    continue
                m = entry.get("metrics", {})
                ci = m.get("ci_f1_match", [None, None]) or [None, None]
                rows.append({
                    "task":    t,
                    "variant": entry.get("variant", ""),
                    "f1":      m.get("f1_match"),
                    "ci_lo":   ci[0] if len(ci) > 0 else None,
                    "ci_hi":   ci[1] if len(ci) > 1 else None,
                })
    return rows


def _plot_main(
    rows: list[dict],
    output_dir: Path,
    tasks: list[str],
    dpi: int,
    fmt: str,
) -> Path | None:
    df = pd.DataFrame(rows)
    if df.empty:
        return None

    variants = df["variant"].unique().tolist()
    n_variants = len(variants)
    n_tasks    = len(tasks)

    fig, axes = plt.subplots(1, n_tasks, figsize=(max(6, 3 * n_tasks), 4.5), sharey=True)
    if n_tasks == 1:
        axes = [axes]

    x = np.arange(n_variants)
    bar_width = 0.55

    for ax, task in zip(axes, tasks):
        sub = df[df["task"] == task]
        f1s  = []
        errs = []
        colors = []
        for v in variants:
            row = sub[sub["variant"] == v]
            if row.empty:
                f1s.append(float("nan"))
                errs.append((float("nan"), float("nan")))
            else:
                f1 = row.iloc[0]["f1"] or float("nan")
                lo = row.iloc[0]["ci_lo"]
                hi = row.iloc[0]["ci_hi"]
                f1s.append(float(f1) if f1 is not None and not math.isnan(f1) else float("nan"))
                errs.append((
                    abs(f1 - lo) if lo is not None and not math.isnan(lo) else 0,
                    abs(hi - f1) if hi is not None and not math.isnan(hi) else 0,
                ))
            _, col = _VARIANT_MARKERS.get(v, ("o", "#888888"))
            colors.append(col)

        yerr = np.array([[e[0] for e in errs], [e[1] for e in errs]])
        # Replace nan errors with 0
        yerr = np.where(np.isnan(yerr), 0, yerr)

        bars = ax.bar(
            x, f1s, width=bar_width,
            color=colors, alpha=0.85,
            yerr=yerr, capsize=4, error_kw={"elinewidth": 1, "capthick": 1},
        )
        ax.axhline(_FLOOR_LINE, color="#888888", linestyle="--", linewidth=0.8, label="precision floor")
        ax.set_title(task.upper(), fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        if ax is axes[0]:
            ax.set_ylabel("F1$_M$", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Ablation Study -- BEM Variant Comparison", fontsize=12, y=1.02)
    fig.tight_layout()

    out_path = output_dir / f"figure_ablation_main.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# K-sensitivity line plot
# ---------------------------------------------------------------------------

def _plot_k_sensitivity(
    a3: dict,
    output_dir: Path,
    tasks: list[str],
    dpi: int,
    fmt: str,
) -> Path | None:
    rows = []
    for task, entries in a3.items():
        for e in entries:
            rows.append({
                "task":  task,
                "K":     e.get("k"),
                "br_m":  e.get("blocking_recall_match"),
                "f1_k":  e.get("bem_f1_at_k"),
            })
    if not rows:
        return None

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for ax, (metric, label) in zip(axes, [("br_m", "Blocking recall (match)"), ("f1_k", "BEM F1$_M$ at $K$")]):
        for task in tasks:
            sub = df[df["task"] == task].sort_values("K")
            if sub.empty:
                continue
            col = _TASK_COLORS.get(task, "#333333")
            ax.plot(sub["K"], sub[metric], marker="o", color=col, label=task.upper(), linewidth=1.5)
            ax.set_xlabel("$K$ (candidates per anchor)", fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.set_xticks(df["K"].dropna().unique().astype(int).tolist())
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)

    fig.suptitle("Ablation A3: K Sensitivity", fontsize=12)
    fig.tight_layout()

    out_path = output_dir / f"figure_ablation_k_sensitivity.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Threshold-sweep heat map
# ---------------------------------------------------------------------------

def _plot_threshold_sweep(
    sweep_df: pd.DataFrame,
    task: str,
    output_dir: Path,
    dpi: int,
    fmt: str,
) -> Path | None:
    if "t_match" not in sweep_df.columns or "m_signals" not in sweep_df.columns:
        return None

    pivot = sweep_df.pivot_table(
        index="m_signals", columns="t_match", values="f1_match", aggfunc="first"
    )
    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 3.5))
    data_arr = pivot.values.astype(float)
    im = ax.imshow(
        data_arr, aspect="auto", origin="lower",
        cmap="RdYlGn", vmin=0.0, vmax=1.0,
    )
    plt.colorbar(im, ax=ax, label="F1$_M$")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns], fontsize=8, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(int(m)) for m in pivot.index], fontsize=8)
    ax.set_xlabel("$t_{match}$ (confidence threshold)", fontsize=10)
    ax.set_ylabel("$m_{signals}$ (min signal categories)", fontsize=10)
    ax.set_title(f"Ablation A6: threshold sweep -- {task.upper()} (F1$_M$)", fontsize=11)

    # Annotate cells
    for i in range(data_arr.shape[0]):
        for j in range(data_arr.shape[1]):
            v = data_arr[i, j]
            if not math.isnan(v):
                text_color = "black" if 0.3 < v < 0.8 else "white"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color=text_color)

    fig.tight_layout()
    out_path = output_dir / f"figure_ablation_sweep_{task}.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
