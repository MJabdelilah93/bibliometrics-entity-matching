"""figures.py — Precision–coverage figure for the BEM evaluation manuscript.

Figure design
-------------
One figure per task (AND, AIN).

  X-axis : Coverage = fraction of binary-gold pairs predicted as MATCH at
           threshold t  (= n_pred_match(t) / n_total_binary).
           As t decreases from 1 to 0, more pairs are predicted match →
           coverage increases, precision decreases.

  Y-axis : Precision_match at threshold t.

  Curves : One line per fair baseline (deterministic, fuzzy, tfidf).
           Dashed line for the auxiliary Scopus-ID comparator (AND only).

  Markers: Diamond (◇) = precision_floor_match operating point.
           Star (★)    = f1_optimal operating point.
           Square (□)  = BEM system (single point; no threshold curve).

The plot is generated from the full diagnostics parquets (101 threshold
points, one row per threshold grid value) already produced by Stage E4.

Requires matplotlib.  If not available, a clear ImportError is raised and
the figure step is skipped in the runner (warning logged).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Colour palette — distinguishable under greyscale and common CVD
_PALETTE = {
    "deterministic": "#1f77b4",   # blue
    "fuzzy":         "#ff7f0e",   # orange
    "tfidf":         "#2ca02c",   # green
    "aux_scopus_id": "#9467bd",   # purple (dashed)
    "bem":           "#d62728",   # red (marker only)
}

_LABELS = {
    "deterministic": "Deterministic",
    "fuzzy":         "Fuzzy",
    "tfidf":         "TF-IDF",
    "aux_scopus_id": "Aux Scopus-ID",
    "bem":           "BEM",
}


def plot_precision_coverage(
    task: str,
    diagnostics_dir: Path,
    operating_points: dict[str, dict[str, dict]],
    bem_point: dict[str, float] | None,
    out_path: Path,
    dpi: int = 300,
    fmt: str = "png",
) -> Path:
    """Generate and save a precision–coverage figure for one task.

    Args:
        task:              'and' | 'ain'
        diagnostics_dir:   Directory containing ``diagnostics_{task}_{bl}.parquet``.
        operating_points:  Nested dict ``{baseline_name: {method: {precision_match, coverage}}}``.
                           Built from the threshold summary in the runner.
        bem_point:         Dict with ``precision_match`` and ``coverage`` for BEM,
                           or None if BEM is not evaluated.
        out_path:          Full path for the output file (extension ignored; ``fmt`` is used).
        dpi:               Dots per inch for raster formats.
        fmt:               Output format ('png' | 'pdf' | 'svg').

    Returns:
        Path to the written figure file.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")     # non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        raise ImportError(
            "[E5] matplotlib is required for figures. Install with:  pip install matplotlib"
        )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    fair_baselines = [bl for bl in ("deterministic", "fuzzy", "tfidf")
                      if _diag_path(diagnostics_dir, task, bl).exists()]
    aux_bl         = "aux_scopus_id" if task == "and" else None

    # --- plot curves from diagnostics parquets ---
    for bl in fair_baselines + ([aux_bl] if aux_bl else []):
        path = _diag_path(diagnostics_dir, task, bl)
        if not path.exists():
            continue

        diag = pd.read_parquet(path)
        n_total_bin = diag["n_total"].iloc[0] if "n_total" in diag.columns else 1
        coverage_curve = diag["n_pred_match"].values / n_total_bin
        precision_curve = diag["precision_match"].values

        # Remove points where n_pred_match == 0 (undefined precision, set to 1.0
        # by convention — those are degenerate points at t=1 with no predictions)
        valid = diag["n_pred_match"].values > 0
        cov_v  = coverage_curve[valid]
        prec_v = precision_curve[valid]

        color  = _PALETTE.get(bl, "#333333")
        label  = _LABELS.get(bl, bl)
        ls     = "--" if bl == aux_bl else "-"
        lw     = 1.4  if bl == aux_bl else 1.8

        # Sort by coverage (ascending) for a left-to-right curve
        order  = np.argsort(cov_v)
        ax.plot(cov_v[order], prec_v[order], color=color, ls=ls, lw=lw,
                label=label, alpha=0.85, zorder=2)

    # --- plot operating points ---
    marker_props = {
        "precision_floor_match": dict(marker="D", s=60, zorder=5, edgecolors="white", lw=0.5),
        "f1_optimal":            dict(marker="*", s=90, zorder=5, edgecolors="white", lw=0.5),
    }

    for bl, methods in operating_points.items():
        color = _PALETTE.get(bl, "#333333")
        for method, pt in methods.items():
            cov  = pt.get("coverage")
            prec = pt.get("precision_match")
            if cov is None or prec is None:
                continue
            if any(v is None or (isinstance(v, float) and v != v) for v in (cov, prec)):
                continue
            props = marker_props.get(method, dict(marker="o", s=50, zorder=5))
            ax.scatter(cov, prec, color=color, **props)

    # --- BEM single point ---
    if bem_point is not None:
        cov  = bem_point.get("coverage")
        prec = bem_point.get("precision_match")
        if cov is not None and prec is not None:
            ax.scatter(cov, prec, marker="s", s=70, color=_PALETTE["bem"],
                       label="BEM", zorder=6, edgecolors="white", lw=0.5)

    # --- precision-floor reference line ---
    ax.axhline(0.90, color="#666666", ls=":", lw=1.0, alpha=0.6, label="0.90 floor")

    # --- Legend for markers (shared across baselines) ---
    legend_handles = [
        ax.get_legend_handles_labels()[0][i]
        for i in range(len(ax.get_legend_handles_labels()[0]))
    ]
    legend_labels = ax.get_legend_handles_labels()[1]

    # Append custom marker entries
    from matplotlib.lines import Line2D
    legend_handles += [
        Line2D([0], [0], marker="D", color="grey", linestyle="None",
               markersize=6, label="Floor op. point"),
        Line2D([0], [0], marker="*", color="grey", linestyle="None",
               markersize=8, label="F1-opt op. point"),
    ]
    legend_labels += ["Floor op. point", "F1-opt op. point"]

    ax.legend(legend_handles, legend_labels, fontsize=8, loc="lower right",
              framealpha=0.9, ncol=2)

    ax.set_xlabel("Coverage  (fraction of pairs predicted MATCH)", fontsize=10)
    ax.set_ylabel("Match Precision", fontsize=10)
    ax.set_title(f"{task.upper()} — Precision vs Coverage", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=8)

    plt.tight_layout()

    # Resolve output path with correct extension
    out_path = out_path.with_suffix(f".{fmt}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, format=fmt, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _diag_path(diagnostics_dir: Path, task: str, baseline_name: str) -> Path:
    return diagnostics_dir / f"diagnostics_{task}_{baseline_name}.parquet"
