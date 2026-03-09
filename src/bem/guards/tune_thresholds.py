"""tune_thresholds.py — DEV-set threshold tuning for C6 guards.

Tunes T_match to meet a precision floor (default 0.98) on the DEV split
of the benchmark.  T_nonmatch is kept at its initial configured value.

Grid: T_match ∈ [0.50 .. 0.99] step 0.01, evaluated from HIGH to LOW.
      The smallest T_match that still meets precision >= precision_floor
      is selected (maximises recall at the target precision).

Tuning uses ONLY the DEV split.  TEST split is never touched here.

Writes runs/<run_id>/manifests/thresholds_tuned_dev.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bem.guards.apply_guards import apply_guards


# ---------------------------------------------------------------------------
# Join decisions with gold labels
# ---------------------------------------------------------------------------

def join_with_gold(
    decisions_df: pd.DataFrame,
    benchmark_path: str | Path,
    split: str = "dev",
) -> pd.DataFrame:
    """Join LLM decisions with benchmark gold labels, filtered to ``split``.

    Parameters
    ----------
    decisions_df : DataFrame from load_llm_decisions_jsonl.
    benchmark_path : Path to benchmark_pairs_{and,ain}.parquet.
    split : "dev" or "test".  Only rows with this split value are kept.

    Returns
    -------
    DataFrame with all columns from decisions_df plus gold_label (and
    stratum if present in the benchmark parquet).  Rows without a
    matching gold label are dropped with a printed warning.
    """
    bp = Path(benchmark_path)
    if not bp.exists():
        raise FileNotFoundError(f"Benchmark parquet not found: {bp}")

    gold_df = pd.read_parquet(bp)
    if "split" in gold_df.columns:
        gold_df = gold_df[gold_df["split"].str.lower() == split.lower()]

    extra_cols = [c for c in ("stratum",) if c in gold_df.columns]
    merged = decisions_df.merge(
        gold_df[["anchor_id", "candidate_id", "gold_label"] + extra_cols],
        on=["anchor_id", "candidate_id"],
        how="inner",
    )

    n_unmatched = len(decisions_df) - len(merged)
    if n_unmatched:
        print(
            f"  [tune] WARN: {n_unmatched} decision row(s) had no matching gold "
            f"label in the {split} split — dropped."
        )
    return merged


# ---------------------------------------------------------------------------
# Grid-search T_match
# ---------------------------------------------------------------------------

def tune_t_match(
    dev_df: pd.DataFrame,
    task: str,
    initial_thresholds: dict[str, float],
    m_signals: int = 2,
    precision_floor: float = 0.98,
    grid_start: float = 0.50,
    grid_end: float = 0.99,
    grid_step: float = 0.01,
) -> dict[str, Any]:
    """Grid-search T_match on the DEV set to meet the precision floor.

    Iterates grid values from HIGH to LOW so that the *smallest* passing
    T_match is returned (maximises recall at the target precision).

    Parameters
    ----------
    dev_df         : Output of join_with_gold.  Required columns: label,
                     confidence, evidence_used, gold_label.
    task           : "AND" or "AIN".
    initial_thresholds : {"t_match": float, "t_nonmatch": float}.
    m_signals      : Minimum independent signal categories for match routing.
    precision_floor : Target precision (default 0.98).
    grid_start / grid_end / grid_step : Grid parameters.

    Returns
    -------
    dict with keys:
        task, tuned_t_match, t_nonmatch, precision, recall,
        n_auto_match, n_gold_match, precision_floor, m_signals,
        grid_evaluated (list of {t, precision, recall, n_auto_match}),
        fallback_to_initial (bool)
    """
    t_nonmatch      = float(initial_thresholds.get("t_nonmatch", 0.85))
    initial_t_match = float(initial_thresholds.get("t_match", 0.85))

    # Build grid from HIGH to LOW
    n_steps = round((grid_end - grid_start) / grid_step) + 1
    grid_values = [
        round(grid_end - i * grid_step, 10)
        for i in range(n_steps)
    ]
    # Clamp to valid range
    grid_values = [t for t in grid_values if grid_start - 1e-9 <= t <= grid_end + 1e-9]

    n_gold_match = int((dev_df["gold_label"] == "match").sum())

    grid_evaluated: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for t in grid_values:
        thresholds_trial = {"t_match": t, "t_nonmatch": t_nonmatch}
        guarded = apply_guards(dev_df, task, thresholds_trial, m_signals)

        auto_match_mask = guarded["label_final"] == "match"
        n_auto_match = int(auto_match_mask.sum())

        if n_auto_match == 0:
            precision_at_t = float("nan")
            recall_at_t    = 0.0
        else:
            tp = int(
                (auto_match_mask & (guarded["gold_label"] == "match")).sum()
            )
            precision_at_t = tp / n_auto_match
            recall_at_t    = tp / n_gold_match if n_gold_match > 0 else 0.0

        entry: dict[str, Any] = {
            "t":            t,
            "precision":    precision_at_t,
            "recall":       recall_at_t,
            "n_auto_match": n_auto_match,
        }
        grid_evaluated.append(entry)

        # Accept the smallest t that meets the floor (we iterate high→low)
        import math
        if not math.isnan(precision_at_t) and precision_at_t >= precision_floor:
            best = entry  # keep overwriting — last accepted = smallest t

    if best is None:
        # No threshold met the floor → fall back to initial
        best = {
            "t":            initial_t_match,
            "precision":    float("nan"),
            "recall":       0.0,
            "n_auto_match": 0,
        }
        fallback = True
        print(
            f"  [{task}] WARN: No T_match in [{grid_start:.2f}..{grid_end:.2f}] "
            f"met precision >= {precision_floor}. "
            f"Falling back to initial T_match={initial_t_match}."
        )
    else:
        fallback = False
        print(
            f"  [{task}] Tuned T_match = {best['t']:.2f}  "
            f"(precision={best['precision']:.4f}, "
            f"recall={best['recall']:.4f}, "
            f"n_auto_match={best['n_auto_match']})"
        )

    return {
        "task":               task,
        "tuned_t_match":      best["t"],
        "t_nonmatch":         t_nonmatch,
        "precision":          best["precision"],
        "recall":             best["recall"],
        "n_auto_match":       best["n_auto_match"],
        "n_gold_match":       n_gold_match,
        "precision_floor":    precision_floor,
        "m_signals":          m_signals,
        "grid_evaluated":     grid_evaluated,
        "fallback_to_initial": fallback,
    }


# ---------------------------------------------------------------------------
# Write manifest
# ---------------------------------------------------------------------------

def write_thresholds_manifest(
    run_dir: str | Path,
    and_result: dict[str, Any] | None,
    ain_result: dict[str, Any] | None,
    m_signals: int,
) -> Path:
    """Write thresholds_tuned_dev.json to run_dir/manifests/.

    Parameters
    ----------
    run_dir    : runs/<run_id>/ directory.
    and_result : Return value of tune_t_match for AND (or None if skipped).
    ain_result : Return value of tune_t_match for AIN (or None if skipped).
    m_signals  : M value used during tuning.

    Returns
    -------
    Path to the written manifest file.
    """
    manifests_dir = Path(run_dir) / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out_path = manifests_dir / "thresholds_tuned_dev.json"

    payload: dict[str, Any] = {
        "timestamp_iso":          datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "m_independent_signals":  m_signals,
        "and":                    and_result,
        "ain":                    ain_result,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path
