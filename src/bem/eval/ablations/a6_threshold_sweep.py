"""a6_threshold_sweep.py -- Ablation A6: threshold-sweep analysis.

Re-applies the C6 guard logic at each point of a (t_match x m_signals) grid
using the raw C5 LLM decisions stored in the benchmark-filtered routing log.
The non-match threshold is held fixed at t_nonmatch_fixed throughout the sweep.

For each (task, t_match, m_signals) combination the standard BEM evaluation
metrics are computed.  The output is a long-format DataFrame suitable for
heat-map visualisation and for computing sensitivity statistics.

Signal categories used by the guard are imported from apply_guards.py so the
sweep is always consistent with the production guard logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bem.eval.evaluation.metrics import compute_all_metrics
from bem.guards.apply_guards import count_signals


def run_a6(
    routing_log_and: Path,
    routing_log_ain: Path,
    tasks: list[str],
    t_match_values: list[float],
    m_signals_values: list[int],
    t_nonmatch_fixed: float,
    rng: np.random.Generator,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    output_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run ablation A6 for all requested tasks.

    Returns:
        Dict keyed by task -> DataFrame of sweep results (one row per grid point).
    """
    log_paths = {"and": routing_log_and, "ain": routing_log_ain}
    results: dict[str, pd.DataFrame] = {}

    for task in tasks:
        log_path = log_paths[task]
        cache_parquet = (output_dir / f"sweep_{task}.parquet") if output_dir else None

        if cache_parquet and cache_parquet.exists() and not force:
            print(f"  [A6/{task.upper()}] loading cached sweep from {cache_parquet}")
            results[task] = pd.read_parquet(cache_parquet)
            continue

        if not log_path.exists():
            print(f"  [A6/{task.upper()}] routing log not found -- skipping: {log_path}")
            continue

        df = pd.read_parquet(log_path)
        print(
            f"  [A6/{task.upper()}] sweeping "
            f"{len(t_match_values)} x {len(m_signals_values)} = "
            f"{len(t_match_values) * len(m_signals_values)} grid points ..."
        )

        gold  = df["gold_label"].astype(str)
        label = df["label"].astype(str)
        conf  = df["confidence"].astype(float)

        # Pre-compute signal counts once (task-specific category table)
        ev_lists = df["evidence_used"].tolist()
        signals_per_pair = np.array(
            [count_signals(_safe_list(ev), task)[0] for ev in ev_lists], dtype=int
        )

        rows: list[dict] = []
        for t_match in t_match_values:
            for m_signals in m_signals_values:
                predicted = _apply_grid_point(
                    label, conf, signals_per_pair,
                    t_match=t_match,
                    t_nonmatch=t_nonmatch_fixed,
                    m_signals=m_signals,
                )
                m = compute_all_metrics(gold, pd.Series(predicted), rng, n_bootstrap, alpha)
                ci = m.get("ci_f1_match", (float("nan"), float("nan")))
                rows.append({
                    "task":               task,
                    "t_match":            t_match,
                    "m_signals":          m_signals,
                    "t_nonmatch":         t_nonmatch_fixed,
                    "precision_match":    float(m.get("precision_match", float("nan"))),
                    "recall_match":       float(m.get("recall_match",    float("nan"))),
                    "f1_match":           float(m.get("f1_match",        float("nan"))),
                    "ci_f1_lo":           float(ci[0]),
                    "ci_f1_hi":           float(ci[1]),
                    "coverage":           float(m.get("coverage",        float("nan"))),
                    "uncertain_rate":     float(m.get("uncertain_rate",  float("nan"))),
                    "precision_nonmatch": float(m.get("precision_nonmatch", float("nan"))),
                    "recall_nonmatch":    float(m.get("recall_nonmatch",    float("nan"))),
                    "f1_nonmatch":        float(m.get("f1_nonmatch",        float("nan"))),
                    "three_way_accuracy": float(m.get("three_way_accuracy", float("nan"))),
                    "macro_f1_binary":    float(m.get("macro_f1_binary",    float("nan"))),
                })

        sweep_df = pd.DataFrame(rows)
        results[task] = sweep_df

        if output_dir and cache_parquet:
            output_dir.mkdir(parents=True, exist_ok=True)
            sweep_df.to_parquet(cache_parquet, index=False)
            print(f"  [A6/{task.upper()}] sweep saved -> {cache_parquet}")

        # Quick summary: best F1 per m_signals
        print(f"  [A6/{task.upper()}] best F1_match per m_signals (across t_match sweep):")
        for m_sig in m_signals_values:
            sub = sweep_df[sweep_df["m_signals"] == m_sig]
            if sub.empty:
                continue
            best = sub.loc[sub["f1_match"].idxmax()]
            print(
                f"    m_signals={m_sig}  best_F1={best['f1_match']:.3f}  "
                f"at t_match={best['t_match']:.2f}  coverage={best['coverage']:.3f}"
            )

    return results


# ---------------------------------------------------------------------------
# Guard re-application (matches apply_guards.py logic, without DataFrame I/O)
# ---------------------------------------------------------------------------

def _apply_grid_point(
    label: pd.Series,
    confidence: pd.Series,
    signals_count: np.ndarray,
    t_match: float,
    t_nonmatch: float,
    m_signals: int,
) -> list[str]:
    """Apply C6 guard with given parameters and return predicted labels."""
    predicted: list[str] = []
    for lbl, conf, n_sig in zip(label, confidence, signals_count):
        if lbl == "error":
            predicted.append("uncertain")
        elif lbl == "non-match" and conf >= t_nonmatch:
            predicted.append("non-match")
        elif lbl == "match" and conf >= t_match and n_sig >= m_signals:
            predicted.append("match")
        else:
            predicted.append("uncertain")
    return predicted


def _safe_list(ev: Any) -> list[str]:
    if isinstance(ev, list):
        return [str(x) for x in ev]
    if isinstance(ev, np.ndarray):
        return [str(x) for x in ev.tolist()]
    return []
