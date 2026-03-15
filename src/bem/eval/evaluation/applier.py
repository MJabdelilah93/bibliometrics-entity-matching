"""applier.py — Apply tuned thresholds to baseline scores.

For each (baseline_name, method) combination, converts continuous scores into
three-class predictions: 'match', 'non-match', or 'uncertain'.

Methods
-------
precision_floor_match / f1_optimal  : single threshold t
    score >= t  -> 'match'
    score <  t  -> 'non-match'
    coverage    = 1.0 (no abstention)

two_threshold                        : band (t_low, t_high)
    score >= t_high              -> 'match'
    score <  t_low               -> 'non-match'
    t_low <= score < t_high      -> 'uncertain'
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Load thresholds artefact
# ---------------------------------------------------------------------------

def load_thresholds(thresholds_json_path: Path) -> dict:
    """Load a ``thresholds_<task>.json`` file.

    Args:
        thresholds_json_path: Path produced by Stage E4.

    Returns:
        Parsed dict keyed by ``baselines.<name>.<method>``.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    if not thresholds_json_path.exists():
        raise FileNotFoundError(
            f"[E5] Thresholds file not found: {thresholds_json_path}\n"
            "Run Stage E4 first:  python -m bem.eval --no-dry-run --stage tune-thresholds"
        )
    return json.loads(thresholds_json_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Core applier
# ---------------------------------------------------------------------------

def apply_threshold(
    scores: np.ndarray,
    method_dict: dict,
) -> np.ndarray:
    """Convert continuous scores to three-class predictions using a tuned threshold.

    Args:
        scores:      Float array of scores in [0, 1].
        method_dict: Dict from ``thresholds_<task>.json``; contains either
                     ``threshold`` (single) or ``t_low`` / ``t_high`` (two-threshold).
                     If ``threshold`` is ``None`` (floor not achieved), returns all
                     'uncertain'.

    Returns:
        Object array of str: 'match', 'non-match', or 'uncertain'.
    """
    predicted = np.full(len(scores), "uncertain", dtype=object)

    method = method_dict.get("method", "")

    if method == "two_threshold":
        t_low  = method_dict.get("t_low")
        t_high = method_dict.get("t_high")
        if t_low is None or t_high is None:
            return predicted   # all uncertain
        predicted[scores >= t_high] = "match"
        predicted[scores <  t_low]  = "non-match"

    else:
        # f1_optimal / precision_floor_match — single threshold
        t = method_dict.get("threshold")
        if t is None:
            return predicted   # floor not achieved — all uncertain
        predicted[scores >= t] = "match"
        predicted[scores <  t] = "non-match"

    return predicted


# ---------------------------------------------------------------------------
# Apply all configured methods for one (task, baseline_name)
# ---------------------------------------------------------------------------

def apply_all_methods(
    master_df: pd.DataFrame,
    task: str,
    baseline_name: str,
    thresholds_dict: dict,
    methods: list[str],
) -> list[dict]:
    """Apply every requested threshold method for one (task × baseline).

    Args:
        master_df:       Long-format baseline scores master file.
        task:            Task string ('and' | 'ain').
        baseline_name:   E.g. 'fuzzy', 'tfidf', 'deterministic'.
        thresholds_dict: Parsed ``thresholds_<task>.json``.
        methods:         Which methods to apply from the JSON.

    Returns:
        List of dicts, one per method, each containing:
        ``task``, ``system``, ``baseline_name``, ``method``,
        ``gold_all``, ``predicted_all`` (pd.Series), ``threshold_info`` (dict).
    """
    bl_df = master_df[
        (master_df["task"] == task) & (master_df["baseline_name"] == baseline_name)
    ].copy()

    if len(bl_df) == 0:
        return []

    bl_thresholds = thresholds_dict.get("baselines", {}).get(baseline_name, {})

    results = []
    for method in methods:
        method_info = bl_thresholds.get(method)
        if method_info is None:
            continue  # method not in JSON (e.g. two_threshold disabled)

        scores = bl_df["score"].values
        predicted = apply_threshold(scores, {**method_info, "method": method})

        results.append({
            "task":          task,
            "system":        "baseline",
            "baseline_name": baseline_name,
            "method":        method,
            "gold_all":      bl_df["gold_label"].reset_index(drop=True),
            "predicted_all": pd.Series(predicted),
            "threshold_info": method_info,
            "split":          bl_df["split"].iloc[0] if len(bl_df) > 0 else "unknown",
        })

    return results
