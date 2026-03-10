"""deterministic.py — Baseline A: deterministic rule-based scoring.

AND scoring rule
---------------
Score is derived from two conditions (both computed on fairness-compliant
features; Scopus Author ID is never consulted):

  1. Name exact match  (normalised author name, ID-stripped)
  2. Shared affiliation prefix token

  Score = 1.0  if name_match AND affil_match
        = 0.5  if name_match only
        = 0.0  otherwise

The score is intentionally discrete {0.0, 0.5, 1.0} to support clean
precision / recall curves via threshold sweeping.

AIN scoring rule
----------------
Exact match on normalised affiliation string (affil_norm).

  Score = 1.0  if affil_norm_anchor == affil_norm_candidate  (non-empty)
        = 0.0  otherwise

Notes
-----
- Co-author overlap is NOT used because coauthor_keys in author_instances
  embed Scopus IDs.  Using them would violate the AND fairness constraint.
- AIN year_int is intentionally excluded (AIN evidence boundary).
"""

from __future__ import annotations

import pandas as pd

BASELINE_NAME = "deterministic"


def score_and(df: pd.DataFrame) -> pd.Series:
    """Compute deterministic AND scores for a features DataFrame.

    Args:
        df: Output of :func:`~bem.eval.baselines.features.load_and_features`.

    Returns:
        Float64 Series of length ``len(df)`` with scores in {0.0, 0.5, 1.0}.
    """
    name_a = df["name_clean_anchor"].fillna("").str.strip()
    name_b = df["name_clean_candidate"].fillna("").str.strip()

    name_match = (name_a == name_b) & (name_a != "")

    affil_a = df["affil_prefix_anchor"].fillna("").str.strip()
    affil_b = df["affil_prefix_candidate"].fillna("").str.strip()
    affil_match = (affil_a == affil_b) & (affil_a != "")

    scores = pd.Series(0.0, index=df.index, dtype="float64")
    scores[name_match] = 0.5
    scores[name_match & affil_match] = 1.0
    return scores


def score_ain(df: pd.DataFrame) -> pd.Series:
    """Compute deterministic AIN scores for a features DataFrame.

    Args:
        df: Output of :func:`~bem.eval.baselines.features.load_ain_features`.

    Returns:
        Float64 Series of length ``len(df)`` with scores in {0.0, 1.0}.
    """
    a = df["affil_norm_anchor"].fillna("").str.strip()
    b = df["affil_norm_candidate"].fillna("").str.strip()
    return ((a == b) & (a != "")).astype("float64")


def run(task: str, features_df: pd.DataFrame) -> pd.Series:
    """Dispatch to the correct scorer based on *task*.

    Args:
        task:        ``"and"`` or ``"ain"``.
        features_df: Features DataFrame for that task.

    Returns:
        Float64 score Series.

    Raises:
        ValueError: If *task* is not recognised.
    """
    if task == "and":
        return score_and(features_df)
    if task == "ain":
        return score_ain(features_df)
    raise ValueError(f"[deterministic] Unknown task: {task!r}")
