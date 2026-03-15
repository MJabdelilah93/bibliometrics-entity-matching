"""fuzzy.py — Baseline B: weighted lexical rule model.

AND scoring formula (all weights sum to 1.0)
--------------------------------------------
  score = 0.60 * name_token_sort_ratio   (rapidfuzz; ID-stripped names)
        + 0.20 * affil_prefix_exact      (1.0 if same prefix, 0.0 otherwise)
        + 0.20 * year_closeness          (1.0 if same year; decays linearly)

  year_closeness = max(0, 1 - |year_a - year_b| / YEAR_DECAY_WINDOW)
  YEAR_DECAY_WINDOW = 10   (score = 0 when >= 10 years apart)

AIN scoring formula (all weights sum to 1.0)
--------------------------------------------
  score = 0.50 * affil_token_set_ratio   (rapidfuzz)
        + 0.30 * affil_token_sort_ratio  (rapidfuzz)
        + 0.20 * acronym_jaccard

Fairness
--------
- AND: Author(s) ID is stripped from all name fields before scoring.
  Affil prefix uses record_affil_prefix (token prefix, not full ID-based key).
- AIN: year_int is excluded from scoring.

Requires:  rapidfuzz (in requirements.txt)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz as _fuzz

BASELINE_NAME = "fuzzy"

_YEAR_DECAY = 10.0  # years over which year_closeness decays to 0


# ---------------------------------------------------------------------------
# AND
# ---------------------------------------------------------------------------

def score_and(df: pd.DataFrame) -> pd.Series:
    """Compute weighted fuzzy AND scores.

    Args:
        df: Output of :func:`~bem.eval.baselines.features.load_and_features`.

    Returns:
        Float64 Series with scores in [0, 1].
    """
    name_a = df["name_clean_anchor"].fillna("").astype(str)
    name_b = df["name_clean_candidate"].fillna("").astype(str)

    affil_a = df["affil_prefix_anchor"].fillna("").astype(str).str.strip()
    affil_b = df["affil_prefix_candidate"].fillna("").astype(str).str.strip()

    year_a = pd.to_numeric(df["year_anchor"],    errors="coerce")
    year_b = pd.to_numeric(df["year_candidate"], errors="coerce")

    # Name similarity (rapidfuzz; tokens sorted so order doesn't matter)
    name_sim = np.array([
        _fuzz.token_sort_ratio(a, b) / 100.0
        for a, b in zip(name_a, name_b)
    ], dtype=float)

    # Affil prefix exact match
    affil_match = (
        (affil_a == affil_b) & (affil_a != "")
    ).astype(float).values

    # Year closeness
    year_diff = (year_a - year_b).abs()
    year_score = np.where(
        year_diff.isna(),
        0.0,
        np.clip(1.0 - year_diff.values / _YEAR_DECAY, 0.0, 1.0),
    )

    scores = 0.60 * name_sim + 0.20 * affil_match + 0.20 * year_score
    return pd.Series(scores.astype(float), index=df.index, dtype="float64")


# ---------------------------------------------------------------------------
# AIN
# ---------------------------------------------------------------------------

def score_ain(df: pd.DataFrame) -> pd.Series:
    """Compute weighted fuzzy AIN scores.

    Args:
        df: Output of :func:`~bem.eval.baselines.features.load_ain_features`.

    Returns:
        Float64 Series with scores in [0, 1].
    """
    a_norm = df["affil_norm_anchor"].fillna("").astype(str)
    b_norm = df["affil_norm_candidate"].fillna("").astype(str)

    token_set  = np.array([_fuzz.token_set_ratio(a, b) / 100.0  for a, b in zip(a_norm, b_norm)], dtype=float)
    token_sort = np.array([_fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(a_norm, b_norm)], dtype=float)

    # Acronym Jaccard  (frozensets stored in features DataFrame)
    acro_jac = np.array([
        _jaccard(row["acronyms_anchor"], row["acronyms_candidate"])
        for _, row in df.iterrows()
    ], dtype=float)

    scores = 0.50 * token_set + 0.30 * token_sort + 0.20 * acro_jac
    return pd.Series(scores.astype(float), index=df.index, dtype="float64")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

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
    raise ValueError(f"[fuzzy] Unknown task: {task!r}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _jaccard(a: frozenset, b: frozenset) -> float:
    if not isinstance(a, frozenset):
        a = frozenset()
    if not isinstance(b, frozenset):
        b = frozenset()
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)
