"""features.py — Load and join instance-level features for baseline scoring.

Produces one feature DataFrame per task by joining benchmark prediction pairs
to the corresponding instance parquet (author_instances / affil_instances).

Fairness guarantees
-------------------
AND features
  - ``author_norm`` is stripped of the embedded ``(ScopusID)`` substring before
    any scoring.  The stripped field is called ``*_name_clean``.
  - ``coauthor_keys`` (which encode Scopus IDs) are NOT exposed in the feature
    DataFrame.  Coauthor overlap is not used in any fair baseline.
  - Only ``record_affil_prefix`` (an affiliation prefix token) and ``year_int``
    are retained as auxiliary features.

AIN features
  - ``affil_norm`` (normalised affiliation string) is the main text field.
  - ``affil_acronyms`` is retained for acronym-overlap features.
  - ``year_int`` is intentionally excluded (AIN evidence boundary).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_ID_PATTERN = re.compile(r"\(\d+\)", re.IGNORECASE)


def strip_author_id(text: str) -> str:
    """Remove embedded Scopus author ID (e.g. ``(57221443727)``) from *text*.

    Args:
        text: Raw author_norm string, e.g. ``"benmessaoud, mounir (57221443727)"``.

    Returns:
        Cleaned string, e.g. ``"benmessaoud, mounir"``.
    """
    return _ID_PATTERN.sub("", str(text)).strip()


def to_set(val: object) -> frozenset[str]:
    """Convert a parquet array value (numpy array / list / None) to a frozenset."""
    if val is None:
        return frozenset()
    if isinstance(val, float) and np.isnan(val):
        return frozenset()
    if isinstance(val, (list, np.ndarray)):
        return frozenset(str(x) for x in val if x)
    return frozenset()


def _norm_ids(series: pd.Series) -> pd.Series:
    """Cast a series to plain Python str (handles ArrowString and object dtypes)."""
    return series.astype(str)


# ---------------------------------------------------------------------------
# AND feature loading
# ---------------------------------------------------------------------------

def load_and_features(
    predictions_path: Path,
    author_instances_path: Path,
) -> pd.DataFrame:
    """Join AND benchmark pairs to author-instance features.

    Args:
        predictions_path:      Path to ``predictions_and_<split>.parquet``.
        author_instances_path: Path to ``data/interim/author_instances.parquet``.

    Returns:
        DataFrame with one row per benchmark pair.  Columns:

        From benchmark:
          anchor_id, candidate_id, gold_label, split

        From author_instances (both sides, with _anchor / _candidate suffix):
          name_raw_*    raw author_norm (includes Scopus ID)
          name_clean_*  author_norm with ID stripped  [FAIR FIELD]
          year_*        publication year (int or NaN)
          affil_prefix_*  first affiliation prefix token

    Raises:
        FileNotFoundError: If either input file is missing.
        ValueError: If join produces zero rows.
    """
    for p in (predictions_path, author_instances_path):
        if not p.exists():
            raise FileNotFoundError(f"[features] Required file missing: {p}")

    preds = pd.read_parquet(
        predictions_path,
        columns=["anchor_id", "candidate_id", "gold_label", "split"],
    )
    preds["anchor_id"] = _norm_ids(preds["anchor_id"])
    preds["candidate_id"] = _norm_ids(preds["candidate_id"])

    all_ids = set(preds["anchor_id"].tolist()) | set(preds["candidate_id"].tolist())

    inst = pd.read_parquet(
        author_instances_path,
        columns=["author_instance_id", "author_norm", "year_int", "record_affil_prefix"],
    )
    inst["author_instance_id"] = _norm_ids(inst["author_instance_id"])
    inst = (
        inst[inst["author_instance_id"].isin(all_ids)]
        .drop_duplicates(subset=["author_instance_id"], keep="first")
        .copy()
    )

    anchor = inst.rename(columns={
        "author_instance_id": "anchor_id",
        "author_norm":        "name_raw_anchor",
        "year_int":           "year_anchor",
        "record_affil_prefix": "affil_prefix_anchor",
    })
    candidate = inst.rename(columns={
        "author_instance_id": "candidate_id",
        "author_norm":        "name_raw_candidate",
        "year_int":           "year_candidate",
        "record_affil_prefix": "affil_prefix_candidate",
    })

    df = (
        preds
        .merge(anchor,    on="anchor_id",    how="left")
        .merge(candidate, on="candidate_id", how="left")
    )

    # Strip Scopus IDs → fair name fields
    df["name_clean_anchor"]    = df["name_raw_anchor"].fillna("").apply(strip_author_id)
    df["name_clean_candidate"] = df["name_raw_candidate"].fillna("").apply(strip_author_id)

    if len(df) == 0:
        raise ValueError(
            "[features] AND feature join produced zero rows. "
            "Check that predictions_path and author_instances_path are compatible."
        )

    n_missing_anchor    = df["name_raw_anchor"].isna().sum()
    n_missing_candidate = df["name_raw_candidate"].isna().sum()
    if n_missing_anchor or n_missing_candidate:
        print(
            f"[features] AND: {n_missing_anchor} anchor / {n_missing_candidate} candidate "
            "IDs not found in author_instances. Scores for these pairs will be 0.0."
        )

    return df


# ---------------------------------------------------------------------------
# AIN feature loading
# ---------------------------------------------------------------------------

def load_ain_features(
    predictions_path: Path,
    affil_instances_path: Path,
) -> pd.DataFrame:
    """Join AIN benchmark pairs to affiliation-instance features.

    Args:
        predictions_path:     Path to ``predictions_ain_<split>.parquet``.
        affil_instances_path: Path to ``data/interim/affil_instances.parquet``.

    Returns:
        DataFrame with one row per benchmark pair.  Columns:

        From benchmark:
          anchor_id, candidate_id, gold_label, split

        From affil_instances (both sides):
          affil_norm_*     normalised affiliation string  [FAIR FIELD]
          acronyms_*       frozenset of extracted acronyms  [FAIR FIELD]

        Note: year_int is intentionally NOT included (AIN evidence boundary).

    Raises:
        FileNotFoundError: If either input file is missing.
        ValueError: If join produces zero rows.
    """
    for p in (predictions_path, affil_instances_path):
        if not p.exists():
            raise FileNotFoundError(f"[features] Required file missing: {p}")

    preds = pd.read_parquet(
        predictions_path,
        columns=["anchor_id", "candidate_id", "gold_label", "split"],
    )
    preds["anchor_id"] = _norm_ids(preds["anchor_id"])
    preds["candidate_id"] = _norm_ids(preds["candidate_id"])

    all_ids = set(preds["anchor_id"].tolist()) | set(preds["candidate_id"].tolist())

    inst = pd.read_parquet(
        affil_instances_path,
        columns=["affil_instance_id", "affil_norm", "affil_acronyms"],
    )
    inst["affil_instance_id"] = _norm_ids(inst["affil_instance_id"])
    inst = (
        inst[inst["affil_instance_id"].isin(all_ids)]
        .drop_duplicates(subset=["affil_instance_id"], keep="first")
        .copy()
    )

    # Convert acronyms to frozenset for Jaccard computation
    inst["affil_acronyms"] = inst["affil_acronyms"].apply(to_set)

    anchor = inst.rename(columns={
        "affil_instance_id": "anchor_id",
        "affil_norm":        "affil_norm_anchor",
        "affil_acronyms":    "acronyms_anchor",
    })
    candidate = inst.rename(columns={
        "affil_instance_id": "candidate_id",
        "affil_norm":        "affil_norm_candidate",
        "affil_acronyms":    "acronyms_candidate",
    })

    df = (
        preds
        .merge(anchor,    on="anchor_id",    how="left")
        .merge(candidate, on="candidate_id", how="left")
    )

    if len(df) == 0:
        raise ValueError(
            "[features] AIN feature join produced zero rows. "
            "Check that predictions_path and affil_instances_path are compatible."
        )

    n_missing_anchor    = df["affil_norm_anchor"].isna().sum()
    n_missing_candidate = df["affil_norm_candidate"].isna().sum()
    if n_missing_anchor or n_missing_candidate:
        print(
            f"[features] AIN: {n_missing_anchor} anchor / {n_missing_candidate} candidate "
            "IDs not found in affil_instances. Scores for these pairs will be 0.0."
        )

    # Fill NaN acronyms with empty frozenset
    df["acronyms_anchor"] = df["acronyms_anchor"].apply(
        lambda x: x if isinstance(x, frozenset) else frozenset()
    )
    df["acronyms_candidate"] = df["acronyms_candidate"].apply(
        lambda x: x if isinstance(x, frozenset) else frozenset()
    )

    return df
