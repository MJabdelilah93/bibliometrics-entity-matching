"""union_find.py — Cluster entity mentions via union-find (disjoint sets).

Evidence boundary: merges are derived only from auto-routed and human-adjudicated
pair decisions within this pipeline run. No external entity registries.

TODOs:
- Implement a path-compressed, union-by-rank disjoint-set structure.
- Accept a DataFrame of (left_id, right_id) match pairs.
- Return a Series mapping each record_id to its canonical cluster_id.
- 'uncertain' and 'non-match' pairs must never be merged.
- Cluster IDs should be deterministic given a fixed set of match pairs
  (sort by (left_id, right_id) before processing to ensure stability).
"""

from __future__ import annotations

import pandas as pd


def build_clusters(match_pairs: pd.DataFrame) -> pd.Series:
    """Cluster entity mentions using union-find.

    Args:
        match_pairs: DataFrame with ``left_id`` and ``right_id`` columns,
            containing only confirmed match decisions.

    Returns:
        Series indexed by ``record_id`` with values being canonical
        ``cluster_id`` strings (stable across reruns given same input).

    TODO: implement.
    """
    raise NotImplementedError("build_clusters is not yet implemented.")
