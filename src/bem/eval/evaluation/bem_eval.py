"""bem_eval.py — Load and prepare BEM routing predictions for evaluation.

BEM routing decisions come from ``predictions_{task}_{split}.parquet``
produced by Stage E1 (materialise_inputs).  The ``label_final`` column
contains the routing output of Stage C6.

BEM label_final → evaluation class mapping
-------------------------------------------
  'match'          -> 'match'
  'non-match'      -> 'non-match'
  'routed_to_human', 'uncertain', or any other value -> 'uncertain'

BEM is evaluated as a complete system (threshold + routing logic built in),
not as a scorer.  There is no additional threshold tuning for BEM here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Labels that map to confident 'match'
_MATCH_LABELS    = {"match"}
# Labels that map to confident 'non-match'
_NONMATCH_LABELS = {"non-match"}


def load_bem_predictions(
    predictions_path: Path,
    task: str,
) -> pd.DataFrame:
    """Load BEM predictions from a materialised predictions parquet.

    Args:
        predictions_path: Path to ``predictions_{task}_{split}.parquet``.
        task:             Task string ('and' | 'ain') — used only for logging.

    Returns:
        DataFrame with columns:
          ``anchor_id``, ``candidate_id``, ``gold_label``, ``predicted``,
          ``label_final_raw`` (the original label_final value).

    Raises:
        FileNotFoundError: If the predictions file is missing.
        ValueError: If required columns are absent.
    """
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"[E5] BEM predictions file not found: {predictions_path}\n"
            "Run Stage E1 first:  python -m bem.eval --no-dry-run --stage materialise-inputs"
        )

    df = pd.read_parquet(predictions_path)

    required = {"anchor_id", "candidate_id", "gold_label", "label_final"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[E5] predictions_{task}_*.parquet is missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df["label_final_raw"] = df["label_final"].astype(str)
    df["predicted"] = df["label_final_raw"].apply(_map_label)

    label_dist = df["label_final_raw"].value_counts().to_dict()
    pred_dist  = df["predicted"].value_counts().to_dict()
    print(
        f"  [BEM/{task.upper()}] label_final distribution: {label_dist}\n"
        f"  [BEM/{task.upper()}] mapped predictions:       {pred_dist}"
    )

    return df[["anchor_id", "candidate_id", "gold_label", "predicted", "label_final_raw"]]


def _map_label(raw: str) -> str:
    """Map a BEM label_final string to the standard three-class evaluation label."""
    if raw in _MATCH_LABELS:
        return "match"
    if raw in _NONMATCH_LABELS:
        return "non-match"
    return "uncertain"
