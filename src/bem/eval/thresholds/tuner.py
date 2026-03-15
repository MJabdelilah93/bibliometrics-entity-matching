"""tuner.py — Core threshold-tuning algorithms.

No I/O.  No config dependency.  Operates on numpy arrays.  Easily unit-tested.

Evaluation protocol
-------------------
- Positive class : 'match'     (gold_label == 'match')
- Negative class : 'non-match' (gold_label == 'non-match')
- Excluded       : 'uncertain' — reported as a count but never used in any
                   precision / recall / F computation.

Tuning methods
--------------
1. ``f1_optimal``          argmax F-beta(t) over the threshold grid.
                           Single threshold; coverage = 1.0.

2. ``precision_floor_match``  The lowest threshold (= maximum recall) at which
                           precision_match >= floor.  This is the PRIMARY
                           operating point in the BEM evaluation policy.

3. ``two_threshold``       Independent precision-floor tuning for each class:
                             t_high = lowest t   s.t. precision_match(t) >= floor
                             t_low  = highest t  s.t. precision_nonmatch(t) >= floor
                           Pairs with t_low <= score < t_high are abstained
                           (uncertain band — would route to human review in BEM).
                           If t_low > t_high (floors too strict to separate), the
                           band is collapsed: t_low is clamped to t_high.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Grid helper
# ---------------------------------------------------------------------------

def make_threshold_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Return a linearly-spaced threshold array from *start* to *stop* inclusive.

    Uses ``np.linspace`` so the result is numerically stable even for non-
    representable float steps.
    """
    n = round((stop - start) / step) + 1
    return np.linspace(start, stop, n)


# ---------------------------------------------------------------------------
# Gold-label encoding
# ---------------------------------------------------------------------------

def encode_gold(gold_series: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
    """Map gold_label strings to a binary int array, excluding 'uncertain' rows.

    Args:
        gold_series: Series with values ``'match'``, ``'non-match'``, or
                     ``'uncertain'``.

    Returns:
        Tuple ``(binary_mask, gold_binary, n_uncertain)`` where:

        * ``binary_mask`` — boolean array (same length as *gold_series*);
          True for rows that are 'match' or 'non-match'.
        * ``gold_binary``  — int array of 1 (match) / 0 (non-match) for the
          rows selected by *binary_mask*.
        * ``n_uncertain``  — number of excluded 'uncertain' rows.
    """
    binary_mask = gold_series.isin(["match", "non-match"])
    n_uncertain = int((~binary_mask).sum())
    gold_binary = (gold_series[binary_mask] == "match").astype(int).values
    return binary_mask.values, gold_binary, n_uncertain


# ---------------------------------------------------------------------------
# Precision / recall curves
# ---------------------------------------------------------------------------

def compute_match_pr_curve(
    scores: np.ndarray,
    gold: np.ndarray,
    thresholds: np.ndarray,
    f1_beta: float = 1.0,
) -> pd.DataFrame:
    """Precision/recall/F-beta curve for the MATCH class.

    At each threshold *t*, pairs with ``score >= t`` are predicted MATCH.

    Convention when no pairs are predicted (n_pred == 0):
    precision = 1.0, recall = 0.0, f_beta = 0.0.

    Args:
        scores:     Float array, same length as *gold*.
        gold:       Binary int array (1 = match, 0 = non-match).
        thresholds: 1-D array of threshold values to evaluate.
        f1_beta:    Beta for the F-beta metric (1.0 = standard F1).

    Returns:
        DataFrame with one row per threshold.  Columns:
        ``threshold``, ``precision_match``, ``recall_match``, ``f_beta_match``,
        ``n_pred_match``, ``n_total``.
    """
    n_true_match = int((gold == 1).sum())
    beta_sq = f1_beta ** 2
    rows = []

    for t in thresholds:
        pred = scores >= t
        tp    = int((pred & (gold == 1)).sum())
        fp    = int((pred & (gold == 0)).sum())
        n_pred = int(pred.sum())

        prec   = tp / n_pred        if n_pred > 0                   else 1.0
        rec    = tp / n_true_match  if n_true_match > 0             else 0.0
        f_beta = (
            (1 + beta_sq) * prec * rec / (beta_sq * prec + rec)
            if (prec + rec) > 0 else 0.0
        )

        rows.append({
            "threshold":       float(t),
            "precision_match": prec,
            "recall_match":    rec,
            "f_beta_match":    f_beta,
            "n_pred_match":    n_pred,
            "n_total":         len(gold),
        })

    return pd.DataFrame(rows)


def compute_nonmatch_pr_curve(
    scores: np.ndarray,
    gold: np.ndarray,
    thresholds: np.ndarray,
    f1_beta: float = 1.0,
) -> pd.DataFrame:
    """Precision/recall/F-beta curve for the NON-MATCH class.

    At each threshold *t*, pairs with ``score < t`` are predicted NON-MATCH.

    Convention when no pairs are predicted (n_pred == 0):
    precision = 1.0, recall = 0.0, f_beta = 0.0.

    Returns:
        DataFrame with columns:
        ``threshold``, ``precision_nonmatch``, ``recall_nonmatch``,
        ``f_beta_nonmatch``, ``n_pred_nonmatch``, ``n_total``.
    """
    n_true_nonmatch = int((gold == 0).sum())
    beta_sq = f1_beta ** 2
    rows = []

    for t in thresholds:
        pred   = scores < t
        tn     = int((pred & (gold == 0)).sum())
        fn_nm  = int((pred & (gold == 1)).sum())   # false non-matches
        n_pred = int(pred.sum())

        prec   = tn / n_pred           if n_pred > 0              else 1.0
        rec    = tn / n_true_nonmatch  if n_true_nonmatch > 0     else 0.0
        f_beta = (
            (1 + beta_sq) * prec * rec / (beta_sq * prec + rec)
            if (prec + rec) > 0 else 0.0
        )

        rows.append({
            "threshold":          float(t),
            "precision_nonmatch": prec,
            "recall_nonmatch":    rec,
            "f_beta_nonmatch":    f_beta,
            "n_pred_nonmatch":    n_pred,
            "n_total":            len(gold),
        })

    return pd.DataFrame(rows)


def merge_diagnostics(
    match_curve: pd.DataFrame,
    nonmatch_curve: pd.DataFrame,
) -> pd.DataFrame:
    """Merge match and non-match PR curves into a single diagnostics DataFrame.

    Returns:
        Wide DataFrame keyed by ``threshold`` with all six curve columns plus
        ``n_total``.
    """
    return match_curve.merge(
        nonmatch_curve.drop(columns="n_total"),
        on="threshold",
    )


# ---------------------------------------------------------------------------
# Tuning functions — each returns a plain dict for easy JSON serialisation
# ---------------------------------------------------------------------------

def tune_f1_optimal(
    match_curve: pd.DataFrame,
    baseline_name: str,
    task: str,
    n_uncertain: int,
    f1_beta: float,
) -> dict:
    """Return the operating point that maximises F-beta for the match class.

    Returns a dict with keys:
    ``method``, ``baseline_name``, ``task``, ``threshold``,
    ``precision_match``, ``recall_match``, ``f_beta_match``,
    ``n_pred_match``, ``n_total_binary``, ``n_uncertain_excluded``,
    ``f1_beta``, ``achieved_floor``, ``note``.
    """
    best_idx = match_curve["f_beta_match"].idxmax()
    best = match_curve.loc[best_idx]

    return {
        "method":               "f1_optimal",
        "baseline_name":        baseline_name,
        "task":                 task,
        "threshold":            float(best["threshold"]),
        "precision_match":      float(best["precision_match"]),
        "recall_match":         float(best["recall_match"]),
        "f_beta_match":         float(best["f_beta_match"]),
        "n_pred_match":         int(best["n_pred_match"]),
        "n_total_binary":       int(best["n_total"]),
        "n_uncertain_excluded": n_uncertain,
        "f1_beta":              f1_beta,
        "achieved_floor":       None,   # not applicable to this method
        "note":                 "",
    }


def tune_precision_floor_match(
    match_curve: pd.DataFrame,
    baseline_name: str,
    task: str,
    floor: float,
    n_uncertain: int,
    f1_beta: float,
) -> dict:
    """Lowest threshold (= maximum recall) at which precision_match >= floor.

    This is the PRIMARY operating point in the BEM evaluation policy.

    Returns a dict with the same keys as :func:`tune_f1_optimal`, plus
    ``precision_floor`` and an updated ``achieved_floor`` bool.
    """
    valid = match_curve[
        (match_curve["precision_match"] >= floor)
        & (match_curve["n_pred_match"] > 0)
    ]

    if valid.empty:
        return {
            "method":               "precision_floor_match",
            "baseline_name":        baseline_name,
            "task":                 task,
            "threshold":            None,
            "precision_match":      None,
            "recall_match":         None,
            "f_beta_match":         None,
            "n_pred_match":         None,
            "n_total_binary":       int(match_curve["n_total"].iloc[0]),
            "n_uncertain_excluded": n_uncertain,
            "precision_floor":      floor,
            "achieved_floor":       False,
            "f1_beta":              f1_beta,
            "note":                 f"No threshold achieves precision_match >= {floor:.2f}",
        }

    # Minimum threshold among valid rows = maximum recall at the floor
    best = valid.loc[valid["threshold"].idxmin()]

    return {
        "method":               "precision_floor_match",
        "baseline_name":        baseline_name,
        "task":                 task,
        "threshold":            float(best["threshold"]),
        "precision_match":      float(best["precision_match"]),
        "recall_match":         float(best["recall_match"]),
        "f_beta_match":         float(best["f_beta_match"]),
        "n_pred_match":         int(best["n_pred_match"]),
        "n_total_binary":       int(best["n_total"]),
        "n_uncertain_excluded": n_uncertain,
        "precision_floor":      floor,
        "achieved_floor":       True,
        "f1_beta":              f1_beta,
        "note":                 f"Lowest threshold achieving precision_match >= {floor:.2f}",
    }


def tune_two_threshold(
    match_curve: pd.DataFrame,
    nonmatch_curve: pd.DataFrame,
    scores: np.ndarray,
    gold: np.ndarray,
    baseline_name: str,
    task: str,
    floor_match: float,
    floor_nonmatch: float,
    n_uncertain: int,
) -> dict:
    """Compute (t_low, t_high) via independent precision-floor tuning.

    ``t_high`` = lowest threshold with ``precision_match >= floor_match``.
    ``t_low``  = highest threshold with ``precision_nonmatch >= floor_nonmatch``.

    Semantics:
      * score >= t_high  -> predict MATCH     (confident)
      * score <  t_low   -> predict NON-MATCH (confident)
      * t_low <= score < t_high -> ABSTAIN    (uncertain band)

    If ``t_low > t_high`` (the precision floors are too tight to leave a
    meaningful separation), the band is collapsed by clamping t_low = t_high,
    which corresponds to no abstention.

    Returns:
        Dict with keys:
        ``method``, ``baseline_name``, ``task``,
        ``t_low``, ``t_high``, ``precision_floor_match``,
        ``precision_floor_nonmatch``,
        ``definite_precision_match``, ``definite_recall_match``,
        ``definite_precision_nonmatch``, ``definite_recall_nonmatch``,
        ``coverage``, ``abstention_rate``, ``n_in_band``,
        ``n_total_binary``, ``n_uncertain_excluded``,
        ``achieved_floor_match``, ``achieved_floor_nonmatch``, ``note``.
    """
    notes: list[str] = []

    # --- t_high: lowest t with precision_match >= floor_match ----------------
    valid_high = match_curve[
        (match_curve["precision_match"] >= floor_match)
        & (match_curve["n_pred_match"] > 0)
    ]
    if valid_high.empty:
        t_high = 1.0
        achieved_high = False
        notes.append(f"no t_high achieves precision_match >= {floor_match:.2f}")
    else:
        t_high = float(valid_high.loc[valid_high["threshold"].idxmin(), "threshold"])
        achieved_high = True

    # --- t_low: highest t with precision_nonmatch >= floor_nonmatch ----------
    valid_low = nonmatch_curve[
        (nonmatch_curve["precision_nonmatch"] >= floor_nonmatch)
        & (nonmatch_curve["n_pred_nonmatch"] > 0)
    ]
    if valid_low.empty:
        t_low = 0.0
        achieved_low = False
        notes.append(f"no t_low achieves precision_nonmatch >= {floor_nonmatch:.2f}")
    else:
        t_low = float(valid_low.loc[valid_low["threshold"].idxmax(), "threshold"])
        achieved_low = True

    # --- collapse band if t_low > t_high -------------------------------------
    if t_low > t_high:
        notes.append(
            f"band collapsed: t_low ({t_low:.2f}) > t_high ({t_high:.2f}); "
            f"t_low clamped to t_high"
        )
        t_low = t_high

    # --- compute stats at the chosen (t_low, t_high) -------------------------
    pred_match    = scores >= t_high
    pred_nonmatch = scores <  t_low
    in_band       = (scores >= t_low) & (scores < t_high)

    n_true_match    = int((gold == 1).sum())
    n_true_nonmatch = int((gold == 0).sum())
    n_total         = len(scores)

    tp    = int((pred_match    & (gold == 1)).sum())
    fp    = int((pred_match    & (gold == 0)).sum())
    tn    = int((pred_nonmatch & (gold == 0)).sum())
    fn_nm = int((pred_nonmatch & (gold == 1)).sum())
    n_band = int(in_band.sum())

    def_prec_m  = tp / (tp + fp)       if (tp + fp) > 0       else 1.0
    def_rec_m   = tp / n_true_match    if n_true_match > 0     else 0.0
    def_prec_nm = tn / (tn + fn_nm)    if (tn + fn_nm) > 0    else 1.0
    def_rec_nm  = tn / n_true_nonmatch if n_true_nonmatch > 0  else 0.0

    coverage        = (n_total - n_band) / n_total if n_total > 0 else 0.0
    abstention_rate = n_band              / n_total if n_total > 0 else 0.0

    return {
        "method":                       "two_threshold",
        "baseline_name":                baseline_name,
        "task":                         task,
        "t_low":                        t_low,
        "t_high":                       t_high,
        "precision_floor_match":        floor_match,
        "precision_floor_nonmatch":     floor_nonmatch,
        "definite_precision_match":     def_prec_m,
        "definite_recall_match":        def_rec_m,
        "definite_precision_nonmatch":  def_prec_nm,
        "definite_recall_nonmatch":     def_rec_nm,
        "coverage":                     coverage,
        "abstention_rate":              abstention_rate,
        "n_in_band":                    n_band,
        "n_total_binary":               n_total,
        "n_uncertain_excluded":         n_uncertain,
        "achieved_floor_match":         achieved_high,
        "achieved_floor_nonmatch":      achieved_low,
        "note":                         "; ".join(notes),
    }
