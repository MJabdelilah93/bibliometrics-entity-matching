"""metrics.py — Compute all evaluation metrics with bootstrap confidence intervals.

Evaluation protocol (locked)
-----------------------------
Positive class   : 'match'
Negative class   : 'non-match'
Excluded         : gold_label == 'uncertain'  (reported as n_uncertain_gold)

Denominators
------------
coverage         = n_auto_decided / n_total
                   n_total       = ALL pairs (including uncertain gold)
                   n_auto_decided = pairs where predicted in {'match', 'non-match'}

precision_match  = TP / (TP + FP)
                   numerator + denominator : definite-match predictions only

recall_match     = TP / (TP + FN + FN_abstained)
                   denominator : ALL true matches in binary gold
                   (abstained true matches count as misses per spec rule 3)

three_way_accuracy = (TP + TN) / n_gold_binary
                   ALL binary-gold pairs in denominator (abstained = missed)

macro_f1_binary  = (f1_match + f1_nonmatch) / 2   over binary gold

Bootstrap CI     : pair-level resample, percentile method, 1 000 resamples by default.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_all_metrics(
    gold_all: pd.Series,
    predicted_all: pd.Series,
    rng: np.random.Generator,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute all evaluation metrics for one (system, task, split) combination.

    Args:
        gold_all:       Gold labels for ALL pairs; values 'match', 'non-match',
                        or 'uncertain'.
        predicted_all:  Predicted labels for ALL pairs; same value set.
        rng:            Seeded random generator for reproducible bootstrap.
        n_bootstrap:    Number of bootstrap resamples.
        alpha:          Significance level for CI (0.05 → 95 % CI).

    Returns:
        Dict with all metrics listed in the module docstring.
    """
    gold_all      = gold_all.astype(str).reset_index(drop=True)
    predicted_all = predicted_all.astype(str).reset_index(drop=True)
    if len(gold_all) != len(predicted_all):
        raise ValueError("gold_all and predicted_all must have the same length")

    n_total = len(gold_all)

    # --- Coverage -------------------------------------------------------
    n_auto_decided = int(predicted_all.isin(["match", "non-match"]).sum())
    n_uncertain_predicted = n_total - n_auto_decided
    coverage       = n_auto_decided / n_total if n_total > 0 else 0.0
    uncertain_rate = 1.0 - coverage

    # --- Filter to binary gold ------------------------------------------
    binary_mask   = gold_all.isin(["match", "non-match"])
    n_gold_binary = int(binary_mask.sum())
    n_uncertain_gold = n_total - n_gold_binary

    if n_gold_binary == 0:
        return _empty_metrics(n_total, n_uncertain_gold, coverage, uncertain_rate)

    gold_bin = gold_all[binary_mask].values          # 'match' | 'non-match'
    pred_bin = predicted_all[binary_mask].values     # 'match' | 'non-match' | 'uncertain'

    # --- Confusion matrix (on binary gold) ------------------------------
    tp  = int(((pred_bin == "match")     & (gold_bin == "match")).sum())
    fp  = int(((pred_bin == "match")     & (gold_bin == "non-match")).sum())
    tn  = int(((pred_bin == "non-match") & (gold_bin == "non-match")).sum())
    fn  = int(((pred_bin == "non-match") & (gold_bin == "match")).sum())

    # Abstained counts on binary gold
    abs_on_match    = int(((pred_bin == "uncertain") & (gold_bin == "match")).sum())
    abs_on_nonmatch = int(((pred_bin == "uncertain") & (gold_bin == "non-match")).sum())
    n_abstained_on_binary = abs_on_match + abs_on_nonmatch

    n_true_match    = tp + fn + abs_on_match
    n_true_nonmatch = tn + fp + abs_on_nonmatch

    # --- Primary: match class -------------------------------------------
    prec_m = tp / (tp + fp)           if (tp + fp) > 0       else 1.0
    rec_m  = tp / n_true_match        if n_true_match > 0    else 0.0
    f1_m   = _f1(prec_m, rec_m)

    # --- Secondary: non-match class -------------------------------------
    prec_nm = tn / (tn + fn)          if (tn + fn) > 0       else 1.0
    rec_nm  = tn / n_true_nonmatch    if n_true_nonmatch > 0 else 0.0
    f1_nm   = _f1(prec_nm, rec_nm)

    # --- Three-way accuracy + macro F1 ----------------------------------
    three_way_acc = (tp + tn) / n_gold_binary if n_gold_binary > 0 else 0.0
    macro_f1      = (f1_m + f1_nm) / 2.0

    # --- Bootstrap CIs (pair-level, on binary gold pairs) ---------------
    ci_prec_m  = _bootstrap_ci(gold_bin, pred_bin, _bootstrap_prec_match,  rng, n_bootstrap, alpha)
    ci_rec_m   = _bootstrap_ci(gold_bin, pred_bin, _bootstrap_rec_match,   rng, n_bootstrap, alpha)
    ci_f1_m    = _bootstrap_ci(gold_bin, pred_bin, _bootstrap_f1_match,    rng, n_bootstrap, alpha)
    ci_prec_nm = _bootstrap_ci(gold_bin, pred_bin, _bootstrap_prec_nonmatch, rng, n_bootstrap, alpha)
    ci_rec_nm  = _bootstrap_ci(gold_bin, pred_bin, _bootstrap_rec_nonmatch,  rng, n_bootstrap, alpha)
    ci_f1_nm   = _bootstrap_ci(gold_bin, pred_bin, _bootstrap_f1_nonmatch,   rng, n_bootstrap, alpha)

    return {
        # Counts
        "n_total":              n_total,
        "n_gold_binary":        n_gold_binary,
        "n_uncertain_gold":     n_uncertain_gold,
        "n_auto_decided":       n_auto_decided,
        "n_uncertain_predicted": n_uncertain_predicted,
        "n_abstained_on_binary": n_abstained_on_binary,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_true_match":         n_true_match,
        "n_true_nonmatch":      n_true_nonmatch,
        # Coverage
        "coverage":             coverage,
        "uncertain_rate":       uncertain_rate,
        # Match class (primary)
        "precision_match":      prec_m,
        "recall_match":         rec_m,
        "f1_match":             f1_m,
        "ci_precision_match":   ci_prec_m,
        "ci_recall_match":      ci_rec_m,
        "ci_f1_match":          ci_f1_m,
        # Non-match class (secondary)
        "precision_nonmatch":   prec_nm,
        "recall_nonmatch":      rec_nm,
        "f1_nonmatch":          f1_nm,
        "ci_precision_nonmatch": ci_prec_nm,
        "ci_recall_nonmatch":    ci_rec_nm,
        "ci_f1_nonmatch":        ci_f1_nm,
        # Aggregate
        "three_way_accuracy":   three_way_acc,
        "macro_f1_binary":      macro_f1,
        # Meta
        "n_bootstrap":          n_bootstrap,
        "bootstrap_alpha":      alpha,
    }


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    gold: np.ndarray,
    predicted: np.ndarray,
    metric_fn,
    rng: np.random.Generator,
    n_samples: int,
    alpha: float,
) -> tuple[float, float]:
    n = len(gold)
    values = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, n)
        v = metric_fn(gold[idx], predicted[idx])
        if not (v is None or (isinstance(v, float) and math.isnan(v))):
            values.append(v)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    values.sort()
    lo_idx = max(0, int(math.floor(alpha / 2 * len(values))))
    hi_idx = min(len(values) - 1, int(math.ceil((1 - alpha / 2) * len(values))) - 1)
    return (float(values[lo_idx]), float(values[hi_idx]))


def _bootstrap_prec_match(g, p) -> float:
    tp = ((p == "match") & (g == "match")).sum()
    fp = ((p == "match") & (g == "non-match")).sum()
    return tp / (tp + fp) if (tp + fp) > 0 else float("nan")


def _bootstrap_rec_match(g, p) -> float:
    tp  = ((p == "match")     & (g == "match")).sum()
    abs_= ((p == "uncertain") & (g == "match")).sum()
    fn  = ((p == "non-match") & (g == "match")).sum()
    denom = tp + fn + abs_
    return tp / denom if denom > 0 else float("nan")


def _bootstrap_f1_match(g, p) -> float:
    prec = _bootstrap_prec_match(g, p)
    rec  = _bootstrap_rec_match(g, p)
    return _f1(prec, rec)


def _bootstrap_prec_nonmatch(g, p) -> float:
    tn = ((p == "non-match") & (g == "non-match")).sum()
    fn = ((p == "non-match") & (g == "match")).sum()
    return tn / (tn + fn) if (tn + fn) > 0 else float("nan")


def _bootstrap_rec_nonmatch(g, p) -> float:
    tn   = ((p == "non-match") & (g == "non-match")).sum()
    abs_ = ((p == "uncertain") & (g == "non-match")).sum()
    fp   = ((p == "match")     & (g == "non-match")).sum()
    denom = tn + fp + abs_
    return tn / denom if denom > 0 else float("nan")


def _bootstrap_f1_nonmatch(g, p) -> float:
    prec = _bootstrap_prec_nonmatch(g, p)
    rec  = _bootstrap_rec_nonmatch(g, p)
    return _f1(prec, rec)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f1(precision: float, recall: float) -> float:
    if precision is None or recall is None:
        return float("nan")
    if math.isnan(precision) or math.isnan(recall):
        return float("nan")
    denom = precision + recall
    return 2 * precision * recall / denom if denom > 0 else 0.0


def _empty_metrics(
    n_total: int,
    n_uncertain_gold: int,
    coverage: float,
    uncertain_rate: float,
) -> dict[str, Any]:
    nan = float("nan")
    return {
        "n_total": n_total, "n_gold_binary": 0, "n_uncertain_gold": n_uncertain_gold,
        "n_auto_decided": int(n_total * coverage), "n_uncertain_predicted": 0,
        "n_abstained_on_binary": 0,
        "tp": 0, "fp": 0, "tn": 0, "fn": 0,
        "n_true_match": 0, "n_true_nonmatch": 0,
        "coverage": coverage, "uncertain_rate": uncertain_rate,
        "precision_match": nan, "recall_match": nan, "f1_match": nan,
        "ci_precision_match": (nan, nan), "ci_recall_match": (nan, nan),
        "ci_f1_match": (nan, nan),
        "precision_nonmatch": nan, "recall_nonmatch": nan, "f1_nonmatch": nan,
        "ci_precision_nonmatch": (nan, nan), "ci_recall_nonmatch": (nan, nan),
        "ci_f1_nonmatch": (nan, nan),
        "three_way_accuracy": nan, "macro_f1_binary": nan,
        "n_bootstrap": 0, "bootstrap_alpha": 0.05,
    }
