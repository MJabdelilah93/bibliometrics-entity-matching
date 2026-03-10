"""slicer.py -- Robustness slice computation for BEM (E7).

Slices computed for each task:
  overall   : all benchmark pairs (reference; matches E5 result)
  Q1        : pairs whose anchor instance is from query frame Q1
  Q2        : pairs whose anchor instance is from query frame Q2
  stratum_* : if a 'stratum' column is present in the routing log (not
              available in the current dev2 benchmarks -- documented clearly)

Each slice is evaluated with the same compute_all_metrics() function used in
E5 so results are directly comparable.  The BEM prediction is taken from
label_final in the benchmark-filtered routing log.

If the Q1/Q2 join fails (e.g., missing instance file), the slice is omitted
and the reason is recorded in the returned metadata rather than raising an
exception.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bem.eval.evaluation.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_robustness_slices(
    routing_log: pd.DataFrame,
    task: str,
    instances_and: Path,
    instances_ain: Path,
    compute_q1_q2: bool,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float,
) -> tuple[list[dict], dict[str, str]]:
    """Compute per-slice metrics for one task.

    Args:
        routing_log: Benchmark-filtered routing log (1 000 rows per task).
        task: 'and' or 'ain'.
        instances_and: Path to author_instances.parquet.
        instances_ain: Path to affil_instances.parquet.
        compute_q1_q2: Whether to attempt the Q1/Q2 join.
        rng: Seeded random generator.
        n_bootstrap: Bootstrap resamples.
        alpha: Bootstrap alpha (e.g. 0.05 -> 95% CI).

    Returns:
        (slice_results, notes)
        slice_results: list of dicts, one per slice.
        notes: dict of {slice_name: note_string} for missing/skipped slices.
    """
    notes: dict[str, str] = {}
    slices: dict[str, pd.DataFrame] = {"overall": routing_log}

    # -----------------------------------------------------------------
    # Q1 / Q2 slicing
    # -----------------------------------------------------------------
    if compute_q1_q2:
        qf = _add_query_frames(routing_log["anchor_id"], task, instances_and, instances_ain)
        if qf is None:
            notes["Q1"] = "skipped: Q1/Q2 join failed (see log)"
            notes["Q2"] = "skipped: Q1/Q2 join failed (see log)"
        else:
            log_with_qf = routing_log.copy()
            log_with_qf["_query_frame"] = qf.values
            for qval in ("Q1", "Q2"):
                sub = log_with_qf[log_with_qf["_query_frame"] == qval]
                if sub.empty:
                    notes[qval] = f"skipped: no {qval} pairs in benchmark split"
                else:
                    slices[qval] = sub
    else:
        notes["Q1"] = "skipped: compute_q1_q2 is false in config"
        notes["Q2"] = "skipped: compute_q1_q2 is false in config"

    # -----------------------------------------------------------------
    # Stratum slicing
    # -----------------------------------------------------------------
    if "stratum" in routing_log.columns:
        for s in routing_log["stratum"].dropna().unique():
            sub = routing_log[routing_log["stratum"] == s]
            if not sub.empty:
                slices[f"stratum_{s}"] = sub
    else:
        notes["stratum"] = (
            "not available: 'stratum' column absent from benchmark "
            "(dev2 benchmarks do not carry stratum labels)"
        )

    # -----------------------------------------------------------------
    # Compute metrics for each slice
    # -----------------------------------------------------------------
    results = []
    for slice_name, sub_df in slices.items():
        gold = sub_df["gold_label"].astype(str)
        pred = sub_df["label_final"].astype(str).apply(
            lambda x: x if x in ("match", "non-match") else "uncertain"
        )
        m = compute_all_metrics(gold, pred, rng, n_bootstrap, alpha)
        ci = m.get("ci_f1_match", (float("nan"), float("nan")))
        results.append({
            "task":             task,
            "slice":            slice_name,
            "n_total":          m.get("n_total", len(sub_df)),
            "n_gold_binary":    m.get("n_gold_binary"),
            "n_uncertain_gold": m.get("n_uncertain_gold"),
            "n_auto_decided":   m.get("n_auto_decided"),
            "coverage":         m.get("coverage"),
            "uncertain_rate":   m.get("uncertain_rate"),
            "precision_match":  m.get("precision_match"),
            "recall_match":     m.get("recall_match"),
            "f1_match":         m.get("f1_match"),
            "ci_f1_lo":         ci[0] if len(ci) > 0 else None,
            "ci_f1_hi":         ci[1] if len(ci) > 1 else None,
            "precision_nonmatch": m.get("precision_nonmatch"),
            "recall_nonmatch":    m.get("recall_nonmatch"),
            "f1_nonmatch":        m.get("f1_nonmatch"),
            "macro_f1_binary":    m.get("macro_f1_binary"),
        })

    return results, notes


# ---------------------------------------------------------------------------
# Q1/Q2 join
# ---------------------------------------------------------------------------

def _add_query_frames(
    anchor_ids: pd.Series,
    task: str,
    instances_and: Path,
    instances_ain: Path,
) -> pd.Series | None:
    """Join anchor_ids to query_frame via the relevant instance table.

    The instance parquets carry a 'query_frame' column directly (Q1/Q2),
    so no secondary join to records_normalised is needed.

    Returns:
        pd.Series of query_frame values aligned with anchor_ids, or None on
        failure.
    """
    try:
        if task == "and":
            inst = pd.read_parquet(
                instances_and, columns=["author_instance_id", "query_frame"]
            )
            id_col = "author_instance_id"
        else:
            inst = pd.read_parquet(
                instances_ain, columns=["affil_instance_id", "query_frame"]
            )
            id_col = "affil_instance_id"

        inst_renamed = inst.rename(columns={id_col: "anchor_id"})
        # Drop duplicates: each anchor_id maps to exactly one query_frame.
        inst_unique = inst_renamed.drop_duplicates(subset="anchor_id")

        merged = (
            anchor_ids
            .to_frame("anchor_id")
            .merge(inst_unique, on="anchor_id", how="left")
        )
        return merged["query_frame"].reset_index(drop=True)

    except Exception as exc:
        print(f"  [E7/slicer] Q1/Q2 join failed for {task.upper()}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Table formatters
# ---------------------------------------------------------------------------

def to_dataframe(slice_results: list[dict]) -> pd.DataFrame:
    """Convert slice result list to a flat DataFrame."""
    return pd.DataFrame(slice_results)


def to_markdown(df: pd.DataFrame, task: str) -> str:
    lines = [
        f"## Robustness slices -- {task.upper()}\n",
        "Columns: N = total pairs; Cov = coverage; Prec/Rec/F1 = match class "
        "(primary); CI = 95% bootstrap; F1_NM = non-match F1.\n",
        "| Slice | N | Cov | Prec_M | Rec_M | F1_M | CI_95 | F1_NM |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['slice']} "
            f"| {_fmt_int(row.get('n_total'))} "
            f"| {_fmt(row.get('coverage'))} "
            f"| {_fmt(row.get('precision_match'))} "
            f"| {_fmt(row.get('recall_match'))} "
            f"| {_fmt(row.get('f1_match'))} "
            f"| {_fmt_ci(row.get('ci_f1_lo'), row.get('ci_f1_hi'))} "
            f"| {_fmt(row.get('f1_nonmatch'))} |"
        )
    return "\n".join(lines) + "\n"


def to_latex(df: pd.DataFrame, task: str) -> str:
    header = (
        "\\begin{table}[ht]\n\\centering\n"
        f"\\caption{{Robustness slices -- {task.upper()} (dev split). "
        "Metrics computed with bootstrap 95\\% CI (1\\,000 resamples). "
        "Q1/Q2 = query frame of anchor instance. "
        "Stratum data not available for current benchmarks.}}\n"
        f"\\label{{tab:robustness_{task}}}\n"
        "\\begin{tabular}{lrrrrrrl}\n\\toprule\n"
        "Slice & $N$ & Cov & Prec$_M$ & Rec$_M$ & F1$_M$ & CI$_{95}$ & F1$_{NM}$ \\\\\n"
        "\\midrule\n"
    )
    body_lines = []
    for _, row in df.iterrows():
        body_lines.append(
            f"{_tex(str(row['slice']))} & "
            f"{_fmt_int(row.get('n_total'))} & "
            f"{_fmt(row.get('coverage'))} & "
            f"{_fmt(row.get('precision_match'))} & "
            f"{_fmt(row.get('recall_match'))} & "
            f"{_fmt(row.get('f1_match'))} & "
            f"{_fmt_ci_tex(row.get('ci_f1_lo'), row.get('ci_f1_hi'))} & "
            f"{_fmt(row.get('f1_nonmatch'))} \\\\"
        )
    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return header + "\n".join(body_lines) + "\n" + footer


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{float(v):.3f}"


def _fmt_int(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return str(int(v))


def _fmt_ci(lo, hi) -> str:
    if lo is None or hi is None:
        return "--"
    try:
        if math.isnan(float(lo)) or math.isnan(float(hi)):
            return "--"
    except (TypeError, ValueError):
        return "--"
    return f"[{float(lo):.3f}, {float(hi):.3f}]"


def _fmt_ci_tex(lo, hi) -> str:
    if lo is None or hi is None:
        return "--"
    try:
        if math.isnan(float(lo)) or math.isnan(float(hi)):
            return "--"
    except (TypeError, ValueError):
        return "--"
    return f"[{float(lo):.3f},\\,{float(hi):.3f}]"


def _tex(s: str) -> str:
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
