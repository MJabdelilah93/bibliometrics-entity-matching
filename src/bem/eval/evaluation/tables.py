"""tables.py — Format evaluation metrics as CSV, Markdown, and LaTeX tables.

Main table structure (one per task)
------------------------------------
Each row represents one (system, method) combination.  Columns are grouped
into three panels separated by vertical whitespace in the manuscript:

  Panel A  : match precision, recall, F1 [95% CI], coverage, uncertain%
  Panel B  : non-match precision, recall, F1
  Panel C  : three-way accuracy, macro-F1

The LaTeX output uses the booktabs package.  BEM is shown first, followed
by the three classical baselines, followed (optionally) by the auxiliary
Scopus-ID comparator in an italic section.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Build flat metrics DataFrame from a list of result dicts
# ---------------------------------------------------------------------------

def build_metrics_df(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten a list of per-(system, method) result dicts into a DataFrame.

    Each result dict must have:
      task, system, display_name, method, split, + all keys from compute_all_metrics.

    Returns:
        Wide DataFrame ready for inspection or export.
    """
    rows = []
    for r in results:
        m = r.get("metrics", {})
        ci_f1_m   = m.get("ci_f1_match",    (float("nan"), float("nan")))
        ci_prec_m = m.get("ci_precision_match", (float("nan"), float("nan")))
        ci_rec_m  = m.get("ci_recall_match",    (float("nan"), float("nan")))

        rows.append({
            # Identity
            "task":           r.get("task"),
            "split":          r.get("split"),
            "system":         r.get("system"),
            "display_name":   r.get("display_name"),
            "method":         r.get("method"),
            "is_auxiliary":   r.get("is_auxiliary", False),
            # Counts
            "n_total":        m.get("n_total"),
            "n_gold_binary":  m.get("n_gold_binary"),
            "n_uncertain_gold": m.get("n_uncertain_gold"),
            "n_auto_decided": m.get("n_auto_decided"),
            # Coverage
            "coverage":       m.get("coverage"),
            "uncertain_rate": m.get("uncertain_rate"),
            # Match (primary)
            "precision_match": m.get("precision_match"),
            "recall_match":    m.get("recall_match"),
            "f1_match":        m.get("f1_match"),
            "ci_f1_match_lo":  ci_f1_m[0],
            "ci_f1_match_hi":  ci_f1_m[1],
            "ci_prec_match_lo": ci_prec_m[0],
            "ci_prec_match_hi": ci_prec_m[1],
            # Non-match (secondary)
            "precision_nonmatch": m.get("precision_nonmatch"),
            "recall_nonmatch":    m.get("recall_nonmatch"),
            "f1_nonmatch":        m.get("f1_nonmatch"),
            # Aggregate
            "three_way_accuracy": m.get("three_way_accuracy"),
            "macro_f1_binary":    m.get("macro_f1_binary"),
            # Threshold info
            "threshold_value": r.get("threshold_value"),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v * 100:.1f}%"


def _fmt_ci(lo, hi, decimals: int = 3) -> str:
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (lo, hi)):
        return ""
    return f"[{lo:.{decimals}f}, {hi:.{decimals}f}]"


def _method_label(method: str) -> str:
    return {
        "precision_floor_match": "Floor(0.90)",
        "f1_optimal":            "F1-opt",
        "two_threshold":         "2-thresh",
        "bem_routing":           "auto-route",
    }.get(method, method)


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def to_markdown(df: pd.DataFrame, task: str, split: str) -> str:
    """Format the metrics DataFrame as a GitHub-Flavored Markdown table.

    Returns:
        Markdown string with a header note about the evaluation split and
        denominator conventions.
    """
    lines: list[str] = [
        f"## {task.upper()} evaluation — {split} split\n",
        (
            "Columns: Prec_M / Rec_M / F1_M / Cov / Unc% = match precision, recall, F1, "
            "coverage, uncertain rate.  "
            "CI = bootstrap 95 % CI for F1_M.  "
            "Denominators: precision_M = TP/(TP+FP); recall_M = TP/all true matches "
            "(including abstained); Coverage = auto-decided / all pairs.\n"
        ),
        "| System | Method | Prec_M | Rec_M | F1_M | CI_F1_M | Cov | Unc% "
        "| Prec_NM | Rec_NM | F1_NM | 3W-Acc | Macro-F1 |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: "
        "| ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, row in df.iterrows():
        m_label  = _method_label(row.get("method", ""))
        ci_str   = _fmt_ci(row.get("ci_f1_match_lo"), row.get("ci_f1_match_hi"))
        name     = row.get("display_name", "")
        if row.get("is_auxiliary"):
            name = f"*{name}*"

        lines.append(
            f"| {name} | {m_label} "
            f"| {_fmt(row.get('precision_match'))} "
            f"| {_fmt(row.get('recall_match'))} "
            f"| {_fmt(row.get('f1_match'))} "
            f"| {ci_str} "
            f"| {_fmt(row.get('coverage'))} "
            f"| {_fmt_pct(row.get('uncertain_rate'))} "
            f"| {_fmt(row.get('precision_nonmatch'))} "
            f"| {_fmt(row.get('recall_nonmatch'))} "
            f"| {_fmt(row.get('f1_nonmatch'))} "
            f"| {_fmt(row.get('three_way_accuracy'))} "
            f"| {_fmt(row.get('macro_f1_binary'))} |"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# LaTeX table (booktabs)
# ---------------------------------------------------------------------------

def to_latex(df: pd.DataFrame, task: str, split: str) -> str:
    """Format the metrics DataFrame as a LaTeX booktabs table.

    Returns:
        LaTeX string including ``\\begin{table}`` / ``\\end{table}`` wrapper.
    """
    task_u  = task.upper()
    split_c = split.capitalize()

    header = rf"""\begin{{table}}[ht]
\centering
\caption{{{task_u} evaluation on the {split_c} split. Uncertain gold labels excluded from
precision/recall/F1. Coverage $=$ auto-decided\,/\,all pairs. CI $=$ bootstrap 95\%
confidence interval (1\,000 resamples). Primary operating point: precision floor
$\geq 0.90$ (Floor).}}
\label{{tab:{task.lower()}-eval-{split.lower()}}}
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{llccccccccc}}
\toprule
System & Method &
  $\text{{Prec}}_M$ & $\text{{Rec}}_M$ & $\text{{F1}}_M$ [95\%\,CI] &
  Cov & Unc\% &
  $\text{{Prec}}_{{NM}}$ & $\text{{F1}}_{{NM}}$ &
  3W-Acc \\
\midrule"""

    body_lines: list[str] = []
    prev_system = None
    in_aux = False

    for _, row in df.iterrows():
        system    = row.get("system", "")
        name      = _tex_escape(str(row.get("display_name", "")))
        method    = _method_label(row.get("method", ""))
        is_aux    = bool(row.get("is_auxiliary", False))

        # Insert midrule between system groups
        if prev_system is not None and system != prev_system and not in_aux:
            body_lines.append(r"\midrule")
        if is_aux and not in_aux:
            body_lines.append(r"\midrule")
            in_aux = True
        prev_system = system

        ci_str = _fmt_ci(
            row.get("ci_f1_match_lo"), row.get("ci_f1_match_hi"), decimals=3
        )
        # Wrap CI in smaller text to save column space
        ci_cell = rf"\small {ci_str}" if ci_str else "—"

        fmt_name = rf"\textit{{{name}}}" if is_aux else name

        line = (
            f"  {fmt_name} & {method} & "
            f"{_fmt(row.get('precision_match'))} & "
            f"{_fmt(row.get('recall_match'))} & "
            f"{_fmt(row.get('f1_match'))} {ci_cell} & "
            f"{_fmt(row.get('coverage'))} & "
            f"{_fmt_pct(row.get('uncertain_rate'))} & "
            f"{_fmt(row.get('precision_nonmatch'))} & "
            f"{_fmt(row.get('f1_nonmatch'))} & "
            f"{_fmt(row.get('three_way_accuracy'))} \\\\"
        )
        body_lines.append(line)

    footer = r"""\bottomrule
\end{tabular}
\end{table}"""

    return "\n".join([header] + body_lines + [footer]) + "\n"


def _tex_escape(s: str) -> str:
    return (
        s.replace("&", r"\&")
         .replace("%", r"\%")
         .replace("_", r"\_")
         .replace("#", r"\#")
         .replace("{", r"\{")
         .replace("}", r"\}")
    )
