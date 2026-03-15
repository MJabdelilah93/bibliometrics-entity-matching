"""tables.py -- Manuscript-ready tables for the ablation stage (E6).

Exports:
  build_ablation_tables(results, output_dir, tasks) -> dict of paths written

Tables produced:

  Main table (for paper body)
    ablation_main.{csv,md,tex}
    -- A2 (no-guards) and A4/A5 (missing fields) side-by-side with full BEM.
      One row per (task, variant). Columns: Prec_M, Rec_M, F1_M, CI_F1_M, Cov, Unc%.

  K-sensitivity table (supplement)
    ablation_k_sensitivity.{csv,md,tex}
    -- A3 results: blocking recall and BEM metrics at K = 10 / 25 / 50.

  Threshold-sweep summary table (supplement)
    ablation_threshold_sweep_{task}.{csv,md,tex}
    -- A6 grid: F1_M at each (t_match, m_signals) combination.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_ablation_tables(
    results: dict[str, Any],
    output_dir: Path,
    tasks: list[str],
) -> dict[str, Path]:
    """Build and write all ablation tables.

    Args:
        results: dict from runner.run_ablations() (keyed by ablation name).
        output_dir: Destination directory.
        tasks: Task names to include.

    Returns:
        Dict of {label: path} for every file written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # --- Main ablation table (A2 + A4 + A5 + A1) ---
    main_rows = _collect_main_rows(results, tasks)
    if main_rows:
        df_main = pd.DataFrame(main_rows)
        written.update(_write_main_table(df_main, output_dir))

    # --- K-sensitivity table (A3) ---
    a3 = results.get("a3_k_sensitivity", {})
    if a3:
        written.update(_write_k_sensitivity_table(a3, output_dir))

    # --- Threshold-sweep summary (A6) ---
    a6 = results.get("a6_threshold_sweep", {})
    if a6:
        for task, sweep_df in a6.items():
            if isinstance(sweep_df, pd.DataFrame) and not sweep_df.empty:
                written.update(_write_threshold_sweep_table(sweep_df, task, output_dir))

    return written


# ---------------------------------------------------------------------------
# Main ablation table
# ---------------------------------------------------------------------------

def _collect_main_rows(results: dict, tasks: list[str]) -> list[dict]:
    """Flatten A1 / A2 / A4 / A5 results into one row per (task, variant)."""
    rows: list[dict] = []

    def _add(ablation_key: str, task_filter: str | None = None) -> None:
        data = results.get(ablation_key, {})
        if not data:
            return
        # data is either dict[task -> list[dict]] or dict[variant -> list[dict]]
        for key, variant_list in data.items():
            if not isinstance(variant_list, list):
                continue
            for entry in variant_list:
                if not isinstance(entry, dict):
                    continue
                t = entry.get("task", key)
                if task_filter and t != task_filter:
                    continue
                if t not in tasks:
                    continue
                m = entry.get("metrics", {})
                ci = m.get("ci_f1_match", [None, None]) or [None, None]
                rows.append({
                    "ablation":    ablation_key,
                    "variant":     entry.get("variant", key),
                    "description": entry.get("description", ""),
                    "task":        t,
                    "prec_m":      m.get("precision_match"),
                    "rec_m":       m.get("recall_match"),
                    "f1_m":        m.get("f1_match"),
                    "ci_f1_lo":    ci[0] if len(ci) > 0 else None,
                    "ci_f1_hi":    ci[1] if len(ci) > 1 else None,
                    "coverage":    m.get("coverage"),
                    "unc_pct":     (m.get("uncertain_rate") or 0.0) * 100,
                    "prec_nm":     m.get("precision_nonmatch"),
                    "rec_nm":      m.get("recall_nonmatch"),
                    "f1_nm":       m.get("f1_nonmatch"),
                    "macro_f1":    m.get("macro_f1_binary"),
                })

    for key in ("a1_single_prompt", "a2_no_guards",
                "a4_missing_fields_and", "a5_missing_fields_ain"):
        _add(key)

    return rows


def _write_main_table(df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    written = {}
    # Sort by task, then ablation, then variant
    df = df.sort_values(["task", "ablation", "variant"]).reset_index(drop=True)

    # CSV
    csv_path = output_dir / "ablation_main.csv"
    df.to_csv(csv_path, index=False)
    written["ablation_main_csv"] = csv_path

    # Markdown
    md_path = output_dir / "ablation_main.md"
    md_path.write_text(_main_to_markdown(df), encoding="utf-8")
    written["ablation_main_md"] = md_path

    # LaTeX
    tex_path = output_dir / "ablation_main.tex"
    tex_path.write_text(_main_to_latex(df), encoding="utf-8")
    written["ablation_main_tex"] = tex_path

    return written


def _main_to_markdown(df: pd.DataFrame) -> str:
    lines = [
        "## Ablation study -- main results\n",
        "Columns: Prec_M / Rec_M / F1_M / CI_F1_M = match precision, recall, F1, 95% CI.  "
        "Cov = coverage (auto-decided / total).  Unc% = uncertain rate.\n",
        "| Task | Ablation | Variant | Prec_M | Rec_M | F1_M | CI_F1_M | Cov | Unc% |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for _, row in df.iterrows():
        ci_str = _fmt_ci(row.get("ci_f1_lo"), row.get("ci_f1_hi"))
        lines.append(
            f"| {row['task'].upper()} "
            f"| {row['ablation']} "
            f"| {row['variant']} "
            f"| {_fmt(row.get('prec_m'))} "
            f"| {_fmt(row.get('rec_m'))} "
            f"| {_fmt(row.get('f1_m'))} "
            f"| {ci_str} "
            f"| {_fmt(row.get('coverage'))} "
            f"| {_fmt_pct(row.get('unc_pct'))} |"
        )
    return "\n".join(lines) + "\n"


def _main_to_latex(df: pd.DataFrame) -> str:
    header = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Ablation study (dev split). "
        "Prec$_M$ / Rec$_M$ / F1$_M$: match precision, recall, F1 at 95\\% CI. "
        "Cov: auto-decided / total pairs.}\n"
        "\\label{tab:ablation_main}\n"
        "\\begin{tabular}{llllllll}\n"
        "\\toprule\n"
        "Task & Ablation & Variant & Prec$_M$ & Rec$_M$ & F1$_M$ & CI$_{95}$ & Cov \\\\\n"
        "\\midrule\n"
    )
    body_lines = []
    prev_task = None
    for _, row in df.iterrows():
        task = row["task"].upper()
        if prev_task and task != prev_task:
            body_lines.append("\\midrule")
        prev_task = task
        ci_str = _fmt_ci_tex(row.get("ci_f1_lo"), row.get("ci_f1_hi"))
        body_lines.append(
            f"{task} & {_tex(row['ablation'])} & {_tex(row['variant'])} & "
            f"{_fmt(row.get('prec_m'))} & {_fmt(row.get('rec_m'))} & "
            f"{_fmt(row.get('f1_m'))} & {ci_str} & "
            f"{_fmt(row.get('coverage'))} \\\\"
        )
    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    return header + "\n".join(body_lines) + "\n" + footer


# ---------------------------------------------------------------------------
# K-sensitivity table
# ---------------------------------------------------------------------------

def _write_k_sensitivity_table(a3: dict, output_dir: Path) -> dict[str, Path]:
    rows = []
    for task, entries in a3.items():
        for e in entries:
            rows.append({
                "task":                   task.upper(),
                "K":                      e.get("k"),
                "blocking_recall_match":  e.get("blocking_recall_match"),
                "blocking_recall_all":    e.get("blocking_recall_all"),
                "n_match_survived":       e.get("n_match_survived"),
                "n_match_total":          e.get("n_match_total"),
                "bem_f1_at_k":            e.get("bem_f1_at_k"),
                "bem_precision_at_k":     e.get("bem_precision_at_k"),
                "bem_recall_at_k":        e.get("bem_recall_at_k"),
                "bem_coverage_at_k":      e.get("bem_coverage_at_k"),
            })
    if not rows:
        return {}

    df = pd.DataFrame(rows).sort_values(["task", "K"])
    written: dict[str, Path] = {}

    csv_path = output_dir / "ablation_k_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    written["ablation_k_csv"] = csv_path

    md_lines = [
        "## Ablation A3: K sensitivity -- blocking recall at K\n",
        "| Task | K | BR_match | BR_all | Survived | Total_match | BEM_F1@K | BEM_Cov@K |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in df.iterrows():
        md_lines.append(
            f"| {row['task']} | {row['K']} "
            f"| {_fmt(row.get('blocking_recall_match'))} "
            f"| {_fmt(row.get('blocking_recall_all'))} "
            f"| {int(row['n_match_survived']) if pd.notna(row['n_match_survived']) else '--'} "
            f"| {int(row['n_match_total']) if pd.notna(row['n_match_total']) else '--'} "
            f"| {_fmt(row.get('bem_f1_at_k'))} "
            f"| {_fmt(row.get('bem_coverage_at_k'))} |"
        )
    md_path = output_dir / "ablation_k_sensitivity.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    written["ablation_k_md"] = md_path

    # LaTeX
    tex = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{Ablation A3: blocking recall at $K$. "
        "BR$_M$ = fraction of gold-match benchmark pairs generated at cap $K$. "
        "BEM metrics computed on surviving pairs only; pairs dropped by $K$ counted as abstained.}\n"
        "\\label{tab:k_sensitivity}\n"
        "\\begin{tabular}{llllll}\n\\toprule\n"
        "Task & $K$ & BR$_M$ & BR$_{all}$ & BEM F1@$K$ & BEM Cov@$K$ \\\\\n\\midrule\n"
    )
    prev_task = None
    for _, row in df.iterrows():
        task = str(row["task"])
        if prev_task and task != prev_task:
            tex += "\\midrule\n"
        prev_task = task
        tex += (
            f"{task} & {int(row['K'])} & {_fmt(row.get('blocking_recall_match'))} & "
            f"{_fmt(row.get('blocking_recall_all'))} & "
            f"{_fmt(row.get('bem_f1_at_k'))} & {_fmt(row.get('bem_coverage_at_k'))} \\\\\n"
        )
    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    tex_path = output_dir / "ablation_k_sensitivity.tex"
    tex_path.write_text(tex, encoding="utf-8")
    written["ablation_k_tex"] = tex_path

    return written


# ---------------------------------------------------------------------------
# Threshold-sweep summary table
# ---------------------------------------------------------------------------

def _write_threshold_sweep_table(
    sweep_df: pd.DataFrame, task: str, output_dir: Path
) -> dict[str, Path]:
    written: dict[str, Path] = {}
    task_upper = task.upper()

    # Pivot: rows = m_signals, columns = t_match, values = f1_match
    if "t_match" not in sweep_df.columns:
        return written

    pivot = sweep_df.pivot_table(
        index="m_signals", columns="t_match", values="f1_match", aggfunc="first"
    )
    pivot.index.name   = "m_signals"
    pivot.columns.name = "t_match"

    csv_path = output_dir / f"ablation_threshold_sweep_{task}.csv"
    pivot.to_csv(csv_path)
    written[f"ablation_sweep_{task}_csv"] = csv_path

    # Markdown pivot table
    md_lines = [
        f"## Ablation A6: threshold sweep -- {task_upper} (F1_match)\n",
        "Rows: `m_signals` (minimum signal categories for auto-match).  "
        "Columns: `t_match` (confidence threshold).  "
        "Values: F1_match on dev benchmark.\n",
    ]
    col_headers = ["m_signals \\ t_match"] + [f"{c:.2f}" for c in pivot.columns]
    md_lines.append("| " + " | ".join(col_headers) + " |")
    md_lines.append("| " + " | ".join(["---:"] * len(col_headers)) + " |")
    for m_sig, row in pivot.iterrows():
        cells = [str(int(m_sig))] + [_fmt(v) for v in row.values]
        md_lines.append("| " + " | ".join(cells) + " |")
    md_path = output_dir / f"ablation_threshold_sweep_{task}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    written[f"ablation_sweep_{task}_md"] = md_path

    # LaTeX
    col_fmt = "l" + "r" * len(pivot.columns)
    tex = (
        f"\\begin{{table}}[ht]\n\\centering\n"
        f"\\caption{{Ablation A6 ({task_upper}): F1$_M$ at each (t\\_match, m\\_signals) grid point.}}\n"
        f"\\label{{tab:threshold_sweep_{task}}}\n"
        f"\\begin{{tabular}}{{{col_fmt}}}\n\\toprule\n"
    )
    tex += "$m$ \\ $t$ & " + " & ".join(f"{c:.2f}" for c in pivot.columns) + " \\\\\n\\midrule\n"
    for m_sig, row in pivot.iterrows():
        tex += str(int(m_sig)) + " & " + " & ".join(_fmt(v) for v in row.values) + " \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    tex_path = output_dir / f"ablation_threshold_sweep_{task}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    written[f"ablation_sweep_{task}_tex"] = tex_path

    return written


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{float(v):.3f}"


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{float(v):.1f}%"


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
    """Escape LaTeX special characters in a string."""
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
