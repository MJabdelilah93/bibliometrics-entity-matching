"""apply_guards.py — Confidence guards and auto-routing logic (C6).

B2 Guard: auto-route a pair when BOTH conditions hold:
  1. confidence >= t_match  (or <= t_nonmatch for non-match direction)
  2. evidence_used contains at least M distinct signal *categories*

Signal categories are semantic buckets derived from the evidence_used list
returned by the LLM.  Category membership is determined by keyword lookup
on each evidence_used item (case-insensitive substring match).

AND signal categories
---------------------
  NAME       : name, author_name, surname, initials, first_name, last_name,
               given_name, full_name, name_similarity
  COAUTHOR   : coauthor, co-author, collaborator, shared_coauthor
  AFFIL      : affiliation, affil, institution, university, department,
               lab, laboratory, org, organisation, organization

AIN signal categories
---------------------
  STRING         : string, affil_string, text, raw_string, normalised,
                   string_similarity, str_sim, levenshtein
  ACRONYM        : acronym, abbreviation, acro, acronym_overlap
  LINKED_AUTHORS : linked_author, linked_authors, author_list,
                   linked_author_overlap

Routing outcomes
----------------
  auto_match   : label=="match"     AND confidence >= t_match  AND signals >= M
  auto_nonmatch: label=="non-match" AND confidence >= t_nonmatch
  adjudication : everything else (routed_to_human=True)

Output columns added to the DataFrame
--------------------------------------
  label_final      str   "match" | "non-match" | "uncertain"
  override_reason  str   empty unless guard overrides the LLM label
  signals_count    int   number of distinct signal categories fired
  fired_categories str   pipe-separated category names
  routed_to_human  bool  True when sent to adjudication queue
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Signal category keyword tables
# ---------------------------------------------------------------------------

AND_SIGNAL_CATEGORIES: dict[str, set[str]] = {
    "NAME": {
        "name", "author_name", "surname", "initials", "first_name",
        "last_name", "given_name", "full_name", "name_similarity",
    },
    "COAUTHOR": {
        "coauthor", "co-author", "coauthors", "co-authors",
        "collaborator", "collaborators", "shared_coauthor",
    },
    "AFFIL": {
        "affiliation", "affil", "institution", "university",
        "department", "lab", "laboratory", "org", "organisation",
        "organization", "affiliation_similarity",
    },
}

AIN_SIGNAL_CATEGORIES: dict[str, set[str]] = {
    "STRING": {
        "string", "affil_string", "text", "raw_string", "normalised",
        "string_similarity", "str_sim", "levenshtein",
    },
    "ACRONYM": {
        "acronym", "abbreviation", "acro", "acronyms",
        "acronym_overlap",
    },
    "LINKED_AUTHORS": {
        "linked_author", "linked_authors", "author_list",
        "linked_author_overlap",
    },
}

_CATEGORY_TABLES: dict[str, dict[str, set[str]]] = {
    "AND": AND_SIGNAL_CATEGORIES,
    "AIN": AIN_SIGNAL_CATEGORIES,
}


# ---------------------------------------------------------------------------
# Load decisions
# ---------------------------------------------------------------------------

def load_llm_decisions_jsonl(path: str | Path) -> pd.DataFrame:
    """Load an LLM decisions JSONL file into a tidy DataFrame.

    Each line is expected to contain a ``decision`` sub-object with
    ``label``, ``confidence``, ``evidence_used``, ``reason_code``, and
    optionally ``abstention_reason``.

    Error lines (missing ``decision`` key) are included with
    label="error", confidence=0.0, evidence_used=[].

    Returns
    -------
    DataFrame with columns:
        anchor_id, candidate_id, task, label, confidence, evidence_used
        (Python list), reason_code, abstention_reason, backend, retry_count
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Decisions JSONL not found: {p}")

    records = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            dec = obj.get("decision") or {}
            records.append({
                "anchor_id":         obj.get("anchor_id", ""),
                "candidate_id":      obj.get("candidate_id", ""),
                "task":              obj.get("task", ""),
                "label":             dec.get("label", "error") if dec else "error",
                "confidence":        float(dec.get("confidence", 0.0)) if dec else 0.0,
                "evidence_used":     dec.get("evidence_used", []) if dec else [],
                "reason_code":       dec.get("reason_code", obj.get("error", "")) if dec else obj.get("error", ""),
                "abstention_reason": dec.get("abstention_reason") if dec else None,
                "backend":           obj.get("backend", ""),
                "retry_count":       int(obj.get("retry_count", 0)),
            })

    if not records:
        return pd.DataFrame(columns=[
            "anchor_id", "candidate_id", "task", "label", "confidence",
            "evidence_used", "reason_code", "abstention_reason",
            "backend", "retry_count",
        ])

    df = pd.DataFrame(records)

    # Deduplicate: a pair may appear multiple times when resume appended entries.
    # Keep the last non-error entry per pair; fall back to the last entry if all
    # entries for that pair are errors.
    #
    # Strategy: assign a sort key so that error rows come before non-error rows,
    # then groupby-last picks the last non-error row when one exists, or the last
    # error row otherwise.  This avoids applying a lambda over empty DataFrames
    # (which returns a DataFrame rather than a Series in some pandas versions,
    # causing groupby to crash with KeyError).
    # _is_error: 1 for error rows, 0 for non-error rows.
    # Sort descending on _is_error so errors (1) come first within each pair,
    # non-errors (0) come last → groupby-last picks the last non-error when
    # one exists, otherwise the last error row.
    df["_is_error"] = (df["label"] == "error").astype(int)
    df_sorted = df.sort_values(
        ["anchor_id", "candidate_id", "_is_error"],
        ascending=[True, True, False],
        kind="stable",
    )
    result = (
        df_sorted
        .groupby(["anchor_id", "candidate_id"], sort=False)
        .last()
        .reset_index()
    )
    result = result.drop(columns=["_is_error"], errors="ignore")
    df.drop(columns=["_is_error"], inplace=True, errors="ignore")
    return result


# ---------------------------------------------------------------------------
# Signal counting
# ---------------------------------------------------------------------------

def count_signals(evidence_used: list[str], task: str) -> tuple[int, list[str]]:
    """Count distinct signal categories fired for a pair.

    Each item in ``evidence_used`` is matched (case-insensitive substring)
    against the keyword table for ``task``.  A category is counted once
    regardless of how many items match it.

    Returns
    -------
    (count, fired_categories)
        count             — number of distinct categories fired
        fired_categories  — sorted list of category names
    """
    table = _CATEGORY_TABLES.get(task.upper(), {})
    fired: set[str] = set()
    for item in evidence_used:
        item_lower = item.lower().strip()
        for category, keywords in table.items():
            if any(kw in item_lower for kw in keywords):
                fired.add(category)
    return len(fired), sorted(fired)


# ---------------------------------------------------------------------------
# Apply guards
# ---------------------------------------------------------------------------

def apply_guards(
    df: pd.DataFrame,
    task: str,
    thresholds: dict[str, float],
    m_signals: int = 2,
) -> pd.DataFrame:
    """Apply B2 confidence guards to LLM decisions.

    Parameters
    ----------
    df : DataFrame from load_llm_decisions_jsonl (or compatible).
         Required columns: anchor_id, candidate_id, label, confidence,
         evidence_used.
    task : "AND" or "AIN".
    thresholds : dict with keys ``t_match`` and ``t_nonmatch``.
    m_signals : Minimum distinct signal categories required to auto-route
                to *match* (precision floor).

    Returns
    -------
    DataFrame with added columns:
        label_final, override_reason, signals_count,
        fired_categories, routed_to_human.
    """
    t_match    = float(thresholds.get("t_match", 0.85))
    t_nonmatch = float(thresholds.get("t_nonmatch", 0.85))

    out = df.copy()

    label_finals: list[str] = []
    override_reasons: list[str] = []
    signals_counts: list[int] = []
    fired_categories_col: list[str] = []
    routed_to_human: list[bool] = []

    for _, row in out.iterrows():
        label      = str(row.get("label", "uncertain"))
        confidence = float(row.get("confidence", 0.0))
        ev_used    = row.get("evidence_used", [])
        if not isinstance(ev_used, list):
            ev_used = []

        n_signals, fired = count_signals(ev_used, task)
        fired_str = "|".join(fired)

        # Error rows → always adjudication
        if label == "error":
            label_finals.append("uncertain")
            override_reasons.append("error_row")
            signals_counts.append(0)
            fired_categories_col.append("")
            routed_to_human.append(True)
            continue

        # Non-match guard: confidence-only (no signal count check)
        if label == "non-match" and confidence >= t_nonmatch:
            label_finals.append("non-match")
            override_reasons.append("")
            signals_counts.append(n_signals)
            fired_categories_col.append(fired_str)
            routed_to_human.append(False)
            continue

        # Match guard: confidence + M independent signals
        if label == "match" and confidence >= t_match:
            if n_signals >= m_signals:
                label_finals.append("match")
                override_reasons.append("")
                signals_counts.append(n_signals)
                fired_categories_col.append(fired_str)
                routed_to_human.append(False)
            else:
                # High confidence but insufficient independent signals
                label_finals.append("uncertain")
                override_reasons.append(
                    f"match_guard_insufficient_signals:{n_signals}<{m_signals}"
                )
                signals_counts.append(n_signals)
                fired_categories_col.append(fired_str)
                routed_to_human.append(True)
            continue

        # Uncertain band or LLM abstention
        label_finals.append("uncertain")
        override_reasons.append("")
        signals_counts.append(n_signals)
        fired_categories_col.append(fired_str)
        routed_to_human.append(True)

    out["label_final"]      = label_finals
    out["override_reason"]  = override_reasons
    out["signals_count"]    = signals_counts
    out["fired_categories"] = fired_categories_col
    out["routed_to_human"]  = routed_to_human
    return out


# ---------------------------------------------------------------------------
# Benchmark-filtered routing log
# ---------------------------------------------------------------------------

def filter_routing_to_benchmark(
    routing_log_path: str | Path,
    benchmark_path: str | Path,
    out_path: str | Path,
) -> dict[str, Any]:
    """Inner-join a full routing log to benchmark pairs and write a filtered parquet.

    The full routing log may contain more rows than the benchmark (e.g. when
    a resume run processes additional non-benchmark pairs).  This function
    produces a ``*_bm.parquet`` that contains exactly the benchmark pairs,
    suitable for C7 aggregation and evaluation.

    Parameters
    ----------
    routing_log_path : Path to ``routing_log_{task}.parquet``.
    benchmark_path   : Path to ``dev2_benchmark_pairs_{task}.parquet``.
    out_path         : Destination for the filtered ``routing_log_{task}_bm.parquet``.

    Returns
    -------
    dict with keys: task_rows, bm_rows, label_final_dist, routed_to_human_count.

    Raises
    ------
    AssertionError if the inner-join row count != benchmark row count.
    FileNotFoundError if either input is missing.
    """
    rlog = Path(routing_log_path)
    bmp  = Path(benchmark_path)
    outp = Path(out_path)

    if not rlog.exists():
        raise FileNotFoundError(f"Routing log not found: {rlog}")
    if not bmp.exists():
        raise FileNotFoundError(f"Benchmark parquet not found: {bmp}")

    routing_df = pd.read_parquet(rlog)
    bench_df   = pd.read_parquet(bmp)[["anchor_id", "candidate_id", "gold_label"]]

    n_bench = len(bench_df)
    merged  = bench_df.merge(
        routing_df,
        on=["anchor_id", "candidate_id"],
        how="inner",
    )

    assert len(merged) == n_bench, (
        f"Benchmark-filtered routing log has {len(merged)} rows but expected {n_bench}. "
        f"Some benchmark pairs are missing from the routing log."
    )

    outp.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(outp, index=False)

    return {
        "routing_rows":        len(routing_df),
        "bm_rows":             len(merged),
        "label_final_dist":    merged["label_final"].value_counts().to_dict(),
        "routed_to_human":     int(merged["routed_to_human"].sum()),
    }
