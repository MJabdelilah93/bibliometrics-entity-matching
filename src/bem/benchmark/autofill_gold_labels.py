"""autofill_gold_labels.py — Deterministic auto-fill of clear gold labels.

Conservative deterministic rules fill gold_label ONLY for high-confidence cases
derived exclusively from fields already present in the annotation-packet CSVs.
All other rows are left untouched.  No LLM is involved at any step.

Usage
-----
    python -m bem.benchmark.autofill_gold_labels \\
        --and_in  data/derived/annotation_packets_and.csv \\
        --ain_in  data/derived/annotation_packets_ain.csv \\
        --and_out data/derived/annotation_packets_and.csv \\
        --ain_out data/derived/annotation_packets_ain.csv \\
        --overwrite false

Options
-------
    --and_in      AND annotation packets CSV (default: data/derived/annotation_packets_and.csv)
    --ain_in      AIN annotation packets CSV (default: data/derived/annotation_packets_ain.csv)
    --and_out     Output path for AND CSV (default: same as --and_in)
    --ain_out     Output path for AIN CSV (default: same as --ain_in)
    --overwrite   'true' to overwrite existing gold_label values; 'false' to skip
                  already-labelled rows (default: false)

AND rules applied
-----------------
AND_MATCH    name_sim >= 0.97
             AND (coauthor_overlap >= 1 OR affil_sim >= 0.95)
             AND (surname_match == True  if surnames are extractable)

AND_NONMATCH (surnames extractable AND surnames differ AND name_sim <= 0.85)
             OR (name_sim <= 0.70)

AIN rules applied
-----------------
AIN_MATCH    str_sim >= 0.95
             AND (acronym_overlap OR token_jaccard >= 0.60)
             AND (str_sim >= 0.97 if fallback == true)

AIN_NONMATCH str_sim <= 0.45
             AND token_jaccard <= 0.20
             AND acronym_overlap == False

Audit columns added to every output row
----------------------------------------
    auto_filled   "true" / "false"
    auto_rule     rule tag (e.g. AND_MATCH_NAME+COAUTH) or "" / "skipped:already_filled"
    auto_metrics  compact signal string (e.g. "name_sim=0.982, co_ov=2, affil_sim=0.87")

Rows that already have a gold_label and --overwrite=false receive
auto_filled="false", auto_rule="skipped:already_filled", auto_metrics="".
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz


# ---------------------------------------------------------------------------
# A) Parsing helpers
# ---------------------------------------------------------------------------

_NUM_PAREN_RE = re.compile(r"\s*\(\d{6,20}\)")


def strip_numeric_parentheses(s: str) -> str:
    """Remove ' (digits)' suffixes where parentheses contain ONLY digits, length 6–20.

    Parenthesised tokens containing any non-digit character are left untouched,
    so department acronyms like '(LMCE)' or '(UM6P)' are preserved.

    Examples:
        "ez-zahraouy, hamid (7004513174)" → "ez-zahraouy, hamid"
        "lazfi, souad (55998718300)"       → "lazfi, souad"
        "university (LMCE)"                → "university (LMCE)"
    """
    return _NUM_PAREN_RE.sub("", s).strip()


def safe_str(v: object) -> str:
    """Return str(v), or '' for None / float NaN / 'nan' / 'none'."""
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none") else s


def normalise_name_for_compare(s: object) -> str:
    """safe_str → strip_numeric_parentheses → lowercase → collapse whitespace."""
    t = strip_numeric_parentheses(safe_str(s))
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def split_people_list(s: object) -> list[str]:
    """Split a pipe-separated coauthor / linked-author string into normalised names.

    Strips the truncation marker '…', applies strip_numeric_parentheses, and
    normalises each name via normalise_name_for_compare.
    """
    raw = safe_str(s)
    if not raw:
        return []
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    # drop the truncation marker written by _pipe_join
    parts = [p for p in parts if p != "…"]
    return [normalise_name_for_compare(p) for p in parts]


def surname_initial(name: str) -> tuple[str, str]:
    """Extract (surname, first_initial) from a normalised name string.

    surname      = last whitespace-separated token.
    first_initial = first character of the first whitespace-separated token.

    Returns ("", "") when the name is empty or has fewer than 2 tokens (i.e.
    a reliable surname / initial pair cannot be extracted).
    """
    n = normalise_name_for_compare(name)
    tokens = n.split()
    if len(tokens) < 2:
        return ("", "")
    surname = tokens[-1]
    initial = tokens[0][0] if tokens[0] else ""
    return (surname, initial)


# ---------------------------------------------------------------------------
# B) Token helpers
# ---------------------------------------------------------------------------

def _alnum_tokens(s: str) -> set[str]:
    """Lowercase alphanumeric tokens of length >= 2."""
    return {t for t in re.split(r"\W+", s.lower()) if len(t) >= 2}


def _token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity of alnum-token sets (length >= 2)."""
    ta = _alnum_tokens(a)
    tb = _alnum_tokens(b)
    if not ta and not tb:
        return 0.0          # both empty → treat as uninformative
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _parse_acronyms(s: object) -> set[str]:
    """Parse a pipe-separated acronym string into a set of uppercase tokens."""
    raw = safe_str(s)
    if not raw:
        return set()
    return {p.strip().upper() for p in raw.split("|") if p.strip() and p.strip() != "…"}


# ---------------------------------------------------------------------------
# C) AND signal extraction
# ---------------------------------------------------------------------------

def _and_signals(row: pd.Series) -> dict:
    """Compute AND comparison signals from annotation-packet row fields."""
    anch_name  = normalise_name_for_compare(row.get("anchor_author_norm",    ""))
    cand_name  = normalise_name_for_compare(row.get("candidate_author_norm", ""))
    anch_affil = normalise_name_for_compare(row.get("anchor_affiliations_norm",    ""))
    cand_affil = normalise_name_for_compare(row.get("candidate_affiliations_norm", ""))

    # Guard: empty names → signals uninformative
    if not anch_name and not cand_name:
        return {
            "name_sim": 0.0, "affil_sim": 0.0, "coauthor_overlap_count": 0,
            "surnames_extractable": False, "surname_match": None,
            "sur_a": "", "sur_b": "",
        }

    name_sim  = fuzz.token_sort_ratio(anch_name,  cand_name)  / 100.0
    affil_sim = (
        fuzz.token_set_ratio(anch_affil, cand_affil) / 100.0
        if (anch_affil or cand_affil) else 0.0
    )

    # Coauthor overlap on normalised name strings
    anch_coauth = set(split_people_list(row.get("anchor_coauthors_norm",    "")))
    cand_coauth = set(split_people_list(row.get("candidate_coauthors_norm", "")))
    coauthor_overlap_count = (
        len(anch_coauth & cand_coauth) if (anch_coauth and cand_coauth) else 0
    )

    # Surname / first-initial extraction
    (sur_a, init_a) = surname_initial(anch_name)
    (sur_b, init_b) = surname_initial(cand_name)
    surnames_extractable = bool(sur_a and sur_b)
    surname_match = (sur_a == sur_b) if surnames_extractable else None

    return {
        "name_sim":               name_sim,
        "affil_sim":              affil_sim,
        "coauthor_overlap_count": coauthor_overlap_count,
        "surnames_extractable":   surnames_extractable,
        "surname_match":          surname_match,
        "sur_a":                  sur_a,
        "sur_b":                  sur_b,
    }


def _apply_and_rules(sig: dict) -> tuple[str | None, str, str]:
    """Apply AND auto-fill rules.

    Returns:
        (label, rule_tag, metrics_str)
        label is None if no rule fires (row left unchanged).
    """
    ns = sig["name_sim"]
    af = sig["affil_sim"]
    co = sig["coauthor_overlap_count"]
    se = sig["surnames_extractable"]
    sm = sig["surname_match"]

    name_detail = (
        f", sur_a={sig['sur_a']!r}, sur_b={sig['sur_b']!r}" if se else ""
    )
    metrics = f"name_sim={ns:.3f}, co_ov={co}, affil_sim={af:.3f}{name_detail}"

    # --- AND_MATCH ---
    if ns >= 0.97:
        corroboration = (co >= 1) or (af >= 0.95)
        # If surnames are extractable, they must agree
        name_ok = (not se) or (sm is True)
        if corroboration and name_ok:
            rule = "AND_MATCH_NAME+" + ("COAUTH" if co >= 1 else "AFFIL")
            return ("match", rule, metrics)

    # --- AND_NONMATCH ---
    # Case 1: extractable surnames disagree AND name similarity is low
    if se and (sm is False) and ns <= 0.85:
        return ("non-match", "AND_NONMATCH_SURNAME_MISMATCH", metrics)
    # Case 2: name similarity is very low regardless of surname info
    if ns <= 0.70:
        return ("non-match", "AND_NONMATCH_LOWNAME", metrics)

    return (None, "", metrics)


# ---------------------------------------------------------------------------
# D) AIN signal extraction
# ---------------------------------------------------------------------------

def _ain_signals(row: pd.Series) -> dict:
    """Compute AIN comparison signals from annotation-packet row fields."""
    anch_norm = normalise_name_for_compare(row.get("anchor_affil_norm",    ""))
    cand_norm = normalise_name_for_compare(row.get("candidate_affil_norm", ""))

    # Guard: both empty → uninformative
    if not anch_norm and not cand_norm:
        return {"str_sim": 0.0, "tok_jac": 0.0, "acronym_overlap": False, "fallback": False}

    str_sim = fuzz.token_set_ratio(anch_norm, cand_norm) / 100.0
    tok_jac = _token_jaccard(anch_norm, cand_norm)

    anch_acros = _parse_acronyms(row.get("anchor_affil_acronyms",    ""))
    cand_acros = _parse_acronyms(row.get("candidate_affil_acronyms", ""))
    acronym_overlap = bool(anch_acros & cand_acros) if (anch_acros and cand_acros) else False

    # fallback column is optional; treat absence as False
    fallback_raw = safe_str(row.get("anchor_linked_authors_fallback", ""))
    fallback = fallback_raw.lower() in ("true", "1", "yes")

    return {
        "str_sim":        str_sim,
        "tok_jac":        tok_jac,
        "acronym_overlap": acronym_overlap,
        "fallback":        fallback,
    }


def _apply_ain_rules(sig: dict) -> tuple[str | None, str, str]:
    """Apply AIN auto-fill rules.

    Returns:
        (label, rule_tag, metrics_str)
        label is None if no rule fires.
    """
    ss = sig["str_sim"]
    tj = sig["tok_jac"]
    ao = sig["acronym_overlap"]
    fb = sig["fallback"]

    metrics = f"str_sim={ss:.3f}, jac={tj:.3f}, acro={ao}, fallback={fb}"

    # --- AIN_MATCH ---
    if ss >= 0.95:
        corroboration = ao or (tj >= 0.60)
        if corroboration:
            # Stricter threshold when the linked-author heuristic fell back
            if fb and ss < 0.97:
                pass  # insufficient confidence — leave for human
            else:
                rule = "AIN_MATCH_STR+" + ("ACRO" if ao else "JAC")
                return ("match", rule, metrics)

    # --- AIN_NONMATCH ---
    if ss <= 0.45 and tj <= 0.20 and not ao:
        return ("non-match", "AIN_NONMATCH_STR+JAC", metrics)

    return (None, "", metrics)


# ---------------------------------------------------------------------------
# E) Generic per-task processor
# ---------------------------------------------------------------------------

def _process_task(
    df: pd.DataFrame,
    signal_fn,
    rule_fn,
    overwrite: bool,
) -> tuple[pd.DataFrame, int, int, int]:
    """Apply auto-fill rules row-by-row to a task DataFrame.

    Adds or updates columns: auto_filled, auto_rule, auto_metrics.
    Never overwrites an existing gold_label unless overwrite=True.

    Returns:
        (updated_df, filled_match, filled_nonmatch, unchanged)
    """
    # Ensure audit columns are present (preserves any existing values on first pass)
    for col in ("auto_filled", "auto_rule", "auto_metrics"):
        if col not in df.columns:
            df[col] = ""

    filled_match    = 0
    filled_nonmatch = 0
    unchanged       = 0

    for idx, row in df.iterrows():
        existing = safe_str(row.get("gold_label", "")).strip()
        already_filled = existing != ""

        if already_filled and not overwrite:
            df.at[idx, "auto_filled"]  = "false"
            df.at[idx, "auto_rule"]    = "skipped:already_filled"
            df.at[idx, "auto_metrics"] = ""
            unchanged += 1
            continue

        sig   = signal_fn(row)
        label, rule, metrics = rule_fn(sig)

        df.at[idx, "auto_metrics"] = metrics

        if label is not None:
            df.at[idx, "gold_label"]  = label
            df.at[idx, "auto_filled"] = "true"
            df.at[idx, "auto_rule"]   = rule
            if label == "match":
                filled_match += 1
            else:
                filled_nonmatch += 1
        else:
            df.at[idx, "auto_filled"] = "false"
            df.at[idx, "auto_rule"]   = ""
            unchanged += 1

    return df, filled_match, filled_nonmatch, unchanged


# ---------------------------------------------------------------------------
# F) CLI entry point
# ---------------------------------------------------------------------------

DEFAULT_AND_IN = "data/derived/annotation_packets_and.csv"
DEFAULT_AIN_IN = "data/derived/annotation_packets_ain.csv"


def main(argv: list[str] | None = None) -> None:  # noqa: C901
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic auto-fill of clear gold labels in annotation packet CSVs. "
            "No LLM is used. Only high-confidence cases are filled."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--and_in", default=DEFAULT_AND_IN,
        help=f"AND annotation packets CSV (default: {DEFAULT_AND_IN}).",
    )
    parser.add_argument(
        "--ain_in", default=DEFAULT_AIN_IN,
        help=f"AIN annotation packets CSV (default: {DEFAULT_AIN_IN}).",
    )
    parser.add_argument(
        "--and_out", default=None,
        help="Output path for AND CSV (default: same as --and_in).",
    )
    parser.add_argument(
        "--ain_out", default=None,
        help="Output path for AIN CSV (default: same as --ain_in).",
    )
    parser.add_argument(
        "--overwrite", default="false",
        help="'true' to overwrite existing gold_label values (default: false).",
    )
    args = parser.parse_args(argv)

    and_out  = args.and_out or args.and_in
    ain_out  = args.ain_out or args.ain_in
    overwrite = args.overwrite.strip().lower() == "true"

    print(f"overwrite={overwrite}")
    print()

    # ---- AND ----
    and_path = Path(args.and_in)
    if not and_path.exists():
        print(f"[AND] File not found: {and_path} — skipping.")
    else:
        print(f"[AND] Loading {and_path} …")
        and_df = pd.read_csv(and_path, dtype=str).fillna("")
        print(f"[AND] {len(and_df):,} rows loaded.")

        and_df, m, nm, unc = _process_task(
            and_df, _and_signals, _apply_and_rules, overwrite
        )

        Path(and_out).parent.mkdir(parents=True, exist_ok=True)
        and_df.to_csv(and_out, index=False)
        print(
            f"[AND] auto-filled  match={m:,}  non-match={nm:,}  unchanged={unc:,}"
        )
        print(f"[AND] Written → {and_out}")
    print()

    # ---- AIN ----
    ain_path = Path(args.ain_in)
    if not ain_path.exists():
        print(f"[AIN] File not found: {ain_path} — skipping.")
    else:
        print(f"[AIN] Loading {ain_path} …")
        ain_df = pd.read_csv(ain_path, dtype=str).fillna("")
        print(f"[AIN] {len(ain_df):,} rows loaded.")

        ain_df, m, nm, unc = _process_task(
            ain_df, _ain_signals, _apply_ain_rules, overwrite
        )

        Path(ain_out).parent.mkdir(parents=True, exist_ok=True)
        ain_df.to_csv(ain_out, index=False)
        print(
            f"[AIN] auto-filled  match={m:,}  non-match={nm:,}  unchanged={unc:,}"
        )
        print(f"[AIN] Written → {ain_out}")
    print()

    print("Done.")
    print("  Review auto_filled=true rows before running pack_benchmark_pairs.")
    print("  Human judgement is still required for all unchanged rows.")


if __name__ == "__main__":
    main()
