"""auto_label_packets.py — Deterministic auto-labeller for annotation-packet CSVs.

Produces *_code.csv outputs containing a conservative machine-assigned
``gold_label_auto`` column.  The original annotation-packet files are NEVER
modified.  Auto labels are NOT gold-standard; human labels remain authoritative.

Usage
-----
    python -m vs2.benchmark.auto_label_packets \\
        --in_dir  data/derived \\
        --out_dir data/derived \\
        --config  configs/run_config.yaml

Options
-------
    --in_dir   Directory containing annotation_packets_and.csv / _ain.csv
               (default: data/derived)
    --out_dir  Directory for output *_code.csv files + manifest
               (default: data/derived)
    --config   Path to run_config.yaml (default: configs/run_config.yaml)

Outputs
-------
    <out_dir>/annotation_packets_and_code.csv
    <out_dir>/annotation_packets_ain_code.csv
    <out_dir>/auto_label_manifest.json

Label semantics
---------------
    match      — very strong evidence that both sides are the same real-world entity
    non-match  — strong evidence that they are different entities
    uncertain  — default; insufficient evidence for either extreme

No LLM is involved.  All rules are deterministic functions of the CSV fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Required column schemas
# ---------------------------------------------------------------------------

AND_REQUIRED_COLS: list[str] = [
    "task", "split", "anchor_id", "candidate_id", "similarity_score",
    "anchor_author_norm", "candidate_author_norm",
    "anchor_coauthors_norm", "candidate_coauthors_norm",
    "anchor_affiliations_norm", "candidate_affiliations_norm",
    "gold_label", "notes",
]

AIN_REQUIRED_COLS: list[str] = [
    "task", "split", "anchor_id", "candidate_id", "similarity_score",
    "anchor_affil_raw", "candidate_affil_raw",
    "anchor_affil_norm", "candidate_affil_norm",
    "anchor_affil_acronyms", "candidate_affil_acronyms",
    "anchor_linked_authors_norm", "candidate_linked_authors_norm",
    "gold_label", "notes",
]

# ---------------------------------------------------------------------------
# A) Text helpers (deterministic, no external data)
# ---------------------------------------------------------------------------

_NUM_PAREN_RE = re.compile(r"\s*\(\d{6,20}\)")


def norm_basic(s: object) -> str:
    """NFKC normalise, strip, collapse whitespace, casefold.

    Returns '' for None / NaN / non-string inputs.
    """
    if s is None:
        return ""
    try:
        import math as _math
        if isinstance(s, float) and _math.isnan(s):
            return ""
    except Exception:
        pass
    text = str(s)
    if text.strip().lower() in ("nan", "none", ""):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text


def strip_numeric_parens(s: str) -> str:
    """Remove ' (digits)' substrings where parentheses contain ONLY digits (length 6–20).

    Parenthesised tokens containing any non-digit character are preserved,
    so institutional acronyms like '(LMCE)' or '(UM6P)' are untouched.

    Examples:
        "ez-zahraouy, hamid (7004513174)" → "ez-zahraouy, hamid"
        "lazfi, souad (55998718300)"       → "lazfi, souad"
        "university (LMCE)"                → "university (LMCE)"
    """
    return _NUM_PAREN_RE.sub("", s).strip()


def split_pipe_list(s: object) -> set[str]:
    """Split on ' | ', strip, drop empty items and the '…' truncation marker.

    Applies norm_basic + strip_numeric_parens to every element so that
    names are comparable across anchor and candidate sides.
    """
    raw = norm_basic(s)
    if not raw:
        return set()
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    cleaned = set()
    for p in parts:
        p = p.strip()
        if not p or p == "…":
            continue
        cleaned.add(strip_numeric_parens(norm_basic(p)))
    return cleaned


def tokenise(s: object) -> set[str]:
    """Casefold, extract alphanumeric tokens of length >= 2."""
    text = norm_basic(s)
    return {t for t in re.split(r"\W+", text) if len(t) >= 2}


def parse_acronyms_pipe(s: object) -> set[str]:
    """Split on '|' or ' | ', strip, drop empties; return set of uppercase tokens."""
    raw = norm_basic(s)
    if not raw:
        return set()
    parts = re.split(r"\s*\|\s*", raw)
    result = set()
    for p in parts:
        p = p.strip()
        if p and p != "…":
            result.add(p.upper())
    return result


# ---------------------------------------------------------------------------
# B) Schema validation
# ---------------------------------------------------------------------------

def _validate_schema(df: pd.DataFrame, required: list[str], label: str) -> None:
    """Raise ValueError with an actionable message if required columns are missing."""
    observed = list(df.columns)
    missing  = [c for c in required if c not in df.columns]
    extra    = [c for c in observed if c not in required]
    if missing:
        raise ValueError(
            f"[{label}] Schema validation failed.\n"
            f"  Missing columns : {missing}\n"
            f"  Extra columns   : {extra}\n"
            f"  Observed header : {observed}"
        )


# ---------------------------------------------------------------------------
# C) Signal computation
# ---------------------------------------------------------------------------

def _and_signals(row: pd.Series) -> dict[str, Any]:
    """Compute AND comparison signals from a single annotation-packet row."""
    anch_name = strip_numeric_parens(norm_basic(row.get("anchor_author_norm",    "")))
    cand_name = strip_numeric_parens(norm_basic(row.get("candidate_author_norm", "")))

    if not anch_name and not cand_name:
        # Both empty: signals uninformative
        return {"name_sim": 0.0, "coauthor_overlap_count": 0, "affil_sim": 0.0}

    name_sim = fuzz.ratio(anch_name, cand_name) / 100.0

    anch_coauth = split_pipe_list(row.get("anchor_coauthors_norm",    ""))
    cand_coauth = split_pipe_list(row.get("candidate_coauthors_norm", ""))
    coauthor_overlap_count = len(anch_coauth & cand_coauth) if (anch_coauth and cand_coauth) else 0

    anch_affil = norm_basic(row.get("anchor_affiliations_norm",    ""))
    cand_affil = norm_basic(row.get("candidate_affiliations_norm", ""))
    affil_sim = (
        fuzz.token_set_ratio(anch_affil, cand_affil) / 100.0
        if (anch_affil or cand_affil) else 0.0
    )

    return {
        "name_sim":               name_sim,
        "coauthor_overlap_count": coauthor_overlap_count,
        "affil_sim":              affil_sim,
    }


def _ain_signals(row: pd.Series) -> dict[str, Any]:
    """Compute AIN comparison signals from a single annotation-packet row."""
    anch_norm = norm_basic(row.get("anchor_affil_norm",    ""))
    cand_norm = norm_basic(row.get("candidate_affil_norm", ""))

    if not anch_norm and not cand_norm:
        return {"affil_str_sim": 0.0, "token_jaccard": 0.0, "acronym_overlap": False}

    affil_str_sim = fuzz.token_set_ratio(anch_norm, cand_norm) / 100.0

    anch_tok = tokenise(anch_norm)
    cand_tok = tokenise(cand_norm)
    if anch_tok or cand_tok:
        token_jaccard = (
            len(anch_tok & cand_tok) / len(anch_tok | cand_tok)
            if (anch_tok and cand_tok) else 0.0
        )
    else:
        token_jaccard = 0.0

    anch_acros = parse_acronyms_pipe(row.get("anchor_affil_acronyms",    ""))
    cand_acros = parse_acronyms_pipe(row.get("candidate_affil_acronyms", ""))
    acronym_overlap = bool(anch_acros & cand_acros) if (anch_acros and cand_acros) else False

    return {
        "affil_str_sim":  affil_str_sim,
        "token_jaccard":  token_jaccard,
        "acronym_overlap": acronym_overlap,
    }


# ---------------------------------------------------------------------------
# D) Label rules
# ---------------------------------------------------------------------------

def _and_label(sig: dict[str, Any], thresholds: dict[str, float]) -> str:
    """Apply conservative AND labelling rules.

    Returns 'match', 'non-match', or 'uncertain'.
    """
    ns  = sig["name_sim"]
    co  = sig["coauthor_overlap_count"]
    af  = sig["affil_sim"]
    t_m  = thresholds["name_sim_match"]
    t_am = thresholds["affil_sim_match"]
    t_nm = thresholds["name_sim_nonmatch"]

    if ns >= t_m and (co >= 1 or af >= t_am):
        return "match"
    if ns <= t_nm:
        return "non-match"
    return "uncertain"


def _ain_label(sig: dict[str, Any], thresholds: dict[str, float]) -> str:
    """Apply conservative AIN labelling rules.

    Returns 'match', 'non-match', or 'uncertain'.
    """
    ss  = sig["affil_str_sim"]
    tj  = sig["token_jaccard"]
    ao  = sig["acronym_overlap"]
    t_m  = thresholds["affil_str_sim_match"]
    t_tj = thresholds["token_jaccard_match"]
    t_nm = thresholds["affil_str_sim_nonmatch"]

    if ss >= t_m and (tj >= t_tj or ao):
        return "match"
    if ss <= t_nm:
        return "non-match"
    return "uncertain"


# ---------------------------------------------------------------------------
# E) Per-task processor
# ---------------------------------------------------------------------------

def _process_and(
    df: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Add auto-label and signal columns to an AND packet DataFrame."""
    out = df.copy()
    labels, name_sims, coauth_ovs, affil_sims = [], [], [], []

    for _, row in df.iterrows():
        sig   = _and_signals(row)
        label = _and_label(sig, thresholds)
        labels.append(label)
        name_sims.append(round(sig["name_sim"], 4))
        coauth_ovs.append(sig["coauthor_overlap_count"])
        affil_sims.append(round(sig["affil_sim"], 4))

    out["gold_label_auto"]        = labels
    out["gold_label_source"]      = "auto"
    out["gold_label"]             = labels        # fill the gold_label column in _code output
    out["name_sim"]               = name_sims
    out["coauthor_overlap_count"] = coauth_ovs
    out["affil_sim"]              = affil_sims
    return out


def _process_ain(
    df: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Add auto-label and signal columns to an AIN packet DataFrame."""
    out = df.copy()
    labels, str_sims, jaccards, acro_ovs = [], [], [], []

    for _, row in df.iterrows():
        sig   = _ain_signals(row)
        label = _ain_label(sig, thresholds)
        labels.append(label)
        str_sims.append(round(sig["affil_str_sim"], 4))
        jaccards.append(round(sig["token_jaccard"], 4))
        acro_ovs.append(sig["acronym_overlap"])

    out["gold_label_auto"]   = labels
    out["gold_label_source"] = "auto"
    out["gold_label"]        = labels
    out["affil_str_sim"]     = str_sims
    out["token_jaccard"]     = jaccards
    out["acronym_overlap"]   = acro_ovs
    return out


# ---------------------------------------------------------------------------
# F) Manifest helper
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest of a file, or 'MISSING' if not found."""
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _label_counts(series: "pd.Series[str]") -> dict[str, Any]:
    n = len(series)
    counts = series.value_counts().to_dict()
    pct = {
        k: round(v / n * 100, 2) if n else 0.0
        for k, v in counts.items()
    }
    return {"counts": counts, "pct": pct, "total": n}


# ---------------------------------------------------------------------------
# G) CLI entry point
# ---------------------------------------------------------------------------

DEFAULT_IN_DIR  = "data/derived"
DEFAULT_OUT_DIR = "data/derived"
DEFAULT_CONFIG  = "configs/run_config.yaml"

AND_IN_CSV  = "annotation_packets_and.csv"
AIN_IN_CSV  = "annotation_packets_ain.csv"
MANIFEST_OUT = "auto_label_manifest.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic auto-labeller for annotation-packet CSVs. "
            "Writes *_code.csv outputs; never modifies original files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--in_dir", default=DEFAULT_IN_DIR,
        help=f"Directory containing annotation_packets_*.csv (default: {DEFAULT_IN_DIR}).",
    )
    parser.add_argument(
        "--out_dir", default=DEFAULT_OUT_DIR,
        help=f"Directory for output *_code.csv and manifest (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help=f"Path to run_config.yaml (default: {DEFAULT_CONFIG}).",
    )
    args = parser.parse_args(argv)

    in_dir   = Path(args.in_dir)
    out_dir  = Path(args.out_dir)
    cfg_path = Path(args.config)

    # -- Load config --
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("PyYAML is required: pip install pyyaml") from exc

    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    al_cfg   = cfg.get("auto_labeling", {})
    suffix   = al_cfg.get("output_suffix", "_code")
    and_thr  = al_cfg.get("and", {})
    ain_thr  = al_cfg.get("ain", {})

    # Validate threshold keys present
    for key in ("name_sim_match", "affil_sim_match", "name_sim_nonmatch"):
        if key not in and_thr:
            raise KeyError(f"Missing auto_labeling.and.{key} in {cfg_path}")
    for key in ("affil_str_sim_match", "token_jaccard_match", "affil_str_sim_nonmatch"):
        if key not in ain_thr:
            raise KeyError(f"Missing auto_labeling.ain.{key} in {cfg_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "config_path":   str(cfg_path),
        "thresholds": {
            "and": dict(and_thr),
            "ain": dict(ain_thr),
        },
        "input_files":  {},
        "output_files": {},
        "label_counts": {},
    }

    # ---- AND ----
    and_in_path  = in_dir  / AND_IN_CSV
    and_out_path = out_dir / (Path(AND_IN_CSV).stem + suffix + ".csv")

    if not and_in_path.exists():
        print(f"[AND] Input not found: {and_in_path} — skipping.")
    else:
        print(f"[AND] Loading {and_in_path} …")
        and_df = pd.read_csv(and_in_path, dtype=str).fillna("")
        print(f"[AND] {len(and_df):,} rows loaded.")

        _validate_schema(and_df, AND_REQUIRED_COLS, "AND")

        and_out = _process_and(and_df, and_thr)
        and_out.to_csv(and_out_path, index=False)

        lc = _label_counts(and_out["gold_label_auto"])
        manifest["input_files"]["and"] = {
            "path":       str(and_in_path),
            "size_bytes": and_in_path.stat().st_size,
            "sha256":     _sha256_file(and_in_path),
        }
        manifest["output_files"]["and"] = {
            "path":       str(and_out_path),
            "size_bytes": and_out_path.stat().st_size,
            "sha256":     _sha256_file(and_out_path),
        }
        manifest["label_counts"]["and"] = lc

        print(
            f"[AND] match={lc['counts'].get('match', 0):,}  "
            f"non-match={lc['counts'].get('non-match', 0):,}  "
            f"uncertain={lc['counts'].get('uncertain', 0):,}"
        )
        print(f"[AND] Written → {and_out_path}")
    print()

    # ---- AIN ----
    ain_in_path  = in_dir  / AIN_IN_CSV
    ain_out_path = out_dir / (Path(AIN_IN_CSV).stem + suffix + ".csv")

    if not ain_in_path.exists():
        print(f"[AIN] Input not found: {ain_in_path} — skipping.")
    else:
        print(f"[AIN] Loading {ain_in_path} …")
        ain_df = pd.read_csv(ain_in_path, dtype=str).fillna("")
        print(f"[AIN] {len(ain_df):,} rows loaded.")

        _validate_schema(ain_df, AIN_REQUIRED_COLS, "AIN")

        ain_out = _process_ain(ain_df, ain_thr)
        ain_out.to_csv(ain_out_path, index=False)

        lc = _label_counts(ain_out["gold_label_auto"])
        manifest["input_files"]["ain"] = {
            "path":       str(ain_in_path),
            "size_bytes": ain_in_path.stat().st_size,
            "sha256":     _sha256_file(ain_in_path),
        }
        manifest["output_files"]["ain"] = {
            "path":       str(ain_out_path),
            "size_bytes": ain_out_path.stat().st_size,
            "sha256":     _sha256_file(ain_out_path),
        }
        manifest["label_counts"]["ain"] = lc

        print(
            f"[AIN] match={lc['counts'].get('match', 0):,}  "
            f"non-match={lc['counts'].get('non-match', 0):,}  "
            f"uncertain={lc['counts'].get('uncertain', 0):,}"
        )
        print(f"[AIN] Written → {ain_out_path}")
    print()

    # ---- Manifest ----
    manifest_path = out_dir / MANIFEST_OUT
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"Manifest → {manifest_path}")
    print()
    print("Note: gold_label_auto values are NOT gold-standard.")
    print("      Human labels remain authoritative for all downstream evaluation.")


if __name__ == "__main__":
    main()
