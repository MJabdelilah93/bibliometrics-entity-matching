"""build_annotation_packets.py — Enrich annotation task CSVs with evidence fields for human review.

Reads the annotation task CSVs produced by sample_benchmark_tasks.py, joins each
pair to the instance and record tables, and writes enriched CSVs whose columns
contain all Scopus-derived evidence needed to assign a gold label in a spreadsheet.

NO labels are suggested or generated.  NO external data sources are used.
Author(s) ID is NEVER included (AND evidence boundary).
Title / source / year are NEVER included in the AIN output (topical-inference boundary).

Usage
-----
    python -m bem.benchmark.build_annotation_packets \\
        --in_dir  data/derived \\
        --out_dir data/derived \\
        --only_dev false

Options
-------
    --in_dir      Directory containing annotation_tasks_*.csv (default: data/derived)
    --out_dir     Directory for output annotation_packets_*.csv (default: data/derived)
    --only_dev    true ->keep only split="dev" rows; false ->keep all (default: false)
    --author_instances  Path to author_instances.parquet
    --affil_instances   Path to affil_instances.parquet
    --records_norm      Path to records_normalised.parquet

Outputs
-------
    <out_dir>/annotation_packets_and.csv   — AND evidence, one row per pair
    <out_dir>/annotation_packets_ain.csv   — AIN evidence, one row per pair

Evidence columns
----------------
AND:
    task, split, anchor_id, candidate_id, similarity_score,
    anchor_author_norm, candidate_author_norm,
    anchor_coauthors_norm, candidate_coauthors_norm,   (pipe-separated, first 30)
    anchor_affiliations_norm, candidate_affiliations_norm,  (≤300 chars)
    anchor_title, candidate_title,                     (≤300 chars)
    anchor_source_title, candidate_source_title,
    anchor_year, candidate_year,
    gold_label, notes

AIN:
    task, split, anchor_id, candidate_id, similarity_score,
    anchor_affil_raw, candidate_affil_raw,              (≤300 chars)
    anchor_affil_norm, candidate_affil_norm,            (≤300 chars)
    anchor_affil_acronyms, candidate_affil_acronyms,   (pipe-separated)
    anchor_linked_authors_norm, candidate_linked_authors_norm,  (pipe-separated, first 30)
    anchor_linked_authors_fallback, candidate_linked_authors_fallback,
    gold_label, notes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from bem.llm_verify.evidence_cards import build_and_evidence, build_ain_evidence
from bem.normalise.normalise import strip_scopus_author_id


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_AUTHOR_INSTANCES = "data/interim/author_instances.parquet"
DEFAULT_AFFIL_INSTANCES  = "data/interim/affil_instances.parquet"
DEFAULT_RECORDS_NORM     = "data/interim/records_normalised.parquet"

_AFFIL_MAX   = 300   # max chars for affiliation / title fields
_TITLE_MAX   = 300   # max chars for title fields
_COAUTH_MAX  = 30    # max co-author / linked-author names before truncation

AND_TASK_CSV = "annotation_tasks_and.csv"
AIN_TASK_CSV = "annotation_tasks_ain.csv"
AND_OUT_CSV  = "annotation_packets_and.csv"
AIN_OUT_CSV  = "annotation_packets_ain.csv"


def _task_csv_names(prefix: str) -> tuple[str, str, str, str]:
    """Return (and_task, ain_task, and_out, ain_out) filenames for a given prefix.

    An empty prefix keeps the default names.  A non-empty prefix is prepended
    with an underscore separator, e.g. prefix='dev2' →
        dev2_annotation_tasks_and.csv, dev2_annotation_packets_and.csv.
    """
    p = f"{prefix.rstrip('_')}_" if prefix else ""
    return (
        f"{p}annotation_tasks_and.csv",
        f"{p}annotation_tasks_ain.csv",
        f"{p}annotation_packets_and.csv",
        f"{p}annotation_packets_ain.csv",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trunc(s: str, n: int) -> str:
    """Truncate string to n characters, appending '…' if cut."""
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _pipe_join(names: list[str], max_n: int) -> str:
    """Join up to max_n strings with ' | '; append '…' suffix if truncated."""
    if not names:
        return ""
    head = names[:max_n]
    suffix = " | …" if len(names) > max_n else ""
    return " | ".join(head) + suffix


def _load_task_csv(path: Path, task: str, only_dev: bool) -> pd.DataFrame | None:
    """Load and optionally filter an annotation task CSV."""
    if not path.exists():
        print(f"  [{task}] Task CSV not found: {path} — skipping.")
        return None
    df = pd.read_csv(path, dtype=str).fillna("")
    if only_dev:
        df = df[df["split"].str.lower() == "dev"].copy()
        print(f"  [{task}] Filtered to dev split: {len(df):,} rows")
    else:
        print(f"  [{task}] Loaded {len(df):,} rows")
    if df.empty:
        print(f"  [{task}] No rows to process.")
        return None
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Flatteners
# ---------------------------------------------------------------------------

def _flatten_and(card: dict) -> dict:
    """Flatten a build_and_evidence result to a flat dict of CSV columns."""
    anch = card["anchor"]
    cand = card["candidate"]
    return {
        "anchor_author_norm":        strip_scopus_author_id(anch["author_norm"]),
        "candidate_author_norm":     strip_scopus_author_id(cand["author_norm"]),
        "anchor_coauthors_norm":     _pipe_join(
            [strip_scopus_author_id(n) for n in anch["coauthors_norm"]], _COAUTH_MAX
        ),
        "candidate_coauthors_norm":  _pipe_join(
            [strip_scopus_author_id(n) for n in cand["coauthors_norm"]], _COAUTH_MAX
        ),
        "anchor_affiliations_norm":  _trunc(anch["affiliations_norm"], _AFFIL_MAX),
        "candidate_affiliations_norm": _trunc(cand["affiliations_norm"], _AFFIL_MAX),
        "anchor_title":              _trunc(anch["record"].get("title", ""), _TITLE_MAX),
        "candidate_title":           _trunc(cand["record"].get("title", ""), _TITLE_MAX),
        "anchor_source_title":       anch["record"].get("source_title", ""),
        "candidate_source_title":    cand["record"].get("source_title", ""),
        "anchor_year":               anch["record"].get("year", ""),
        "candidate_year":            cand["record"].get("year", ""),
    }


def _flatten_ain(card: dict) -> dict:
    """Flatten a build_ain_evidence result to a flat dict of CSV columns."""
    anch = card["anchor"]
    cand = card["candidate"]
    return {
        "anchor_affil_raw":                    _trunc(anch["affil_raw"],  _AFFIL_MAX),
        "candidate_affil_raw":                 _trunc(cand["affil_raw"],  _AFFIL_MAX),
        "anchor_affil_norm":                   _trunc(anch["affil_norm"], _AFFIL_MAX),
        "candidate_affil_norm":                _trunc(cand["affil_norm"], _AFFIL_MAX),
        "anchor_affil_acronyms":               _pipe_join(anch["affil_acronyms"], 99),
        "candidate_affil_acronyms":            _pipe_join(cand["affil_acronyms"], 99),
        "anchor_linked_authors_norm":          _pipe_join(anch["linked_authors_norm"], _COAUTH_MAX),
        "candidate_linked_authors_norm":       _pipe_join(cand["linked_authors_norm"], _COAUTH_MAX),
        "anchor_linked_authors_fallback":      str(anch["linked_authors_fallback"]).lower(),
        "candidate_linked_authors_fallback":   str(cand["linked_authors_fallback"]).lower(),
    }


# ---------------------------------------------------------------------------
# Column order definitions
# ---------------------------------------------------------------------------

AND_COLS = [
    "task", "split", "anchor_id", "candidate_id", "similarity_score",
    "anchor_author_norm", "candidate_author_norm",
    "anchor_coauthors_norm", "candidate_coauthors_norm",
    "anchor_affiliations_norm", "candidate_affiliations_norm",
    "anchor_title", "candidate_title",
    "anchor_source_title", "candidate_source_title",
    "anchor_year", "candidate_year",
    "gold_label", "notes",
]

AIN_COLS = [
    "task", "split", "anchor_id", "candidate_id", "similarity_score",
    "anchor_affil_raw", "candidate_affil_raw",
    "anchor_affil_norm", "candidate_affil_norm",
    "anchor_affil_acronyms", "candidate_affil_acronyms",
    "anchor_linked_authors_norm", "candidate_linked_authors_norm",
    "anchor_linked_authors_fallback", "candidate_linked_authors_fallback",
    "gold_label", "notes",
]


# ---------------------------------------------------------------------------
# Per-task processing
# ---------------------------------------------------------------------------

def _process_and(
    task_df: pd.DataFrame,
    auth_inst_df: pd.DataFrame,
    records_norm_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build AND annotation packet rows from task_df."""
    rows: list[dict] = []
    n_errors = 0

    for _, task_row in task_df.iterrows():
        anchor_id    = str(task_row["anchor_id"])
        candidate_id = str(task_row["candidate_id"])

        try:
            card = build_and_evidence(anchor_id, candidate_id, auth_inst_df, records_norm_df)
        except (KeyError, Exception) as exc:
            n_errors += 1
            # Keep a minimal row so the annotator can still identify the pair
            rows.append({
                "task":           task_row.get("task", "AND"),
                "split":          task_row.get("split", ""),
                "anchor_id":      anchor_id,
                "candidate_id":   candidate_id,
                "similarity_score": task_row.get("similarity_score", ""),
                "anchor_author_norm": f"[ERROR: {exc}]",
                "candidate_author_norm": "",
                "anchor_coauthors_norm": "",
                "candidate_coauthors_norm": "",
                "anchor_affiliations_norm": "",
                "candidate_affiliations_norm": "",
                "anchor_title": "",
                "candidate_title": "",
                "anchor_source_title": "",
                "candidate_source_title": "",
                "anchor_year": "",
                "candidate_year": "",
                "gold_label": task_row.get("gold_label", ""),
                "notes": task_row.get("notes", ""),
            })
            continue

        flat = _flatten_and(card)
        rows.append({
            "task":             task_row.get("task", "AND"),
            "split":            task_row.get("split", ""),
            "anchor_id":        anchor_id,
            "candidate_id":     candidate_id,
            "similarity_score": task_row.get("similarity_score", ""),
            **flat,
            "gold_label": task_row.get("gold_label", ""),
            "notes":      task_row.get("notes", ""),
        })

    if n_errors:
        print(f"  [AND] {n_errors} evidence-build errors (pair kept with error marker).")

    return pd.DataFrame(rows, columns=AND_COLS)


def _process_ain(
    task_df: pd.DataFrame,
    affil_inst_df: pd.DataFrame,
    records_norm_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build AIN annotation packet rows from task_df."""
    rows: list[dict] = []
    n_errors = 0

    for _, task_row in task_df.iterrows():
        anchor_id    = str(task_row["anchor_id"])
        candidate_id = str(task_row["candidate_id"])

        try:
            card = build_ain_evidence(anchor_id, candidate_id, affil_inst_df, records_norm_df)
        except (KeyError, Exception) as exc:
            n_errors += 1
            rows.append({
                "task":           task_row.get("task", "AIN"),
                "split":          task_row.get("split", ""),
                "anchor_id":      anchor_id,
                "candidate_id":   candidate_id,
                "similarity_score": task_row.get("similarity_score", ""),
                "anchor_affil_raw": f"[ERROR: {exc}]",
                "candidate_affil_raw": "",
                "anchor_affil_norm": "",
                "candidate_affil_norm": "",
                "anchor_affil_acronyms": "",
                "candidate_affil_acronyms": "",
                "anchor_linked_authors_norm": "",
                "candidate_linked_authors_norm": "",
                "anchor_linked_authors_fallback": "",
                "candidate_linked_authors_fallback": "",
                "gold_label": task_row.get("gold_label", ""),
                "notes": task_row.get("notes", ""),
            })
            continue

        flat = _flatten_ain(card)
        rows.append({
            "task":             task_row.get("task", "AIN"),
            "split":            task_row.get("split", ""),
            "anchor_id":        anchor_id,
            "candidate_id":     candidate_id,
            "similarity_score": task_row.get("similarity_score", ""),
            **flat,
            "gold_label": task_row.get("gold_label", ""),
            "notes":      task_row.get("notes", ""),
        })

    if n_errors:
        print(f"  [AIN] {n_errors} evidence-build errors (pair kept with error marker).")

    return pd.DataFrame(rows, columns=AIN_COLS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build enriched annotation packets for human gold-label assignment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--in_dir", default="data/derived",
        help="Directory containing annotation_tasks_*.csv (default: data/derived).",
    )
    parser.add_argument(
        "--out_dir", default="data/derived",
        help="Directory for output annotation_packets_*.csv (default: data/derived).",
    )
    parser.add_argument(
        "--only_dev", default="false",
        help="'true' to process only split=dev rows; 'false' for all (default: false).",
    )
    parser.add_argument(
        "--prefix", default="",
        help="Filename prefix for task CSVs and output packets.  "
             "E.g. 'dev2' reads dev2_annotation_tasks_*.csv and writes "
             "dev2_annotation_packets_*.csv (default: empty ->standard names).",
    )
    parser.add_argument(
        "--author_instances", default=DEFAULT_AUTHOR_INSTANCES,
        help=f"Path to author_instances.parquet (default: {DEFAULT_AUTHOR_INSTANCES}).",
    )
    parser.add_argument(
        "--affil_instances", default=DEFAULT_AFFIL_INSTANCES,
        help=f"Path to affil_instances.parquet (default: {DEFAULT_AFFIL_INSTANCES}).",
    )
    parser.add_argument(
        "--records_norm", default=DEFAULT_RECORDS_NORM,
        help=f"Path to records_normalised.parquet (default: {DEFAULT_RECORDS_NORM}).",
    )
    args = parser.parse_args(argv)

    in_dir   = Path(args.in_dir)
    out_dir  = Path(args.out_dir)
    only_dev = args.only_dev.strip().lower() == "true"

    and_task_csv, ain_task_csv, and_out_csv, ain_out_csv = _task_csv_names(args.prefix)

    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load supporting tables (shared across both tasks) --
    for label, path in [
        ("author_instances", args.author_instances),
        ("affil_instances",  args.affil_instances),
        ("records_norm",     args.records_norm),
    ]:
        if not Path(path).exists():
            print(f"ERROR: {label} not found: {path}")
            print("Run the pipeline first:  python -m bem --config configs/run_config.yaml")
            sys.exit(1)

    print("Loading supporting parquets …")
    auth_inst_df   = pd.read_parquet(args.author_instances)
    affil_inst_df  = pd.read_parquet(args.affil_instances)
    records_norm_df = pd.read_parquet(args.records_norm)
    print(f"  author_instances  : {len(auth_inst_df):,} rows")
    print(f"  affil_instances   : {len(affil_inst_df):,} rows")
    print(f"  records_normalised: {len(records_norm_df):,} rows")
    print()

    only_dev_label = "dev only" if only_dev else "all splits"

    # ---- AND ----
    and_task_df = _load_task_csv(in_dir / and_task_csv, "AND", only_dev)
    if and_task_df is not None:
        print(f"  [AND] Building evidence packets ({only_dev_label}) …")
        and_out_df = _process_and(and_task_df, auth_inst_df, records_norm_df)
        and_out_path = out_dir / and_out_csv
        and_out_df.to_csv(and_out_path, index=False)
        print(f"  [AND] Written {len(and_out_df):,} rows ->{and_out_path}")
    print()

    # ---- AIN ----
    ain_task_df = _load_task_csv(in_dir / ain_task_csv, "AIN", only_dev)
    if ain_task_df is not None:
        print(f"  [AIN] Building evidence packets ({only_dev_label}) …")
        ain_out_df = _process_ain(ain_task_df, affil_inst_df, records_norm_df)
        ain_out_path = out_dir / ain_out_csv
        ain_out_df.to_csv(ain_out_path, index=False)
        print(f"  [AIN] Written {len(ain_out_df):,} rows ->{ain_out_path}")
    print()

    print("Next step:")
    print("  Open annotation_packets_and.csv / annotation_packets_ain.csv")
    print("  Note: Author IDs have been removed from name strings.")
    print("  Fill the 'gold_label' column for each row:")
    print("    match / non-match / uncertain   (no LLM assistance)")
    print("  Save in-place, then run:")
    print("    python -m bem.benchmark.pack_benchmark_pairs --in_dir data/derived")


if __name__ == "__main__":
    main()
