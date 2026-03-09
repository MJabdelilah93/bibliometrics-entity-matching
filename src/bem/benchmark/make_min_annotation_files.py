"""make_min_annotation_files.py -- Generate minimal evidence CSVs for manual annotation.

Reads dev2 annotation packet CSVs and writes two stripped-down, Excel-friendly
files containing only the evidence columns needed to decide match / non-match /
uncertain.  A numeric gold_label_code column (initially empty) is appended so
the annotator can fill it directly in Excel:

    0 = uncertain
    1 = non-match
    2 = match

Usage
-----
    python -m bem.benchmark.make_min_annotation_files --in_dir data/derived --prefix dev2

Outputs
-------
    <in_dir>/<prefix>_min_annotate_and.csv
    <in_dir>/<prefix>_min_annotate_ain.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Column specs
# ---------------------------------------------------------------------------

AND_REQUIRED = [
    "anchor_id", "candidate_id", "similarity_score",
    "anchor_author_norm", "candidate_author_norm",
    "anchor_coauthors_norm", "candidate_coauthors_norm",
    "anchor_affiliations_norm", "candidate_affiliations_norm",
    "anchor_title", "candidate_title",
    "anchor_source_title", "candidate_source_title",
    "anchor_year", "candidate_year",
]

AIN_REQUIRED = [
    "anchor_id", "candidate_id", "similarity_score",
    "anchor_affil_raw", "candidate_affil_raw",
    "anchor_affil_norm", "candidate_affil_norm",
    "anchor_affil_acronyms", "candidate_affil_acronyms",
    "anchor_linked_authors_norm", "candidate_linked_authors_norm",
    "anchor_linked_authors_fallback", "candidate_linked_authors_fallback",
]

# Truncation limits
COAUTHOR_MAX   = 30    # max pipe-separated names
AFFIL_MAX      = 300   # chars
TITLE_MAX      = 200   # chars
AUTHOR_LIST_MAX = 30   # max pipe-separated names for linked_authors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate_pipe_list(value: str, max_items: int) -> str:
    """Keep at most max_items pipe-separated tokens."""
    if not value:
        return value
    parts = [p.strip() for p in value.split("|")]
    if len(parts) <= max_items:
        return value
    return " | ".join(parts[:max_items]) + f" | ... (+{len(parts) - max_items} more)"


def _truncate_str(value: str, max_chars: int) -> str:
    if not value or len(value) <= max_chars:
        return value
    return value[:max_chars] + "..."


def _build_and_min(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["anchor_id"]    = df["anchor_id"]
    out["candidate_id"] = df["candidate_id"]
    out["similarity_score"] = df["similarity_score"]

    out["anchor_author_norm"]    = df["anchor_author_norm"]
    out["candidate_author_norm"] = df["candidate_author_norm"]

    out["anchor_coauthors_norm"]    = df["anchor_coauthors_norm"].apply(
        lambda v: _truncate_pipe_list(str(v) if pd.notna(v) else "", COAUTHOR_MAX)
    )
    out["candidate_coauthors_norm"] = df["candidate_coauthors_norm"].apply(
        lambda v: _truncate_pipe_list(str(v) if pd.notna(v) else "", COAUTHOR_MAX)
    )

    out["anchor_affiliations_norm"]    = df["anchor_affiliations_norm"].apply(
        lambda v: _truncate_str(str(v) if pd.notna(v) else "", AFFIL_MAX)
    )
    out["candidate_affiliations_norm"] = df["candidate_affiliations_norm"].apply(
        lambda v: _truncate_str(str(v) if pd.notna(v) else "", AFFIL_MAX)
    )

    out["anchor_title"]    = df["anchor_title"].apply(
        lambda v: _truncate_str(str(v) if pd.notna(v) else "", TITLE_MAX)
    )
    out["candidate_title"] = df["candidate_title"].apply(
        lambda v: _truncate_str(str(v) if pd.notna(v) else "", TITLE_MAX)
    )

    out["anchor_source_title"]    = df["anchor_source_title"]
    out["candidate_source_title"] = df["candidate_source_title"]
    out["anchor_year"]    = df["anchor_year"]
    out["candidate_year"] = df["candidate_year"]

    out["gold_label_code"] = ""
    out["notes"] = ""
    return out


def _build_ain_min(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["anchor_id"]    = df["anchor_id"]
    out["candidate_id"] = df["candidate_id"]
    out["similarity_score"] = df["similarity_score"]

    out["anchor_affil_raw"]    = df["anchor_affil_raw"].apply(
        lambda v: _truncate_str(str(v) if pd.notna(v) else "", AFFIL_MAX)
    )
    out["candidate_affil_raw"] = df["candidate_affil_raw"].apply(
        lambda v: _truncate_str(str(v) if pd.notna(v) else "", AFFIL_MAX)
    )

    out["anchor_affil_norm"]    = df["anchor_affil_norm"].apply(
        lambda v: _truncate_str(str(v) if pd.notna(v) else "", AFFIL_MAX)
    )
    out["candidate_affil_norm"] = df["candidate_affil_norm"].apply(
        lambda v: _truncate_str(str(v) if pd.notna(v) else "", AFFIL_MAX)
    )

    out["anchor_affil_acronyms"]    = df["anchor_affil_acronyms"]
    out["candidate_affil_acronyms"] = df["candidate_affil_acronyms"]

    out["anchor_linked_authors_norm"]    = df["anchor_linked_authors_norm"].apply(
        lambda v: _truncate_pipe_list(str(v) if pd.notna(v) else "", AUTHOR_LIST_MAX)
    )
    out["candidate_linked_authors_norm"] = df["candidate_linked_authors_norm"].apply(
        lambda v: _truncate_pipe_list(str(v) if pd.notna(v) else "", AUTHOR_LIST_MAX)
    )

    # Combine the two fallback flags into one readable column
    def _fmt_fallback(row: pd.Series) -> str:
        a = str(row.get("anchor_linked_authors_fallback", "")).strip().lower()
        c = str(row.get("candidate_linked_authors_fallback", "")).strip().lower()
        a_bool = a in ("true", "1", "yes")
        c_bool = c in ("true", "1", "yes")
        if not a_bool and not c_bool:
            return "false"
        parts = []
        if a_bool:
            parts.append("anchor")
        if c_bool:
            parts.append("candidate")
        return "true (" + "+".join(parts) + ")"

    out["linked_authors_fallback"] = df.apply(_fmt_fallback, axis=1)

    out["gold_label_code"] = ""
    out["notes"] = ""
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate minimal evidence CSVs for manual annotation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--in_dir",  default="data/derived",
                        help="Directory containing annotation packet CSVs (default: data/derived).")
    parser.add_argument("--prefix",  default="dev2",
                        help="Filename prefix, e.g. 'dev2' (default: dev2).")
    args = parser.parse_args(argv)

    in_dir = Path(args.in_dir)
    p = f"{args.prefix.rstrip('_')}_" if args.prefix else ""

    specs = {
        "AND": {
            "in_file":  in_dir / f"{p}annotation_packets_and.csv",
            "out_file": in_dir / f"{p}min_annotate_and.csv",
            "required": AND_REQUIRED,
            "builder":  _build_and_min,
        },
        "AIN": {
            "in_file":  in_dir / f"{p}annotation_packets_ain.csv",
            "out_file": in_dir / f"{p}min_annotate_ain.csv",
            "required": AIN_REQUIRED,
            "builder":  _build_ain_min,
        },
    }

    for task, cfg in specs.items():
        in_path  = cfg["in_file"]
        out_path = cfg["out_file"]
        required = cfg["required"]
        builder  = cfg["builder"]

        print(f"[{task}] Reading {in_path.name} ...")
        if not in_path.exists():
            raise FileNotFoundError(f"[{task}] Input not found: {in_path}")

        df = pd.read_csv(in_path, dtype=str).fillna("")

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"[{task}] {in_path.name} is missing required columns: {sorted(missing)}\n"
                f"  Found columns: {list(df.columns)}"
            )

        out_df = builder(df)
        out_df.to_csv(out_path, index=False)
        print(f"[{task}] Written {len(out_df):,} rows -> {out_path.name}  "
              f"({len(out_df.columns)} columns)")

    print()
    print("Next steps:")
    print(f"  1. Open {p}min_annotate_and.csv and {p}min_annotate_ain.csv in Excel.")
    print("  2. Fill the 'gold_label_code' column for every row:")
    print("       0 = uncertain   1 = non-match   2 = match")
    print("  3. Save (keep CSV format).")
    print(f"  4. Run: python -m bem.benchmark.apply_min_labels_to_dev2 "
          f"--in_dir {args.in_dir} --prefix {args.prefix}")


if __name__ == "__main__":
    main()
