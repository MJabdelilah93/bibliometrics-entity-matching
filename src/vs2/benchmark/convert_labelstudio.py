"""convert_labelstudio.py — Convert Label Studio annotation exports to VS2 benchmark parquets.

Reads a Label Studio export (CSV or JSON) containing pairwise AND / AIN annotation
decisions and writes two benchmark parquet files:

    data/derived/benchmark_pairs_and.parquet
    data/derived/benchmark_pairs_ain.parquet

Each output file has the schema:
    anchor_id    (str)  — author_instance_id or affil_instance_id (SHA-256 hex)
    candidate_id (str)  — same type
    task         (str)  — "AND" or "AIN"
    gold_label   (str)  — "match" | "non-match" | "uncertain"
    split        (str)  — "train" | "dev" | "test"  (if present in export, else None)
    stratum      (str)  — annotation stratum tag (if present in export, else None)

Usage
-----
    python -m vs2.benchmark.convert_labelstudio \\
        --input path/to/labelstudio_export.csv \\
        --format auto \\
        --task auto \\
        --out_and data/derived/benchmark_pairs_and.parquet \\
        --out_ain data/derived/benchmark_pairs_ain.parquet \\
        --author_instances data/interim/author_instances.parquet \\
        --affil_instances  data/interim/affil_instances.parquet

Format detection (--format auto)
---------------------------------
  CSV  — expects columns: anchor_id, candidate_id, task, gold_label; optional: split, stratum
  JSON — Label Studio list-of-tasks format:
           [{"id":1, "data":{"anchor_id":"...", "candidate_id":"...", "task":"..."},
             "annotations":[{"result":[{"value":{"choices":["match"]}}]}]}, ...]
         The first annotation's first result choice is used as gold_label.

Task detection (--task auto)
-----------------------------
  Looks at the "task" column / field. Rows with task=="AND" → AND output; "AIN" → AIN output.
  If --task AND or --task AIN is given, all rows are forced to that task.

ID validation
-------------
  Each anchor_id and candidate_id is checked against the appropriate instance parquet.
  Unknown IDs are reported in a mismatch summary printed to stdout. They are NOT written
  to the output files (to keep the benchmark clean).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Column / field name aliases accepted in CSV exports
# ---------------------------------------------------------------------------

_ANCHOR_ALIASES    = {"anchor_id", "anchor", "id_anchor", "instance_a", "a_id"}
_CANDIDATE_ALIASES = {"candidate_id", "candidate", "id_candidate", "instance_b", "b_id"}
_TASK_ALIASES      = {"task", "entity_type", "type"}
_LABEL_ALIASES     = {"gold_label", "label", "annotation", "choice", "decision"}
_SPLIT_ALIASES     = {"split", "fold", "partition"}
_STRATUM_ALIASES   = {"stratum", "strata", "stratum_tag", "tier"}


def _pick_col(df: pd.DataFrame, aliases: set[str], required: bool = True) -> str | None:
    """Return the first matching column name found in df.columns (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    if required:
        raise ValueError(
            f"Could not find any of {sorted(aliases)} among columns: {list(df.columns)}"
        )
    return None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV export from Label Studio (or any hand-crafted CSV)."""
    df = pd.read_csv(path, dtype=str).fillna("")
    return df


def _load_json(path: Path) -> pd.DataFrame:
    """Load a Label Studio JSON list-of-tasks export and flatten to a DataFrame."""
    raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("JSON export must be a list of task objects at the top level.")

    rows: list[dict[str, str]] = []
    for task in raw:
        data: dict[str, Any] = task.get("data", {})

        # Collect annotations — use first annotation, first result choice
        gold_label = ""
        annotations: list[dict] = task.get("annotations", [])
        if annotations:
            results: list[dict] = annotations[0].get("result", [])
            for r in results:
                choices = r.get("value", {}).get("choices", [])
                if choices:
                    gold_label = str(choices[0]).lower().strip()
                    break

        row: dict[str, str] = {
            "anchor_id":    str(data.get("anchor_id",    data.get("anchor",    ""))),
            "candidate_id": str(data.get("candidate_id", data.get("candidate", ""))),
            "task":         str(data.get("task",         data.get("entity_type", ""))).upper().strip(),
            "gold_label":   gold_label or str(data.get("gold_label", data.get("label", ""))),
            "split":        str(data.get("split",   "")),
            "stratum":      str(data.get("stratum", "")),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _detect_format(path: Path, fmt: str) -> str:
    """Return 'csv' or 'json' based on fmt hint or file extension."""
    if fmt in ("csv", "json"):
        return fmt
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in (".json", ".jsonl"):
        return "json"
    raise ValueError(
        f"Cannot detect format from extension '{suffix}'. "
        "Use --format csv or --format json explicitly."
    )


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_LABEL_MAP = {
    "match":     "match",
    "same":      "match",
    "positive":  "match",
    "yes":       "match",
    "1":         "match",
    "non-match": "non-match",
    "nonmatch":  "non-match",
    "different": "non-match",
    "negative":  "non-match",
    "no":        "non-match",
    "0":         "non-match",
    "uncertain": "uncertain",
    "unsure":    "uncertain",
    "skip":      "uncertain",
    "abstain":   "uncertain",
}


def _normalise_label(raw: str) -> str:
    """Map raw annotation string to canonical label; returns '' on unknown."""
    return _LABEL_MAP.get(raw.strip().lower(), "")


# ---------------------------------------------------------------------------
# ID validation
# ---------------------------------------------------------------------------

def _validate_ids(
    df: pd.DataFrame,
    task: str,
    author_inst_path: Path | None,
    affil_inst_path: Path | None,
) -> tuple[pd.DataFrame, int, int]:
    """Drop rows whose anchor_id or candidate_id are unknown. Return (clean_df, n_kept, n_dropped)."""
    if task == "AND":
        if author_inst_path is None or not author_inst_path.exists():
            print(
                f"  [WARN] author_instances parquet not found — skipping ID validation for AND."
            )
            return df, len(df), 0
        inst_df = pd.read_parquet(author_inst_path, columns=["author_instance_id"])
        valid_ids: set[str] = set(inst_df["author_instance_id"].astype(str))
    else:  # AIN
        if affil_inst_path is None or not affil_inst_path.exists():
            print(
                f"  [WARN] affil_instances parquet not found — skipping ID validation for AIN."
            )
            return df, len(df), 0
        inst_df = pd.read_parquet(affil_inst_path, columns=["affil_instance_id"])
        valid_ids = set(inst_df["affil_instance_id"].astype(str))

    mask_anchor    = df["anchor_id"].isin(valid_ids)
    mask_candidate = df["candidate_id"].isin(valid_ids)
    mask_valid = mask_anchor & mask_candidate

    n_bad_anchor    = int((~mask_anchor).sum())
    n_bad_candidate = int((~mask_candidate).sum())
    n_dropped       = int((~mask_valid).sum())

    if n_dropped > 0:
        print(f"  [{task}] ID validation: {n_bad_anchor} unknown anchor_ids, "
              f"{n_bad_candidate} unknown candidate_ids → {n_dropped} rows dropped.")
        # Print a sample of bad IDs
        bad_anchors = df.loc[~mask_anchor, "anchor_id"].unique()[:5]
        if len(bad_anchors):
            print(f"  [{task}]   Sample unknown anchor_ids: {list(bad_anchors)}")
        bad_cands = df.loc[~mask_candidate, "candidate_id"].unique()[:5]
        if len(bad_cands):
            print(f"  [{task}]   Sample unknown candidate_ids: {list(bad_cands)}")

    clean_df = df[mask_valid].copy()
    return clean_df, int(mask_valid.sum()), n_dropped


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def convert(
    input_path: Path,
    fmt: str,
    task_override: str,
    out_and: Path,
    out_ain: Path,
    author_inst_path: Path | None,
    affil_inst_path: Path | None,
) -> None:
    """Run the full conversion pipeline."""

    # 1. Load
    detected_fmt = _detect_format(input_path, fmt)
    print(f"Loading {detected_fmt.upper()} export: {input_path}")

    if detected_fmt == "csv":
        raw_df = _load_csv(input_path)
        # Remap column names to canonical names
        anchor_col    = _pick_col(raw_df, _ANCHOR_ALIASES)
        candidate_col = _pick_col(raw_df, _CANDIDATE_ALIASES)
        label_col     = _pick_col(raw_df, _LABEL_ALIASES)
        task_col      = _pick_col(raw_df, _TASK_ALIASES, required=(task_override == "auto"))
        split_col     = _pick_col(raw_df, _SPLIT_ALIASES,  required=False)
        stratum_col   = _pick_col(raw_df, _STRATUM_ALIASES, required=False)

        df = pd.DataFrame({
            "anchor_id":    raw_df[anchor_col].astype(str),
            "candidate_id": raw_df[candidate_col].astype(str),
            "gold_label":   raw_df[label_col].astype(str),
            "task":         raw_df[task_col].astype(str).str.upper().str.strip()
                            if task_col else pd.Series([""] * len(raw_df)),
            "split":        raw_df[split_col].astype(str) if split_col else pd.Series([""] * len(raw_df)),
            "stratum":      raw_df[stratum_col].astype(str) if stratum_col else pd.Series([""] * len(raw_df)),
        })
    else:
        df = _load_json(input_path)

    print(f"  Loaded {len(df)} rows.")

    # 2. Apply task override
    if task_override != "auto":
        df["task"] = task_override.upper()

    # 3. Normalise gold_label
    df["gold_label"] = df["gold_label"].apply(_normalise_label)

    # 4. Drop rows with empty/unrecognised label
    n_before = len(df)
    df = df[df["gold_label"] != ""].copy()
    n_bad_label = n_before - len(df)
    if n_bad_label:
        print(f"  Dropped {n_bad_label} rows with unrecognised gold_label.")

    # 5. Drop rows with empty anchor_id or candidate_id
    df = df[(df["anchor_id"] != "") & (df["candidate_id"] != "")].copy()

    # 6. Replace empty strings with None for optional columns
    for col in ("split", "stratum"):
        df[col] = df[col].replace("", None)

    # 7. Split by task
    and_df = df[df["task"] == "AND"].copy()
    ain_df = df[df["task"] == "AIN"].copy()

    tasks_present = []
    if not and_df.empty:
        tasks_present.append("AND")
    if not ain_df.empty:
        tasks_present.append("AIN")
    if not tasks_present:
        print("ERROR: No AND or AIN rows found after filtering. Check --task and column names.")
        sys.exit(1)

    # 8. ID validation + output
    output_cols = ["anchor_id", "candidate_id", "task", "gold_label", "split", "stratum"]

    for task, task_df, out_path in [
        ("AND", and_df, out_and),
        ("AIN", ain_df, out_ain),
    ]:
        if task_df.empty:
            print(f"  [{task}] No rows — skipping output.")
            continue

        clean_df, n_kept, n_dropped = _validate_ids(
            task_df, task, author_inst_path, affil_inst_path
        )

        if clean_df.empty:
            print(f"  [{task}] All rows dropped after ID validation — no output written.")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        clean_df[output_cols].reset_index(drop=True).to_parquet(out_path, index=False)

        label_counts = clean_df["gold_label"].value_counts().to_dict()
        split_counts = clean_df["split"].value_counts(dropna=False).to_dict()
        print(f"\n  [{task}] Written {n_kept} rows → {out_path}")
        print(f"  [{task}]   Labels : {label_counts}")
        print(f"  [{task}]   Splits : {split_counts}")
        if n_dropped:
            print(f"  [{task}]   Dropped (unknown IDs): {n_dropped}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Label Studio export to VS2 benchmark parquets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the Label Studio export file (CSV or JSON).",
    )
    parser.add_argument(
        "--format", dest="fmt", default="auto", choices=["auto", "csv", "json"],
        help="Export format. 'auto' detects from file extension (default: auto).",
    )
    parser.add_argument(
        "--task", default="auto", choices=["auto", "AND", "AIN"],
        help="Force all rows to task AND or AIN; 'auto' reads from the task column (default: auto).",
    )
    parser.add_argument(
        "--out_and",
        default="data/derived/benchmark_pairs_and.parquet",
        help="Output path for AND benchmark parquet (default: data/derived/benchmark_pairs_and.parquet).",
    )
    parser.add_argument(
        "--out_ain",
        default="data/derived/benchmark_pairs_ain.parquet",
        help="Output path for AIN benchmark parquet (default: data/derived/benchmark_pairs_ain.parquet).",
    )
    parser.add_argument(
        "--author_instances",
        default="data/interim/author_instances.parquet",
        help="Path to author_instances.parquet for ID validation (default: data/interim/author_instances.parquet).",
    )
    parser.add_argument(
        "--affil_instances",
        default="data/interim/affil_instances.parquet",
        help="Path to affil_instances.parquet for ID validation (default: data/interim/affil_instances.parquet).",
    )
    args = parser.parse_args(argv)

    convert(
        input_path=Path(args.input),
        fmt=args.fmt,
        task_override=args.task,
        out_and=Path(args.out_and),
        out_ain=Path(args.out_ain),
        author_inst_path=Path(args.author_instances),
        affil_inst_path=Path(args.affil_instances),
    )


if __name__ == "__main__":
    main()
