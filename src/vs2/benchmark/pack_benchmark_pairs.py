"""pack_benchmark_pairs.py — Validate completed annotation CSVs and write benchmark parquets.

Reads the human-annotated evidence-packet CSVs (preferred) or task-template CSVs
(fallback), validates all rows, and writes the two benchmark parquets that C5
verification reads.

Usage
-----
    python -m vs2.benchmark.pack_benchmark_pairs --in_dir data/derived

    # Pack dev2 annotation packets (quota-stratified benchmark):
    python -m vs2.benchmark.pack_benchmark_pairs --in_dir data/derived --prefix dev2

    # Force old task-template behaviour:
    python -m vs2.benchmark.pack_benchmark_pairs --in_dir data/derived --use_packets false

Input resolution (per task, in priority order)
----------------------------------------------
    1. {prefix_}annotation_packets_{and,ain}.csv   ← preferred
    2. {prefix_}annotation_tasks_{and,ain}.csv     ← fallback

    Override with --use_packets false to always use annotation_tasks_*.csv.

Outputs
-------
    <in_dir>/{prefix_}benchmark_pairs_and.parquet
    <in_dir>/{prefix_}benchmark_pairs_ain.parquet
    <in_dir>/benchmark_id_mismatches.csv   (only if unknown IDs are detected)

Validation rules
----------------
1. Required columns must be present: anchor_id, candidate_id, task, gold_label, split.
2. gold_label must be one of: match, non-match, uncertain (case-insensitive).
   Rows with an empty or unrecognised gold_label are rejected as unlabelled.
3. split must be "dev" or "test" (case-insensitive).
4. anchor_id and candidate_id must be non-empty strings.
5. anchor_id / candidate_id must exist in the appropriate instance parquet:
   - AND ->data/interim/author_instances.parquet  (column: author_instance_id)
   - AIN ->data/interim/affil_instances.parquet   (column: affil_instance_id)
   Unknown IDs are written to benchmark_id_mismatches.csv and the script exits
   with a non-zero code so the user can investigate.

Output parquet schema
---------------------
    anchor_id    str
    candidate_id str
    task         str   ("AND" or "AIN")
    gold_label   str   ("match" | "non-match" | "uncertain")
    split        str   ("dev" | "test")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_LABELS = {"match", "non-match", "uncertain"}
VALID_SPLITS = {"dev", "test"}

LABEL_MAP = {
    "match":     "match",
    "same":      "match",
    "positive":  "match",
    "yes":       "match",
    "non-match": "non-match",
    "nonmatch":  "non-match",
    "different": "non-match",
    "negative":  "non-match",
    "no":        "non-match",
    "uncertain": "uncertain",
    "unsure":    "uncertain",
    "skip":      "uncertain",
    "abstain":   "uncertain",
}

INSTANCE_PATHS = {
    "AND": "data/interim/author_instances.parquet",
    "AIN": "data/interim/affil_instances.parquet",
}
INSTANCE_ID_COL = {
    "AND": "author_instance_id",
    "AIN": "affil_instance_id",
}

# Preferred input: evidence packets (annotated in place)
IN_PACKET_NAMES = {
    "AND": "annotation_packets_and.csv",
    "AIN": "annotation_packets_ain.csv",
}
# Fallback input: original task templates
IN_TASK_NAMES = {
    "AND": "annotation_tasks_and.csv",
    "AIN": "annotation_tasks_ain.csv",
}

OUT_NAMES = {
    "AND": "benchmark_pairs_and.parquet",
    "AIN": "benchmark_pairs_ain.parquet",
}

OUTPUT_COLS = ["anchor_id", "candidate_id", "task", "gold_label", "split"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_label(raw: str) -> str:
    return LABEL_MAP.get(raw.strip().lower(), "")


def _normalise_split(raw: str) -> str:
    v = raw.strip().lower()
    return v if v in VALID_SPLITS else ""


def _prefixed_names(prefix: str) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Return (packet_names, task_names, out_names) dicts with prefix applied."""
    p = f"{prefix.rstrip('_')}_" if prefix else ""
    packet_names = {t: f"{p}annotation_packets_{t.lower()}.csv" for t in ("AND", "AIN")}
    task_names   = {t: f"{p}annotation_tasks_{t.lower()}.csv"   for t in ("AND", "AIN")}
    out_names    = {t: f"{p}benchmark_pairs_{t.lower()}.parquet" for t in ("AND", "AIN")}
    return packet_names, task_names, out_names


def _resolve_input(
    in_dir: Path,
    task: str,
    use_packets: bool,
    packet_names: dict[str, str],
    task_names: dict[str, str],
) -> Path | None:
    """Return the input CSV path to use, printing which file was chosen."""
    packet_path = in_dir / packet_names[task]
    task_path   = in_dir / task_names[task]

    if use_packets and packet_path.exists():
        print(f"  [{task}] Input : {packet_path.name}  (evidence packets — preferred)")
        return packet_path

    if use_packets and not packet_path.exists():
        print(f"  [{task}] NOTE  : {packet_path.name} not found; "
              f"falling back to {task_path.name}")

    if task_path.exists():
        print(f"  [{task}] Input : {task_path.name}  (task templates — fallback)")
        return task_path

    print(f"  [{task}] Input CSV not found "
          f"(tried: {packet_path.name}, {task_path.name}) — skipping.")
    return None


def _load_csv(path: Path, task: str) -> pd.DataFrame:
    """Load and return annotation CSV with basic dtype normalisation."""
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"anchor_id", "candidate_id", "gold_label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[{task}] {path.name} is missing required columns: {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )
    return df


def _validate_ids(
    df: pd.DataFrame,
    task: str,
    instance_path: Path,
    mismatch_rows: list[dict],
) -> pd.DataFrame:
    """Drop rows whose IDs are absent in the instance parquet; accumulate mismatches."""
    id_col = INSTANCE_ID_COL[task]
    if not instance_path.exists():
        print(f"  [{task}] WARN: instance parquet not found at {instance_path}. "
              "Skipping ID validation.")
        return df

    inst_ids: set[str] = set(
        pd.read_parquet(instance_path, columns=[id_col])[id_col].astype(str)
    )

    mask_anchor    = df["anchor_id"].isin(inst_ids)
    mask_candidate = df["candidate_id"].isin(inst_ids)
    bad_mask       = ~(mask_anchor & mask_candidate)

    if bad_mask.any():
        for _, row in df[bad_mask].iterrows():
            mismatch_rows.append({
                "task":            task,
                "anchor_id":       row["anchor_id"],
                "candidate_id":    row["candidate_id"],
                "anchor_known":    row["anchor_id"] in inst_ids,
                "candidate_known": row["candidate_id"] in inst_ids,
            })

    return df[~bad_mask].copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Pack completed annotation CSVs into benchmark parquets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--in_dir",
        default="data/derived",
        help="Directory containing annotation CSVs (default: data/derived).",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help=(
            "Filename prefix for input CSVs and output parquets.  "
            "E.g. 'dev2' reads dev2_annotation_packets_*.csv and writes "
            "dev2_benchmark_pairs_*.parquet (default: empty = standard names)."
        ),
    )
    parser.add_argument(
        "--in_prefix",
        default=None,
        help="Alias for --prefix (for backward compatibility).",
    )
    parser.add_argument(
        "--use_packets",
        default="true",
        help=(
            "'true' (default): prefer annotation_packets_*.csv over annotation_tasks_*.csv. "
            "'false': always use annotation_tasks_*.csv."
        ),
    )
    parser.add_argument(
        "--author_instances",
        default=INSTANCE_PATHS["AND"],
        help=f"Path to author_instances.parquet (default: {INSTANCE_PATHS['AND']}).",
    )
    parser.add_argument(
        "--affil_instances",
        default=INSTANCE_PATHS["AIN"],
        help=f"Path to affil_instances.parquet (default: {INSTANCE_PATHS['AIN']}).",
    )
    args = parser.parse_args(argv)

    in_dir       = Path(args.in_dir)
    use_packets  = args.use_packets.strip().lower() != "false"
    # --in_prefix is an alias for --prefix; --prefix takes precedence
    prefix = args.prefix if args.prefix else (args.in_prefix or "")
    instance_paths = {
        "AND": Path(args.author_instances),
        "AIN": Path(args.affil_instances),
    }

    packet_names, task_names, out_names = _prefixed_names(prefix)

    prefix_label = repr(prefix) if prefix else "(none)"
    print(f"prefix={prefix_label}  use_packets={use_packets}")
    print()

    mismatch_rows: list[dict] = []
    any_error = False

    for task in ("AND", "AIN"):
        csv_path = _resolve_input(in_dir, task, use_packets, packet_names, task_names)
        if csv_path is None:
            continue

        # -- Load --
        try:
            df = _load_csv(csv_path, task)
        except ValueError as exc:
            print(f"  [{task}] ERROR: {exc}")
            any_error = True
            continue

        n_raw = len(df)
        print(f"  [{task}] Loaded {n_raw:,} rows")

        # -- Normalise label --
        df["gold_label"] = df["gold_label"].apply(_normalise_label)
        unlabelled = df[df["gold_label"] == ""]
        if not unlabelled.empty:
            print(f"  [{task}] WARN: {len(unlabelled):,} rows have an empty or "
                  "unrecognised gold_label — these rows are excluded.")
            print(f"  [{task}]   First few row indices: {unlabelled.index[:5].tolist()}")
        df = df[df["gold_label"] != ""].copy()

        # -- Normalise split --
        df["split"] = df["split"].apply(_normalise_split)
        bad_split = df[df["split"] == ""]
        if not bad_split.empty:
            print(f"  [{task}] WARN: {len(bad_split):,} rows have an unrecognised split "
                  "(expected dev/test) — these rows are excluded.")
        df = df[df["split"] != ""].copy()

        # -- Drop empty IDs --
        bad_ids = df[(df["anchor_id"] == "") | (df["candidate_id"] == "")]
        if not bad_ids.empty:
            print(f"  [{task}] WARN: {len(bad_ids):,} rows have empty anchor_id or "
                  "candidate_id — excluded.")
        df = df[(df["anchor_id"] != "") & (df["candidate_id"] != "")].copy()

        # -- ID validation --
        n_before_id = len(df)
        df = _validate_ids(df, task, instance_paths[task], mismatch_rows)
        n_id_dropped = n_before_id - len(df)
        if n_id_dropped:
            print(f"  [{task}] {n_id_dropped:,} rows dropped: IDs not found in "
                  f"{instance_paths[task].name}")

        if df.empty:
            print(f"  [{task}] ERROR: No valid rows remain after validation.")
            any_error = True
            continue

        # -- Set task column --
        df["task"] = task

        # -- Write parquet --
        out_path = in_dir / out_names[task]
        df[OUTPUT_COLS].reset_index(drop=True).to_parquet(out_path, index=False)

        # -- Summary --
        label_counts = df["gold_label"].value_counts().to_dict()
        split_counts = df["split"].value_counts().to_dict()
        print(f"  [{task}] Written {len(df):,} rows ->{out_path}")
        print(f"  [{task}]   Labels : {label_counts}")
        print(f"  [{task}]   Splits : {split_counts}")
        print()

    # -- Write mismatch report if needed --
    if mismatch_rows:
        mismatch_path = in_dir / "benchmark_id_mismatches.csv"
        pd.DataFrame(mismatch_rows).to_csv(mismatch_path, index=False)
        print(f"ID mismatch report written ->{mismatch_path}")
        print("Check the report, fix the IDs in your annotation CSVs, then re-run.")
        any_error = True

    if any_error:
        sys.exit(1)

    print("Benchmark parquets are ready.")
    print("Next step — enable C5 verification and run the pipeline:")
    print("  Ensure configs/run_config.yaml has:")
    print("    verification:")
    print("      enabled: true")
    print("      scope: benchmark")
    print("    llm:")
    print("      backend: requests_only")
    print()
    print("  Then run:")
    print("    python -m vs2 --config configs/run_config.yaml")


if __name__ == "__main__":
    main()
