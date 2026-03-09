"""apply_min_labels_to_dev2.py -- Apply numeric annotation codes back to dev2 packet files.

Reads the filled-in minimal annotation CSVs (gold_label_code 0/1/2), maps them
to canonical string labels, merges back into the original dev2 annotation
packet files, and then calls pack_benchmark_pairs to produce the final
benchmark parquets.

Label mapping
-------------
    0 -> "uncertain"
    1 -> "non-match"
    2 -> "match"

Usage
-----
    python -m vs2.benchmark.apply_min_labels_to_dev2 --in_dir data/derived --prefix dev2

Inputs
------
    <in_dir>/<prefix>_min_annotate_and.csv   (filled by annotator)
    <in_dir>/<prefix>_min_annotate_ain.csv   (filled by annotator)
    <in_dir>/<prefix>_annotation_packets_and.csv
    <in_dir>/<prefix>_annotation_packets_ain.csv

Outputs
-------
    <in_dir>/_backup_dev2_labels_<YYYYMMDD_HHMMSS>/   (backup of original packets)
    <in_dir>/<prefix>_annotation_packets_and.csv      (gold_label column updated)
    <in_dir>/<prefix>_annotation_packets_ain.csv      (gold_label column updated)
    <in_dir>/<prefix>_benchmark_pairs_and.parquet
    <in_dir>/<prefix>_benchmark_pairs_ain.parquet
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


LABEL_MAP = {
    "0": "uncertain",
    "1": "non-match",
    "2": "match",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_codes(df: pd.DataFrame, task: str, path: Path) -> pd.Series:
    """Return normalised gold_label_code series (strings '0'/'1'/'2') or raise."""
    if "gold_label_code" not in df.columns:
        raise ValueError(
            f"[{task}] 'gold_label_code' column not found in {path.name}.\n"
            f"  Columns present: {list(df.columns)}"
        )

    raw = df["gold_label_code"].astype(str).str.strip()

    blank_mask = raw.isin(["", "nan"])
    if blank_mask.any():
        n = blank_mask.sum()
        idx = df.index[blank_mask][:5].tolist()
        raise ValueError(
            f"[{task}] {n:,} rows have an empty gold_label_code in {path.name}.\n"
            f"  First row indices with blanks: {idx}\n"
            "  Fill all rows before running this script."
        )

    # Normalise numeric representations: "0.0" -> "0", "1.0" -> "1", etc.
    def _norm(v: str) -> str:
        try:
            return str(int(float(v)))
        except (ValueError, OverflowError):
            return v

    raw = raw.apply(_norm)

    invalid_mask = ~raw.isin(LABEL_MAP)
    if invalid_mask.any():
        bad_vals = df.loc[invalid_mask, "gold_label_code"].unique().tolist()
        raise ValueError(
            f"[{task}] Invalid gold_label_code values in {path.name}: {bad_vals}\n"
            "  Allowed values: 0 (uncertain), 1 (non-match), 2 (match)."
        )

    return raw


def _make_backup(in_dir: Path, prefix: str, timestamp: str) -> Path:
    backup_dir = in_dir / f"_backup_{prefix}_labels_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    p = f"{prefix.rstrip('_')}_" if prefix else ""
    for task in ("and", "ain"):
        src = in_dir / f"{p}annotation_packets_{task}.csv"
        if src.exists():
            shutil.copy2(src, backup_dir / src.name)
            print(f"  Backed up: {src.name} -> {backup_dir.name}/")
    return backup_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Apply numeric annotation codes to dev2 packet files and pack parquets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--in_dir",  default="data/derived",
                        help="Directory containing annotation files (default: data/derived).")
    parser.add_argument("--prefix",  default="dev2",
                        help="Filename prefix, e.g. 'dev2' (default: dev2).")
    parser.add_argument("--skip_pack", action="store_true",
                        help="Skip calling pack_benchmark_pairs after label merge.")
    args = parser.parse_args(argv)

    in_dir    = Path(args.in_dir)
    prefix    = args.prefix.rstrip("_")
    p         = f"{prefix}_" if prefix else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    any_error = False

    # -----------------------------------------------------------------------
    # Step 1 — validate all inputs before touching anything
    # -----------------------------------------------------------------------
    print("Validating inputs ...")
    validated: dict[str, tuple[pd.DataFrame, pd.Series]] = {}

    for task in ("AND", "AIN"):
        t = task.lower()
        min_path    = in_dir / f"{p}min_annotate_{t}.csv"
        packet_path = in_dir / f"{p}annotation_packets_{t}.csv"

        for path in (min_path, packet_path):
            if not path.exists():
                print(f"  [{task}] ERROR: File not found: {path}")
                any_error = True

        if any_error:
            continue

        min_df = pd.read_csv(min_path, dtype=str).fillna("")
        try:
            codes = _validate_codes(min_df, task, min_path)
        except ValueError as exc:
            print(f"  [{task}] ERROR: {exc}")
            any_error = True
            continue

        validated[task] = (min_df, codes)
        label_counts = codes.map(LABEL_MAP).value_counts().to_dict()
        print(f"  [{task}] {len(min_df):,} rows validated.  Label distribution: {label_counts}")

    if any_error:
        print("\nFix the errors above and re-run.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 2 — backup original packet files
    # -----------------------------------------------------------------------
    print(f"\nCreating backup ...")
    backup_dir = _make_backup(in_dir, prefix, timestamp)
    print(f"  Backup folder: {backup_dir}")

    # -----------------------------------------------------------------------
    # Step 3 — merge labels into packet files
    # -----------------------------------------------------------------------
    print("\nMerging labels into packet files ...")
    for task in ("AND", "AIN"):
        t = task.lower()
        min_df, codes = validated[task]
        packet_path   = in_dir / f"{p}annotation_packets_{t}.csv"

        # Build merge key -> label mapping
        min_df = min_df.copy()
        min_df["_code_norm"] = codes
        min_df["_gold_label"] = min_df["_code_norm"].map(LABEL_MAP)

        label_lookup = min_df.set_index(["anchor_id", "candidate_id"])["_gold_label"]

        # Load packet
        packet_df = pd.read_csv(packet_path, dtype=str).fillna("")
        n_before = len(packet_df)

        # Merge
        idx = pd.MultiIndex.from_frame(packet_df[["anchor_id", "candidate_id"]])
        mapped = label_lookup.reindex(idx).values

        n_missing = pd.isnull(mapped).sum()
        if n_missing > 0:
            raise RuntimeError(
                f"[{task}] {n_missing:,} rows in {packet_path.name} have no matching "
                "entry in the minimal annotation file. "
                "The files may be out of sync — re-run make_min_annotation_files."
            )

        packet_df["gold_label"] = mapped
        assert len(packet_df) == n_before, "Row count changed — aborting."

        # Verify every row got a non-empty label
        empty = (packet_df["gold_label"] == "") | packet_df["gold_label"].isna()
        if empty.any():
            raise RuntimeError(
                f"[{task}] {empty.sum():,} rows ended up with an empty gold_label — aborting."
            )

        packet_df.to_csv(packet_path, index=False)
        label_counts = packet_df["gold_label"].value_counts().to_dict()
        print(f"  [{task}] Written {n_before:,} rows -> {packet_path.name}")
        print(f"  [{task}]   gold_label distribution: {label_counts}")

    # -----------------------------------------------------------------------
    # Step 4 — pack benchmark parquets
    # -----------------------------------------------------------------------
    if args.skip_pack:
        print("\n--skip_pack set: skipping pack_benchmark_pairs.")
        return

    print("\nPacking benchmark parquets ...")
    cmd = [
        sys.executable, "-m", "vs2.benchmark.pack_benchmark_pairs",
        "--in_dir", str(in_dir),
        "--prefix", prefix,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("\nERROR: pack_benchmark_pairs exited with a non-zero code.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
