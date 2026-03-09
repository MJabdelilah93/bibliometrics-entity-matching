"""manifests.py — Write and read run artefact manifests.

Manifests record the full provenance of each pipeline run: config hashes,
input file hashes, model parameters, and output file locations.

Evidence boundary: all helpers operate only on local file paths and
in-memory data structures; no external services are contacted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunManifest:
    """Provenance record for a single VS2 pipeline run."""

    vs2_run_id: str
    timestamp_iso: str
    timezone: str
    config_path: str
    config_hash: str
    schema_headers_path: str
    schema_headers_hash: str
    model_id: str
    top_k: int
    truncation_author_count: int


# ---------------------------------------------------------------------------
# D) Checksum helpers
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    """Return hex SHA-256 digest of the file at *path*.

    Args:
        path: Path to any local file.

    Returns:
        64-character lowercase hex digest string.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    """Return hex SHA-256 digest of *text* encoded as UTF-8.

    Args:
        text: Any Python string.

    Returns:
        64-character lowercase hex digest string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# D) DataFrame statistics helper
# ---------------------------------------------------------------------------

def compute_dataframe_stats(
    q1_df: Any,  # pd.DataFrame — avoid importing pandas at module top level
    q2_df: Any,  # pd.DataFrame
) -> dict[str, Any]:
    """Compute row counts and per-column missingness for Q1 and Q2 frames.

    Missingness is defined as NaN *or* empty / whitespace-only string.

    Args:
        q1_df: Ingested Q1 DataFrame (output of ingest_scopus_exports).
        q2_df: Ingested Q2 DataFrame (output of ingest_scopus_exports).

    Returns:
        Dict with:
          - row_counts: total and per-source-file counts for Q1 and Q2.
          - missingness: per-column null/empty counts for Q1, Q2, and combined.
    """
    import pandas as pd

    def _row_counts_per_file(df: Any) -> dict[str, int]:
        if df.empty or "source_file" not in df.columns:
            return {}
        return {str(k): int(v) for k, v in df.groupby("source_file").size().items()}

    def _missingness(df: Any) -> dict[str, int]:
        result: dict[str, int] = {}
        for col in df.columns:
            ser = df[col]
            if ser.dtype == object:
                is_missing = ser.isna() | (ser.fillna("").str.strip() == "")
            else:
                is_missing = ser.isna()
            result[col] = int(is_missing.sum())
        return result

    frames = [df for df in (q1_df, q2_df) if not df.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return {
        "row_counts": {
            "Q1": {
                "total": len(q1_df),
                "per_file": _row_counts_per_file(q1_df),
            },
            "Q2": {
                "total": len(q2_df),
                "per_file": _row_counts_per_file(q2_df),
            },
        },
        "missingness": {
            "Q1": _missingness(q1_df) if not q1_df.empty else {},
            "Q2": _missingness(q2_df) if not q2_df.empty else {},
            "combined": _missingness(combined) if not combined.empty else {},
        },
    }


# ---------------------------------------------------------------------------
# D) Export manifest writer
# ---------------------------------------------------------------------------

def write_export_manifest(
    run_dir: str | Path,
    q1_paths: list[str | Path],
    q2_paths: list[str | Path],
    schema_headers_path: str | Path,
    dataframe_stats: dict[str, Any],
) -> Path:
    """Write export_manifest.json for the C2 ingestion artefact.

    Records:
      - Input files: path, size_bytes, sha256 for each Q1/Q2 CSV.
      - schema_headers_hash: SHA-256 of the schema headers file.
      - Row counts per file and per frame.
      - Missingness summary per column for Q1, Q2, and combined frames.

    Args:
        run_dir: Path to the current run directory (runs/<run_id>/).
        q1_paths: List of Q1 CSV file paths (existing files only).
        q2_paths: List of Q2 CSV file paths (existing files only).
        schema_headers_path: Path to configs/schema_headers.txt.
        dataframe_stats: Pre-computed stats dict from compute_dataframe_stats().

    Returns:
        Path to the written export_manifest.json file.

    Evidence boundary: all hashes are computed locally from file content;
    no network calls are made.
    """
    run_dir = Path(run_dir)
    manifests_dir = run_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    def _file_entry(p: Path) -> dict[str, Any]:
        return {
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }

    manifest: dict[str, Any] = {
        "schema_headers_hash": sha256_file(schema_headers_path),
        "input_files": {
            "Q1": [_file_entry(Path(p)) for p in q1_paths if Path(p).exists()],
            "Q2": [_file_entry(Path(p)) for p in q2_paths if Path(p).exists()],
        },
        **dataframe_stats,
    }

    out_path = manifests_dir / "export_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# C) Normalisation manifest + log writers
# ---------------------------------------------------------------------------

def write_normalisation_manifest(
    run_dir: str | Path,
    normalisation_rules_path: str | Path,
    rules_hash: str,
    summary_stats: dict[str, Any],
) -> Path:
    """Write normalisation_manifest.json for the C3 normalisation artefact.

    Records:
      - normalisation_rules_path and its SHA-256.
      - Per-column non-null counts for all 7 new normalised columns.
      - Per-column changed-value counts for text-normalised columns.
      - Top-20 most frequent affiliation acronyms (with counts).
      - Rules version stamps from normalisation_rules.yaml.

    Args:
        run_dir: Path to the current run directory (runs/<run_id>/).
        normalisation_rules_path: Path to configs/normalisation_rules.yaml.
        rules_hash: Pre-computed SHA-256 of the rules file.
        summary_stats: Stats dict returned by apply_normalisation().

    Returns:
        Path to the written normalisation_manifest.json file.

    Evidence boundary: hashes computed locally; no network calls.
    """
    run_dir = Path(run_dir)
    manifests_dir = run_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "normalisation_rules_path": str(normalisation_rules_path),
        "normalisation_rules_hash": rules_hash,
        **summary_stats,
    }

    out_path = manifests_dir / "normalisation_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


def write_normalisation_log(
    run_dir: str | Path,
    summary_stats: dict[str, Any],
) -> Path:
    """Write normalisation_log.jsonl: one JSON object per stat item.

    Line types written:
      - {"stat_type": "nonnull_count",  "column": ..., "count": ...}
      - {"stat_type": "changed_count",  "column": ..., "count": ...}
      - {"stat_type": "top_acronym",    "acronym": ..., "count": ...}
      - {"stat_type": "rules_version",  "key": ...,    "version": ...}

    Args:
        run_dir: Path to the current run directory.
        summary_stats: Stats dict returned by apply_normalisation().

    Returns:
        Path to the written normalisation_log.jsonl file.
    """
    run_dir = Path(run_dir)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    for col, count in summary_stats.get("column_nonnull_counts", {}).items():
        lines.append(json.dumps({"stat_type": "nonnull_count", "column": col, "count": count}))

    for col, count in summary_stats.get("changed_counts", {}).items():
        lines.append(json.dumps({"stat_type": "changed_count", "column": col, "count": count}))

    for entry in summary_stats.get("top_20_acronyms", []):
        lines.append(json.dumps({"stat_type": "top_acronym", **entry}))

    for key, version in summary_stats.get("normalisation_rules_versions", {}).items():
        lines.append(json.dumps({"stat_type": "rules_version", "key": key, "version": version}))

    out_path = logs_dir / "normalisation_log.jsonl"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# E) Candidate manifest writer
# ---------------------------------------------------------------------------

def write_candidate_manifest(
    run_dir: str | Path,
    manifest_dict: dict[str, Any],
) -> Path:
    """Write candidate_manifest.json for the C4 candidate generation artefact.

    Records pipeline parameters (top_k, truncation_author_count, year_window,
    token_prefix_len, prefix_chars), AND/AIN pass definitions, instance and
    candidate counts, top-20 block sizes per pass, and SHA-256 checksums of
    all four output parquet files.

    Args:
        run_dir: Path to the current run directory (runs/<run_id>/).
        manifest_dict: Pre-built manifest dict (constructed in run.py from
            constants exported by generate.py and the stats dict returned by
            run_candidate_generation).

    Returns:
        Path to the written candidate_manifest.json file.

    Evidence boundary: hashes computed locally from parquet file content;
    no external services are contacted.
    """
    run_dir = Path(run_dir)
    manifests_dir = run_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    out_path = manifests_dir / "candidate_manifest.json"
    out_path.write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Existing run manifest helpers (preserved from C1)
# ---------------------------------------------------------------------------

def write_run_manifest(manifest: RunManifest, manifest_dir: Path) -> Path:
    """Serialise a RunManifest to JSON.

    Args:
        manifest: Populated RunManifest instance.
        manifest_dir: Directory to write ``run_manifest.json`` into.

    Returns:
        Path to the written manifest file.
    """
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_path = manifest_dir / "run_manifest.json"
    out_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return out_path


def load_run_manifest(manifest_path: Path) -> RunManifest:
    """Load and validate a run manifest from JSON.

    Args:
        manifest_path: Path to an existing ``run_manifest.json``.

    Returns:
        Populated RunManifest instance.
    """
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return RunManifest(**data)
