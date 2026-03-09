"""ingest.py — Load and validate raw Scopus UI CSV exports.

Evidence boundary: Scopus UI exports only (33-column schema defined in
configs/schema_headers.txt).  No external sources are ingested here.

All column values are read as str (dtype=str) to prevent implicit type
coercion.  Missing values are preserved as NaN / empty string — no data is
"fixed" beyond safe type parsing.  No rows are dropped at any stage.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# A) Schema header validation
# ---------------------------------------------------------------------------

def load_expected_headers(schema_headers_path: str | Path) -> list[str]:
    """Load expected column headers from the schema definition file.

    Reads configs/schema_headers.txt, stripping surrounding whitespace and
    optional surrounding double-quotes from each non-blank line.

    Args:
        schema_headers_path: Path to the schema headers text file.

    Returns:
        List of normalised column header strings in declaration order.

    Evidence boundary: this list defines the 33-column Scopus UI contract;
    no headers are added, removed, or reordered here.
    """
    path = Path(schema_headers_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    headers: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Remove optional surrounding double-quotes
        if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
            stripped = stripped[1:-1]
        headers.append(stripped)
    return headers


def read_csv_headers(csv_path: str | Path) -> list[str]:
    """Read only the header row from a CSV file.

    Normalises each header by stripping surrounding whitespace and optional
    surrounding double-quotes.

    Args:
        csv_path: Path to the raw Scopus CSV export.

    Returns:
        List of normalised column header strings as found in the file.
    """
    path = Path(csv_path)
    # nrows=0 loads only the header; utf-8-sig handles the BOM Scopus adds
    df_header = pd.read_csv(path, nrows=0, dtype=str, encoding="utf-8-sig")
    headers: list[str] = []
    for col in df_header.columns:
        stripped = col.strip()
        if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
            stripped = stripped[1:-1]
        headers.append(stripped)
    return headers


def validate_headers(expected: list[str], observed: list[str]) -> None:
    """Assert that observed CSV headers exactly match expected schema headers.

    Checks for missing columns, extra columns, and column-order differences.

    Args:
        expected: Headers from load_expected_headers().
        observed: Headers from read_csv_headers().

    Raises:
        ValueError: With a detailed diff showing missing columns, extra
            columns, and the first shared-column index where order differs.

    Evidence boundary: this guard ensures no column is silently dropped or
    reordered before downstream processing.
    """
    expected_set = set(expected)
    observed_set = set(observed)

    missing = [c for c in expected if c not in observed_set]
    extra = [c for c in observed if c not in expected_set]

    # Find first order mismatch among shared columns (preserving each side's order)
    order_error: str | None = None
    shared_expected = [c for c in expected if c in observed_set]
    shared_observed = [c for c in observed if c in expected_set]
    for i, (exp_col, obs_col) in enumerate(zip(shared_expected, shared_observed)):
        if exp_col != obs_col:
            order_error = (
                f"First order mismatch at shared-column index {i}: "
                f"expected '{exp_col}', observed '{obs_col}'"
            )
            break

    if missing or extra or order_error:
        parts: list[str] = ["CSV header mismatch:"]
        if missing:
            parts.append(f"  Missing columns ({len(missing)}): {missing}")
        if extra:
            parts.append(f"  Extra columns ({len(extra)}): {extra}")
        if order_error:
            parts.append(f"  Order error: {order_error}")
        raise ValueError("\n".join(parts))


# ---------------------------------------------------------------------------
# B) Ingestion of multiple CSV batches per query frame
# ---------------------------------------------------------------------------

def ingest_scopus_exports(
    csv_paths: list[str | Path],
    schema_headers_path: str | Path,
    query_frame: str,
) -> pd.DataFrame:
    """Ingest and validate multiple Scopus UI CSV exports for one query frame.

    For each CSV in csv_paths:
      - Validates headers against the 33-column schema contract.
      - Reads the full CSV with all columns as str (no implicit coercion).
      - Adds provenance columns: query_frame, source_file, row_id_in_file.
    Then concatenates all batches into a single DataFrame for the frame.

    Args:
        csv_paths: List of paths to raw Scopus CSV exports for this frame.
        schema_headers_path: Path to configs/schema_headers.txt.
        query_frame: Frame label, e.g. "Q1" or "Q2".

    Returns:
        Concatenated DataFrame with all 33 original columns (as str) plus
        query_frame (str), source_file (str), row_id_in_file (int).

    Raises:
        FileNotFoundError: If any csv_path does not exist.
        ValueError: If any CSV's headers fail schema validation.

    Evidence boundary: all column values are str; no rows are dropped; no
    values are coerced beyond safe string parsing.
    """
    expected_headers = load_expected_headers(schema_headers_path)
    batches: list[pd.DataFrame] = []

    for raw_path in csv_paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        # Validate headers before loading the full file
        observed_headers = read_csv_headers(path)
        validate_headers(expected_headers, observed_headers)

        # Read full CSV; utf-8-sig handles BOM; dtype=str prevents coercion
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=True)

        # Normalise column names to match expected (strips quotes / whitespace)
        col_map = {orig: norm for orig, norm in zip(df.columns, observed_headers)}
        df = df.rename(columns=col_map)

        # Add provenance columns
        df["query_frame"] = query_frame
        df["source_file"] = path.name
        df["row_id_in_file"] = range(len(df))

        batches.append(df)

    if not batches:
        return pd.DataFrame()

    return pd.concat(batches, ignore_index=True)


# ---------------------------------------------------------------------------
# C) Canonical records table builder
# ---------------------------------------------------------------------------

def _sha256_eid(eid_value: object) -> str:
    """Return hex SHA-256 of the EID string value.

    If EID is NaN or empty the empty string is hashed; the caller records a
    warning in the export manifest missingness count.

    Args:
        eid_value: Raw EID cell value (str or NaN).

    Returns:
        64-character lowercase hex digest string.
    """
    raw = str(eid_value) if pd.notna(eid_value) else ""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_canonical_records(
    q1_df: pd.DataFrame,
    q2_df: pd.DataFrame,
    expected_headers: list[str],
) -> pd.DataFrame:
    """Combine Q1 and Q2 ingested frames into a single canonical records table.

    Adds record_id = sha256(EID) as a stable opaque identifier for downstream
    stages.  Does not drop any rows or alter any field values.

    Args:
        q1_df: Ingested DataFrame for query frame Q1.
        q2_df: Ingested DataFrame for query frame Q2.
        expected_headers: The 33 schema column names (from load_expected_headers).

    Returns:
        DataFrame with all 33 original columns (str) plus:
          - query_frame     (str)
          - source_file     (str)
          - row_id_in_file  (int)
          - record_id       (str, hex SHA-256 of EID)
        Column order: 33 schema columns → provenance → record_id.

    Evidence boundary: EID is hashed to produce a stable opaque identifier;
    no rows are dropped; no field values are modified.
    """
    frames = [df for df in (q1_df, q2_df) if not df.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Compute record_id from EID column
    combined["record_id"] = combined["EID"].apply(_sha256_eid)

    # Enforce column order: 33 schema columns + provenance + record_id
    provenance_cols = ["query_frame", "source_file", "row_id_in_file", "record_id"]
    present_schema = [c for c in expected_headers if c in combined.columns]
    present_prov = [c for c in provenance_cols if c in combined.columns]
    combined = combined[present_schema + present_prov]

    return combined
