"""test_config_load.py — Minimal config and schema smoke-tests (no pytest dependency).

Run with:
    python tests/test_config_load.py

Checks:
  1. configs/run_config.yaml can be loaded and all required keys are present.
  2. configs/schema_headers.txt can be loaded and contains exactly 33 headers.

No actual CSV files are required for these tests.
"""

from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_PATH = Path("configs/run_config.yaml")

REQUIRED_TOP_LEVEL_KEYS = [
    "project",
    "inputs",
    "candidate_generation",
    "ranking",
    "llm",
    "guards_and_routing",
    "outputs",
]

REQUIRED_NESTED = {
    "project": ["name", "timezone"],
    "inputs": ["schema_headers_path", "q1_csv_paths", "q2_csv_paths"],
    "candidate_generation": ["top_k", "truncation_author_count"],
    "llm": ["model_id", "temperature", "top_p", "max_tokens"],
    "guards_and_routing": ["m_independent_signals", "precision_floor_target", "thresholds_initial"],
    "outputs": ["prefer_parquet", "llm_trace_format"],
}


def test_config_loads() -> None:
    assert CONFIG_PATH.exists(), f"Config file not found: {CONFIG_PATH}"
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert isinstance(cfg, dict), "Config must deserialise to a dict."

    for key in REQUIRED_TOP_LEVEL_KEYS:
        assert key in cfg, f"Missing top-level key in config: '{key}'"

    for section, keys in REQUIRED_NESTED.items():
        for key in keys:
            assert key in cfg[section], (
                f"Missing key '{key}' in config section '{section}'"
            )

    # Spot-check values
    assert cfg["candidate_generation"]["top_k"] == 50, "top_k must be 50"
    assert cfg["candidate_generation"]["truncation_author_count"] == 100
    assert cfg["llm"]["temperature"] == 0, "temperature must be 0 for reproducibility"
    assert cfg["llm"]["model_id"] == "claude-sonnet-4-6"

    print("PASS — configs/run_config.yaml loaded and validated.")


SCHEMA_HEADERS_PATH = Path("configs/schema_headers.txt")
EXPECTED_HEADER_COUNT = 33


def test_schema_headers_load() -> None:
    """Load expected headers and assert the count is exactly 33.

    Does NOT require any CSV files to be present.
    """
    assert SCHEMA_HEADERS_PATH.exists(), (
        f"Schema headers file not found: {SCHEMA_HEADERS_PATH}"
    )

    # Use the same normalisation logic as the ingest module (inline here to
    # keep this test free of import-side-effects from pandas).
    lines = SCHEMA_HEADERS_PATH.read_text(encoding="utf-8").splitlines()
    headers: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
            stripped = stripped[1:-1]
        headers.append(stripped)

    assert len(headers) == EXPECTED_HEADER_COUNT, (
        f"Expected {EXPECTED_HEADER_COUNT} schema headers, "
        f"got {len(headers)}: {headers}"
    )

    # EID must be the last column (Scopus UI contract)
    assert headers[-1] == "EID", (
        f"Last schema column must be 'EID', got '{headers[-1]}'"
    )

    print(
        f"PASS — configs/schema_headers.txt contains exactly "
        f"{len(headers)} headers; last column is 'EID'."
    )


if __name__ == "__main__":
    test_config_loads()
    test_schema_headers_load()
