"""test_load_llm_decisions.py — Unit tests for load_llm_decisions_jsonl().

Tests the exact JSONL schema produced by the verifier:
  { run_id, task, anchor_id, candidate_id, ...,
    decision: { label, confidence, evidence_used, reason_code, abstention_reason },
    retry_count, backend, timestamp }
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from vs2.guards.apply_guards import load_llm_decisions_jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "decisions.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_nested_schema(tmp_path):
    """Top-level anchor_id/candidate_id + nested decision dict."""
    rows = [
        {
            "run_id": "test_run",
            "task": "AND",
            "anchor_id": "aaa",
            "candidate_id": "bbb",
            "prompt_version": "v1",
            "model_id": "claude-sonnet-4-6",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 800,
            "decision": {
                "label": "match",
                "confidence": 0.92,
                "evidence_used": ["author_norm", "coauthors_norm"],
                "reason_code": "name_exact_affil_match_nearby_years",
                "abstention_reason": None,
            },
            "retry_count": 0,
            "backend": "anthropic_api",
            "timestamp": "2026-03-08T22:56:46+00:00",
        },
        {
            "run_id": "test_run",
            "task": "AND",
            "anchor_id": "ccc",
            "candidate_id": "ddd",
            "decision": {
                "label": "non-match",
                "confidence": 0.88,
                "evidence_used": ["author_norm", "affiliations_norm"],
                "reason_code": "different_surname",
                "abstention_reason": None,
            },
            "retry_count": 0,
            "backend": "anthropic_api",
            "timestamp": "2026-03-08T22:57:00+00:00",
        },
    ]
    p = _write_jsonl(tmp_path, rows)
    df = load_llm_decisions_jsonl(p)

    assert len(df) == 2
    assert set(df.columns) >= {"anchor_id", "candidate_id", "label", "confidence", "evidence_used"}
    assert df.loc[df["anchor_id"] == "aaa", "label"].iloc[0] == "match"
    assert df.loc[df["anchor_id"] == "aaa", "confidence"].iloc[0] == pytest.approx(0.92)
    assert df.loc[df["anchor_id"] == "ccc", "label"].iloc[0] == "non-match"
    # evidence_used must be a Python list
    ev = df.loc[df["anchor_id"] == "aaa", "evidence_used"].iloc[0]
    assert isinstance(ev, list)
    assert "author_norm" in ev


def test_dedup_keeps_last_non_error(tmp_path):
    """When a pair appears twice (resume), keep the last non-error entry."""
    rows = [
        {
            "anchor_id": "aaa", "candidate_id": "bbb", "task": "AND",
            "decision": {"label": "uncertain", "confidence": 0.5,
                         "evidence_used": [], "reason_code": "first", "abstention_reason": None},
            "retry_count": 0, "backend": "anthropic_api",
        },
        {
            "anchor_id": "aaa", "candidate_id": "bbb", "task": "AND",
            "decision": {"label": "match", "confidence": 0.9,
                         "evidence_used": ["author_norm"], "reason_code": "second", "abstention_reason": None},
            "retry_count": 1, "backend": "anthropic_api",
        },
    ]
    p = _write_jsonl(tmp_path, rows)
    df = load_llm_decisions_jsonl(p)

    assert len(df) == 1
    assert df.iloc[0]["label"] == "match"
    assert df.iloc[0]["reason_code"] == "second"


def test_error_row_missing_decision(tmp_path):
    """Rows with no 'decision' key get label='error', confidence=0.0."""
    rows = [
        {
            "anchor_id": "aaa", "candidate_id": "bbb", "task": "AIN",
            "error": "timeout", "retry_count": 2, "backend": "anthropic_api",
        },
    ]
    p = _write_jsonl(tmp_path, rows)
    df = load_llm_decisions_jsonl(p)

    assert len(df) == 1
    assert df.iloc[0]["label"] == "error"
    assert df.iloc[0]["confidence"] == pytest.approx(0.0)


def test_error_then_success_dedup(tmp_path):
    """If a pair has an error entry then a success entry, keep the success."""
    rows = [
        {
            "anchor_id": "x1", "candidate_id": "y1", "task": "AIN",
            "error": "api_error", "retry_count": 0, "backend": "anthropic_api",
        },
        {
            "anchor_id": "x1", "candidate_id": "y1", "task": "AIN",
            "decision": {"label": "non-match", "confidence": 0.87,
                         "evidence_used": ["string"], "reason_code": "sp_disjoint",
                         "abstention_reason": None},
            "retry_count": 1, "backend": "anthropic_api",
        },
    ]
    p = _write_jsonl(tmp_path, rows)
    df = load_llm_decisions_jsonl(p)

    assert len(df) == 1
    assert df.iloc[0]["label"] == "non-match"


def test_empty_file(tmp_path):
    """Empty file returns empty DataFrame with required columns."""
    p = _write_jsonl(tmp_path, [])
    df = load_llm_decisions_jsonl(p)
    assert len(df) == 0
    assert "anchor_id" in df.columns
    assert "candidate_id" in df.columns


def test_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_llm_decisions_jsonl(tmp_path / "nonexistent.jsonl")


def test_real_schema_sample(tmp_path):
    """Reproduces the exact schema from the 20260308 run."""
    rows = [
        {
            "run_id": "20260308_235149_s9nqge",
            "task": "AND",
            "anchor_id": "003c8a9d7a6b56a3723265bcfb0394b831a4daf07b92b38cd9048711129a3c70",
            "candidate_id": "4d33e6ffe681cd44d169b9df555d0b84080fe20bfc075fef6e987c50339c6561",
            "prompt_version": "v1",
            "model_id": "claude-sonnet-4-6",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 800,
            "decision": {
                "label": "match",
                "confidence": 0.82,
                "evidence_used": ["author_norm", "affiliations_norm",
                                  "authors_with_affiliations_norm", "coauthors_norm"],
                "reason_code": "name_exact_affil_match_nearby_years",
                "abstention_reason": None,
            },
            "retry_count": 0,
            "backend": "anthropic_api",
            "usage": {"input_tokens": 1517, "output_tokens": 89},
            "timestamp": "2026-03-08T22:56:46+00:00",
        }
    ]
    p = _write_jsonl(tmp_path, rows)
    df = load_llm_decisions_jsonl(p)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["anchor_id"] == "003c8a9d7a6b56a3723265bcfb0394b831a4daf07b92b38cd9048711129a3c70"
    assert row["candidate_id"] == "4d33e6ffe681cd44d169b9df555d0b84080fe20bfc075fef6e987c50339c6561"
    assert row["label"] == "match"
    assert row["confidence"] == pytest.approx(0.82)
    assert row["backend"] == "anthropic_api"
    assert row["retry_count"] == 0
