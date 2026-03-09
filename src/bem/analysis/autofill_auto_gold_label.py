"""autofill_auto_gold_label.py — Fill auto_gold_label columns in annotation packet CSVs.

Reads the dev annotation packet CSVs produced by build_annotation_packets.py,
calls the Anthropic API for each unannotated row, and writes results back to the
SAME files in-place.  A timestamped backup is created before any modification.

Evidence boundaries (enforced by the existing evidence card builders):
  AND : Author(s) ID is NEVER included.
  AIN : Title / year / source are NEVER included.

Resume support: rows where auto_gold_label is already non-empty are skipped.

Usage
-----
    python -m bem.analysis.autofill_auto_gold_label ^
        --prefix dev2 ^
        --in_dir data/derived ^
        --backend anthropic_api ^
        --model_id claude-sonnet-4-6 ^
        --temperature 0 ^
        --max_tokens 400 ^
        --rpm 30 ^
        --resume true ^
        --backup true

Added columns (appended in-place)
----------------------------------
    auto_gold_label     match | non-match | uncertain
    auto_confidence     float 0..1
    auto_reason_code    short snake_case string
    auto_evidence_used  pipe-separated list of field names
    auto_timestamp_utc  ISO-8601 UTC
    auto_model_id       model identifier
    auto_backend        backend used

Error log
---------
    <in_dir>/auto_label_errors_and.jsonl
    <in_dir>/auto_label_errors_ain.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from bem.llm_verify.evidence_cards import build_and_evidence, build_ain_evidence
from bem.llm_verify.verifier import check_and_truncate_evidence


# ---------------------------------------------------------------------------
# Load .env before any os.environ access
# ---------------------------------------------------------------------------

_repo_root = Path(__file__).resolve().parents[3]   # bem/
load_dotenv(dotenv_path=_repo_root / ".env", override=False)


# ---------------------------------------------------------------------------
# Output column definitions
# ---------------------------------------------------------------------------

AUTO_COLS = [
    "auto_gold_label",
    "auto_confidence",
    "auto_reason_code",
    "auto_evidence_used",
    "auto_timestamp_utc",
    "auto_model_id",
    "auto_backend",
]


# ---------------------------------------------------------------------------
# Decision schema (same as LLMDecisionV1 in verifier.py)
# ---------------------------------------------------------------------------

class _AutoDecision(BaseModel):
    label:         str   = Field(pattern=r"^(match|non-match|uncertain)$")
    confidence:    float = Field(ge=0.0, le=1.0)
    reason_code:   str
    evidence_used: list[str]


# ---------------------------------------------------------------------------
# Prompts (defined inline; JSON-only, abstain-first)
# ---------------------------------------------------------------------------

_AND_PROMPT = """\
You are a bibliometric identity-resolution expert.
Decide whether the two Scopus author instances below refer to the SAME real-world person.

EVIDENCE (JSON):
{{EVIDENCE_CARD}}

Respond with ONLY a single valid JSON object — no markdown, no text outside it:
{
  "label": "<match|non-match|uncertain>",
  "confidence": <float 0.0–1.0>,
  "reason_code": "<short_snake_case>",
  "evidence_used": ["<field1>", ...]
}

Decision rules (apply in order):
1. Default to "uncertain" when evidence is incomplete, ambiguous, or contradictory.
2. "match" requires strong positive evidence across at least 2 independent fields
   (e.g. matching normalised name AND overlapping co-authors or affiliations).
3. "non-match" requires a CLEAR contradiction (incompatible names, or completely
   different affiliations with no co-author overlap).
4. Author ID is NOT present in the evidence — do not reference it.
5. confidence: 1.0 = certain, 0.0 = no idea.\
"""

_AIN_PROMPT = """\
You are a bibliometric identity-resolution expert.
Decide whether the two Scopus affiliation instances below refer to the SAME
real-world institution or organisational unit.

EVIDENCE (JSON):
{{EVIDENCE_CARD}}

Respond with ONLY a single valid JSON object — no markdown, no text outside it:
{
  "label": "<match|non-match|uncertain>",
  "confidence": <float 0.0–1.0>,
  "reason_code": "<short_snake_case>",
  "evidence_used": ["<field1>", ...]
}

Decision rules (apply in order):
1. Default to "uncertain" when evidence is incomplete or ambiguous.
2. "match" requires strong agreement in string similarity, acronyms, or
   linked-author overlap.
3. "non-match" requires a CLEAR contradiction: very low string similarity
   AND no acronym overlap AND no linked-author support.
4. Title, year, and source are NOT in the evidence — do not reference them.
5. confidence: 1.0 = certain, 0.0 = no idea.\
"""

_PROMPTS: dict[str, str] = {"AND": _AND_PROMPT, "AIN": _AIN_PROMPT}


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RPMLimiter:
    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / max(rpm, 1)
        self._last: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        gap = self._interval - (now - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


# ---------------------------------------------------------------------------
# Error classification (simplified)
# ---------------------------------------------------------------------------

def _classify_exc(exc: Exception) -> tuple[str, int | None, str | None]:
    """Return (error_type, http_status, request_id)."""
    try:
        import anthropic as _a
        rid = getattr(exc, "request_id", None)
        if isinstance(exc, _a.AuthenticationError):
            return "AUTH", 401, rid
        if isinstance(exc, _a.PermissionDeniedError):
            return "PERMISSION", 403, rid
        if isinstance(exc, _a.RateLimitError):
            return "RATE_LIMIT", 429, rid
        if isinstance(exc, _a.BadRequestError):
            return "INVALID_REQUEST", 400, rid
        if isinstance(exc, _a.NotFoundError):
            return "MODEL_NOT_FOUND", 404, rid
        if isinstance(exc, _a.APIConnectionError):
            return "NETWORK", None, None
        if isinstance(exc, _a.APIStatusError):
            return "API_ERROR", int(exc.status_code), rid
    except ImportError:
        pass
    msg = str(exc).lower()
    if "empty_response_content" in msg:
        return "INVALID_RESPONSE", None, None
    if "auth" in msg or "api key" in msg:
        return "AUTH", None, None
    return "UNKNOWN", None, None


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

def _write_jsonl(fh: Any, obj: dict) -> None:
    fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
    fh.flush()


# ---------------------------------------------------------------------------
# Anthropic API call
# ---------------------------------------------------------------------------

def _call_api(
    prompt: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call the Anthropic API and return raw response text.

    Raises ValueError("empty_response_content") if response has no text blocks.
    """
    import os
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set.  Add it to .env or export it."
        )

    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    if not msg.content:
        raise ValueError("empty_response_content")
    parts = [b.text for b in msg.content if hasattr(b, "text")]
    result = "".join(parts).strip()
    if not result:
        raise ValueError("empty_response_content")
    return result


# ---------------------------------------------------------------------------
# Parse + validate response, with format-error retries
# ---------------------------------------------------------------------------

def _parse_decision(raw: str) -> _AutoDecision:
    """Parse raw JSON string into _AutoDecision.  Raises on failure."""
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        ).strip()
    return _AutoDecision(**json.loads(text))


def _call_with_retry(
    prompt: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    limiter: _RPMLimiter,
) -> _AutoDecision:
    """Call API and validate; retry up to max_retries times on format errors only."""
    retry_prompt = prompt
    for attempt in range(max_retries + 1):
        limiter.wait()
        raw = _call_api(retry_prompt, model_id, temperature, max_tokens)
        try:
            return _parse_decision(raw)
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
            if attempt >= max_retries:
                raise ValueError(
                    f"Format error after {attempt} retries: {exc}. "
                    f"Last response: {raw[:200]!r}"
                ) from exc
            # Append a FORMAT FIX instruction and retry
            retry_prompt = (
                retry_prompt
                + f"\n\n--- FORMAT FIX (attempt {attempt + 1}/{max_retries}) ---\n"
                f"Your previous response was invalid. Error: {exc}\n"
                "Output ONLY a single valid JSON object with keys: "
                "label, confidence, reason_code, evidence_used. No markdown."
            )
    raise RuntimeError("Unreachable")


# ---------------------------------------------------------------------------
# Error record builder
# ---------------------------------------------------------------------------

def _make_error_rec(
    row_index: int,
    anchor_id: str,
    candidate_id: str,
    exc: Exception,
) -> dict:
    err_type, http_status, req_id = _classify_exc(exc)
    return {
        "timestamp":    datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "row_index":    row_index,
        "anchor_id":    anchor_id,
        "candidate_id": candidate_id,
        "error_type":   err_type,
        "error_message": str(exc)[:500],
        "http_status":  http_status,
        "request_id":   req_id,
    }


# ---------------------------------------------------------------------------
# Build filled prompt from evidence card
# ---------------------------------------------------------------------------

def _build_prompt(evidence_card: dict, task: str) -> str:
    card, _ = check_and_truncate_evidence(evidence_card)
    evidence_json = json.dumps(card, indent=2, ensure_ascii=False, default=str)
    return _PROMPTS[task].replace("{{EVIDENCE_CARD}}", evidence_json)


# ---------------------------------------------------------------------------
# Per-CSV processing
# ---------------------------------------------------------------------------

def _process_csv(
    task: str,
    csv_path: Path,
    auth_inst_df: pd.DataFrame,
    affil_inst_df: pd.DataFrame,
    records_norm_df: pd.DataFrame,
    model_id: str,
    temperature: float,
    max_tokens: int,
    backend: str,
    limiter: _RPMLimiter,
    max_retries: int,
    do_resume: bool,
    do_backup: bool,
    error_log_path: Path,
) -> dict[str, Any]:
    """Process one annotation packet CSV in-place.

    Returns stats dict.
    """
    print(f"\n  [{task}] Loading {csv_path} ...")
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    # Ensure auto_* columns exist
    for col in AUTO_COLS:
        if col not in df.columns:
            df[col] = ""

    # Count already-filled (for resume)
    n_filled_before = int((df["auto_gold_label"] != "").sum())

    if do_backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = csv_path.with_name(f"{csv_path.stem}.bak_{ts}.csv")
        df.to_csv(backup_path, index=False)
        print(f"  [{task}] Backup written -> {backup_path.name}")

    stats: dict[str, Any] = {
        "total":          len(df),
        "filled_now":     0,
        "skipped_resume": 0,
        "errors":         0,
    }

    if not do_resume:
        # Clear all auto_* columns if not resuming
        for col in AUTO_COLS:
            df[col] = ""
        n_filled_before = 0

    if n_filled_before:
        print(f"  [{task}] Resume: {n_filled_before} rows already filled — will skip them.")

    with open(error_log_path, "a", encoding="utf-8") as err_fh:
        for idx in range(len(df)):
            row = df.iloc[idx]

            # Resume skip
            if row["auto_gold_label"] != "":
                stats["skipped_resume"] += 1
                continue

            anchor_id    = str(row.get("anchor_id", ""))
            candidate_id = str(row.get("candidate_id", ""))

            # --- Build evidence ---
            try:
                if task == "AND":
                    evidence_card = build_and_evidence(
                        anchor_id, candidate_id, auth_inst_df, records_norm_df
                    )
                else:
                    evidence_card = build_ain_evidence(
                        anchor_id, candidate_id, affil_inst_df, records_norm_df
                    )
            except Exception as exc:
                stats["errors"] += 1
                _write_jsonl(err_fh, _make_error_rec(idx, anchor_id, candidate_id, exc))
                _fill_error(df, idx, model_id, backend, "evidence_build_error")
                df.to_csv(csv_path, index=False)
                print(f"  [{task}] row {idx}: evidence error — {type(exc).__name__}: {exc}")
                continue

            # --- Call LLM or stub ---
            try:
                if backend == "requests_only":
                    decision = _AutoDecision(
                        label="uncertain",
                        confidence=0.0,
                        reason_code="requests_only_stub",
                        evidence_used=[],
                    )
                else:
                    prompt = _build_prompt(evidence_card, task)
                    decision = _call_with_retry(
                        prompt=prompt,
                        model_id=model_id,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        max_retries=max_retries,
                        limiter=limiter,
                    )
            except Exception as exc:
                stats["errors"] += 1
                _write_jsonl(err_fh, _make_error_rec(idx, anchor_id, candidate_id, exc))
                _fill_error(df, idx, model_id, backend, "api_error")
                df.to_csv(csv_path, index=False)
                err_type, _, _ = _classify_exc(exc)
                print(f"  [{task}] row {idx}: API error [{err_type}] — {str(exc)[:120]}")
                continue

            # --- Fill row ---
            _fill_decision(df, idx, decision, model_id, backend)
            stats["filled_now"] += 1

            # Save after every row (protects against crashes)
            df.to_csv(csv_path, index=False)

            # Progress print every 50 rows
            done = stats["filled_now"] + stats["errors"]
            remaining = stats["total"] - stats["skipped_resume"] - done
            if done % 50 == 0 and done > 0:
                print(f"  [{task}] progress: {done} done, {remaining} remaining ...")

    print(f"  [{task}] Done.")
    return stats


# ---------------------------------------------------------------------------
# In-place column setters
# ---------------------------------------------------------------------------

def _fill_decision(
    df: pd.DataFrame,
    idx: int,
    decision: _AutoDecision,
    model_id: str,
    backend: str,
) -> None:
    df.at[idx, "auto_gold_label"]    = decision.label
    df.at[idx, "auto_confidence"]    = f"{decision.confidence:.4f}"
    df.at[idx, "auto_reason_code"]   = decision.reason_code
    df.at[idx, "auto_evidence_used"] = "|".join(decision.evidence_used)
    df.at[idx, "auto_timestamp_utc"] = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    df.at[idx, "auto_model_id"]      = model_id
    df.at[idx, "auto_backend"]       = backend


def _fill_error(
    df: pd.DataFrame,
    idx: int,
    model_id: str,
    backend: str,
    reason: str,
) -> None:
    df.at[idx, "auto_gold_label"]    = "uncertain"
    df.at[idx, "auto_confidence"]    = "0.0000"
    df.at[idx, "auto_reason_code"]   = reason
    df.at[idx, "auto_evidence_used"] = ""
    df.at[idx, "auto_timestamp_utc"] = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    df.at[idx, "auto_model_id"]      = model_id
    df.at[idx, "auto_backend"]       = backend


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(task: str, csv_path: Path, stats: dict[str, Any], df: pd.DataFrame) -> None:
    print(f"\n  [{task}] Summary for {csv_path.name}:")
    print(f"    total_rows     : {stats['total']}")
    print(f"    filled_now     : {stats['filled_now']}")
    print(f"    skipped_resume : {stats['skipped_resume']}")
    print(f"    error_rows     : {stats['errors']}")
    # Distribution
    dist = df["auto_gold_label"].value_counts(dropna=False).to_dict()
    print(f"    auto_gold_label distribution:")
    for label in ["match", "non-match", "uncertain", ""]:
        n = dist.get(label, 0)
        if n or label == "":
            display = label if label else "<empty>"
            pct = 100.0 * n / max(stats["total"], 1)
            print(f"      {display:<14}  {n:>5}  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Auto-fill auto_gold_label columns in annotation packet CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--prefix",      default="dev2",
                        help="Filename prefix (default: dev2).")
    parser.add_argument("--in_dir",      default="data/derived",
                        help="Directory containing annotation_packets_*.csv.")
    parser.add_argument("--backend",     default="anthropic_api",
                        choices=["anthropic_api", "requests_only"],
                        help="LLM backend (default: anthropic_api).")
    parser.add_argument("--model_id",    default="claude-sonnet-4-6")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p",       type=float, default=1.0,
                        help="Kept for reference; not sent when temperature is set.")
    parser.add_argument("--max_tokens",  type=int, default=400)
    parser.add_argument("--rpm",         type=int, default=30,
                        help="Max requests per minute (default: 30).")
    parser.add_argument("--max_retries", type=int, default=2,
                        help="Max retries on format errors (default: 2).")
    parser.add_argument("--resume",      default="true",
                        help="'true' to skip already-filled rows (default: true).")
    parser.add_argument("--backup",      default="true",
                        help="'true' to write a timestamped backup before editing.")
    parser.add_argument("--author_instances", default="data/interim/author_instances.parquet")
    parser.add_argument("--affil_instances",  default="data/interim/affil_instances.parquet")
    parser.add_argument("--records_norm",     default="data/interim/records_normalised.parquet")
    args = parser.parse_args(argv)

    do_resume = args.resume.strip().lower() == "true"
    do_backup = args.backup.strip().lower() == "true"
    in_dir    = Path(args.in_dir)
    prefix    = args.prefix.rstrip("_") + "_" if args.prefix else ""

    # Resolve CSV paths
    and_csv = in_dir / f"{prefix}annotation_packets_and.csv"
    ain_csv = in_dir / f"{prefix}annotation_packets_ain.csv"

    # Error log paths
    and_err_log = in_dir / f"auto_label_errors_and.jsonl"
    ain_err_log = in_dir / f"auto_label_errors_ain.jsonl"

    print("autofill_auto_gold_label")
    print(f"  prefix    : {args.prefix!r}")
    print(f"  backend   : {args.backend}")
    print(f"  model_id  : {args.model_id}")
    print(f"  rpm       : {args.rpm}")
    print(f"  resume    : {do_resume}")
    print(f"  backup    : {do_backup}")
    print(f"  AND csv   : {and_csv}")
    print(f"  AIN csv   : {ain_csv}")

    # Check input CSV files exist
    missing = [p for p in [and_csv, ain_csv] if not p.exists()]
    if missing:
        for p in missing:
            print(f"\nERROR: not found: {p}")
        sys.exit(1)

    # Fail-fast API key check (anthropic_api only)
    if args.backend == "anthropic_api":
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("\nERROR: ANTHROPIC_API_KEY is not set.")
            print("Add it to .env or export it before running.")
            sys.exit(1)
        print("  api_key   : SET")
    else:
        print("  api_key   : N/A (requests_only mode)")

    # Load supporting parquets
    for label, path in [
        ("author_instances", args.author_instances),
        ("affil_instances",  args.affil_instances),
        ("records_norm",     args.records_norm),
    ]:
        if not Path(path).exists():
            print(f"\nERROR: {label} not found: {path}")
            sys.exit(1)

    print("\nLoading supporting parquets ...")
    auth_inst_df    = pd.read_parquet(args.author_instances)
    affil_inst_df   = pd.read_parquet(args.affil_instances)
    records_norm_df = pd.read_parquet(args.records_norm)
    print(f"  author_instances  : {len(auth_inst_df):,} rows")
    print(f"  affil_instances   : {len(affil_inst_df):,} rows")
    print(f"  records_normalised: {len(records_norm_df):,} rows")

    limiter = _RPMLimiter(args.rpm)

    # ---- AND ----
    and_df_final: pd.DataFrame | None = None
    and_stats = _process_csv(
        task="AND",
        csv_path=and_csv,
        auth_inst_df=auth_inst_df,
        affil_inst_df=affil_inst_df,
        records_norm_df=records_norm_df,
        model_id=args.model_id,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        backend=args.backend,
        limiter=limiter,
        max_retries=args.max_retries,
        do_resume=do_resume,
        do_backup=do_backup,
        error_log_path=and_err_log,
    )
    and_df_final = pd.read_csv(and_csv, dtype=str).fillna("")

    # ---- AIN ----
    ain_stats = _process_csv(
        task="AIN",
        csv_path=ain_csv,
        auth_inst_df=auth_inst_df,
        affil_inst_df=affil_inst_df,
        records_norm_df=records_norm_df,
        model_id=args.model_id,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        backend=args.backend,
        limiter=limiter,
        max_retries=args.max_retries,
        do_resume=do_resume,
        do_backup=do_backup,
        error_log_path=ain_err_log,
    )
    ain_df_final = pd.read_csv(ain_csv, dtype=str).fillna("")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    _print_summary("AND", and_csv, and_stats, and_df_final)
    _print_summary("AIN", ain_csv, ain_stats, ain_df_final)

    if and_err_log.exists() and and_err_log.stat().st_size > 0:
        print(f"\n  AND error log : {and_err_log}")
    if ain_err_log.exists() and ain_err_log.stat().st_size > 0:
        print(f"  AIN error log : {ain_err_log}")

    print("\nDone.")


if __name__ == "__main__":
    main()
