"""verifier.py — LLM-based pairwise verification with structured JSON decisions (C5).

Evidence boundary: the LLM receives only field values derived from the Scopus CSV.
  AND: Author(s) ID is NEVER included.
  AIN: title / source / year are NEVER included (prevents topical-inference bias).

The LLM may only output one of: {"match", "non-match", "uncertain"}.
"uncertain" is treated as abstention and defaults to NO MERGE downstream.

IMPORTANT — gold annotation independence:
  This module is used for automated pipeline verification only.  Gold-standard
  labels are assigned by the human researcher from primary evidence, entirely
  independent of any LLM suggestions.

Retries are allowed ONLY for JSON / schema format errors (max 2 retries).
They are NEVER issued to obtain a "better" or "different" label.

Backends:
  requests_only  — writes each request envelope as a JSONL line to
                   runs/<run_id>/logs/llm_requests_{task}.jsonl; does NOT call
                   any model; records a stub decision (label="uncertain",
                   confidence=0.0, reason_code="requests_only_stub").
  anthropic_api  — calls the Anthropic Python SDK; requires the environment
                   variable ANTHROPIC_API_KEY to be set.

Error classification codes (written to llm_errors_{task}.jsonl):
  AUTH              — authentication failure (bad/missing API key)
  PERMISSION        — key exists but lacks permission for this model/endpoint
  BILLING           — account credit / subscription issue
  RATE_LIMIT        — 429 too many requests
  MODEL_NOT_FOUND   — model ID does not exist
  INVALID_REQUEST   — bad request payload (400)
  NETWORK           — connection-level failure
  UNKNOWN           — unclassified error
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator


# ---------------------------------------------------------------------------
# Decision schema (Pydantic v2)
# ---------------------------------------------------------------------------

class LLMDecisionV1(BaseModel):
    """Structured LLM decision — output_schema_version v1."""

    label: Literal["match", "non-match", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_used: list[str]
    reason_code: str
    abstention_reason: str | None = None


# Stub used by the requests_only backend
_STUB_DECISION = LLMDecisionV1(
    label="uncertain",
    confidence=0.0,
    evidence_used=[],
    reason_code="requests_only_stub",
    abstention_reason="Backend is requests_only; no API call was made.",
)

# Payload size thresholds for defensive logging
_WARN_EVIDENCE_CHARS = 30_000   # warn if evidence JSON > this
_WARN_PROMPT_CHARS   = 60_000   # warn if filled prompt > this
_TRUNCATE_STR_CHARS  = 400      # max chars for any single string field when truncating


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RPMLimiter:
    """Enforces a minimum inter-request interval derived from max_requests_per_minute."""

    def __init__(self, max_rpm: int) -> None:
        self._interval: float = 60.0 / max(max_rpm, 1)
        self._last: float = 0.0

    def wait_if_needed(self) -> None:
        now = time.monotonic()
        gap = self._interval - (now - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompt(path: str | Path) -> str:
    """Load a prompt template from disk.

    Args:
        path: Path to a ``.txt`` prompt file.

    Returns:
        Raw prompt string (contains ``{{EVIDENCE_CARD}}`` placeholder).

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Request envelope builder
# ---------------------------------------------------------------------------

def make_request(
    task: str,
    evidence_card: dict[str, Any],
    prompt_text: str,
    model_id: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 800,
    prompt_version: str = "v1",
) -> dict[str, Any]:
    """Build a self-contained request envelope.

    The ``filled_prompt`` field contains the prompt text with ``{{EVIDENCE_CARD}}``
    replaced by the JSON-serialised evidence card.  All other fields are kept for
    logging and reproducibility.

    Args:
        task: "AND" or "AIN".
        evidence_card: Dict produced by build_and_evidence or build_ain_evidence.
        prompt_text: Raw prompt template (with ``{{EVIDENCE_CARD}}`` placeholder).
        model_id: Model identifier string.
        temperature: Sampling temperature.
        top_p: Top-p nucleus sampling parameter.
        max_tokens: Maximum tokens in model response.
        prompt_version: Version tag for the prompt template.

    Returns:
        Request envelope dict.
    """
    evidence_json = json.dumps(evidence_card, indent=2, ensure_ascii=False, default=str)
    filled = prompt_text.replace("{{EVIDENCE_CARD}}", evidence_json)
    return {
        "task": task,
        "model_id": model_id,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "prompt_version": prompt_version,
        "evidence_card": evidence_card,
        "filled_prompt": filled,
        "evidence_json_chars": len(evidence_json),
        "prompt_chars": len(filled),
    }


# ---------------------------------------------------------------------------
# Payload size check + defensive truncation
# ---------------------------------------------------------------------------

def _truncate_strings_in_obj(obj: Any, max_chars: int = _TRUNCATE_STR_CHARS) -> Any:
    """Recursively truncate all string values in a dict/list to max_chars."""
    if isinstance(obj, str):
        return obj[:max_chars] + ("…" if len(obj) > max_chars else "")
    if isinstance(obj, dict):
        return {k: _truncate_strings_in_obj(v, max_chars) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_strings_in_obj(v, max_chars) for v in obj]
    return obj


def check_and_truncate_evidence(
    evidence_card: dict[str, Any],
    warn_chars: int = _WARN_EVIDENCE_CHARS,
) -> tuple[dict[str, Any], bool]:
    """Return (possibly-truncated evidence_card, was_truncated).

    If the serialised evidence card exceeds warn_chars, string fields are
    truncated to _TRUNCATE_STR_CHARS.  The truncation is deterministic and
    logged by the caller.
    """
    evidence_json = json.dumps(evidence_card, ensure_ascii=False, default=str)
    if len(evidence_json) <= warn_chars:
        return evidence_card, False
    truncated = _truncate_strings_in_obj(evidence_card, _TRUNCATE_STR_CHARS)
    return truncated, True  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def _classify_error(exc: Exception) -> tuple[str, int | None, str | None]:
    """Classify an API exception into a standard code.

    Returns
    -------
    (classification, http_status, request_id)
      classification — one of AUTH / PERMISSION / BILLING / RATE_LIMIT /
                       MODEL_NOT_FOUND / INVALID_REQUEST / NETWORK / UNKNOWN
      http_status    — integer HTTP status code if available, else None
      request_id     — Anthropic request-id header value if available, else None
    """
    try:
        import anthropic as _a
        request_id: str | None = getattr(exc, "request_id", None)

        if isinstance(exc, _a.AuthenticationError):
            return "AUTH", 401, request_id
        if isinstance(exc, _a.PermissionDeniedError):
            return "PERMISSION", 403, request_id
        if isinstance(exc, _a.RateLimitError):
            return "RATE_LIMIT", 429, request_id
        if isinstance(exc, _a.BadRequestError):
            # "usage limits" errors arrive as 400 BadRequestError — classify as
            # BILLING so the caller can treat them as a hard-stop, not a retry.
            msg_lower = str(exc).lower()
            if "usage" in msg_lower and "limit" in msg_lower:
                return "BILLING", 400, request_id
            return "INVALID_REQUEST", 400, request_id
        if isinstance(exc, _a.NotFoundError):
            return "MODEL_NOT_FOUND", 404, request_id
        if isinstance(exc, _a.APIConnectionError):
            return "NETWORK", None, None
        if isinstance(exc, _a.APIStatusError):
            status = int(exc.status_code)
            msg_lower = str(exc).lower()
            if status == 401:
                return "AUTH", status, request_id
            if status == 402 or "billing" in msg_lower or "credit" in msg_lower:
                return "BILLING", status, request_id
            if status == 403:
                return "PERMISSION", status, request_id
            if status == 404:
                return "MODEL_NOT_FOUND", status, request_id
            if status == 429:
                return "RATE_LIMIT", status, request_id
            if status == 400:
                return "INVALID_REQUEST", status, request_id
            return "UNKNOWN", status, request_id
    except ImportError:
        pass

    # Fallback: keyword-based classification for non-Anthropic exceptions
    msg_lower = str(exc).lower()
    # Empty / malformed response from the API
    if isinstance(exc, (IndexError, ValueError)) and (
        "empty_response_content" in msg_lower
        or "list index out of range" in msg_lower
    ):
        return "INVALID_RESPONSE", None, None
    if "auth" in msg_lower or "unauthorized" in msg_lower or "api key" in msg_lower:
        return "AUTH", None, None
    if "billing" in msg_lower or "credit" in msg_lower or (
        "usage" in msg_lower and "limit" in msg_lower
    ):
        return "BILLING", None, None
    if "rate" in msg_lower or "429" in msg_lower:
        return "RATE_LIMIT", None, None
    if "connect" in msg_lower or "timeout" in msg_lower or "network" in msg_lower:
        return "NETWORK", None, None
    return "UNKNOWN", None, None


# ---------------------------------------------------------------------------
# Validation + retry
# ---------------------------------------------------------------------------

def validate_or_retry(
    initial_response: str,
    call_fn: Callable[[str], str],
    original_prompt: str,
    max_retries: int = 2,
) -> tuple[LLMDecisionV1, int]:
    """Parse and validate a model response; retry on JSON / schema errors.

    Retries are triggered ONLY by:
      - ``json.JSONDecodeError``: response is not valid JSON.
      - ``pydantic.ValidationError``: JSON does not match LLMDecisionV1.

    Each retry appends a FORMAT FIX instruction to the original prompt and calls
    ``call_fn`` with the amended prompt.  The retry counter is returned so it can
    be logged.

    Args:
        initial_response: Raw text from the first model call.
        call_fn: Callable that accepts a prompt string and returns a raw response.
                 Not called for ``requests_only`` backend (no retries needed).
        original_prompt: The original filled prompt (used to construct retry prompts).
        max_retries: Maximum number of retry attempts (default 2).

    Returns:
        (decision, retry_count) where retry_count is 0 if no retry was needed.

    Raises:
        ValueError: if the response cannot be parsed after all retries.
    """
    response_text = initial_response
    prompt_for_retry = original_prompt

    for attempt in range(max_retries + 1):
        try:
            data = json.loads(response_text)
            decision = LLMDecisionV1(**data)
            return decision, attempt
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            if attempt >= max_retries:
                raise ValueError(
                    f"Response failed validation after {max_retries} retries. "
                    f"Last error: {exc}. Last response: {response_text[:200]!r}"
                ) from exc
            # Build a FORMAT FIX retry prompt
            prompt_for_retry = (
                prompt_for_retry
                + f"\n\n--- FORMAT FIX (attempt {attempt + 1}/{max_retries}) ---\n"
                f"Your previous response was invalid. Error: {exc}\n"
                "Output ONLY a single valid JSON object matching the required schema. "
                "No markdown, no explanation outside the JSON."
            )
            response_text = call_fn(prompt_for_retry)

    raise RuntimeError("Unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _call_anthropic_api(
    filled_prompt: str,
    model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    """Call the Anthropic API and return (response_text, usage_metadata).

    Requires the ``ANTHROPIC_API_KEY`` environment variable to be set.

    Args:
        filled_prompt: Fully substituted prompt string.
        model_id: Model identifier (e.g. "claude-sonnet-4-6").
        temperature: Sampling temperature.
        top_p: Top-p nucleus parameter.
        max_tokens: Maximum tokens in response.

    Returns:
        (response_text, usage) where usage = {"input_tokens": int, "output_tokens": int}.

    Raises:
        RuntimeError: if ANTHROPIC_API_KEY is not set.
        ImportError: if the ``anthropic`` package is not installed.
    """
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set.\n"
            "Either set it (export ANTHROPIC_API_KEY=sk-ant-...) or switch\n"
            "llm.backend to 'requests_only' in run_config.yaml."
        )

    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "Package 'anthropic' is required for the anthropic_api backend.\n"
            "Install it with: pip install anthropic"
        ) from exc

    client = anthropic.Anthropic(api_key=api_key)

    # The Anthropic API rejects requests that set BOTH temperature and top_p.
    # We always use temperature for deterministic output; top_p is omitted.
    msg = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": filled_prompt}],
    )
    usage = {
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
    }
    return _extract_response_text(msg), usage


def _extract_response_text(msg: Any) -> str:
    """Safely extract text from an Anthropic SDK message response.

    Raises ValueError("empty_response_content") instead of IndexError
    when the content list is empty or contains no text blocks.
    """
    if not msg.content:
        raise ValueError("empty_response_content")
    text_parts = [block.text for block in msg.content if hasattr(block, "text")]
    result = "".join(text_parts).strip()
    if not result:
        raise ValueError("empty_response_content")
    return result


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def write_verification_diagnostics(
    manifests_dir: str | Path,
    llm_cfg: dict[str, Any],
    smoke_test: bool,
    smoke_n: int,
) -> Path:
    """Write runtime diagnostics to manifests/anthropic_diagnostics.json.

    Prints a safe summary (never prints the API key).

    Returns path to the written file.
    """
    import os

    manifests_dir = Path(manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
    model_id: str = llm_cfg.get("model_id", "claude-sonnet-4-6")
    backend: str = llm_cfg.get("backend", "requests_only")

    try:
        import anthropic as _a
        sdk_version: str = getattr(_a, "__version__", "unknown")
    except ImportError:
        sdk_version = "NOT_INSTALLED"

    diagnostics: dict[str, Any] = {
        "timestamp_iso":          _now_iso(),
        "anthropic_key_set":      key_set,
        "model_id":               model_id,
        "backend":                backend,
        "anthropic_sdk_version":  sdk_version,
        "smoke_test":             smoke_test,
        "smoke_pairs_per_task":   smoke_n,
    }

    out_path = manifests_dir / "anthropic_diagnostics.json"
    out_path.write_text(
        json.dumps(diagnostics, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"  ANTHROPIC_API_KEY  : {'SET' if key_set else 'MISSING'}")
    print(f"  model_id           : {model_id}")
    print(f"  anthropic SDK      : {sdk_version}")
    print(f"  smoke_test         : {smoke_test} (pairs/task={smoke_n})")
    print(f"  diagnostics        : {out_path}")

    return out_path


# ---------------------------------------------------------------------------
# Core verification loop
# ---------------------------------------------------------------------------

_FAIL_FAST_CONSECUTIVE = 3   # stop after this many consecutive API errors


def run_verification(
    pairs_df: pd.DataFrame,
    task: str,
    evidence_fn: Callable[[str, str], dict[str, Any]],
    config: dict[str, Any],
    run_id: str,
    logs_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify a set of candidate pairs using the configured LLM backend.

    For each pair in ``pairs_df``:
      1. Builds an evidence card via ``evidence_fn(anchor_id, candidate_id)``.
      2. Serialises a request envelope via ``make_request()``.
      3. Writes the request to ``llm_requests_{task}.jsonl`` (always).
      4. For ``requests_only``: records a stub decision and moves on.
         For ``anthropic_api``: calls the model, validates, retries on format errors.
      5. Writes the decision to ``llm_decisions_{task}.jsonl``.
      6. API errors are written to ``llm_errors_{task}.jsonl`` with structured
         classification.  After _FAIL_FAST_CONSECUTIVE consecutive errors the
         loop aborts with a clear message.

    Args:
        pairs_df: DataFrame with at least anchor_id, candidate_id, task columns.
        task: "AND" or "AIN".
        evidence_fn: Callable(anchor_id, candidate_id) -> evidence_card dict.
        config: Full pipeline config dict (top-level from run_config.yaml).
        run_id: Current bem_run_id string.
        logs_dir: Path to runs/<run_id>/logs/.

    Returns:
        (decision_records, stats) where:
          decision_records — list of per-pair dicts (same structure as JSONL lines).
          stats — {"pairs_attempted", "decisions_written", "requests_written",
                   "errors", "fallback_count", "fail_fast_triggered"}.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    llm_cfg: dict[str, Any] = config.get("llm", {})
    model_id: str = llm_cfg.get("model_id", "claude-sonnet-4-6")
    temperature: float = float(llm_cfg.get("temperature", 0.0))
    top_p: float = float(llm_cfg.get("top_p", 1.0))
    max_tokens: int = int(llm_cfg.get("max_tokens", 800))
    backend: str = llm_cfg.get("backend", "requests_only")
    prompt_version: str = llm_cfg.get("output_schema_version", "v1")
    max_retries: int = int(
        llm_cfg.get("retry_policy", {}).get("max_retries", 2)
    )
    max_rpm: int = int(
        llm_cfg.get("rate_limit", {}).get("max_requests_per_minute", 60)
    )

    # Load task-specific prompt
    prompt_key = "prompt_and_path" if task == "AND" else "prompt_ain_path"
    prompt_path = llm_cfg.get(prompt_key, f"configs/prompts/{task.lower()}_verifier_v1.txt")
    prompt_text = load_prompt(prompt_path)

    # Rate limiter (only meaningful for anthropic_api)
    limiter = _RPMLimiter(max_rpm)

    # Log file paths
    req_path = logs_dir / f"llm_requests_{task.lower()}.jsonl"
    dec_path = logs_dir / f"llm_decisions_{task.lower()}.jsonl"
    err_path = logs_dir / f"llm_errors_{task.lower()}.jsonl"

    # --- Resume: load already-decided (anchor_id, candidate_id) pairs ---
    # Only pairs with a real (non-None) decision are skipped.
    # Error records (decision=None) are retried so transient failures are healed.
    already_decided: set[tuple[str, str]] = set()
    if dec_path.exists():
        with open(dec_path, encoding="utf-8") as _fh:
            for _line in _fh:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _obj = json.loads(_line)
                    _a   = _obj.get("anchor_id", "")
                    _c   = _obj.get("candidate_id", "")
                    _dec = _obj.get("decision")
                    # Only count as decided if there is a real decision object
                    if _a and _c and _dec is not None:
                        already_decided.add((_a, _c))
                except Exception:
                    pass

    # Retroactive healing: INVALID_RESPONSE errors from prior runs are permanently
    # unsolvable (empty API response).  Write a forced-uncertain decision for each
    # one that isn't already in the decisions file, then add to already_decided so
    # the pair loop skips it entirely.
    _to_heal: list[tuple[str, str]] = []
    if err_path.exists():
        with open(err_path, encoding="utf-8") as _efh:
            for _line in _efh:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _obj = json.loads(_line)
                    if _obj.get("classification") == "INVALID_RESPONSE":
                        _a = _obj.get("anchor_id", "")
                        _c = _obj.get("candidate_id", "")
                        if _a and _c and (_a, _c) not in already_decided:
                            _to_heal.append((_a, _c))
                            already_decided.add((_a, _c))
                except Exception:
                    pass

    if _to_heal:
        _forced_dec = LLMDecisionV1(
            label="uncertain",
            confidence=0.0,
            evidence_used=[],
            reason_code="invalid_response",
            abstention_reason="empty_response_content_healed_on_resume",
        ).model_dump()
        with open(dec_path, "a", encoding="utf-8") as _dhfh:
            for _a, _c in _to_heal:
                _write_jsonl(_dhfh, {
                    "run_id":       run_id,
                    "task":         task,
                    "anchor_id":    _a,
                    "candidate_id": _c,
                    "backend":      backend,
                    "model_id":     model_id,
                    "decision":     _forced_dec,
                    "timestamp":    _now_iso(),
                })
        print(f"  [{task}] Healed {len(_to_heal)} prior INVALID_RESPONSE error(s) as forced-uncertain.")

    n_skipped = len(already_decided)
    if n_skipped:
        print(f"  [{task}] Resume: skipping {n_skipped} already-decided pairs.")

    # Append to existing logs when resuming; start fresh otherwise
    file_mode = "a" if already_decided else "w"

    decision_records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "pairs_attempted":    0,
        "decisions_written":  0,
        "requests_written":   0,
        "errors":             0,
        "fallback_count":     0,
        "fail_fast_triggered": False,
        "billing_limit_hit":  False,
        "skipped_already_decided": n_skipped,
    }

    consecutive_errors = 0

    with (
        open(req_path, file_mode, encoding="utf-8") as req_fh,
        open(dec_path, file_mode, encoding="utf-8") as dec_fh,
        open(err_path, file_mode, encoding="utf-8") as err_fh,
    ):
        for _, row in pairs_df.iterrows():
            anchor_id    = str(row["anchor_id"])
            candidate_id = str(row["candidate_id"])

            # Skip pairs already decided in a previous run
            if (anchor_id, candidate_id) in already_decided:
                continue

            stats["pairs_attempted"] += 1

            # -- Build evidence card --
            try:
                evidence_card = evidence_fn(anchor_id, candidate_id)
            except Exception as exc:
                stats["errors"] += 1
                consecutive_errors += 1
                err_rec = _make_error_record(
                    run_id=run_id, task=task, backend=backend,
                    model_id=model_id, anchor_id=anchor_id,
                    candidate_id=candidate_id, exc=exc,
                    phase="evidence_build",
                )
                _write_jsonl(err_fh, err_rec)
                _write_jsonl(dec_fh, {**err_rec, "decision": None})
                if consecutive_errors >= _FAIL_FAST_CONSECUTIVE:
                    stats["fail_fast_triggered"] = True
                    break
                continue

            # Track linked_authors_fallback for stats
            for side in ("anchor", "candidate"):
                if evidence_card.get(side, {}).get("linked_authors_fallback", False):
                    stats["fallback_count"] += 1

            # -- Payload size check + defensive truncation --
            evidence_card, was_truncated = check_and_truncate_evidence(evidence_card)
            if was_truncated:
                _write_jsonl(err_fh, {
                    "run_id": run_id, "task": task,
                    "anchor_id": anchor_id, "candidate_id": candidate_id,
                    "error_type": "INVALID_REQUEST",
                    "classification": "INVALID_REQUEST",
                    "error_message": (
                        "Evidence card exceeded size threshold; "
                        f"strings truncated to {_TRUNCATE_STR_CHARS} chars."
                    ),
                    "phase": "payload_truncation",
                    "timestamp": _now_iso(),
                })

            # -- Build request envelope --
            req = make_request(
                task=task,
                evidence_card=evidence_card,
                prompt_text=prompt_text,
                model_id=model_id,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                prompt_version=prompt_version,
            )
            _write_jsonl(req_fh, {
                "run_id": run_id,
                "task": task,
                "anchor_id": anchor_id,
                "candidate_id": candidate_id,
                **{k: v for k, v in req.items() if k != "evidence_card"},
                "timestamp": _now_iso(),
            })
            stats["requests_written"] += 1

            # Warn if prompt is unusually large
            if req["prompt_chars"] > _WARN_PROMPT_CHARS:
                _write_jsonl(err_fh, {
                    "run_id": run_id, "task": task,
                    "anchor_id": anchor_id, "candidate_id": candidate_id,
                    "classification": "INVALID_REQUEST",
                    "error_message": (
                        f"Prompt is very large ({req['prompt_chars']} chars); "
                        "may be rejected by the API."
                    ),
                    "phase": "payload_size_warn",
                    "timestamp": _now_iso(),
                })

            # -- Call backend --
            retry_count = 0
            usage: dict[str, Any] = {}
            decision: LLMDecisionV1

            if backend == "requests_only":
                decision = _STUB_DECISION
                consecutive_errors = 0  # reset on success
            else:
                # anthropic_api
                limiter.wait_if_needed()
                try:
                    raw_response, usage = _call_anthropic_api(
                        filled_prompt=req["filled_prompt"],
                        model_id=model_id,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                    )

                    def _retry_call(retry_prompt: str) -> str:
                        limiter.wait_if_needed()
                        text, _ = _call_anthropic_api(
                            retry_prompt, model_id, temperature, top_p, max_tokens
                        )
                        return text

                    decision, retry_count = validate_or_retry(
                        initial_response=raw_response,
                        call_fn=_retry_call,
                        original_prompt=req["filled_prompt"],
                        max_retries=max_retries,
                    )
                    consecutive_errors = 0  # reset on success

                except Exception as exc:
                    stats["errors"] += 1
                    err_rec = _make_error_record(
                        run_id=run_id, task=task, backend=backend,
                        model_id=model_id, anchor_id=anchor_id,
                        candidate_id=candidate_id, exc=exc,
                        phase="api_call",
                        retry_count=retry_count,
                        prompt_chars=req["prompt_chars"],
                        evidence_json_chars=req["evidence_json_chars"],
                    )
                    _write_jsonl(err_fh, err_rec)

                    if err_rec["classification"] == "INVALID_RESPONSE":
                        # Empty/malformed response: write a forced-uncertain decision
                        # so the resume logic treats this pair as already decided and
                        # never retries it.
                        forced = LLMDecisionV1(
                            label="uncertain",
                            confidence=0.0,
                            evidence_used=[],
                            reason_code="invalid_response",
                            abstention_reason="empty_response_content",
                        )
                        _write_jsonl(dec_fh, {
                            "run_id":         run_id,
                            "task":           task,
                            "anchor_id":      anchor_id,
                            "candidate_id":   candidate_id,
                            "prompt_version": prompt_version,
                            "model_id":       model_id,
                            "temperature":    temperature,
                            "max_tokens":     max_tokens,
                            "decision":       forced.model_dump(),
                            "retry_count":    retry_count,
                            "backend":        backend,
                            "usage":          {},
                            "timestamp":      _now_iso(),
                        })
                        stats["decisions_written"] += 1
                        # Not counted as a consecutive error — pair is settled.
                    elif err_rec["classification"] == "BILLING":
                        # Usage/billing limit: stop immediately without burning
                        # retries.  The pair is left with decision=None so the
                        # resume logic will retry it when the limit resets.
                        _write_jsonl(dec_fh, {**err_rec, "decision": None})
                        stats["billing_limit_hit"] = True
                        stats["fail_fast_triggered"] = True
                        break
                    else:
                        # Transient/infrastructure error: write decision=None so
                        # the resume logic retries this pair next run.
                        _write_jsonl(dec_fh, {**err_rec, "decision": None})
                        consecutive_errors += 1
                        if consecutive_errors >= _FAIL_FAST_CONSECUTIVE:
                            stats["fail_fast_triggered"] = True
                            break

                    continue

            # -- Write decision --
            dec_record: dict[str, Any] = {
                "run_id":         run_id,
                "task":           task,
                "anchor_id":      anchor_id,
                "candidate_id":   candidate_id,
                "prompt_version": prompt_version,
                "model_id":       model_id,
                "temperature":    temperature,
                "top_p":          top_p,
                "max_tokens":     max_tokens,
                "decision":       decision.model_dump(),
                "retry_count":    retry_count,
                "backend":        backend,
                "usage":          usage,
                "timestamp":      _now_iso(),
            }
            _write_jsonl(dec_fh, dec_record)
            decision_records.append(dec_record)
            stats["decisions_written"] += 1

    return decision_records, stats


# ---------------------------------------------------------------------------
# Error record builder
# ---------------------------------------------------------------------------

def _make_error_record(
    run_id: str,
    task: str,
    backend: str,
    model_id: str,
    anchor_id: str,
    candidate_id: str,
    exc: Exception,
    phase: str,
    retry_count: int = 0,
    prompt_chars: int = 0,
    evidence_json_chars: int = 0,
) -> dict[str, Any]:
    classification, http_status, request_id = _classify_error(exc)
    return {
        "run_id":              run_id,
        "task":                task,
        "anchor_id":           anchor_id,
        "candidate_id":        candidate_id,
        "backend":             backend,
        "model_id":            model_id,
        "phase":               phase,
        "classification":      classification,
        "error_type":          type(exc).__name__,
        "error_message":       str(exc),
        "http_status":         http_status,
        "request_id":          request_id,
        "retry_count":         retry_count,
        "prompt_chars":        prompt_chars,
        "evidence_json_chars": evidence_json_chars,
        "timestamp":           _now_iso(),
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _write_jsonl(fh: Any, obj: dict[str, Any]) -> None:
    """Write a single JSON object as one line to an open file handle."""
    fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
