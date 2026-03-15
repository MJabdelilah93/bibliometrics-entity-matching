"""token_cost.py -- Token and cost extraction for BEM LLM runs (E7).

Reads llm_decisions_{task}.jsonl and extracts per-decision token usage,
timing, and retry/error accounting.

JSONL schema (per line):
  anchor_id, candidate_id, task, run_id
  timestamp           -- ISO 8601 UTC (used for duration estimates)
  backend             -- 'anthropic_api' | 'requests_only'
  model_id            -- e.g. 'claude-sonnet-4-6'
  temperature, top_p, max_tokens
  retry_count         -- 0 = first attempt succeeded
  usage               -- {input_tokens: int, output_tokens: int}
  decision            -- {label, confidence, evidence_used, reason_code, abstention_reason}

IMPORTANT framing:
  - Cost estimates are computed from usage fields + configurable pricing.
  - Prices are snapshot-dependent.  Verify against official pricing before
    citing in any paper.
  - If usage fields are absent (e.g. 'requests_only' backend), the function
    reports 'not_available' and does NOT invent values.
  - Wall-clock stage runtime is NOT captured in the JSONL.  Only the span
    between first and last decision timestamp is available as a lower bound.
  - Retry/cache/invalid-response accounting uses retry_count and decision.label.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Token usage loader
# ---------------------------------------------------------------------------

def load_token_usage(decisions_path: Path) -> dict[str, Any]:
    """Parse llm_decisions JSONL and return a usage summary dict.

    Returns a dict with keys:
        available         -- bool: whether usage data was found
        reason            -- str (only if not available)
        n_decisions       -- total lines parsed
        n_with_usage      -- lines with a usage field
        n_stubs           -- backend == 'requests_only' (no real API call)
        n_retries         -- decisions where retry_count > 0
        n_api_errors      -- decisions where label indicates API error
        total_input_tokens
        total_output_tokens
        total_tokens
        avg_input_per_decision
        avg_output_per_decision
        model_ids         -- set of model IDs seen
        timestamp_first   -- ISO string of first decision
        timestamp_last    -- ISO string of last decision
        duration_seconds  -- span from first to last decision (API time only)
        duration_note     -- explains what 'duration' means and does NOT cover
    """
    if not decisions_path.exists():
        return {
            "available": False,
            "reason": f"file not found: {decisions_path}",
        }

    input_tokens:  list[int] = []
    output_tokens: list[int] = []
    n_decisions    = 0
    n_stubs        = 0
    n_retries      = 0
    n_api_errors   = 0
    timestamps:    list[str] = []
    model_ids:     set[str]  = set()

    with open(decisions_path, encoding="utf-8") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            n_decisions += 1

            backend = obj.get("backend", "")
            if backend == "requests_only":
                n_stubs += 1
                # Stubs have no real usage
                continue

            # Usage
            usage = obj.get("usage") or {}
            inp = usage.get("input_tokens")
            out = usage.get("output_tokens")
            if inp is not None:
                input_tokens.append(int(inp))
            if out is not None:
                output_tokens.append(int(out))

            # Retries
            if int(obj.get("retry_count", 0)) > 0:
                n_retries += 1

            # API errors: decision.label is 'error' or reason_code starts with 'api_error'
            dec = obj.get("decision") or {}
            label = dec.get("label", "")
            reason_code = dec.get("reason_code", "")
            if label == "error" or str(reason_code).startswith("api_error"):
                n_api_errors += 1

            # Model ID
            mid = obj.get("model_id")
            if mid:
                model_ids.add(str(mid))

            # Timestamp
            ts = obj.get("timestamp")
            if ts:
                timestamps.append(str(ts))

    n_with_usage = len(input_tokens)

    if n_with_usage == 0 and n_stubs == n_decisions:
        return {
            "available": False,
            "reason": "all decisions are stub (requests_only backend); no real API calls made",
            "n_decisions": n_decisions,
            "n_stubs":     n_stubs,
        }
    if n_with_usage == 0:
        return {
            "available": False,
            "reason": "no usage fields found in any decision",
            "n_decisions": n_decisions,
        }

    total_in  = sum(input_tokens)
    total_out = sum(output_tokens)

    # Duration from timestamps
    ts_first = ts_last = None
    duration_seconds = None
    if timestamps:
        timestamps_sorted = sorted(timestamps)
        ts_first = timestamps_sorted[0]
        ts_last  = timestamps_sorted[-1]
        try:
            t0 = datetime.fromisoformat(ts_first)
            t1 = datetime.fromisoformat(ts_last)
            duration_seconds = (t1 - t0).total_seconds()
        except ValueError:
            pass

    return {
        "available":               True,
        "n_decisions":             n_decisions,
        "n_with_usage":            n_with_usage,
        "n_stubs":                 n_stubs,
        "n_retries":               n_retries,
        "n_api_errors":            n_api_errors,
        "total_input_tokens":      total_in,
        "total_output_tokens":     total_out,
        "total_tokens":            total_in + total_out,
        "avg_input_per_decision":  total_in  / n_with_usage,
        "avg_output_per_decision": total_out / n_with_usage,
        "model_ids":               sorted(model_ids),
        "timestamp_first":         ts_first,
        "timestamp_last":          ts_last,
        "duration_seconds":        duration_seconds,
        "duration_note": (
            "Span from first to last decision timestamp. "
            "Covers API call time only — excludes C4 blocking, C6 guard "
            "application, C7 clustering, and pipeline setup/teardown. "
            "True wall-clock requires instrumented stage-level timing "
            "(not captured in this run)."
        ),
    }


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------

def compute_cost(
    usage: dict[str, Any],
    price_input_per_million: float,
    price_output_per_million: float,
    pricing_note: str,
) -> dict[str, Any]:
    """Compute estimated cost from token usage.

    IMPORTANT: estimates are snapshot-dependent.  Verify pricing before citing.

    Returns:
        dict with cost fields, or a 'not_available' dict if usage is missing.
    """
    if not usage.get("available"):
        return {
            "available": False,
            "reason": usage.get("reason", "usage not available"),
        }

    total_in  = usage["total_input_tokens"]
    total_out = usage["total_output_tokens"]
    n_pairs   = usage["n_with_usage"]

    cost_input  = total_in  / 1_000_000 * price_input_per_million
    cost_output = total_out / 1_000_000 * price_output_per_million
    cost_total  = cost_input + cost_output

    per_1k = (cost_total / n_pairs * 1000) if n_pairs > 0 else None

    return {
        "available":                  True,
        "price_input_per_million":    price_input_per_million,
        "price_output_per_million":   price_output_per_million,
        "pricing_note":               pricing_note,
        "cost_input_usd":             cost_input,
        "cost_output_usd":            cost_output,
        "cost_total_usd":             cost_total,
        "n_pairs_priced":             n_pairs,
        "cost_per_1k_pairs_usd":      per_1k,
        "tokens_per_1k_pairs": (
            usage["total_tokens"] / n_pairs * 1000 if n_pairs > 0 else None
        ),
        "input_tokens_per_1k_pairs": (
            total_in / n_pairs * 1000 if n_pairs > 0 else None
        ),
        "output_tokens_per_1k_pairs": (
            total_out / n_pairs * 1000 if n_pairs > 0 else None
        ),
    }


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------

def load_environment(diagnostics_path: Path) -> dict[str, Any]:
    """Capture software/model environment details.

    Reads anthropic_diagnostics.json from the BEM run dir (if present) and
    augments with runtime Python/platform info.

    Fields collected:
        python_version, platform, hostname
        pandas_version, numpy_version
        model_id, anthropic_sdk_version, backend
        temperature, max_tokens
        run_timestamp (from diagnostics)
    """
    import platform
    import sys

    env: dict[str, Any] = {
        "python_version": sys.version,
        "platform":       platform.platform(),
        "hostname":       platform.node(),
    }

    try:
        import pandas as pd
        env["pandas_version"] = pd.__version__
    except ImportError:
        env["pandas_version"] = "not installed"

    try:
        import numpy as np
        env["numpy_version"] = np.__version__
    except ImportError:
        env["numpy_version"] = "not installed"

    if diagnostics_path.exists():
        try:
            diag = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            env.update({
                "model_id":              diag.get("model_id"),
                "anthropic_sdk_version": diag.get("anthropic_sdk_version"),
                "backend":               diag.get("backend"),
                "run_timestamp":         diag.get("timestamp_iso"),
            })
        except Exception as exc:
            env["diagnostics_error"] = str(exc)
    else:
        env["diagnostics_note"] = f"anthropic_diagnostics.json not found at {diagnostics_path}"

    return env


# ---------------------------------------------------------------------------
# Aggregation diagnostics (C7 clusters + conflicts)
# ---------------------------------------------------------------------------

def load_aggregation_diagnostics(run_dir: Path, tasks: list[str]) -> dict[str, Any]:
    """Load cluster and conflict CSVs from the BEM run outputs directory.

    Returns a dict keyed by task with sub-dicts containing:
        cluster_count       -- number of distinct entity clusters
        n_singletons        -- clusters of size 1 (self-referential or unlinked)
        n_multi_clusters    -- clusters with >= 2 members
        largest_cluster     -- size of largest cluster
        total_nodes         -- total nodes across all clusters
        conflict_count      -- number of conflicting edges
        conflict_rate       -- conflict_count / total_nodes (or None)
        unresolved_queue    -- conflict_count (unresolved = in conflict file)
        data_available      -- bool
        notes               -- explanatory string
    """
    outputs_dir = run_dir / "outputs"
    result: dict[str, Any] = {}

    for task in tasks:
        cluster_path  = outputs_dir / f"clusters_{task}_bm.csv"
        conflict_path = outputs_dir / f"conflicts_{task}_bm.csv"

        diag: dict[str, Any] = {
            "task": task,
            "data_available": False,
        }

        if not cluster_path.exists():
            diag["notes"] = f"cluster file not found: {cluster_path}"
            result[task] = diag
            continue

        try:
            import pandas as pd
            clusters  = pd.read_csv(cluster_path)
            conflicts = pd.read_csv(conflict_path) if conflict_path.exists() else pd.DataFrame()

            # cluster_count = number of unique entity_ids
            sizes = clusters.groupby("entity_id")["node_id"].count()
            n_clusters = len(sizes)
            largest    = int(sizes.max()) if n_clusters > 0 else 0
            singletons = int((sizes == 1).sum())
            multi      = int((sizes > 1).sum())
            total_nodes = int(len(clusters))

            conflict_count = len(conflicts)
            conflict_rate  = (
                conflict_count / total_nodes if total_nodes > 0 else None
            )

            diag.update({
                "data_available":    True,
                "cluster_count":     n_clusters,
                "n_singletons":      singletons,
                "n_multi_clusters":  multi,
                "largest_cluster":   largest,
                "total_nodes":       total_nodes,
                "conflict_count":    conflict_count,
                "conflict_rate":     conflict_rate,
                "unresolved_queue":  conflict_count,
                "notes": (
                    "Singletons = clusters with exactly 1 member (not yet "
                    "linked to any other instance). "
                    "Conflicts = edges in the conflict file (requires manual "
                    "adjudication). "
                    "Unresolved queue = conflict_count."
                ),
            })
        except Exception as exc:
            diag["notes"] = f"error loading aggregation files: {exc}"

        result[task] = diag

    return result


# ---------------------------------------------------------------------------
# Runtime summary helpers
# ---------------------------------------------------------------------------

def build_runtime_summary(token_usages: dict[str, dict]) -> list[dict]:
    """Build a runtime summary table from per-task token_usage dicts.

    IMPORTANT:
        - Only API call duration is available (timestamp span from JSONL).
        - Per-stage breakdown (C4/C6/C7) is NOT available in this run.
        - All duration values should be treated as lower bounds.
    """
    rows = []

    for task, usage in token_usages.items():
        if not usage.get("available"):
            rows.append({
                "task":             task.upper(),
                "stage":            "C5 (LLM verify)",
                "duration_sec":     None,
                "duration_min":     None,
                "n_pairs":          usage.get("n_decisions"),
                "sec_per_1k_pairs": None,
                "data_available":   False,
                "note":             usage.get("reason", "not available"),
            })
            continue

        dur = usage.get("duration_seconds")
        n   = usage.get("n_with_usage", 0)
        rows.append({
            "task":             task.upper(),
            "stage":            "C5 (LLM verify -- API call span)",
            "duration_sec":     dur,
            "duration_min":     dur / 60 if dur is not None else None,
            "n_pairs":          n,
            "sec_per_1k_pairs": (dur / n * 1000) if (dur and n) else None,
            "data_available":   True,
            "note": (
                "Span from first to last decision timestamp. "
                "Excludes C4/C6/C7 and pipeline overhead."
            ),
        })

    # Stages NOT captured
    for stage_label in (
        "C4 (blocking/candidate generation)",
        "C6 (guard application)",
        "C7 (clustering/union-find)",
    ):
        rows.append({
            "task":             "all",
            "stage":            stage_label,
            "duration_sec":     None,
            "duration_min":     None,
            "n_pairs":          None,
            "sec_per_1k_pairs": None,
            "data_available":   False,
            "note": (
                "Stage-level wall-clock timing not captured in this run. "
                "Requires instrumented pipeline with per-stage timing hooks."
            ),
        })

    return rows


# ---------------------------------------------------------------------------
# Table formatters
# ---------------------------------------------------------------------------

def format_token_table_markdown(rows: list[dict]) -> str:
    lines = [
        "## Efficiency: token and cost summary\n",
        "**Pricing note:** cost estimates are snapshot-dependent "
        "(see `pricing_model_note` in manifest). Verify before citing.\n",
        "| Task | N pairs | Input tok | Output tok | Total tok | "
        "Input tok/1k | Cost (USD) | Cost/1k pairs (USD) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('task', '--')} "
            f"| {_fmt_int(r.get('n_with_usage'))} "
            f"| {_fmt_int(r.get('total_input_tokens'))} "
            f"| {_fmt_int(r.get('total_output_tokens'))} "
            f"| {_fmt_int(r.get('total_tokens'))} "
            f"| {_fmt_big(r.get('input_tokens_per_1k_pairs'))} "
            f"| {_fmt_cost(r.get('cost_total_usd'))} "
            f"| {_fmt_cost(r.get('cost_per_1k_pairs_usd'))} |"
        )
    return "\n".join(lines) + "\n"


def format_token_table_latex(rows: list[dict]) -> str:
    header = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{BEM token and cost summary. "
        "Cost estimates are snapshot-dependent; "
        "see manuscript text for pricing assumptions.}\n"
        "\\label{tab:efficiency_tokens}\n"
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "Task & $N$ & Input tok & Output tok & Total tok/1k & "
        "Cost (USD) & Cost/1k (USD) \\\\\n"
        "\\midrule\n"
    )
    body = []
    for r in rows:
        body.append(
            f"{r.get('task', '--')} & "
            f"{_fmt_int(r.get('n_with_usage'))} & "
            f"{_fmt_int(r.get('total_input_tokens'))} & "
            f"{_fmt_int(r.get('total_output_tokens'))} & "
            f"{_fmt_big(r.get('tokens_per_1k_pairs'))} & "
            f"{_fmt_cost(r.get('cost_total_usd'))} & "
            f"{_fmt_cost(r.get('cost_per_1k_pairs_usd'))} \\\\"
        )
    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return header + "\n".join(body) + "\n" + footer


def format_comparison_markdown(rows: list[dict]) -> str:
    lines = [
        "## Efficiency: BEM vs strongest non-LLM baseline\n",
        "**Note:** cost/token columns apply to BEM only; non-LLM baselines "
        "incur zero API cost. Claims are environment- and snapshot-dependent.\n",
        "| Task | System | F1_M | Coverage | Cost/1k pairs (USD) | Tokens/1k pairs |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        cost = r.get("cost_per_1k_pairs_usd")
        tok  = r.get("tokens_per_1k_pairs")
        lines.append(
            f"| {r.get('task', '--')} "
            f"| {r.get('display_name', r.get('system', '--'))} "
            f"| {_fmt3(r.get('f1_match'))} "
            f"| {_fmt3(r.get('coverage'))} "
            f"| {_fmt_cost(cost) if cost is not None else '0 (no API)'} "
            f"| {_fmt_big(tok) if tok is not None else '0 (no LLM)'} |"
        )
    return "\n".join(lines) + "\n"


def format_comparison_latex(rows: list[dict]) -> str:
    header = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{BEM efficiency vs.~strongest non-LLM baseline. "
        "Cost applies to BEM only; baselines incur zero API cost. "
        "Claims are environment- and snapshot-dependent.}\n"
        "\\label{tab:efficiency_comparison}\n"
        "\\begin{tabular}{llrrrr}\n\\toprule\n"
        "Task & System & F1$_M$ & Coverage & Cost/1k (USD) & Tok/1k \\\\\n"
        "\\midrule\n"
    )
    body = []
    for r in rows:
        cost = r.get("cost_per_1k_pairs_usd")
        tok  = r.get("tokens_per_1k_pairs")
        body.append(
            f"{r.get('task', '--')} & "
            f"{_tex(str(r.get('display_name', r.get('system', '--'))))} & "
            f"{_fmt3(r.get('f1_match'))} & "
            f"{_fmt3(r.get('coverage'))} & "
            f"{_fmt_cost(cost) if cost is not None else '0'} & "
            f"{_fmt_big(tok) if tok is not None else '0'} \\\\"
        )
    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return header + "\n".join(body) + "\n" + footer


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt3(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{float(v):.3f}"


def _fmt_int(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{int(v):,}"


def _fmt_big(v) -> str:
    """Format a large float with comma separators and 0 decimals."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{float(v):,.0f}"


def _fmt_cost(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"${float(v):.2f}"


def _tex(s: str) -> str:
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
