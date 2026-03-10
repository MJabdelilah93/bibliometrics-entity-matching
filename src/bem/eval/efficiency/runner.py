"""runner.py -- Stage E7 efficiency sub-runner.

Produces:

  Supplement-ready (outputs/efficiency/):
    token_detail_{task}.csv        per-task token summary
    cost_summary.csv               aggregated cost (all tasks + total)
    runtime_summary.csv            available timing data + what is NOT captured
    aggregation_diagnostics.csv    cluster count, largest cluster, conflicts
    environment.json               Python/model/SDK/platform details
    efficiency_manifest.json       provenance

  Manuscript-ready (outputs/manuscript/):
    table_efficiency_tokens.{csv,md,tex}       token and cost summary
    table_efficiency_comparison.{csv,md,tex}   BEM vs strongest baseline

FRAMING CONSTRAINTS (enforced in table captions and manifests):
  - All efficiency claims must be framed as snapshot- and environment-dependent.
  - Cost is estimated from per-decision token usage and configurable pricing.
    It is NOT the invoiced amount.
  - Runtime (C5 only) is derived from decision timestamps, NOT wall-clock.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bem.eval.config import EvalConfig
from bem.eval.efficiency.token_cost import (
    build_runtime_summary,
    compute_cost,
    format_comparison_latex,
    format_comparison_markdown,
    format_token_table_latex,
    format_token_table_markdown,
    load_aggregation_diagnostics,
    load_environment,
    load_token_usage,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_efficiency(cfg: EvalConfig) -> dict[str, Any]:
    """Compute and write efficiency summary.

    Returns:
        dict with keys:
            token_usages      -- {task: usage_dict}
            costs             -- {task: cost_dict}
            aggregation       -- {task: diag_dict}
            runtime_rows      -- list[dict] (runtime summary table)
            environment       -- dict
            written           -- {label: Path}
            manifest_path     -- Path of efficiency_manifest.json
    """
    rob_cfg  = cfg.rob_eff
    out_dir  = rob_cfg.efficiency_output_dir
    ms_dir   = cfg.evaluation.manuscript_dir
    run_dir  = cfg.bem_run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ms_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    # ------------------------------------------------------------------
    # 1. Token usage from llm_decisions_{task}.jsonl
    # ------------------------------------------------------------------
    token_usages: dict[str, Any] = {}
    costs:        dict[str, Any] = {}

    for task in cfg.tasks:
        jsonl_path = run_dir / "logs" / f"llm_decisions_{task}.jsonl"
        print(f"  [E7/eff/{task.upper()}] loading token usage from {jsonl_path.name} ...")
        usage = load_token_usage(jsonl_path)
        token_usages[task] = usage

        if usage.get("available"):
            print(
                f"    n_decisions={usage['n_decisions']:,}  "
                f"input_tok={usage['total_input_tokens']:,}  "
                f"output_tok={usage['total_output_tokens']:,}  "
                f"retries={usage['n_retries']}  "
                f"errors={usage['n_api_errors']}"
            )
        else:
            print(f"    [WARN] token usage not available: {usage.get('reason')}")

        # Cost estimate
        cost = compute_cost(
            usage,
            price_input_per_million=rob_cfg.pricing_input_per_million,
            price_output_per_million=rob_cfg.pricing_output_per_million,
            pricing_note=rob_cfg.pricing_model_note,
        )
        costs[task] = cost
        if cost.get("available"):
            print(
                f"    cost_total=${cost['cost_total_usd']:.2f}  "
                f"cost/1k_pairs=${cost['cost_per_1k_pairs_usd']:.2f}"
            )

    # ------------------------------------------------------------------
    # 2. Supplement: token_detail_{task}.csv + cost_summary.csv
    # ------------------------------------------------------------------
    token_rows = []
    cost_rows  = []

    for task in cfg.tasks:
        usage = token_usages[task]
        cost  = costs[task]

        # Token detail
        row = {
            "task":                   task.upper(),
            "n_decisions":            usage.get("n_decisions"),
            "n_with_usage":           usage.get("n_with_usage"),
            "n_stubs":                usage.get("n_stubs"),
            "n_retries":              usage.get("n_retries"),
            "n_api_errors":           usage.get("n_api_errors"),
            "total_input_tokens":     usage.get("total_input_tokens"),
            "total_output_tokens":    usage.get("total_output_tokens"),
            "total_tokens":           usage.get("total_tokens"),
            "avg_input_per_decision": usage.get("avg_input_per_decision"),
            "avg_output_per_decision":usage.get("avg_output_per_decision"),
            "duration_seconds":       usage.get("duration_seconds"),
            "timestamp_first":        usage.get("timestamp_first"),
            "timestamp_last":         usage.get("timestamp_last"),
            "model_ids":              "|".join(usage.get("model_ids") or []),
            "data_available":         usage.get("available", False),
        }
        token_rows.append(row)

        tok_csv = out_dir / f"token_detail_{task}.csv"
        pd.DataFrame([row]).to_csv(tok_csv, index=False)
        written[f"eff_token_detail_{task}_csv"] = tok_csv

        # Cost row
        cost_rows.append({
            "task":                       task.upper(),
            "n_pairs_priced":             cost.get("n_pairs_priced"),
            "price_input_per_million":    cost.get("price_input_per_million"),
            "price_output_per_million":   cost.get("price_output_per_million"),
            "cost_input_usd":             cost.get("cost_input_usd"),
            "cost_output_usd":            cost.get("cost_output_usd"),
            "cost_total_usd":             cost.get("cost_total_usd"),
            "cost_per_1k_pairs_usd":      cost.get("cost_per_1k_pairs_usd"),
            "tokens_per_1k_pairs":        cost.get("tokens_per_1k_pairs"),
            "input_tokens_per_1k_pairs":  cost.get("input_tokens_per_1k_pairs"),
            "output_tokens_per_1k_pairs": cost.get("output_tokens_per_1k_pairs"),
            "pricing_note":               cost.get("pricing_note"),
            "data_available":             cost.get("available", False),
        })

    # Combined total row
    total_in  = sum(
        u.get("total_input_tokens", 0) or 0
        for u in token_usages.values() if u.get("available")
    )
    total_out = sum(
        u.get("total_output_tokens", 0) or 0
        for u in token_usages.values() if u.get("available")
    )
    total_n = sum(
        u.get("n_with_usage", 0) or 0
        for u in token_usages.values() if u.get("available")
    )
    total_cost = sum(
        c.get("cost_total_usd", 0) or 0
        for c in costs.values() if c.get("available")
    )

    cost_rows.append({
        "task":                    "TOTAL",
        "n_pairs_priced":          total_n,
        "price_input_per_million": rob_cfg.pricing_input_per_million,
        "price_output_per_million":rob_cfg.pricing_output_per_million,
        "cost_input_usd":          total_in  / 1_000_000 * rob_cfg.pricing_input_per_million,
        "cost_output_usd":         total_out / 1_000_000 * rob_cfg.pricing_output_per_million,
        "cost_total_usd":          total_cost,
        "cost_per_1k_pairs_usd":   total_cost / total_n * 1000 if total_n > 0 else None,
        "tokens_per_1k_pairs":     (total_in + total_out) / total_n * 1000 if total_n > 0 else None,
        "input_tokens_per_1k_pairs":  total_in  / total_n * 1000 if total_n > 0 else None,
        "output_tokens_per_1k_pairs": total_out / total_n * 1000 if total_n > 0 else None,
        "pricing_note":            rob_cfg.pricing_model_note,
        "data_available":          True,
    })

    cost_csv = out_dir / "cost_summary.csv"
    pd.DataFrame(cost_rows).to_csv(cost_csv, index=False)
    written["eff_cost_summary_csv"] = cost_csv

    # ------------------------------------------------------------------
    # 3. Runtime summary
    # ------------------------------------------------------------------
    runtime_rows = build_runtime_summary(token_usages)
    runtime_csv  = out_dir / "runtime_summary.csv"
    pd.DataFrame(runtime_rows).to_csv(runtime_csv, index=False)
    written["eff_runtime_csv"] = runtime_csv

    # ------------------------------------------------------------------
    # 4. Aggregation diagnostics (C7 clusters + conflicts)
    # ------------------------------------------------------------------
    print("  [E7/eff] loading aggregation diagnostics ...")
    aggregation = load_aggregation_diagnostics(run_dir, cfg.tasks)
    agg_rows = []
    for task, diag in aggregation.items():
        agg_rows.append({
            "task":              task.upper(),
            "cluster_count":     diag.get("cluster_count"),
            "n_singletons":      diag.get("n_singletons"),
            "n_multi_clusters":  diag.get("n_multi_clusters"),
            "largest_cluster":   diag.get("largest_cluster"),
            "total_nodes":       diag.get("total_nodes"),
            "conflict_count":    diag.get("conflict_count"),
            "conflict_rate":     diag.get("conflict_rate"),
            "unresolved_queue":  diag.get("unresolved_queue"),
            "data_available":    diag.get("data_available"),
            "notes":             diag.get("notes"),
        })
        if diag.get("data_available"):
            print(
                f"    {task.upper()}: clusters={diag.get('cluster_count')}  "
                f"largest={diag.get('largest_cluster')}  "
                f"conflicts={diag.get('conflict_count')}  "
                f"unresolved={diag.get('unresolved_queue')}"
            )

    agg_csv = out_dir / "aggregation_diagnostics.csv"
    pd.DataFrame(agg_rows).to_csv(agg_csv, index=False)
    written["eff_aggregation_csv"] = agg_csv

    # ------------------------------------------------------------------
    # 5. Environment
    # ------------------------------------------------------------------
    diag_path = run_dir / "manifests" / "anthropic_diagnostics.json"
    environment = load_environment(diag_path)
    env_path = out_dir / "environment.json"
    env_path.write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    written["eff_environment_json"] = env_path

    # ------------------------------------------------------------------
    # 6. Manuscript tables: token/cost summary
    # ------------------------------------------------------------------
    ms_token_rows = _build_ms_token_rows(cfg.tasks, token_usages, costs)

    # Token table CSV
    ms_tok_csv = ms_dir / "table_efficiency_tokens.csv"
    pd.DataFrame(ms_token_rows).to_csv(ms_tok_csv, index=False)
    written["eff_ms_tok_csv"] = ms_tok_csv

    ms_tok_md = ms_dir / "table_efficiency_tokens.md"
    ms_tok_md.write_text(format_token_table_markdown(ms_token_rows), encoding="utf-8")
    written["eff_ms_tok_md"] = ms_tok_md

    ms_tok_tex = ms_dir / "table_efficiency_tokens.tex"
    ms_tok_tex.write_text(format_token_table_latex(ms_token_rows), encoding="utf-8")
    written["eff_ms_tok_tex"] = ms_tok_tex

    # ------------------------------------------------------------------
    # 7. Manuscript tables: efficiency comparison vs strongest baseline
    # ------------------------------------------------------------------
    metrics_master_path = cfg.evaluation.output_dir / "metrics_master.parquet"
    comp_rows = _build_comparison_rows(
        cfg.tasks, token_usages, costs, metrics_master_path,
        rob_cfg.strongest_baseline,
    )

    ms_cmp_csv = ms_dir / "table_efficiency_comparison.csv"
    pd.DataFrame(comp_rows).to_csv(ms_cmp_csv, index=False)
    written["eff_ms_cmp_csv"] = ms_cmp_csv

    ms_cmp_md = ms_dir / "table_efficiency_comparison.md"
    ms_cmp_md.write_text(format_comparison_markdown(comp_rows), encoding="utf-8")
    written["eff_ms_cmp_md"] = ms_cmp_md

    ms_cmp_tex = ms_dir / "table_efficiency_comparison.tex"
    ms_cmp_tex.write_text(format_comparison_latex(comp_rows), encoding="utf-8")
    written["eff_ms_cmp_tex"] = ms_cmp_tex

    # ------------------------------------------------------------------
    # 8. Manifest
    # ------------------------------------------------------------------
    manifest_path = _write_manifest(cfg, token_usages, costs, aggregation, written, out_dir)

    return {
        "token_usages":   token_usages,
        "costs":          costs,
        "aggregation":    aggregation,
        "runtime_rows":   runtime_rows,
        "environment":    environment,
        "written":        written,
        "manifest_path":  manifest_path,
    }


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_efficiency_summary(
    token_usages: dict,
    costs: dict,
    aggregation: dict,
    manifest_path: Path,
) -> None:
    print("\n" + "=" * 72)
    print("Stage E7 -- Robustness & Efficiency  EFFICIENCY SUMMARY")
    print("=" * 72)

    total_cost = 0.0
    total_tok  = 0

    for task in sorted(token_usages):
        u = token_usages[task]
        c = costs[task]
        if u.get("available"):
            tok = u.get("total_tokens", 0)
            total_tok += tok
            print(
                f"  {task.upper()}: "
                f"n={u['n_with_usage']:,}  "
                f"tok={tok:,}  "
                f"dur={_fmt_dur(u.get('duration_seconds'))}"
            )
        else:
            print(f"  {task.upper()}: token usage not available ({u.get('reason')})")
        if c.get("available"):
            total_cost += c.get("cost_total_usd", 0) or 0
            print(
                f"         cost=${c['cost_total_usd']:.2f}  "
                f"cost/1k=${c['cost_per_1k_pairs_usd']:.2f}"
            )

    print(f"\n  TOTAL cost     : ${total_cost:.2f}")
    print(f"  TOTAL tokens   : {total_tok:,}")
    print("  (Pricing note: snapshot-dependent -- see manifest for assumptions)")

    print("\n  Aggregation diagnostics:")
    for task, diag in aggregation.items():
        if diag.get("data_available"):
            print(
                f"    {task.upper()}: clusters={diag.get('cluster_count')}  "
                f"largest={diag.get('largest_cluster')}  "
                f"conflicts={diag.get('conflict_count')}  "
                f"unresolved={diag.get('unresolved_queue')}"
            )

    print(f"\n  Manifest: {manifest_path}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_ms_token_rows(
    tasks: list[str],
    token_usages: dict,
    costs: dict,
) -> list[dict]:
    """Build rows for the manuscript token/cost table."""
    rows = []
    for task in tasks:
        u = token_usages.get(task, {})
        c = costs.get(task, {})
        rows.append({
            "task":                      task.upper(),
            "n_with_usage":              u.get("n_with_usage"),
            "total_input_tokens":        u.get("total_input_tokens"),
            "total_output_tokens":       u.get("total_output_tokens"),
            "total_tokens":              u.get("total_tokens"),
            "tokens_per_1k_pairs":       c.get("tokens_per_1k_pairs"),
            "input_tokens_per_1k_pairs": c.get("input_tokens_per_1k_pairs"),
            "cost_total_usd":            c.get("cost_total_usd"),
            "cost_per_1k_pairs_usd":     c.get("cost_per_1k_pairs_usd"),
            "data_available":            u.get("available", False),
        })
    return rows


def _build_comparison_rows(
    tasks: list[str],
    token_usages: dict,
    costs: dict,
    metrics_master_path: Path,
    strongest_baseline_cfg: str,
) -> list[dict]:
    """Build rows for the BEM vs strongest baseline comparison table."""
    rows = []

    if not metrics_master_path.exists():
        print(f"  [E7/eff] metrics master not found: {metrics_master_path}")
        return rows

    try:
        master = pd.read_parquet(metrics_master_path)
    except Exception as exc:
        print(f"  [E7/eff] could not load metrics master: {exc}")
        return rows

    for task in tasks:
        task_df = master[master["task"] == task]

        # BEM row (system == 'bem')
        bem_rows = task_df[task_df["system"] == "bem"]
        if bem_rows.empty:
            continue
        bem = bem_rows.iloc[0]

        c = costs.get(task, {})
        u = token_usages.get(task, {})

        rows.append({
            "task":                  task.upper(),
            "system":                "bem",
            "display_name":          "BEM",
            "f1_match":              bem.get("f1_match"),
            "coverage":              bem.get("coverage"),
            "cost_per_1k_pairs_usd": c.get("cost_per_1k_pairs_usd"),
            "tokens_per_1k_pairs":   c.get("tokens_per_1k_pairs"),
            "note":                  "LLM-based; cost/tokens are API estimates",
        })

        # Strongest non-LLM, non-auxiliary baseline
        baseline_df = task_df[
            (task_df["system"] == "baseline") &
            (~task_df.get("is_auxiliary", pd.Series([False]*len(task_df))).fillna(False))
        ]
        if baseline_df.empty:
            continue

        if strongest_baseline_cfg.lower() == "auto":
            # Pick by highest f1_match; break ties by display_name alphabetically
            best_idx = baseline_df["f1_match"].idxmax()
            best_row = baseline_df.loc[best_idx]
        else:
            # Filter by display_name matching the config string
            match = baseline_df[
                baseline_df["display_name"].str.lower()
                == strongest_baseline_cfg.lower()
            ]
            best_row = match.iloc[0] if not match.empty else baseline_df.iloc[0]

        rows.append({
            "task":                  task.upper(),
            "system":                "baseline",
            "display_name":          best_row.get("display_name", "--"),
            "f1_match":              best_row.get("f1_match"),
            "coverage":              best_row.get("coverage"),
            "cost_per_1k_pairs_usd": None,   # non-LLM: no API cost
            "tokens_per_1k_pairs":   None,   # non-LLM: no tokens
            "note":                  "Non-LLM baseline; zero API cost",
        })

    return rows


def _write_manifest(
    cfg: EvalConfig,
    token_usages: dict,
    costs: dict,
    aggregation: dict,
    written: dict,
    out_dir: Path,
) -> Path:
    rob_cfg = cfg.rob_eff
    file_hashes = {}
    for label, path in written.items():
        if isinstance(path, Path) and path.exists():
            file_hashes[label] = _sha256(path)

    # Summarise availability
    availability = {
        task: {
            "token_usage": token_usages.get(task, {}).get("available", False),
            "cost":        costs.get(task, {}).get("available", False),
            "aggregation": aggregation.get(task, {}).get("data_available", False),
        }
        for task in cfg.tasks
    }

    manifest = {
        "stage":             "E7_efficiency",
        "status":            "completed",
        "timestamp":         datetime.now(tz=timezone.utc).isoformat(),
        "tasks":             cfg.tasks,
        "bem_run_dir":       str(cfg.bem_run_dir),
        "pricing_assumptions": {
            "input_cost_per_million":  rob_cfg.pricing_input_per_million,
            "output_cost_per_million": rob_cfg.pricing_output_per_million,
            "model_note":              rob_cfg.pricing_model_note,
            "framing_note": (
                "Cost estimates are snapshot- and environment-dependent. "
                "They are derived from per-decision token usage fields in the "
                "llm_decisions JSONL and the pricing rates above. "
                "Verify official pricing before citing in any publication."
            ),
        },
        "runtime_note": (
            "C5 API call duration is derived from the span between first and "
            "last decision timestamp. This is NOT true wall-clock time for the "
            "full pipeline. Per-stage wall-clock timing was not captured in "
            "this run."
        ),
        "data_availability": availability,
        "output_dir":        str(out_dir),
        "output_files":      file_hashes,
    }
    manifest_path = out_dir / "efficiency_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest_path


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _fmt_dur(sec) -> str:
    if sec is None:
        return "--"
    h = int(sec) // 3600
    m = (int(sec) % 3600) // 60
    s = int(sec) % 60
    return f"{h:02d}h{m:02d}m{s:02d}s"
