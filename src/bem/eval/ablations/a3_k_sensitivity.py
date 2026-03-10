"""a3_k_sensitivity.py -- Ablation A3: K sensitivity at K = 10, 25, 50.

Measures how blocking recall degrades as the per-anchor candidate cap K
decreases.  The candidates parquet produced by C4 already contains a rank
column that gives each pair its 1-indexed position within its anchor s
candidate list, ordered by pass priority (higher-quality blocking passes
receive lower ranks).

A benchmark pair "survives at K" when rank <= K.

Metrics reported per (task, K):
  blocking_recall_match : fraction of gold-match benchmark pairs that survive
  blocking_recall_all   : fraction of ALL benchmark pairs that survive
  n_match_survived      : absolute count of surviving match pairs
  bem_f1_at_k           : BEM F1 restricted to surviving pairs
    (pairs dropped by K -> treated as abstained = "uncertain")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bem.eval.evaluation.metrics import compute_all_metrics


def run_a3(
    candidates_and: Path,
    candidates_ain: Path,
    routing_log_and: Path,
    routing_log_ain: Path,
    benchmark_and: Path,
    benchmark_ain: Path,
    tasks: list[str],
    k_values: list[int],
    rng: np.random.Generator,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    output_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    cand_paths  = {"and": candidates_and,  "ain": candidates_ain}
    log_paths   = {"and": routing_log_and, "ain": routing_log_ain}
    bench_paths = {"and": benchmark_and,   "ain": benchmark_ain}

    results: dict[str, list[dict]] = {}

    for task in tasks:
        cand_path  = cand_paths[task]
        log_path   = log_paths[task]
        bench_path = bench_paths[task]

        cache_path = (output_dir / f"k_sensitivity_{task}.json") if output_dir else None
        if cache_path and cache_path.exists() and not force:
            print(f"  [A3/{task.upper()}] loading cached results from {cache_path}")
            results[task] = json.loads(cache_path.read_text(encoding="utf-8"))
            continue

        missing = [p for p in [cand_path, log_path, bench_path] if not p.exists()]
        if missing:
            print(f"  [A3/{task.upper()}] required files missing -- skipping: {missing}")
            continue

        print(f"  [A3/{task.upper()}] loading candidates (this may take a moment)...")
        cand_df = pd.read_parquet(cand_path, columns=["anchor_id", "candidate_id", "rank"])
        bm_df   = pd.read_parquet(bench_path)
        log_df  = pd.read_parquet(log_path)

        print(f"  [A3/{task.upper()}] {len(cand_df):,} candidates, {len(bm_df)} benchmark pairs")

        bm_anchors = set(bm_df["anchor_id"])
        cand_bm = cand_df[cand_df["anchor_id"].isin(bm_anchors)]

        merged = bm_df[["anchor_id", "candidate_id", "gold_label"]].merge(
            log_df[["anchor_id", "candidate_id", "label_final"]],
            on=["anchor_id", "candidate_id"],
            how="left",
        )
        merged["label_final"] = merged["label_final"].fillna("uncertain")

        merged = merged.merge(
            cand_bm[["anchor_id", "candidate_id", "rank"]],
            on=["anchor_id", "candidate_id"],
            how="left",
        )
        merged["rank"] = merged["rank"].fillna(float("inf"))

        task_results: list[dict] = []
        for k in sorted(k_values):
            row = _compute_at_k(k, merged, rng, n_bootstrap, alpha)
            row["task"] = task
            task_results.append(row)
            print(
                f"    {task.upper()} / K={k:3d}  "
                f"BR_match={row['blocking_recall_match']:.3f}  "
                f"BR_all={row['blocking_recall_all']:.3f}  "
                f"BEM_F1={row['bem_f1_at_k']:.3f}"
            )

        results[task] = task_results

        if output_dir and cache_path:
            output_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(task_results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    return results


def _compute_at_k(
    k: int,
    merged: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float,
) -> dict[str, Any]:
    survives     = merged["rank"] <= k
    is_match     = merged["gold_label"] == "match"
    n_match_tot  = int(is_match.sum())
    n_match_surv = int((is_match & survives).sum())
    br_match     = n_match_surv / n_match_tot if n_match_tot > 0 else 0.0

    n_all_tot  = len(merged)
    n_all_surv = int(survives.sum())
    br_all     = n_all_surv / n_all_tot if n_all_tot > 0 else 0.0

    pred_at_k = merged.apply(
        lambda r: _map_label(str(r["label_final"])) if r["rank"] <= k else "uncertain",
        axis=1,
    )
    m = compute_all_metrics(merged["gold_label"].astype(str), pred_at_k, rng, n_bootstrap, alpha)

    return {
        "ablation":              "a3_k_sensitivity",
        "k":                     k,
        "n_match_total":         n_match_tot,
        "n_match_survived":      n_match_surv,
        "blocking_recall_match": br_match,
        "n_all_total":           n_all_tot,
        "n_all_survived":        n_all_surv,
        "blocking_recall_all":   br_all,
        "bem_f1_at_k":           float(m.get("f1_match",        float("nan"))),
        "bem_precision_at_k":    float(m.get("precision_match", float("nan"))),
        "bem_recall_at_k":       float(m.get("recall_match",    float("nan"))),
        "bem_coverage_at_k":     float(m.get("coverage",        float("nan"))),
        "ci_f1_at_k":            list(m.get("ci_f1_match", [float("nan"), float("nan")])),
    }


def _map_label(lbl: str) -> str:
    return lbl if lbl in ("match", "non-match") else "uncertain"
