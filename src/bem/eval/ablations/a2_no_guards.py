"""a2_no_guards.py -- Ablation A2: full BEM vs same verifier without C6 guards.

Reads the benchmark-filtered routing log (which contains the raw C5 LLM label
alongside the C6 label_final) and re-derives predictions using the C5 label
directly -- no signal-count check, no confidence threshold gate.

Two sub-variants are reported:

  c5_direct  -- raw C5 label used as final prediction:
               'match' -> 'match', 'non-match' -> 'non-match',
               'uncertain'/'error' -> 'uncertain'

  c5_conf    -- C5 label gated only by a confidence threshold (no signal count):
               same mapping but 'match' only accepted when
               confidence >= t_match; 'non-match' only accepted when
               confidence >= t_nonmatch.  Below-threshold decisions become
               'uncertain'.  Uses the default C6 thresholds loaded from the
               thresholds manifest for each task.

Comparison: full BEM (from routing_log_bm label_final) is the reference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bem.eval.evaluation.metrics import compute_all_metrics


# Default guard thresholds when the manifest cannot be located.
_FALLBACK_T = {"t_match": 0.85, "t_nonmatch": 0.85}


def run_a2(
    routing_log_and: Path,
    routing_log_ain: Path,
    thresholds_manifest: Path,
    tasks: list[str],
    rng: np.random.Generator,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    output_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run ablation A2 for all requested tasks.

    Args:
        routing_log_and: Path to routing_log_and_bm.parquet.
        routing_log_ain: Path to routing_log_ain_bm.parquet.
        thresholds_manifest: Path to thresholds_tuned_dev.json (C6 manifest).
        tasks: List of task names, e.g. ['and', 'ain'].
        rng: Seeded random generator.
        n_bootstrap: Bootstrap resamples.
        alpha: CI significance level.
        output_dir: If provided, results are cached as metrics_a2_{task}.json.
        force: Overwrite existing outputs.

    Returns:
        Dict keyed by task -> list of result dicts (one per variant).
    """
    c6_thresholds = _load_c6_thresholds(thresholds_manifest)
    results: dict[str, list[dict]] = {}

    log_paths = {"and": routing_log_and, "ain": routing_log_ain}

    for task in tasks:
        log_path = log_paths[task]
        if not log_path.exists():
            print(f"  [A2/{task.upper()}] routing log not found -- skipping: {log_path}")
            continue

        cache_path = (output_dir / f"metrics_a2_{task}.json") if output_dir else None
        if cache_path and cache_path.exists() and not force:
            print(f"  [A2/{task.upper()}] loading cached results from {cache_path}")
            results[task] = json.loads(cache_path.read_text(encoding="utf-8"))
            continue

        df = pd.read_parquet(log_path)
        print(f"  [A2/{task.upper()}] {len(df)} benchmark pairs loaded")

        gold = df["gold_label"].astype(str)
        t_cfg = c6_thresholds.get(task, _FALLBACK_T)
        t_match    = float(t_cfg.get("t_match",    0.85))
        t_nonmatch = float(t_cfg.get("t_nonmatch", 0.85))

        task_results: list[dict] = []

        # --- variant: c5_direct (C5 label, no threshold, no signal check) ---
        pred_c5 = _map_c5_label(df["label"].astype(str))
        m_c5 = compute_all_metrics(gold, pred_c5, rng, n_bootstrap, alpha)
        task_results.append({
            "ablation":      "a2_no_guards",
            "variant":       "c5_direct",
            "task":          task,
            "description":   "C5 LLM label, no threshold, no signal-count gate",
            "metrics":       _serialise(m_c5),
        })
        _print_variant(task, "c5_direct", m_c5)

        # --- variant: c5_conf (confidence threshold, no signal count) ---
        confidence = df["confidence"].astype(float)
        raw_label  = df["label"].astype(str)
        pred_conf  = _map_c5_conf(raw_label, confidence, t_match, t_nonmatch)
        m_conf = compute_all_metrics(gold, pred_conf, rng, n_bootstrap, alpha)
        task_results.append({
            "ablation":      "a2_no_guards",
            "variant":       "c5_conf",
            "task":          task,
            "description":   (
                f"C5 label gated by confidence only "
                f"(t_match={t_match}, t_nonmatch={t_nonmatch}); no signal-count check"
            ),
            "t_match":       t_match,
            "t_nonmatch":    t_nonmatch,
            "metrics":       _serialise(m_conf),
        })
        _print_variant(task, "c5_conf", m_conf)

        # --- reference: full BEM (label_final from routing log) ---
        pred_bem = _map_bem_label(df["label_final"].astype(str))
        m_bem = compute_all_metrics(gold, pred_bem, rng, n_bootstrap, alpha)
        task_results.append({
            "ablation":      "a2_no_guards",
            "variant":       "bem_full",
            "task":          task,
            "description":   "Full BEM (C5 + C6 guards)",
            "metrics":       _serialise(m_bem),
        })
        _print_variant(task, "bem_full (reference)", m_bem)

        results[task] = task_results

        if output_dir and cache_path:
            output_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(task_results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    return results


# ---------------------------------------------------------------------------
# Prediction mapping helpers
# ---------------------------------------------------------------------------

def _map_c5_label(labels: pd.Series) -> pd.Series:
    """Map raw C5 label to evaluation three-class label (no threshold)."""
    def _map(lbl: str) -> str:
        if lbl == "match":
            return "match"
        if lbl == "non-match":
            return "non-match"
        return "uncertain"
    return labels.map(_map)


def _map_c5_conf(
    labels: pd.Series,
    confidence: pd.Series,
    t_match: float,
    t_nonmatch: float,
) -> pd.Series:
    """Map C5 label to three-class label gated by confidence threshold only."""
    result = []
    for lbl, conf in zip(labels, confidence):
        if lbl == "match" and conf >= t_match:
            result.append("match")
        elif lbl == "non-match" and conf >= t_nonmatch:
            result.append("non-match")
        else:
            result.append("uncertain")
    return pd.Series(result, dtype=str)


def _map_bem_label(label_final: pd.Series) -> pd.Series:
    """Map C6 label_final to three-class evaluation label."""
    def _map(lbl: str) -> str:
        if lbl == "match":
            return "match"
        if lbl == "non-match":
            return "non-match"
        return "uncertain"
    return label_final.map(_map)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_c6_thresholds(manifest_path: Path) -> dict[str, dict[str, float]]:
    """Load per-task t_match / t_nonmatch from the C6 thresholds manifest."""
    if not manifest_path.exists():
        print(f"  [A2] thresholds manifest not found ({manifest_path}); using defaults")
        return {}
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {}
    for task in ("and", "ain"):
        t_match    = raw.get(f"t_match_{task}",    raw.get("t_match",    0.85))
        t_nonmatch = raw.get(f"t_nonmatch_{task}", raw.get("t_nonmatch", 0.85))
        out[task] = {"t_match": float(t_match), "t_nonmatch": float(t_nonmatch)}
    return out


def _serialise(m: dict) -> dict:
    """Convert numpy scalars and tuples to JSON-safe Python types."""
    out = {}
    for k, v in m.items():
        if isinstance(v, tuple):
            out[k] = [float(x) if isinstance(x, (float, np.floating)) else x for x in v]
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            out[k] = None if np.isnan(v) else float(v)
        else:
            out[k] = v
    return out


def _print_variant(task: str, variant: str, m: dict) -> None:
    prec = m.get("precision_match", float("nan"))
    rec  = m.get("recall_match",    float("nan"))
    f1   = m.get("f1_match",        float("nan"))
    cov  = m.get("coverage",        float("nan"))
    ci   = m.get("ci_f1_match", (float("nan"), float("nan")))
    ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "--"
    print(
        f"    {task.upper()} / {variant:30s}  "
        f"Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f} {ci_str}  Cov={cov:.3f}"
    )
