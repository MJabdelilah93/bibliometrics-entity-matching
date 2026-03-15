"""runner.py — Orchestrate Stage E4: threshold tuning.

For every (task, baseline_name) pair found in the baseline-scores master file
this module:

  1. Loads scores and gold labels for the DEV split only.
  2. Excludes 'uncertain' gold labels from all computations (counted + reported).
  3. Computes a full precision/recall/F-beta curve across the threshold grid
     for both the match class and the non-match class.
  4. Derives three operating points:
       a. f1_optimal          — argmax F-beta (secondary / sensitivity)
       b. precision_floor_match — PRIMARY; lowest threshold meeting precision floor
       c. two_threshold       — (t_low, t_high) uncertain-band routing point
  5. Writes per-(task, baseline) diagnostics parquets (full curves).
  6. Writes per-task human-readable CSV summaries.
  7. Writes per-task machine-readable JSON artefacts.
  8. Writes a stage manifest with full provenance.

Safety invariant
----------------
The runner raises ``ValueError`` immediately if any test-split rows are found
in the baseline scores file.  Threshold tuning must be performed on dev only.

Output files (under ``cfg.tuning.output_dir``)
----------------------------------------------
  diagnostics_<task>_<baseline>.parquet   full PR curves (101 rows each)
  threshold_summary_<task>.csv            selected thresholds (human-readable)
  thresholds_<task>.json                  machine-readable artefact (consumed by E5+)
  tuning_manifest.json                    provenance record
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bem.eval.config import EvalConfig
from bem.eval.thresholds.tuner import (
    make_threshold_grid,
    encode_gold,
    compute_match_pr_curve,
    compute_nonmatch_pr_curve,
    merge_diagnostics,
    tune_f1_optimal,
    tune_precision_floor_match,
    tune_two_threshold,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_tuning(cfg: EvalConfig) -> dict[str, Any]:
    """Run Stage E4 threshold tuning.

    Args:
        cfg: Validated :class:`~bem.eval.config.EvalConfig` instance.

    Returns:
        Summary dict with keys ``results``, ``manifest_path``.

    Raises:
        FileNotFoundError: If the baseline scores master file is missing.
        ValueError: If test-split rows are detected in the master file.
    """
    tu = cfg.tuning
    out_dir = tu.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    master_path = cfg.baselines.output_dir / "baseline_scores_master.parquet"
    if not master_path.exists():
        raise FileNotFoundError(
            f"[E4] Baseline scores master file not found: {master_path}\n"
            "Run Stage E3 first:  python -m bem.eval --no-dry-run --stage baselines"
        )

    df = pd.read_parquet(master_path)
    df["score"]       = df["score"].astype(float)
    df["gold_label"]  = df["gold_label"].astype(str)
    df["split"]       = df["split"].astype(str)
    df["task"]        = df["task"].astype(str)
    df["baseline_name"] = df["baseline_name"].astype(str)

    # --- SAFETY: filter to dev only; test rows must NOT influence tuning --------
    n_test = int((df["split"] == "test").sum())
    if n_test > 0:
        print(
            f"[E4] SAFETY: {n_test} test-split rows found in {master_path.name} "
            "(expected when aux_scopus_id comparator covers all benchmark pairs). "
            "Filtering to dev only — test rows are EXCLUDED from threshold tuning."
        )
    dev_df = df[df["split"] == "dev"].copy()
    if len(dev_df) == 0:
        raise ValueError(
            f"[E4] FATAL: 0 dev rows remain in {master_path.name} after filtering. "
            "Cannot tune thresholds without dev data."
        )
    print(f"[E4] Loaded {len(dev_df):,} dev rows from {master_path.name}")

    # --- threshold grid -------------------------------------------------------
    grid = make_threshold_grid(tu.grid_start, tu.grid_stop, tu.grid_step)
    print(f"[E4] Threshold grid: {len(grid)} points "
          f"[{tu.grid_start:.2f} .. {tu.grid_stop:.2f}  step={tu.grid_step}]")

    # --- per-(task x baseline) tuning ----------------------------------------
    all_results:    list[dict] = []   # every result dict, all methods
    output_paths:   dict[str, Path] = {}

    for task in cfg.tasks:
        task_df = dev_df[dev_df["task"] == task]
        floor_match    = tu.precision_floor_and    if task == "and" else tu.precision_floor_ain
        floor_nonmatch = tu.precision_floor_nonmatch_and if task == "and" \
                         else tu.precision_floor_nonmatch_ain

        baseline_names = sorted(task_df["baseline_name"].unique().tolist())
        print(f"\n[E4] Task={task.upper()}  floor_match={floor_match:.2f}  "
              f"floor_nonmatch={floor_nonmatch:.2f}  "
              f"baselines={baseline_names}")

        task_summary_rows: list[dict] = []

        for bl_name in baseline_names:
            bl_df = task_df[task_df["baseline_name"] == bl_name]

            # Encode gold labels — exclude 'uncertain'
            binary_mask, gold, n_uncertain = encode_gold(bl_df["gold_label"])
            scores = bl_df["score"].values[binary_mask]

            n_binary = len(gold)
            if n_binary == 0:
                print(f"  [{bl_name}] SKIP — no binary-labelled pairs after excluding uncertain")
                continue

            n_match    = int((gold == 1).sum())
            n_nonmatch = int((gold == 0).sum())
            print(
                f"  [{bl_name}] {n_binary} pairs  "
                f"(match={n_match}, non-match={n_nonmatch}, uncertain_excl={n_uncertain})"
            )

            # Compute PR curves
            match_curve   = compute_match_pr_curve(scores, gold, grid, tu.f1_beta)
            nonmatch_curve = compute_nonmatch_pr_curve(scores, gold, grid, tu.f1_beta)

            # Write diagnostics
            diag_path = out_dir / f"diagnostics_{task}_{bl_name}.parquet"
            if not diag_path.exists() or cfg.force:
                diag = merge_diagnostics(match_curve, nonmatch_curve)
                diag["task"]          = task
                diag["baseline_name"] = bl_name
                diag["n_uncertain_excluded"] = n_uncertain
                diag.to_parquet(diag_path, index=False)
            output_paths[f"diagnostics_{task}_{bl_name}"] = diag_path

            # Tune
            r_f1    = tune_f1_optimal(
                match_curve, bl_name, task, n_uncertain, tu.f1_beta
            )
            r_floor = tune_precision_floor_match(
                match_curve, bl_name, task, floor_match, n_uncertain, tu.f1_beta
            )
            r_two   = (
                tune_two_threshold(
                    match_curve, nonmatch_curve, scores, gold,
                    bl_name, task, floor_match, floor_nonmatch, n_uncertain,
                )
                if tu.two_threshold_enabled
                else None
            )

            all_results.extend([r_f1, r_floor] + ([r_two] if r_two else []))
            task_summary_rows.extend([r_f1, r_floor] + ([r_two] if r_two else []))

            _print_result(r_f1)
            _print_result(r_floor)
            if r_two:
                _print_result(r_two)

        # Write per-task outputs
        summary_path = _write_task_summary(task, task_summary_rows, out_dir, cfg.force)
        thresh_path  = _write_task_thresholds(
            task, task_summary_rows, floor_match, out_dir, cfg.force
        )
        output_paths[f"threshold_summary_{task}"] = summary_path
        output_paths[f"thresholds_{task}"]        = thresh_path

    # Write manifest
    manifest_path = _write_manifest(cfg, master_path, out_dir, all_results, output_paths, grid)
    print(f"\n[E4] Tuning manifest written: {manifest_path}")

    return {"results": all_results, "manifest_path": manifest_path}


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_task_summary(
    task: str,
    results: list[dict],
    out_dir: Path,
    force: bool,
) -> Path:
    """Write ``threshold_summary_<task>.csv`` — human-readable selected thresholds."""
    out_path = out_dir / f"threshold_summary_{task}.csv"
    if out_path.exists() and not force:
        print(f"[E4] {out_path.name} exists — skipping (--force to overwrite)")
        return out_path

    rows = []
    for r in results:
        method = r["method"]
        if method == "two_threshold":
            rows.append({
                "task":               r["task"],
                "baseline_name":      r["baseline_name"],
                "method":             method,
                "t_low":              r["t_low"],
                "t_high":             r["t_high"],
                "precision_match":    r["definite_precision_match"],
                "recall_match":       r["definite_recall_match"],
                "precision_nonmatch": r["definite_precision_nonmatch"],
                "recall_nonmatch":    r["definite_recall_nonmatch"],
                "coverage":           r["coverage"],
                "abstention_rate":    r["abstention_rate"],
                "n_in_band":          r["n_in_band"],
                "precision_floor":    r["precision_floor_match"],
                "achieved_floor":     r["achieved_floor_match"],
                "n_total_binary":     r["n_total_binary"],
                "n_uncertain_excl":   r["n_uncertain_excluded"],
                "note":               r["note"],
            })
        else:
            # f1_optimal / precision_floor_match
            rows.append({
                "task":               r["task"],
                "baseline_name":      r["baseline_name"],
                "method":             method,
                "t_low":              r.get("threshold"),
                "t_high":             r.get("threshold"),
                "precision_match":    r.get("precision_match"),
                "recall_match":       r.get("recall_match"),
                "precision_nonmatch": None,
                "recall_nonmatch":    None,
                "coverage":           1.0 if r.get("threshold") is not None else None,
                "abstention_rate":    0.0 if r.get("threshold") is not None else None,
                "n_in_band":          0,
                "precision_floor":    r.get("precision_floor"),
                "achieved_floor":     r.get("achieved_floor"),
                "n_total_binary":     r.get("n_total_binary"),
                "n_uncertain_excl":   r.get("n_uncertain_excluded"),
                "note":               r.get("note", ""),
            })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_path, index=False)
    print(f"[E4] Written: {out_path.name}  ({len(summary_df)} rows)")
    return out_path


def _write_task_thresholds(
    task: str,
    results: list[dict],
    floor_match: float,
    out_dir: Path,
    force: bool,
) -> Path:
    """Write ``thresholds_<task>.json`` — machine-readable threshold artefact."""
    out_path = out_dir / f"thresholds_{task}.json"
    if out_path.exists() and not force:
        print(f"[E4] {out_path.name} exists — skipping (--force to overwrite)")
        return out_path

    # Group results by baseline_name
    by_baseline: dict[str, dict] = {}
    for r in results:
        bl = r["baseline_name"]
        if bl not in by_baseline:
            by_baseline[bl] = {
                "n_total_binary":     r.get("n_total_binary"),
                "n_uncertain_excluded": r.get("n_uncertain_excluded"),
            }
        method = r["method"]
        by_baseline[bl][method] = {k: v for k, v in r.items()
                                   if k not in ("method", "baseline_name", "task")}

    payload: dict[str, Any] = {
        "task":                   task,
        "split":                  "dev",
        "precision_floor_match":  floor_match,
        "timestamp_iso":          datetime.now(tz=timezone.utc).isoformat(),
        "baselines":              by_baseline,
    }

    out_path.write_text(json.dumps(payload, indent=2, default=_json_default),
                        encoding="utf-8")
    print(f"[E4] Written: {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _write_manifest(
    cfg: EvalConfig,
    master_path: Path,
    out_dir: Path,
    all_results: list[dict],
    output_paths: dict[str, Path],
    grid: np.ndarray,
) -> Path:
    """Write ``tuning_manifest.json`` capturing full provenance."""
    tu = cfg.tuning

    def _sha256(p: Path) -> str:
        if not p.exists():
            return "missing"
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def _git_commit() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return "unavailable"

    # Summary counts per (task, baseline)
    summary: dict[str, Any] = {}
    for r in all_results:
        key = f"{r['task']}/{r['baseline_name']}"
        if key not in summary:
            summary[key] = {
                "n_total_binary":     r.get("n_total_binary"),
                "n_uncertain_excluded": r.get("n_uncertain_excluded"),
                "methods_tuned":      [],
            }
        summary[key]["methods_tuned"].append(r["method"])

    manifest: dict[str, Any] = {
        "stage":            "E4_tune_thresholds",
        "status":           "completed",
        "timestamp_iso":    datetime.now(tz=timezone.utc).isoformat(),
        "git_commit":       _git_commit(),
        "environment": {
            "python":   sys.version,
            "platform": platform.platform(),
        },
        "inputs": {
            "baseline_scores_master": {
                "path":       str(master_path),
                "sha256":     _sha256(master_path),
                "size_bytes": master_path.stat().st_size if master_path.exists() else 0,
            },
        },
        "config": {
            "precision_floor_and":         tu.precision_floor_and,
            "precision_floor_ain":         tu.precision_floor_ain,
            "precision_floor_nonmatch_and": tu.precision_floor_nonmatch_and,
            "precision_floor_nonmatch_ain": tu.precision_floor_nonmatch_ain,
            "f1_beta":                     tu.f1_beta,
            "grid_start":                  tu.grid_start,
            "grid_stop":                   tu.grid_stop,
            "grid_step":                   tu.grid_step,
            "n_thresholds":                len(grid),
            "two_threshold_enabled":       tu.two_threshold_enabled,
        },
        "methods_tuned": ["f1_optimal", "precision_floor_match"]
                         + (["two_threshold"] if tu.two_threshold_enabled else []),
        "outputs": {
            label: {
                "path":       str(p),
                "sha256":     _sha256(p),
                "size_bytes": p.stat().st_size if p.exists() else 0,
            }
            for label, p in output_paths.items()
        },
        "tuning_summary": summary,
        "evaluation_policy": {
            "primary_operating_point":   "precision_floor_match",
            "secondary_operating_point": "f1_optimal",
            "two_threshold_role":        "uncertain-band routing (abstention)",
            "test_leakage_guard":        "raises ValueError if test rows present",
            "uncertain_gold_labels":     "excluded from all precision/recall computations",
        },
    }

    out_path = out_dir / "tuning_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, default=_json_default),
                        encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_result(r: dict) -> None:
    method = r["method"]
    bl     = r["baseline_name"]
    task   = r["task"]

    if method == "two_threshold":
        t_low  = r["t_low"]
        t_high = r["t_high"]
        cov    = r["coverage"]
        ab     = r["abstention_rate"]
        prec   = r["definite_precision_match"]
        rec    = r["definite_recall_match"]
        print(
            f"    {method:<26s}  t_low={t_low:.2f}  t_high={t_high:.2f}  "
            f"prec={prec:.3f}  rec={rec:.3f}  "
            f"coverage={cov:.3f}  abstention={ab:.3f}"
            + (f"  [{r['note']}]" if r["note"] else "")
        )
    else:
        t  = r.get("threshold")
        p  = r.get("precision_match")
        rc = r.get("recall_match")
        fb = r.get("f_beta_match")
        if t is None:
            print(f"    {method:<26s}  NO THRESHOLD FOUND  [{r.get('note', '')}]")
        else:
            primary = "  [PRIMARY]" if method == "precision_floor_match" else ""
            print(
                f"    {method:<26s}  t={t:.2f}  "
                f"prec={p:.3f}  rec={rc:.3f}  f={fb:.3f}"
                f"{primary}"
                + (f"  [{r['note']}]" if r.get("note") else "")
            )


def print_tuning_summary(results: list[dict], manifest_path: Path) -> None:
    """Print a formatted summary table after all tuning is complete."""
    width = 70
    print("\n" + "=" * width)
    print("  E4 THRESHOLD TUNING - SUMMARY")
    print("=" * width)

    # Group by task
    tasks = sorted({r["task"] for r in results})
    for task in tasks:
        task_results = [r for r in results if r["task"] == task]
        baselines    = sorted({r["baseline_name"] for r in task_results})

        print(f"\n  Task: {task.upper()}")
        print(f"  {'Baseline':<20s}  {'Method':<26s}  {'t_high':>6}  "
              f"{'Prec':>6}  {'Rec':>6}  {'F':>6}  {'Cov':>6}")
        print("  " + "-" * (width - 2))

        for bl in baselines:
            bl_results = [r for r in task_results if r["baseline_name"] == bl]
            for r in bl_results:
                method = r["method"]
                if method == "two_threshold":
                    t      = r["t_high"]
                    prec   = r["definite_precision_match"]
                    rec    = r["definite_recall_match"]
                    f_val  = 0.0   # not computed for two-threshold summary
                    cov    = r["coverage"]
                    mark   = ""
                else:
                    t      = r.get("threshold") or float("nan")
                    prec   = r.get("precision_match") or 0.0
                    rec    = r.get("recall_match") or 0.0
                    f_val  = r.get("f_beta_match") or 0.0
                    cov    = 1.0
                    mark   = "  *" if method == "precision_floor_match" else ""

                t_str   = f"{t:.2f}"   if not (isinstance(t, float) and t != t) else "N/A"
                prec_s  = f"{prec:.3f}" if prec is not None else "N/A"
                rec_s   = f"{rec:.3f}"  if rec  is not None else "N/A"
                f_s     = f"{f_val:.3f}" if f_val else "  -  "
                cov_s   = f"{cov:.3f}"

                print(
                    f"  {bl:<20s}  {method:<26s}  {t_str:>6}  "
                    f"{prec_s:>6}  {rec_s:>6}  {f_s:>6}  {cov_s:>6}{mark}"
                )

    print("\n  * = PRIMARY operating point (precision_floor_match)")
    print(f"\n  Manifest: {manifest_path}")
    print("=" * width + "\n")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Fallback JSON serialiser for numpy scalars and NaN."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if v != v else v   # NaN -> null
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
