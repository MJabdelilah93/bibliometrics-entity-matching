"""runner.py — Stage E5 orchestrator: apply thresholds, compute metrics, export.

Evaluation flow
---------------
For each task (AND, AIN):

  1. Determine the evaluation split.  Test is preferred; if absent, dev is
     used with a prominent WARNING.

  2. For each (baseline_name × method) from the tuned thresholds JSON:
       a. Apply threshold to baseline scores  -> three-class predictions.
       b. Call compute_all_metrics()          -> metrics dict with bootstrap CI.

  3. If include_bem: load BEM routing decisions, compute metrics.

  4. Build a unified metrics DataFrame; run smoke checks.

  5. Write outputs:
       outputs/evaluation/
         metrics_<task>_<split>.json         full per-system metrics
         metrics_master.parquet              all tasks/systems/methods stacked
         evaluation_manifest.json
       outputs/manuscript/
         table_<task>_<split>.csv
         table_<task>_<split>.md
         table_<task>_<split>.tex
         figure_precision_coverage_<task>.<fmt>

Denominators (locked)
---------------------
  coverage         = n_auto_decided / n_total
  precision_match  = TP / (TP + FP)
  recall_match     = TP / (TP + FN + abstained_match)   [all true matches]
  three_way_acc    = (TP + TN) / n_gold_binary
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bem.eval.config import EvalConfig
from bem.eval.evaluation.metrics  import compute_all_metrics
from bem.eval.evaluation.applier  import load_thresholds, apply_all_methods
from bem.eval.evaluation.bem_eval import load_bem_predictions
from bem.eval.evaluation.tables   import (
    build_metrics_df, to_markdown, to_latex,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_evaluation(cfg: EvalConfig) -> dict[str, Any]:
    """Execute Stage E5.

    Args:
        cfg: Validated :class:`~bem.eval.config.EvalConfig`.

    Returns:
        Dict with ``metrics_df`` (DataFrame), ``manifest_path`` (Path).
    """
    ev  = cfg.evaluation
    rng = np.random.default_rng(cfg.random_seed)

    for d in (ev.output_dir, ev.manuscript_dir):
        d.mkdir(parents=True, exist_ok=True)

    master_path = cfg.baselines.output_dir / "baseline_scores_master.parquet"
    if not master_path.exists():
        raise FileNotFoundError(
            f"[E5] Baseline scores master not found: {master_path}\n"
            "Run Stage E3 first."
        )

    master_df = pd.read_parquet(master_path)
    master_df["score"]         = master_df["score"].astype(float)
    master_df["gold_label"]    = master_df["gold_label"].astype(str)
    master_df["split"]         = master_df["split"].astype(str)
    master_df["task"]          = master_df["task"].astype(str)
    master_df["baseline_name"] = master_df["baseline_name"].astype(str)

    all_result_rows: list[dict] = []
    all_metrics_df_parts: list[pd.DataFrame] = []
    output_files: dict[str, Path] = {}

    for task in cfg.tasks:
        print(f"\n{'='*60}")
        print(f"[E5] Task: {task.upper()}")
        print(f"{'='*60}")

        # --- determine evaluation split ---
        split = _resolve_split(master_df, task, cfg)

        # --- load thresholds ---
        thresh_path = cfg.tuning.output_dir / f"thresholds_{task}.json"
        thresholds_dict = load_thresholds(thresh_path)

        # --- collect (system, method) results ---
        task_results: list[dict] = []

        # 1. BEM
        if ev.include_bem:
            preds_path = cfg.eval_inputs_output_dir / f"predictions_{task}_{split}.parquet"
            try:
                bem_df = load_bem_predictions(preds_path, task)
                bem_metrics = compute_all_metrics(
                    bem_df["gold_label"],
                    bem_df["predicted"],
                    rng=rng,
                    n_bootstrap=ev.bootstrap_n_samples,
                    alpha=ev.bootstrap_alpha,
                )
                task_results.append({
                    "task": task, "split": split,
                    "system": "bem", "display_name": "BEM",
                    "method": "bem_routing",
                    "is_auxiliary": False,
                    "threshold_value": None,
                    "metrics": bem_metrics,
                })
                _print_result("BEM", "bem_routing", bem_metrics)
            except FileNotFoundError as e:
                print(f"  [BEM] WARNING: {e}", file=sys.stderr)

        # 2. Baseline systems
        available_baselines = sorted(
            master_df[(master_df["task"] == task) & (master_df["split"] == split)]
            ["baseline_name"].unique().tolist()
        )

        for bl_name in available_baselines:
            is_aux = (bl_name == "aux_scopus_id")
            if is_aux and not ev.include_aux_scopus_id:
                continue
            # Skip aux_scopus_id from main method loop; it goes in a separate block
            row_group = apply_all_methods(
                master_df[master_df["split"] == split],
                task, bl_name, thresholds_dict, ev.methods,
            )
            for row in row_group:
                method_info = row.get("threshold_info", {})
                t_val = method_info.get("threshold") or method_info.get("t_high")
                m = compute_all_metrics(
                    row["gold_all"], row["predicted_all"],
                    rng=rng,
                    n_bootstrap=ev.bootstrap_n_samples,
                    alpha=ev.bootstrap_alpha,
                )
                task_results.append({
                    "task": task, "split": split,
                    "system": "baseline",
                    "display_name": bl_name.replace("_", " ").title(),
                    "baseline_name": bl_name,
                    "method": row["method"],
                    "is_auxiliary": is_aux,
                    "threshold_value": t_val,
                    "metrics": m,
                })
                _print_result(bl_name, row["method"], m)

        # --- smoke checks ---
        _smoke_checks(task_results, task, split)

        # --- build per-task metrics DataFrame ---
        task_df = build_metrics_df(task_results)

        # Preserve system order: BEM first, aux last
        task_df = _sort_rows(task_df)

        all_result_rows.extend(task_results)
        all_metrics_df_parts.append(task_df)

        # --- write per-task JSON ---
        json_path = ev.output_dir / f"metrics_{task}_{split}.json"
        _write_metrics_json(task_results, task, split, cfg, json_path)
        output_files[f"metrics_{task}_{split}"] = json_path
        print(f"\n[E5] Written: {json_path.name}")

        # --- write manuscript tables ---
        for fmt_name, fmt_fn, suffix in [
            ("csv", lambda df: df.to_csv(index=False), ".csv"),
            ("markdown", lambda df: to_markdown(df, task, split), ".md"),
            ("latex", lambda df: to_latex(df, task, split), ".tex"),
        ]:
            out = ev.manuscript_dir / f"table_{task}_{split}{suffix}"
            if not out.exists() or cfg.force:
                content = fmt_fn(task_df)
                if isinstance(content, str):
                    out.write_text(content, encoding="utf-8")
                else:
                    content  # already written by df.to_csv  ... handle below

            # Redo CSV properly
            if fmt_name == "csv":
                out = ev.manuscript_dir / f"table_{task}_{split}.csv"
                if not out.exists() or cfg.force:
                    task_df.to_csv(out, index=False)
            output_files[f"table_{task}_{split}_{fmt_name}"] = out
        print(f"[E5] Tables written: table_{task}_{split}.{{csv,md,tex}}")

        # --- precision–coverage figure ---
        fig_path = ev.manuscript_dir / f"figure_precision_coverage_{task}"
        op_points = _build_operating_points(task_results)
        bem_fig_pt = _bem_figure_point(task_results)
        try:
            from bem.eval.evaluation.figures import plot_precision_coverage
            actual_path = plot_precision_coverage(
                task=task,
                diagnostics_dir=cfg.tuning.output_dir,
                operating_points=op_points,
                bem_point=bem_fig_pt,
                out_path=fig_path,
                dpi=ev.figure_dpi,
                fmt=ev.figure_format,
            )
            print(f"[E5] Figure written: {actual_path.name}")
            output_files[f"figure_precision_coverage_{task}"] = actual_path
        except ImportError as e:
            print(f"[E5] Figure skipped: {e}", file=sys.stderr)

    # --- master parquet (all tasks) ---
    if all_metrics_df_parts:
        master_metrics = pd.concat(all_metrics_df_parts, ignore_index=True)
        master_metrics_path = ev.output_dir / "metrics_master.parquet"
        if not master_metrics_path.exists() or cfg.force:
            master_metrics.to_parquet(master_metrics_path, index=False)
        output_files["metrics_master"] = master_metrics_path
        print(f"\n[E5] Master metrics parquet: {master_metrics_path.name} "
              f"({len(master_metrics)} rows)")

    # --- manifest ---
    manifest_path = _write_manifest(cfg, master_path, output_files)
    print(f"[E5] Evaluation manifest: {manifest_path}")

    return {
        "metrics_df":    pd.concat(all_metrics_df_parts, ignore_index=True) if all_metrics_df_parts else pd.DataFrame(),
        "manifest_path": manifest_path,
    }


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_evaluation_summary(metrics_df: pd.DataFrame, manifest_path: Path) -> None:
    """Print a formatted per-task evaluation summary."""
    width = 78
    print("\n" + "=" * width)
    print("  E5 EVALUATION - SUMMARY")
    print("=" * width)

    for task in sorted(metrics_df["task"].unique()):
        t_df = metrics_df[metrics_df["task"] == task]
        split = t_df["split"].iloc[0] if len(t_df) > 0 else "?"

        print(f"\n  Task: {task.upper()}  |  Split: {split}"
              + ("  [WARNING: not held-out test]" if split == "dev" else ""))
        print(f"\n  {'System':<20} {'Method':<22} {'Prec_M':>7} {'Rec_M':>7} "
              f"{'F1_M':>7} {'CI_F1':>18} {'Cov':>6} {'Unc%':>6}")
        print("  " + "-" * (width - 2))

        for _, row in t_df.iterrows():
            name  = str(row.get("display_name", ""))[:19]
            meth  = _method_short(str(row.get("method", "")))[:21]
            mark  = " *" if row.get("method") == "precision_floor_match" else "  "
            aux   = " (aux)" if row.get("is_auxiliary") else ""

            prec = _fmts(row.get("precision_match"))
            rec  = _fmts(row.get("recall_match"))
            f1   = _fmts(row.get("f1_match"))
            cov  = _fmts(row.get("coverage"))
            unc  = _fmts_pct(row.get("uncertain_rate"))
            lo   = row.get("ci_f1_match_lo")
            hi   = row.get("ci_f1_match_hi")
            ci   = f"[{lo:.3f}, {hi:.3f}]" if _is_num(lo) and _is_num(hi) else "              "

            print(f"  {name:<20} {meth:<22} {prec:>7} {rec:>7} {f1:>7} "
                  f"{ci:>18} {cov:>6} {unc:>6}{mark}{aux}")

    print(f"\n  * = PRIMARY operating point (precision_floor_match)")
    print(f"\n  Manifest: {manifest_path}")
    print("=" * width + "\n")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_split(master_df: pd.DataFrame, task: str, cfg: "EvalConfig | None" = None) -> str:
    """Return 'test' if a test split exists for this task; else warn and return 'dev'.

    Checks two sources (either suffices):
      1. predictions_{task}_test.parquet in eval_inputs_output_dir  (BEM predictions)
      2. test rows in the baseline_scores_master DataFrame
    """
    # Primary check: BEM predictions file for test split
    if cfg is not None:
        preds_test = cfg.eval_inputs_output_dir / f"predictions_{task}_test.parquet"
        if preds_test.exists():
            return "test"

    # Fallback check: baseline scores master contains test rows for this task
    available = master_df[master_df["task"] == task]["split"].unique().tolist()
    if "test" in available:
        return "test"

    print(
        f"\n  [E5] WARNING — No test split found for task={task.upper()}.\n"
        f"  Falling back to 'dev' split.  This is NOT the final held-out evaluation.\n"
        f"  Re-run after test-split annotation is complete.\n",
        file=sys.stderr,
    )
    return "dev"


def _smoke_checks(results: list[dict], task: str, split: str) -> None:
    """Lightweight invariant checks on the computed results."""
    for r in results:
        m = r.get("metrics", {})
        system = r.get("display_name", "?")
        method = r.get("method", "?")
        label  = f"{task}/{system}/{method}"

        # No test leakage
        if r.get("split") == "test":
            assert split == "test", f"[smoke] {label}: split mismatch"

        # Coverage in [0, 1]
        cov = m.get("coverage")
        if cov is not None and not math.isnan(cov):
            assert 0.0 <= cov <= 1.0, f"[smoke] {label}: coverage={cov:.3f} out of range"

        # Precision in [0, 1]
        for key in ("precision_match", "recall_match", "precision_nonmatch", "recall_nonmatch"):
            v = m.get(key)
            if v is not None and not math.isnan(v):
                assert 0.0 <= v <= 1.0, f"[smoke] {label}: {key}={v:.3f} out of range"

        # CI not inverted
        lo, hi = m.get("ci_f1_match", (float("nan"), float("nan")))
        if _is_num(lo) and _is_num(hi):
            assert lo <= hi, f"[smoke] {label}: CI inverted [{lo:.3f}, {hi:.3f}]"

    print(f"  [E5] Smoke checks passed for {task.upper()}/{split} ({len(results)} rows)")


def _build_operating_points(
    results: list[dict],
) -> dict[str, dict[str, dict]]:
    """Build operating-point dict for the figure: {bl: {method: {cov, prec}}}."""
    op: dict[str, dict[str, dict]] = {}
    for r in results:
        if r.get("system") != "baseline":
            continue
        bl     = r.get("baseline_name", "")
        method = r.get("method", "")
        m      = r.get("metrics", {})
        if bl not in op:
            op[bl] = {}
        n_total_bin = m.get("n_gold_binary", 1) or 1
        n_pred = m.get("tp", 0) + m.get("fp", 0)  # predicted match
        op[bl][method] = {
            "precision_match": m.get("precision_match"),
            "coverage":        n_pred / n_total_bin if n_total_bin > 0 else 0.0,
        }
    return op


def _bem_figure_point(results: list[dict]) -> dict | None:
    """Extract BEM's (coverage, precision_match) for the figure."""
    for r in results:
        if r.get("system") == "bem":
            m = r.get("metrics", {})
            n_total_bin = m.get("n_gold_binary", 1) or 1
            n_pred = m.get("tp", 0) + m.get("fp", 0)
            return {
                "precision_match": m.get("precision_match"),
                "coverage":        n_pred / n_total_bin if n_total_bin > 0 else 0.0,
            }
    return None


def _sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows: BEM first, then fair baselines alphabetically, aux last."""
    def _key(row):
        if row["system"] == "bem":
            return (0, "", "")
        if row.get("is_auxiliary"):
            return (2, row.get("display_name", ""), row.get("method", ""))
        return (1, row.get("display_name", ""), row.get("method", ""))
    df = df.copy()
    df["_sort"] = df.apply(_key, axis=1)
    return df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


def _write_metrics_json(
    results: list[dict],
    task: str,
    split: str,
    cfg: EvalConfig,
    out_path: Path,
) -> None:
    """Serialise all metrics for one (task, split) to JSON."""
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if v != v else v
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(type(o))

    serialisable = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in ("gold_all", "predicted_all")}
        m = row.get("metrics", {})
        # Convert tuple CIs to lists
        for k in list(m.keys()):
            if isinstance(m[k], tuple):
                m[k] = list(m[k])
        serialisable.append(row)

    payload = {
        "stage":       "E5_evaluation",
        "task":        task,
        "split":       split,
        "timestamp":   datetime.now(tz=timezone.utc).isoformat(),
        "evaluation_policy": {
            "primary_metric":   "f1_match @ precision_floor_match",
            "coverage_denom":   "n_total (all pairs incl. uncertain gold)",
            "recall_denom":     "all true matches in binary gold (incl. abstained)",
            "uncertain_gold":   "excluded from binary precision/recall/F1",
            "bootstrap_n":      cfg.evaluation.bootstrap_n_samples,
            "bootstrap_alpha":  cfg.evaluation.bootstrap_alpha,
        },
        "results": serialisable,
    }

    out_path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")


def _write_manifest(
    cfg: EvalConfig,
    master_path: Path,
    output_files: dict[str, Path],
) -> Path:
    def _sha256(p: Path) -> str:
        if not p.exists():
            return "missing"
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def _git() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return "unavailable"

    manifest = {
        "stage":         "E5_evaluation",
        "status":        "completed",
        "timestamp_iso": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit":    _git(),
        "environment":   {"python": sys.version, "platform": platform.platform()},
        "inputs": {
            "baseline_scores_master": {
                "path":       str(master_path),
                "sha256":     _sha256(master_path),
                "size_bytes": master_path.stat().st_size if master_path.exists() else 0,
            },
        },
        "config": {
            "methods":               cfg.evaluation.methods,
            "include_bem":           cfg.evaluation.include_bem,
            "include_aux_scopus_id": cfg.evaluation.include_aux_scopus_id,
            "bootstrap_n_samples":   cfg.evaluation.bootstrap_n_samples,
            "bootstrap_alpha":       cfg.evaluation.bootstrap_alpha,
            "random_seed":           cfg.random_seed,
        },
        "outputs": {
            label: {
                "path":       str(p),
                "sha256":     _sha256(p),
                "size_bytes": p.stat().st_size if p.exists() else 0,
            }
            for label, p in output_files.items()
        },
    }

    out_path = cfg.evaluation.output_dir / "evaluation_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


def _print_result(name: str, method: str, m: dict) -> None:
    prec = m.get("precision_match")
    rec  = m.get("recall_match")
    f1   = m.get("f1_match")
    cov  = m.get("coverage")
    lo, hi = m.get("ci_f1_match", (None, None))
    ci_str = f"[{lo:.3f},{hi:.3f}]" if _is_num(lo) and _is_num(hi) else ""
    mark = "  [PRIMARY]" if method == "precision_floor_match" else ""
    print(
        f"  {name:<20} {method:<24}  "
        f"prec={_fmts(prec)}  rec={_fmts(rec)}  "
        f"F1={_fmts(f1)} {ci_str}  cov={_fmts(cov)}{mark}"
    )


def _fmts(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  N/A"
    return f"{v:.3f}"


def _fmts_pct(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  N/A"
    return f"{v*100:.1f}%"


def _is_num(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def _method_short(m: str) -> str:
    return {
        "precision_floor_match": "Floor(0.90)",
        "f1_optimal":            "F1-opt",
        "two_threshold":         "2-thresh",
        "bem_routing":           "auto-route",
    }.get(m, m)
