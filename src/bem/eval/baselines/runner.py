"""runner.py — Orchestrate all classical baseline runs for Stage E3.

For each (task x baseline) combination this module:
  1. Loads or reuses the features DataFrame.
  2. Calls the relevant baseline scorer.
  3. Assembles a per-baseline per-task score DataFrame.
  4. Writes split-specific parquet files.
  5. Assembles the long-format master file (all baselines stacked).
  6. Includes the auxiliary Scopus-ID comparator if configured.
  7. Writes a stage manifest with full provenance.
  8. Runs lightweight smoke checks on every output.

Output files (all under ``cfg.baselines.output_dir``)
------------------------------------------------------
  scores_<task>_<split>_<baseline>.parquet   per-baseline file
  baseline_scores_master.parquet             all baselines stacked (long format)
  baselines_manifest.json                    provenance record

Master DataFrame schema
-----------------------
  anchor_id, candidate_id, task, gold_label, split, baseline_name, score

Per-baseline DataFrame schema
------------------------------
  anchor_id, candidate_id, task, gold_label, split, score
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bem.eval.config import EvalConfig
from bem.eval.baselines import deterministic, fuzzy, tfidf, embedding
from bem.eval.baselines.features import load_and_features, load_ain_features

_AUTHOR_INSTANCES  = Path("data/interim/author_instances.parquet")
_AFFIL_INSTANCES   = Path("data/interim/affil_instances.parquet")

# Minimal columns kept in each per-baseline output
_BASE_COLS = ["anchor_id", "candidate_id", "task", "gold_label", "split"]


# ---------------------------------------------------------------------------
# Reporting dataclass
# ---------------------------------------------------------------------------

@dataclass
class BaselineRunResult:
    task: str
    baseline_name: str
    split: str
    n_pairs: int
    score_min: float
    score_max: float
    score_mean: float
    output_path: Path | None
    skipped: bool = False
    skip_reason: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class BaselineReport:
    results: list[BaselineRunResult] = field(default_factory=list)
    master_path: Path | None = None
    manifest_path: Path | None = None
    all_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_baselines(cfg: EvalConfig) -> BaselineReport:
    """Run all configured baseline scorers for all configured tasks and splits.

    Args:
        cfg: Validated :class:`~bem.eval.config.EvalConfig` instance.

    Returns:
        :class:`BaselineReport` summarising what was written and any warnings.
    """
    bl = cfg.baselines
    out_dir = bl.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = BaselineReport()
    all_scores: list[pd.DataFrame] = []

    task_feature_loaders = {
        "and": lambda: _load_combined_features(
            "and",
            cfg.eval_inputs_output_dir,
            _resolve_instances(_AUTHOR_INSTANCES, cfg),
            load_and_features,
        ),
        "ain": lambda: _load_combined_features(
            "ain",
            cfg.eval_inputs_output_dir,
            _resolve_instances(_AFFIL_INSTANCES, cfg),
            load_ain_features,
        ),
    }

    # Cache feature DataFrames per task (avoid loading twice per task)
    feat_cache: dict[str, pd.DataFrame] = {}

    for task in cfg.tasks:
        print(f"\n[E3] Loading features for task={task.upper()} ...")
        features_df = task_feature_loaders[task]()
        feat_cache[task] = features_df

        for bl_name, scorer_fn in _get_scorers(bl, task, cfg):
            _run_one(
                task=task,
                baseline_name=bl_name,
                features_df=features_df,
                scorer_fn=scorer_fn,
                out_dir=out_dir,
                force=cfg.force,
                report=report,
                all_scores=all_scores,
            )

    # Auxiliary Scopus-ID comparator (AND only)
    if bl.aux_scopus_id_in_master and "and" in cfg.tasks:
        _include_aux_scopus_id(bl, all_scores, report, out_dir, cfg.force)

    # Master file
    if all_scores:
        master_df = pd.concat(all_scores, ignore_index=True)
        master_path = out_dir / "baseline_scores_master.parquet"
        if not master_path.exists() or cfg.force:
            master_df.to_parquet(master_path, index=False)
            print(f"\n[E3] Master file written: {master_path.name}  ({len(master_df):,} rows)")
        else:
            print(f"\n[E3] Master file exists — skipping (--force to overwrite): {master_path.name}")
        report.master_path = master_path
    else:
        print("\n[E3] No baseline scores produced; master file not written.")

    report.manifest_path = _write_manifest(cfg, out_dir, report, all_scores)
    return report


# ---------------------------------------------------------------------------
# Per-baseline runner
# ---------------------------------------------------------------------------

def _run_one(
    task: str,
    baseline_name: str,
    features_df: pd.DataFrame,
    scorer_fn: Any,
    out_dir: Path,
    force: bool,
    report: BaselineReport,
    all_scores: list[pd.DataFrame],
) -> None:
    """Score one (task x baseline) combination and write output."""
    split = "all"   # features_df may contain dev+test rows; split is preserved in the data
    out_path = out_dir / f"scores_{task}_{split}_{baseline_name}.parquet"

    if out_path.exists() and not force:
        print(f"[E3] {task.upper()}/{baseline_name}: exists — skipping (--force to overwrite)")
        # Still load for master file
        existing = pd.read_parquet(out_path)
        existing["baseline_name"] = baseline_name
        all_scores.append(existing[_BASE_COLS + ["score", "baseline_name"]])
        report.results.append(BaselineRunResult(
            task=task, baseline_name=baseline_name, split=split,
            n_pairs=len(existing),
            score_min=float(existing["score"].min()),
            score_max=float(existing["score"].max()),
            score_mean=float(existing["score"].mean()),
            output_path=out_path, skipped=True, skip_reason="output exists",
        ))
        return

    print(f"[E3] {task.upper()}/{baseline_name}: scoring {len(features_df):,} pairs ...")

    try:
        scores = scorer_fn(features_df)
    except Exception as exc:
        warn = f"[E3] {task.upper()}/{baseline_name}: ERROR — {exc}"
        print(warn, file=sys.stderr)
        report.results.append(BaselineRunResult(
            task=task, baseline_name=baseline_name, split=split,
            n_pairs=len(features_df), score_min=0.0, score_max=0.0, score_mean=0.0,
            output_path=None, skipped=True, skip_reason=str(exc),
            warnings=[warn],
        ))
        report.all_warnings.append(warn)
        return

    if scores is None:
        # Embedding skipped by user or non-interactive mode
        warn = f"[E3] {task.upper()}/{baseline_name}: skipped (user declined or non-interactive)"
        report.results.append(BaselineRunResult(
            task=task, baseline_name=baseline_name, split=split,
            n_pairs=len(features_df), score_min=0.0, score_max=0.0, score_mean=0.0,
            output_path=None, skipped=True, skip_reason="user declined or non-interactive",
            warnings=[warn],
        ))
        report.all_warnings.append(warn)
        return

    out_df = features_df[["anchor_id", "candidate_id", "gold_label", "split"]].copy()
    out_df["task"] = task
    out_df["score"] = scores.values

    _smoke_check(out_df, task, baseline_name, split)

    out_df.to_parquet(out_path, index=False)
    print(
        f"[E3] {task.upper()}/{baseline_name}: written {len(out_df):,} rows  "
        f"score=[{scores.min():.3f}, {scores.max():.3f}]  mean={scores.mean():.3f}"
    )

    long_df = out_df.copy()
    long_df["baseline_name"] = baseline_name
    all_scores.append(long_df[_BASE_COLS + ["score", "baseline_name"]])

    report.results.append(BaselineRunResult(
        task=task, baseline_name=baseline_name, split=split,
        n_pairs=len(out_df),
        score_min=float(scores.min()),
        score_max=float(scores.max()),
        score_mean=float(scores.mean()),
        output_path=out_path,
    ))


# ---------------------------------------------------------------------------
# Auxiliary Scopus-ID comparator
# ---------------------------------------------------------------------------

def _include_aux_scopus_id(
    bl: Any,
    all_scores: list[pd.DataFrame],
    report: BaselineReport,
    out_dir: Path,
    force: bool,
) -> None:
    """Append auxiliary Scopus-ID comparator scores to the master file."""
    src = bl.aux_scopus_id_source
    if not src.exists():
        warn = (
            f"[E3-AUX] aux_scopus_id_source not found: {src}. "
            "Run stage materialise-inputs first."
        )
        print(warn)
        report.all_warnings.append(warn)
        return

    aux = pd.read_parquet(src)
    aux["score"]         = aux["scopus_id_comparator_pred"].astype(float)
    aux["baseline_name"] = "aux_scopus_id"
    aux["task"]          = "and"

    cols_needed = _BASE_COLS + ["score", "baseline_name"]
    missing = [c for c in cols_needed if c not in aux.columns]
    if missing:
        warn = f"[E3-AUX] aux file missing columns {missing} — skipping."
        print(warn)
        report.all_warnings.append(warn)
        return

    aux_out = out_dir / "scores_and_dev_aux_scopus_id.parquet"
    if not aux_out.exists() or force:
        aux[cols_needed].to_parquet(aux_out, index=False)
        print(f"[E3-AUX] Auxiliary Scopus-ID comparator written: {aux_out.name}  ({len(aux):,} rows)")

    all_scores.append(aux[cols_needed])
    report.results.append(BaselineRunResult(
        task="and", baseline_name="aux_scopus_id", split="dev",
        n_pairs=len(aux),
        score_min=float(aux["score"].min()),
        score_max=float(aux["score"].max()),
        score_mean=float(aux["score"].mean()),
        output_path=aux_out,
    ))


# ---------------------------------------------------------------------------
# Scorer registry
# ---------------------------------------------------------------------------

def _get_scorers(bl: Any, task: str, cfg: EvalConfig) -> list[tuple[str, Any]]:
    """Return list of (name, callable) for the configured baselines."""
    scorers: list[tuple[str, Any]] = []

    if bl.run_deterministic:
        scorers.append(("deterministic", lambda df: deterministic.run(task, df)))

    if bl.run_fuzzy:
        scorers.append(("fuzzy", lambda df: fuzzy.run(task, df)))

    if bl.run_tfidf:
        # Capture variables for closure
        _max_feat   = bl.tfidf_max_features
        _ngram      = bl.tfidf_ngram_range
        _sublinear  = bl.tfidf_sublinear_tf
        _fit_corpus = bl.tfidf_fit_corpus
        _ai_path    = _resolve_instances(_AUTHOR_INSTANCES, cfg)
        _af_path    = _resolve_instances(_AFFIL_INSTANCES, cfg)
        scorers.append(("tfidf", lambda df, _t=task: tfidf.run(
            _t, df,
            max_features=_max_feat,
            ngram_range=_ngram,
            sublinear_tf=_sublinear,
            fit_corpus=_fit_corpus,
            author_instances_path=_ai_path,
            affil_instances_path=_af_path,
        )))

    if bl.run_embedding:
        _model  = bl.embedding_model
        _bs     = bl.embedding_batch_size
        _dev    = bl.embedding_device
        _req_c  = bl.embedding_require_confirmation
        _seed   = cfg.random_seed
        scorers.append(("embedding", lambda df, _t=task: embedding.run(
            _t, df,
            model_name=_model,
            batch_size=_bs,
            device=_dev,
            require_confirmation=_req_c,
            random_seed=_seed,
        )))

    return scorers


def _load_combined_features(
    task: str,
    eval_inputs_dir: Path,
    instances_path: Path,
    loader_fn: Any,
) -> pd.DataFrame:
    """Load dev + test prediction parquets and return combined feature DataFrame.

    The 'split' column in the returned DataFrame correctly reflects 'dev' or
    'test' for each row, enabling baselines to be scored on all benchmark pairs.
    """
    dev_path  = eval_inputs_dir / f"predictions_{task}_dev.parquet"
    test_path = eval_inputs_dir / f"predictions_{task}_test.parquet"

    dev_df = loader_fn(dev_path, instances_path)

    if test_path.exists():
        test_df = loader_fn(test_path, instances_path)
        combined = pd.concat([dev_df, test_df], ignore_index=True)
        print(
            f"[E3] {task.upper()}: combined dev ({len(dev_df):,}) + "
            f"test ({len(test_df):,}) = {len(combined):,} pairs for scoring"
        )
        return combined

    print(f"[E3] {task.upper()}: test predictions not found — scoring dev only ({len(dev_df):,} pairs)")
    return dev_df


def _resolve_instances(rel_path: Path, cfg: EvalConfig) -> Path:
    """Resolve a relative data path relative to the project root."""
    # project root = config_path.parent (or config_path.parent.parent if inside configs/)
    project_root = cfg.config_path.parent
    if project_root.name == "configs":
        project_root = project_root.parent
    return (project_root / rel_path).resolve()


# ---------------------------------------------------------------------------
# Smoke checks
# ---------------------------------------------------------------------------

def _smoke_check(df: pd.DataFrame, task: str, baseline: str, split: str) -> None:
    """Assert basic invariants on a scored output DataFrame."""
    tag = f"[smoke {task.upper()} {split} {baseline}]"
    errors: list[str] = []

    if "score" not in df.columns:
        errors.append(f"{tag} Missing 'score' column")
    else:
        bad_range = ((df["score"] < 0) | (df["score"] > 1)).sum()
        if bad_range > 0:
            errors.append(f"{tag} {bad_range} scores outside [0, 1]")
        if df["score"].isna().any():
            errors.append(f"{tag} NaN scores present")

    if "gold_label" not in df.columns:
        errors.append(f"{tag} Missing 'gold_label' column")

    dupe_pairs = df.duplicated(subset=["anchor_id", "candidate_id"]).sum()
    if dupe_pairs > 0:
        errors.append(f"{tag} {dupe_pairs} duplicate (anchor_id, candidate_id) pairs")

    if errors:
        for e in errors:
            print(f"[E3] SMOKE FAIL: {e}")
        raise AssertionError("; ".join(errors))

    print(f"[E3] Smoke checks passed: {tag}")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _write_manifest(
    cfg: EvalConfig,
    out_dir: Path,
    report: BaselineReport,
    all_scores: list[pd.DataFrame],
) -> Path:
    results_records = []
    for r in report.results:
        results_records.append({
            "task": r.task,
            "baseline_name": r.baseline_name,
            "split": r.split,
            "n_pairs": r.n_pairs,
            "score_min": r.score_min,
            "score_max": r.score_max,
            "score_mean": r.score_mean,
            "skipped": r.skipped,
            "skip_reason": r.skip_reason,
            "output_path": str(r.output_path) if r.output_path else None,
            "sha256": _sha256(r.output_path) if r.output_path else None,
        })

    manifest: dict[str, Any] = {
        "stage": "E3_baselines",
        "status": "completed",
        "timestamp_iso": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "environment": {"python": sys.version, "platform": platform.platform()},
        "fairness_constraints": {
            "AND": "Author(s) ID stripped from all name fields; coauthor_keys not used",
            "AIN": "year_int excluded from all AIN features",
        },
        "config": {
            "tasks": cfg.tasks,
            "random_seed": cfg.random_seed,
            "tfidf_max_features": cfg.baselines.tfidf_max_features,
            "tfidf_ngram_range": list(cfg.baselines.tfidf_ngram_range),
            "tfidf_sublinear_tf": cfg.baselines.tfidf_sublinear_tf,
            "tfidf_fit_corpus": cfg.baselines.tfidf_fit_corpus,
            "embedding_model": cfg.baselines.embedding_model,
            "embedding_device": cfg.baselines.embedding_device,
        },
        "inputs": {
            "predictions_and_dev":  str(cfg.eval_inputs_output_dir / "predictions_and_dev.parquet"),
            "predictions_and_test": str(cfg.eval_inputs_output_dir / "predictions_and_test.parquet"),
            "predictions_ain_dev":  str(cfg.eval_inputs_output_dir / "predictions_ain_dev.parquet"),
            "predictions_ain_test": str(cfg.eval_inputs_output_dir / "predictions_ain_test.parquet"),
            "author_instances":     str(_AUTHOR_INSTANCES),
            "affil_instances":      str(_AFFIL_INSTANCES),
        },
        "results": results_records,
        "master_path": str(report.master_path) if report.master_path else None,
        "master_sha256": _sha256(report.master_path) if report.master_path else None,
        "all_warnings": report.all_warnings,
        "output_dir": str(out_dir),
    }

    out_path = out_dir / "baselines_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[E3] Baselines manifest written: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_baselines_summary(report: BaselineReport) -> None:
    """Print a human-readable result table to stdout."""
    print("\n" + "=" * 70)
    print("  E3 BASELINES - SUMMARY")
    print("=" * 70)
    header = f"  {'Task':<6} {'Baseline':<16} {'Split':<6} {'N':>6} {'Min':>6} {'Max':>6} {'Mean':>6}  Status"
    print(header)
    print("  " + "-" * 66)
    for r in report.results:
        status = "SKIPPED" if r.skipped else "OK"
        detail = f"  ({r.skip_reason[:30]})" if r.skipped and r.skip_reason else ""
        print(
            f"  {r.task.upper():<6} {r.baseline_name:<16} {r.split:<6} "
            f"{r.n_pairs:>6,} {r.score_min:>6.3f} {r.score_max:>6.3f} "
            f"{r.score_mean:>6.3f}  {status}{detail}"
        )
    print()
    if report.all_warnings:
        print(f"  WARNINGS ({len(report.all_warnings)}):")
        for w in report.all_warnings:
            print(f"    - {w[:100]}")
    else:
        print("  No warnings.")
    print()
    if report.master_path:
        print(f"  Master file : {report.master_path}")
    if report.manifest_path:
        print(f"  Manifest    : {report.manifest_path}")
    print("=" * 70 + "\n")
