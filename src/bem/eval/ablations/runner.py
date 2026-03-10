"""runner.py -- Stage E6 orchestrator for all ablation experiments.

Dispatches to the individual ablation modules (a2_no_guards, a3_k_sensitivity,
a6_threshold_sweep, a1_single_prompt, a4_missing_fields_and,
a5_missing_fields_ain) in order:

  1. Offline / cheap ablations (no API required):
       A2: No-guards
       A3: K sensitivity
       A6: Threshold sweep
  2. API-based ablations (require ANTHROPIC_API_KEY + confirmation):
       A1: Single-prompt LLM
       A4: AND missing-field ablations
       A5: AIN missing-field ablations

Skip/force semantics are consistent with the rest of the eval pipeline:
  - force=False  -> existing outputs are loaded from cache and stage is skipped
  - force=True   -> all outputs are recomputed from scratch

Manifest records: parameters, random seed, git commit (if available),
output file hashes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from bem.eval.config import EvalConfig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_ablations(cfg: EvalConfig) -> dict[str, Any]:
    """Execute all enabled ablation experiments.

    Args:
        cfg: Validated :class:`~bem.eval.config.EvalConfig`.

    Returns:
        dict with keys:
            results   -- per-ablation result dicts / DataFrames
            tables    -- paths of written table files
            figures   -- paths of written figure files
            manifest_path -- Path of ablation_manifest.json
    """
    abl = cfg.ablation
    out = abl.output_dir
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.random_seed)
    n_bs = cfg.evaluation.bootstrap_n_samples
    alpha = cfg.evaluation.bootstrap_alpha

    # Resolved paths used across ablations
    log_and  = cfg.routing_and_bm
    log_ain  = cfg.routing_ain_bm
    bm_and   = cfg.benchmark_and
    bm_ain   = cfg.benchmark_ain
    cand_and = cfg.candidates_and
    cand_ain = cfg.candidates_and.parent / "candidates_ain.parquet"  # sibling
    inst_and = cand_and.parent / "author_instances.parquet"
    inst_ain = cand_and.parent / "affil_instances.parquet"
    rec_norm = cand_and.parent / "records_normalised.parquet"
    prompt_and = cfg.config_path.parent / "prompts" / "and_verifier_v1.txt"
    prompt_ain = cfg.config_path.parent / "prompts" / "ain_verifier_v1.txt"
    thresh_manifest = cfg.thresholds_manifest

    all_results: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # A2: No guards  (offline)
    # ------------------------------------------------------------------
    if abl.run_a2_no_guards:
        print("\n[E6/A2] No-guards ablation ...")
        from bem.eval.ablations.a2_no_guards import run_a2
        all_results["a2_no_guards"] = run_a2(
            routing_log_and=log_and,
            routing_log_ain=log_ain,
            thresholds_manifest=thresh_manifest,
            tasks=cfg.tasks,
            rng=rng,
            n_bootstrap=n_bs,
            alpha=alpha,
            output_dir=out / "a2_no_guards",
            force=cfg.force,
        )
    else:
        print("[E6/A2] skipped (run.a2_no_guards: false)")

    # ------------------------------------------------------------------
    # A3: K sensitivity  (offline -- reads full candidates parquet)
    # ------------------------------------------------------------------
    if abl.run_a3_k_sensitivity:
        print("\n[E6/A3] K-sensitivity ablation ...")
        from bem.eval.ablations.a3_k_sensitivity import run_a3
        all_results["a3_k_sensitivity"] = run_a3(
            candidates_and=cand_and,
            candidates_ain=cand_ain,
            routing_log_and=log_and,
            routing_log_ain=log_ain,
            benchmark_and=bm_and,
            benchmark_ain=bm_ain,
            tasks=cfg.tasks,
            k_values=abl.a3_k_values,
            rng=rng,
            n_bootstrap=n_bs,
            alpha=alpha,
            output_dir=out / "a3_k_sensitivity",
            force=cfg.force,
        )
    else:
        print("[E6/A3] skipped (run.a3_k_sensitivity: false)")

    # ------------------------------------------------------------------
    # A6: Threshold sweep  (offline)
    # ------------------------------------------------------------------
    if abl.run_a6_threshold_sweep:
        print("\n[E6/A6] Threshold-sweep ablation ...")
        from bem.eval.ablations.a6_threshold_sweep import run_a6
        all_results["a6_threshold_sweep"] = run_a6(
            routing_log_and=log_and,
            routing_log_ain=log_ain,
            tasks=cfg.tasks,
            t_match_values=abl.a6_t_match_values,
            m_signals_values=abl.a6_m_signals_values,
            t_nonmatch_fixed=abl.a6_t_nonmatch_fixed,
            rng=rng,
            n_bootstrap=n_bs,
            alpha=alpha,
            output_dir=out / "a6_threshold_sweep",
            force=cfg.force,
        )
    else:
        print("[E6/A6] skipped (run.a6_threshold_sweep: false)")

    # ------------------------------------------------------------------
    # A1: Single-prompt LLM  (expensive -- API calls)
    # ------------------------------------------------------------------
    if abl.run_a1_single_prompt:
        print("\n[E6/A1] Single-prompt LLM ablation ...")
        _confirm_api_ablation("A1: Single-Prompt LLM", abl.a1_require_confirmation)
        from bem.eval.ablations.a1_single_prompt import run_a1
        all_results["a1_single_prompt"] = run_a1(
            routing_log_and=log_and,
            routing_log_ain=log_ain,
            author_instances=inst_and,
            affil_instances=inst_ain,
            records_normalised=rec_norm,
            tasks=cfg.tasks,
            model=abl.a1_model,
            max_pairs_per_task=abl.a1_max_pairs_per_task,
            require_confirmation=abl.a1_require_confirmation,
            rng=rng,
            n_bootstrap=n_bs,
            alpha=alpha,
            output_dir=out / "a1_single_prompt",
            force=cfg.force,
        )
    else:
        print("[E6/A1] skipped (run.a1_single_prompt: false)")

    # ------------------------------------------------------------------
    # A4: AND missing-field ablations  (expensive -- API calls)
    # ------------------------------------------------------------------
    if abl.run_a4_missing_fields_and and "and" in cfg.tasks:
        print("\n[E6/A4] AND missing-field ablations ...")
        _confirm_api_ablation("A4: AND missing fields", abl.a4_require_confirmation)
        from bem.eval.ablations.a4_missing_fields_and import run_a4
        variants = []
        if abl.a4_remove_coauthor:
            variants.append("remove_coauthor")
        if abl.a4_remove_affiliation:
            variants.append("remove_affiliation")
        all_results["a4_missing_fields_and"] = run_a4(
            routing_log_and=log_and,
            author_instances=inst_and,
            affil_instances=inst_ain,
            records_normalised=rec_norm,
            benchmark_and=bm_and,
            and_prompt_path=prompt_and,
            thresholds_manifest=thresh_manifest,
            model=abl.a1_model,  # reuse A1 model setting
            max_pairs=abl.a4_max_pairs,
            require_confirmation=abl.a4_require_confirmation,
            variants=variants,
            rng=rng,
            n_bootstrap=n_bs,
            alpha=alpha,
            output_dir=out / "a4_missing_fields_and",
            force=cfg.force,
        )
    else:
        print("[E6/A4] skipped (run.a4_missing_fields_and: false or 'and' not in tasks)")

    # ------------------------------------------------------------------
    # A5: AIN missing-field ablations  (expensive -- API calls)
    # ------------------------------------------------------------------
    if abl.run_a5_missing_fields_ain and "ain" in cfg.tasks:
        print("\n[E6/A5] AIN missing-field ablations ...")
        _confirm_api_ablation("A5: AIN missing fields", abl.a5_require_confirmation)
        from bem.eval.ablations.a5_missing_fields_ain import run_a5
        variants = []
        if abl.a5_raw_affil_only:
            variants.append("raw_affil_only")
        if abl.a5_remove_author_link:
            variants.append("remove_author_link")
        all_results["a5_missing_fields_ain"] = run_a5(
            routing_log_ain=log_ain,
            author_instances=inst_and,
            affil_instances=inst_ain,
            records_normalised=rec_norm,
            benchmark_ain=bm_ain,
            ain_prompt_path=prompt_ain,
            thresholds_manifest=thresh_manifest,
            model=abl.a1_model,
            max_pairs=abl.a5_max_pairs,
            require_confirmation=abl.a5_require_confirmation,
            variants=variants,
            rng=rng,
            n_bootstrap=n_bs,
            alpha=alpha,
            output_dir=out / "a5_missing_fields_ain",
            force=cfg.force,
        )
    else:
        print("[E6/A5] skipped (run.a5_missing_fields_ain: false or 'ain' not in tasks)")

    # ------------------------------------------------------------------
    # Tables and figures
    # ------------------------------------------------------------------
    print("\n[E6] Writing tables ...")
    from bem.eval.ablations.tables import build_ablation_tables
    tables_dir = out / "tables"
    table_paths = build_ablation_tables(all_results, tables_dir, cfg.tasks)

    print("[E6] Writing figures ...")
    from bem.eval.ablations.figures import build_ablation_figures
    figures_dir = out / "figures"
    figure_paths = build_ablation_figures(
        all_results, figures_dir, cfg.tasks,
        dpi=cfg.evaluation.figure_dpi,
        fmt=cfg.evaluation.figure_format,
    )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    manifest_path = _write_manifest(
        cfg=cfg,
        all_results=all_results,
        table_paths=table_paths,
        figure_paths=figure_paths,
        out_dir=out,
    )

    return {
        "results":       all_results,
        "tables":        table_paths,
        "figures":       figure_paths,
        "manifest_path": manifest_path,
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_ablation_summary(results: dict[str, Any], manifest_path: Path) -> None:
    """Print a short execution summary after E6 completes."""
    print("\n" + "=" * 72)
    print("Stage E6 -- Ablation Study  SUMMARY")
    print("=" * 72)

    ablation_labels = {
        "a1_single_prompt":      "A1  Single-prompt LLM",
        "a2_no_guards":          "A2  No C6 guards",
        "a3_k_sensitivity":      "A3  K sensitivity",
        "a4_missing_fields_and": "A4  AND missing fields",
        "a5_missing_fields_ain": "A5  AIN missing fields",
        "a6_threshold_sweep":    "A6  Threshold sweep",
    }
    for key, label in ablation_labels.items():
        data = results.get(key)
        if data is None:
            status = "SKIPPED"
        elif isinstance(data, dict) and not data:
            status = "no results"
        else:
            status = "OK"
        print(f"  {label:35s}  {status}")

    print(f"\n  Manifest : {manifest_path}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Confirmation checkpoint for expensive API ablations
# ---------------------------------------------------------------------------

def _confirm_api_ablation(label: str, require_confirmation: bool) -> None:
    """Print a notice; raise RuntimeError if non-interactive and confirmation required."""
    import sys
    print(f"\n  [CHECKPOINT] {label} requires ANTHROPIC API calls.")
    if not require_confirmation:
        return
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"[E6] {label}: require_confirmation=true but running non-interactively. "
            "Set require_confirmation: false or run interactively."
        )
    # Individual ablation modules handle the per-task y/N prompts.


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _write_manifest(
    cfg: EvalConfig,
    all_results: dict,
    table_paths: dict,
    figure_paths: dict,
    out_dir: Path,
) -> Path:
    git_commit = _git_commit(cfg.config_path.parent.parent)
    abl = cfg.ablation

    # Collect output file hashes
    file_hashes: dict[str, str] = {}
    for label, path in {**table_paths, **figure_paths}.items():
        if isinstance(path, Path) and path.exists():
            file_hashes[label] = _sha256(path)

    manifest = {
        "stage":       "E6_ablations",
        "status":      "completed",
        "timestamp":   datetime.now(tz=timezone.utc).isoformat(),
        "git_commit":  git_commit,
        "random_seed": cfg.random_seed,
        "tasks":       cfg.tasks,
        "ablations_run": {
            "a1_single_prompt":      abl.run_a1_single_prompt,
            "a2_no_guards":          abl.run_a2_no_guards,
            "a3_k_sensitivity":      abl.run_a3_k_sensitivity,
            "a4_missing_fields_and": abl.run_a4_missing_fields_and,
            "a5_missing_fields_ain": abl.run_a5_missing_fields_ain,
            "a6_threshold_sweep":    abl.run_a6_threshold_sweep,
        },
        "parameters": {
            "a1_model":              abl.a1_model,
            "a1_max_pairs_per_task": abl.a1_max_pairs_per_task,
            "a3_k_values":           abl.a3_k_values,
            "a4_max_pairs":          abl.a4_max_pairs,
            "a5_max_pairs":          abl.a5_max_pairs,
            "a6_t_match_values":     abl.a6_t_match_values,
            "a6_m_signals_values":   abl.a6_m_signals_values,
            "a6_t_nonmatch_fixed":   abl.a6_t_nonmatch_fixed,
        },
        "inputs": {
            "routing_and_bm":    str(cfg.routing_and_bm),
            "routing_ain_bm":    str(cfg.routing_ain_bm),
            "benchmark_and":     str(cfg.benchmark_and),
            "benchmark_ain":     str(cfg.benchmark_ain),
            "thresholds":        str(cfg.thresholds_manifest),
        },
        "output_dir":  str(out_dir),
        "output_files": file_hashes,
    }

    manifest_path = out_dir / "ablation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest_path


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=repo_root, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()
