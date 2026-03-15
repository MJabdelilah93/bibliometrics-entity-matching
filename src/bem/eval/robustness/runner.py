"""runner.py -- Stage E7 robustness sub-runner.

Loads the benchmark-filtered routing logs, computes per-slice metrics
(overall / Q1 / Q2 / stratum if available), and writes:

  Supplement-ready (outputs/robustness/):
    robustness_slices_{task}.csv   -- full metrics table per slice
    robustness_manifest.json       -- provenance

  Manuscript-ready (outputs/manuscript/):
    table_robustness_{task}.{csv,md,tex}
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bem.eval.config import EvalConfig
from bem.eval.robustness.slicer import (
    compute_robustness_slices,
    to_dataframe,
    to_latex,
    to_markdown,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_robustness(cfg: EvalConfig) -> dict[str, Any]:
    """Compute and write robustness slices for all tasks.

    Returns:
        dict with keys:
            slice_dfs    -- {task: pd.DataFrame}
            notes        -- {task: {slice_name: note_string}}
            written      -- {label: Path} for every file written
            manifest_path -- Path of robustness_manifest.json
    """
    rob_cfg = cfg.rob_eff
    out_dir = rob_cfg.robustness_output_dir
    ms_dir  = cfg.evaluation.manuscript_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ms_dir.mkdir(parents=True, exist_ok=True)

    rng       = np.random.default_rng(cfg.random_seed)
    n_bs      = cfg.evaluation.bootstrap_n_samples
    alpha     = cfg.evaluation.bootstrap_alpha
    log_paths = {
        "and": cfg.routing_and_bm,
        "ain": cfg.routing_ain_bm,
    }

    slice_dfs: dict[str, pd.DataFrame] = {}
    all_notes: dict[str, dict[str, str]] = {}
    written:   dict[str, Path] = {}

    for task in cfg.tasks:
        log_path = log_paths[task]
        if not log_path.exists():
            print(f"  [E7/rob/{task.upper()}] routing log not found -- skipping: {log_path}")
            continue

        print(f"  [E7/rob/{task.upper()}] loading routing log ...")
        routing_log = pd.read_parquet(log_path)

        print(f"  [E7/rob/{task.upper()}] computing slices ...")
        results, notes = compute_robustness_slices(
            routing_log=routing_log,
            task=task,
            instances_and=rob_cfg.instances_and,
            instances_ain=rob_cfg.instances_ain,
            compute_q1_q2=rob_cfg.compute_q1_q2,
            rng=rng,
            n_bootstrap=n_bs,
            alpha=alpha,
        )
        df = to_dataframe(results)
        slice_dfs[task] = df
        all_notes[task] = notes

        # --- Supplement: robustness_slices_{task}.csv ---
        supp_csv = out_dir / f"robustness_slices_{task}.csv"
        df.to_csv(supp_csv, index=False)
        written[f"rob_slices_{task}_csv"] = supp_csv
        print(f"    -> {supp_csv}")

        # --- Supplement: notes (append to CSV header comment) ---
        if notes:
            notes_path = out_dir / f"robustness_notes_{task}.json"
            notes_path.write_text(
                json.dumps({"task": task, "notes": notes}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            written[f"rob_notes_{task}_json"] = notes_path

        # --- Manuscript tables ---
        ms_csv = ms_dir / f"table_robustness_{task}.csv"
        df.to_csv(ms_csv, index=False)
        written[f"rob_ms_{task}_csv"] = ms_csv

        ms_md = ms_dir / f"table_robustness_{task}.md"
        ms_md.write_text(to_markdown(df, task), encoding="utf-8")
        written[f"rob_ms_{task}_md"] = ms_md

        ms_tex = ms_dir / f"table_robustness_{task}.tex"
        ms_tex.write_text(to_latex(df, task), encoding="utf-8")
        written[f"rob_ms_{task}_tex"] = ms_tex
        print(f"    -> manuscript: {ms_md.name}, {ms_tex.name}")

        # --- Console summary ---
        _print_slice_summary(task, df, notes)

    # --- Manifest ---
    manifest_path = _write_manifest(cfg, slice_dfs, all_notes, written, out_dir)

    return {
        "slice_dfs":     slice_dfs,
        "notes":         all_notes,
        "written":       written,
        "manifest_path": manifest_path,
    }


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_robustness_summary(slice_dfs: dict, notes: dict, manifest_path: Path) -> None:
    print("\n" + "=" * 72)
    print("Stage E7 -- Robustness & Efficiency  ROBUSTNESS SUMMARY")
    print("=" * 72)
    for task, df in slice_dfs.items():
        print(f"\n  Task: {task.upper()}")
        for _, row in df.iterrows():
            f1  = row.get("f1_match")
            cov = row.get("coverage")
            f1s  = f"{f1:.3f}"  if f1  is not None else "--"
            covs = f"{cov:.3f}" if cov is not None else "--"
            print(f"    {row['slice']:15s}  N={row['n_total']:5d}  F1_M={f1s}  Cov={covs}")
        task_notes = notes.get(task, {})
        for name, msg in task_notes.items():
            print(f"    [{name}] {msg}")
    print(f"\n  Manifest: {manifest_path}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_slice_summary(task: str, df: pd.DataFrame, notes: dict) -> None:
    for _, row in df.iterrows():
        f1  = row.get("f1_match")
        cov = row.get("coverage")
        f1s  = f"{f1:.3f}"  if f1  is not None else "--"
        covs = f"{cov:.3f}" if cov is not None else "--"
        print(
            f"    {row['slice']:10s}  N={row['n_total']:5d}  "
            f"F1_M={f1s}  Cov={covs}"
        )
    for name, msg in notes.items():
        print(f"    [note/{name}] {msg}")


def _write_manifest(
    cfg: EvalConfig,
    slice_dfs: dict,
    notes: dict,
    written: dict,
    out_dir: Path,
) -> Path:
    file_hashes = {}
    for label, path in written.items():
        if isinstance(path, Path) and path.exists():
            file_hashes[label] = _sha256(path)

    manifest = {
        "stage":           "E7_robustness",
        "status":          "completed",
        "timestamp":       datetime.now(tz=timezone.utc).isoformat(),
        "random_seed":     cfg.random_seed,
        "tasks":           cfg.tasks,
        "compute_q1_q2":   cfg.rob_eff.compute_q1_q2,
        "slices_computed": {
            task: [r["slice"] for r in df.to_dict("records")]
            for task, df in slice_dfs.items()
        },
        "notes":           notes,
        "inputs": {
            "routing_and_bm": str(cfg.routing_and_bm),
            "routing_ain_bm": str(cfg.routing_ain_bm),
            "instances_and":  str(cfg.rob_eff.instances_and),
            "instances_ain":  str(cfg.rob_eff.instances_ain),
        },
        "output_dir":   str(out_dir),
        "output_files": file_hashes,
    }
    manifest_path = out_dir / "robustness_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest_path


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()
