"""collector.py -- Stage E8: final manuscript export and smoke checks.

Collects all manuscript-ready artefacts produced by stages E3-E7 into a
single directory (outputs/manuscript/) and runs existence smoke checks.

What this stage does
--------------------
1. Copies ablation tables (CSV / MD / TEX) from outputs/ablations/tables/
   to outputs/manuscript/ so all table formats live in one place.
2. Copies all figures into outputs/manuscript/figures/.
   Sources: outputs/manuscript/ (E5 figures) and outputs/ablations/figures/ (E6).
3. Writes a comprehensive manuscript_inventory.csv listing every file,
   its source stage, size (bytes), and SHA-256 hash.
4. Runs existence smoke checks against a declared list of expected files.
   Missing files are reported clearly — not hidden.
5. Writes manuscript_manifest.json for provenance.

What this stage does NOT do
---------------------------
- It does not modify existing artefacts.
- It does not delete any outputs.
- If a source file is absent (e.g. because a stage was skipped or used
  force=false on a fresh run), the inventory records it as MISSING.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bem.eval.config import EvalConfig


# ---------------------------------------------------------------------------
# Catalogue of expected manuscript artefacts
# ---------------------------------------------------------------------------

def _declare_expected(cfg: EvalConfig) -> list[dict]:
    """Return the full list of expected manuscript artefacts.

    Each entry:
        label       -- human-readable name
        stage       -- which eval stage produces this file
        source      -- Path where the stage writes it originally
        dest_rel    -- Path relative to manuscript_dir (None = already in manuscript_dir)
        required    -- bool: True = smoke-check failure if missing
    """
    ms   = cfg.evaluation.manuscript_dir          # outputs/manuscript
    ev   = cfg.evaluation.output_dir              # outputs/evaluation
    abl  = cfg.ablation.output_dir                # outputs/ablations
    rob  = cfg.rob_eff.robustness_output_dir      # outputs/robustness
    eff  = cfg.rob_eff.efficiency_output_dir      # outputs/efficiency
    thr  = cfg.tuning.output_dir                  # outputs/thresholds

    tasks = cfg.tasks
    expected: list[dict] = []

    def _add(label, stage, source, dest_rel=None, required=True):
        expected.append({
            "label":    label,
            "stage":    stage,
            "source":   Path(source),
            "dest_rel": Path(dest_rel) if dest_rel else None,
            "required": required,
        })

    # --- E4: threshold summaries (supplement only) ---
    for task in tasks:
        _add(
            f"threshold_summary_{task}",
            "E4",
            thr / f"threshold_summary_{task}.csv",
            required=True,
        )

    # --- E5: main evaluation tables (already in manuscript_dir) ---
    for task in tasks:
        for ext in ("csv", "md", "tex"):
            _add(
                f"table_{task}_dev.{ext}",
                "E5",
                ms / f"table_{task}_dev.{ext}",
                required=True,
            )
        _add(
            f"figure_precision_coverage_{task}",
            "E5",
            ms / f"figure_precision_coverage_{task}.{cfg.evaluation.figure_format}",
            dest_rel=f"figures/figure_precision_coverage_{task}.{cfg.evaluation.figure_format}",
            required=True,
        )

    # --- E6: ablation tables (need copying to manuscript_dir) ---
    abl_tables = [
        ("ablation_main",            "E6"),
        ("ablation_k_sensitivity",   "E6"),
    ] + [
        (f"ablation_threshold_sweep_{task}", "E6")
        for task in tasks
    ]
    for stem, stage in abl_tables:
        for ext in ("csv", "md", "tex"):
            _add(
                f"{stem}.{ext}",
                stage,
                abl / "tables" / f"{stem}.{ext}",
                dest_rel=f"{stem}.{ext}",
                required=True,
            )

    # --- E6: ablation figures (need copying to manuscript_dir/figures/) ---
    abl_figs = [
        "figure_ablation_main",
        "figure_ablation_k_sensitivity",
    ] + [f"figure_ablation_sweep_{task}" for task in tasks]
    fmt = cfg.evaluation.figure_format
    for stem in abl_figs:
        _add(
            stem,
            "E6",
            abl / "figures" / f"{stem}.{fmt}",
            dest_rel=f"figures/{stem}.{fmt}",
            required=True,
        )

    # --- E7: robustness tables (already in manuscript_dir) ---
    for task in tasks:
        for ext in ("csv", "md", "tex"):
            _add(
                f"table_robustness_{task}.{ext}",
                "E7",
                ms / f"table_robustness_{task}.{ext}",
                required=True,
            )

    # --- E7: efficiency tables (already in manuscript_dir) ---
    for stem in ("table_efficiency_tokens", "table_efficiency_comparison"):
        for ext in ("csv", "md", "tex"):
            _add(
                f"{stem}.{ext}",
                "E7",
                ms / f"{stem}.{ext}",
                required=True,
            )

    # --- E7: supplement files (informational; not required to be in ms_dir) ---
    for task in tasks:
        _add(
            f"robustness_slices_{task}.csv",
            "E7",
            rob / f"robustness_slices_{task}.csv",
            required=False,
        )
    _add("cost_summary.csv",              "E7", eff / "cost_summary.csv",       required=False)
    _add("aggregation_diagnostics.csv",   "E7", eff / "aggregation_diagnostics.csv", required=False)
    _add("runtime_summary.csv",           "E7", eff / "runtime_summary.csv",    required=False)
    _add("environment.json",              "E7", eff / "environment.json",        required=False)

    return expected


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_export(cfg: EvalConfig) -> dict[str, Any]:
    """Collect, copy, smoke-check, and inventory all manuscript artefacts.

    Returns:
        dict with keys:
            inventory_df   -- pd.DataFrame of all items
            missing        -- list of required items that are absent
            written        -- {label: Path} for files written by this stage
            manifest_path  -- Path of manuscript_manifest.json
    """
    ms_dir = cfg.evaluation.manuscript_dir
    ms_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ms_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    expected = _declare_expected(cfg)
    inventory_rows: list[dict] = []
    written: dict[str, Path] = {}
    missing: list[str] = []
    copied: list[str] = []

    for item in expected:
        src      = item["source"]
        dest_rel = item["dest_rel"]
        label    = item["label"]
        stage    = item["stage"]
        required = item["required"]

        # Determine destination path (None = source already IS the destination)
        if dest_rel is not None:
            dest = ms_dir / dest_rel
        else:
            dest = src    # file already lives in manuscript_dir

        # Existence check on the source
        if not src.exists():
            status = "MISSING"
            size   = None
            sha256 = None
            if required:
                missing.append(label)
                print(f"  [E8] MISSING (required): {label}  <- {src}")
            else:
                print(f"  [E8] absent  (optional): {label}")
        else:
            # Copy if the destination differs from source and doesn't exist yet
            if dest != src and not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                copied.append(label)
                written[f"copied_{label}"] = dest

            final_path = dest if dest != src else src
            size   = final_path.stat().st_size
            sha256 = _sha256(final_path)
            status = "OK"

        inventory_rows.append({
            "label":     label,
            "stage":     stage,
            "source":    str(src),
            "dest":      str(dest),
            "status":    status,
            "required":  required,
            "size_bytes": size,
            "sha256":    sha256,
        })

    inventory_df = pd.DataFrame(inventory_rows)

    # Write inventory
    inv_csv = ms_dir / "manuscript_inventory.csv"
    inventory_df.to_csv(inv_csv, index=False)
    written["manuscript_inventory_csv"] = inv_csv

    # Write manifest
    manifest_path = _write_manifest(cfg, inventory_df, missing, written, ms_dir)
    written["manuscript_manifest_json"] = manifest_path

    if copied:
        print(f"\n  [E8] Copied {len(copied)} file(s) into {ms_dir}")
    if missing:
        print(f"\n  [E8] {len(missing)} required file(s) MISSING:")
        for m in missing:
            print(f"       - {m}")

    return {
        "inventory_df":  inventory_df,
        "missing":       missing,
        "written":       written,
        "manifest_path": manifest_path,
    }


# ---------------------------------------------------------------------------
# Smoke checks
# ---------------------------------------------------------------------------

def run_smoke_checks(inventory_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Run existence smoke checks from the inventory.

    Returns:
        (passed_labels, failed_labels)
    """
    passed = []
    failed = []
    required_rows = inventory_df[inventory_df["required"]]
    for _, row in required_rows.iterrows():
        if row["status"] == "OK":
            passed.append(row["label"])
        else:
            failed.append(row["label"])
    return passed, failed


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_export_summary(
    inventory_df: pd.DataFrame,
    missing: list[str],
    manifest_path: Path,
) -> None:
    print("\n" + "=" * 72)
    print("Stage E8 -- Manuscript Export  SUMMARY")
    print("=" * 72)

    ok_required  = (inventory_df["status"] == "OK")  & inventory_df["required"]
    ok_optional  = (inventory_df["status"] == "OK")  & ~inventory_df["required"]
    mis_required = (inventory_df["status"] != "OK")  & inventory_df["required"]
    mis_optional = (inventory_df["status"] != "OK")  & ~inventory_df["required"]

    print(f"\n  Required files present  : {ok_required.sum():3d}")
    print(f"  Required files MISSING  : {mis_required.sum():3d}  {'<-- ACTION NEEDED' if mis_required.sum() else ''}")
    print(f"  Optional files present  : {ok_optional.sum():3d}")
    print(f"  Optional files absent   : {mis_optional.sum():3d}")

    # Group by stage
    print("\n  Files by stage:")
    for stage in sorted(inventory_df["stage"].unique()):
        sub = inventory_df[inventory_df["stage"] == stage]
        n_ok  = (sub["status"] == "OK").sum()
        n_mis = (sub["status"] != "OK").sum()
        bar = "OK" if n_mis == 0 else f"{n_mis} MISSING"
        print(f"    {stage}  {n_ok:2d} present  {bar}")

    if missing:
        print("\n  Missing required files:")
        for m in missing:
            print(f"    - {m}")
        print(
            "\n  To generate missing files, run the relevant stage(s) with "
            "--no-dry-run first."
        )
    else:
        print("\n  All required files present. Manuscript export is complete.")

    print(f"\n  Inventory : {manifest_path.parent / 'manuscript_inventory.csv'}")
    print(f"  Manifest  : {manifest_path}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_manifest(
    cfg: EvalConfig,
    inventory_df: pd.DataFrame,
    missing: list[str],
    written: dict,
    ms_dir: Path,
) -> Path:
    n_ok  = int((inventory_df["status"] == "OK").sum())
    n_req = int(inventory_df["required"].sum())
    n_mis = int((inventory_df["required"] & (inventory_df["status"] != "OK")).sum())

    manifest = {
        "stage":      "E8_export",
        "status":     "completed",
        "timestamp":  datetime.now(tz=timezone.utc).isoformat(),
        "tasks":      cfg.tasks,
        "summary": {
            "total_declared": len(inventory_df),
            "present":        n_ok,
            "missing_required": n_mis,
            "complete":       n_mis == 0,
        },
        "missing_required": missing,
        "source_dirs": {
            "manuscript":  str(cfg.evaluation.manuscript_dir),
            "evaluation":  str(cfg.evaluation.output_dir),
            "ablations":   str(cfg.ablation.output_dir),
            "robustness":  str(cfg.rob_eff.robustness_output_dir),
            "efficiency":  str(cfg.rob_eff.efficiency_output_dir),
            "thresholds":  str(cfg.tuning.output_dir),
        },
        "output_dir":  str(ms_dir),
    }
    manifest_path = ms_dir / "manuscript_manifest.json"
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
