"""io_check.py — Artefact presence check (Stage E0 gate).

All mandatory inputs are verified before any computation starts.
Missing inputs raise :class:`ArtefactCheckError` immediately (fail fast).

Usage
-----
    from bem.eval.io_check import check_artefacts, print_artefact_report
    report = check_artefacts(cfg)
    print_artefact_report(report)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bem.eval.config import EvalConfig


class ArtefactCheckError(RuntimeError):
    """Raised when one or more mandatory input artefacts are missing."""


@dataclass
class ArtefactEntry:
    label: str
    path: Path
    required: bool
    present: bool
    size_bytes: int = 0


@dataclass
class ArtefactReport:
    entries: list[ArtefactEntry] = field(default_factory=list)
    all_required_present: bool = True


def check_artefacts(cfg: EvalConfig) -> ArtefactReport:
    """Verify every input artefact referenced in *cfg* exists on disk.

    Args:
        cfg: Validated :class:`EvalConfig` instance.

    Returns:
        :class:`ArtefactReport` summarising presence/absence of all inputs.

    Raises:
        ArtefactCheckError: If any required artefact is missing.
    """
    candidates: list[tuple[str, Path, bool]] = [
        # (label, path, required)
        ("AND benchmark pairs (dev2)",      cfg.benchmark_and,      True),
        ("AIN benchmark pairs (dev2)",      cfg.benchmark_ain,      True),
        ("AND bm routing log",              cfg.routing_and_bm,     "and" in cfg.tasks),
        ("AIN bm routing log",              cfg.routing_ain_bm,     "ain" in cfg.tasks),
        ("Thresholds manifest",             cfg.thresholds_manifest, True),
    ]

    report = ArtefactReport()
    missing_required: list[str] = []

    for label, path, required in candidates:
        present = path.exists()
        size = path.stat().st_size if present else 0
        entry = ArtefactEntry(
            label=label,
            path=path,
            required=bool(required),
            present=present,
            size_bytes=size,
        )
        report.entries.append(entry)
        if required and not present:
            missing_required.append(f"  - {label}: {path}")

    if missing_required:
        report.all_required_present = False
        lines = "\n".join(missing_required)
        raise ArtefactCheckError(
            f"[E0] Missing mandatory artefacts — cannot proceed:\n{lines}"
        )

    return report


def print_artefact_report(report: ArtefactReport) -> None:
    """Print a human-readable artefact check table to stdout."""
    print("\n=== Artefact Check (E0) ===")
    w_label = max(len(e.label) for e in report.entries) + 2
    header = f"{'Artefact':<{w_label}}  {'Required':<8}  {'Status':<8}  {'Size'}"
    print(header)
    print("-" * len(header))
    for e in report.entries:
        status = "OK" if e.present else "MISSING"
        req_str = "yes" if e.required else "no"
        size_str = f"{e.size_bytes:,} B" if e.present else "—"
        flag = "" if e.present else "  <-- MISSING"
        print(f"{e.label:<{w_label}}  {req_str:<8}  {status:<8}  {size_str}{flag}")
    print()
    if report.all_required_present:
        print("[E0] All required artefacts present.")
    else:
        print("[E0] FAILED — see missing artefacts above.")
    print()


def print_dry_run_plan(cfg: EvalConfig) -> None:
    """Print the plan that would be executed if dry_run were False."""
    print("\n=== Dry-Run Plan ===")
    print(f"  Tasks           : {cfg.tasks}")
    print(f"  Benchmark AND   : {cfg.benchmark_and}")
    print(f"  Benchmark AIN   : {cfg.benchmark_ain}")
    print(f"  Routing AND bm  : {cfg.routing_and_bm}")
    print(f"  Routing AIN bm  : {cfg.routing_ain_bm}")
    print(f"  Thresholds      : {cfg.thresholds_manifest}")
    print(f"  Output dir      : {cfg.output_dir}")
    print(f"  Random seed     : {cfg.random_seed}")
    print(f"  Force           : {cfg.force}")
    print()
    print("  Stages that WOULD run:")
    print("    E1  Prediction join  -> predictions_{and,ain}.parquet")
    print("    E2  Metrics          -> [NOT YET IMPLEMENTED]")
    print("    E3  Baselines        -> [NOT YET IMPLEMENTED]")
    print()
    print("  Re-run without --dry-run (or set dry_run: false in config) to execute.")
    print()
