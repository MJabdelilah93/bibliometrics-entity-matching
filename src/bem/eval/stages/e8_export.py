"""e8_export.py -- Stage E8: final manuscript export and smoke checks.

Thin wrapper that calls the export collector and prints a summary.

Collects all manuscript-ready artefacts produced by E3-E7 into
outputs/manuscript/ (tables) and outputs/manuscript/figures/ (figures).
Runs existence smoke checks and writes a manifest + inventory CSV.
"""

from __future__ import annotations

from bem.eval.config import EvalConfig
from bem.eval.export.collector import (
    print_export_summary,
    run_export,
    run_smoke_checks,
)


def run(cfg: EvalConfig) -> None:
    """Execute Stage E8: manuscript export and smoke checks."""
    result = run_export(cfg)

    passed, failed = run_smoke_checks(result["inventory_df"])
    print_export_summary(
        inventory_df=result["inventory_df"],
        missing=result["missing"],
        manifest_path=result["manifest_path"],
    )

    if failed:
        # Non-fatal: print clearly but do not abort pipeline
        print(
            f"\n[E8] WARNING: {len(failed)} required file(s) are missing. "
            "Run the relevant stage(s) first to generate them."
        )
