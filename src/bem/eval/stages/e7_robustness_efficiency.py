"""e7_robustness_efficiency.py -- Stage E7: Robustness & Efficiency summary.

Thin wrapper that calls the robustness and efficiency sub-runners in sequence
and prints a combined summary.

Robustness:
  - Per-slice metrics (overall / Q1 / Q2 / stratum if available)
  - Outputs: outputs/robustness/ + outputs/manuscript/table_robustness_*.{csv,md,tex}

Efficiency:
  - Token usage and cost estimates from llm_decisions JSONL
  - Runtime summary (C5 API span; C4/C6/C7 not available)
  - Aggregation diagnostics (cluster count, largest cluster, conflicts)
  - Environment capture (Python / model / SDK / platform)
  - Outputs: outputs/efficiency/ + outputs/manuscript/table_efficiency_*.{csv,md,tex}
"""

from __future__ import annotations

from pathlib import Path

from bem.eval.config import EvalConfig
from bem.eval.robustness.runner import print_robustness_summary, run_robustness
from bem.eval.efficiency.runner import print_efficiency_summary, run_efficiency


def run(cfg: EvalConfig) -> None:
    """Execute Stage E7: Robustness & Efficiency."""

    # --- Robustness slices ---
    print("\n[E7] Computing robustness slices ...")
    rob_result = run_robustness(cfg)
    print_robustness_summary(
        slice_dfs=rob_result["slice_dfs"],
        notes=rob_result["notes"],
        manifest_path=rob_result["manifest_path"],
    )

    # --- Efficiency summary ---
    print("\n[E7] Computing efficiency summary ...")
    eff_result = run_efficiency(cfg)
    print_efficiency_summary(
        token_usages=eff_result["token_usages"],
        costs=eff_result["costs"],
        aggregation=eff_result["aggregation"],
        manifest_path=eff_result["manifest_path"],
    )

    print(
        f"\n[E7] Done.  "
        f"Robustness -> {cfg.rob_eff.robustness_output_dir}  "
        f"Efficiency -> {cfg.rob_eff.efficiency_output_dir}"
    )
