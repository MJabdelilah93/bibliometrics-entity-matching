"""e4_tune_thresholds.py — Stage E4 thin wrapper.

Delegates entirely to :mod:`bem.eval.thresholds.runner`.
"""

from __future__ import annotations

from bem.eval.config import EvalConfig
from bem.eval.thresholds.runner import run_tuning, print_tuning_summary


def run(cfg: EvalConfig) -> None:
    """Execute Stage E4: threshold tuning on the dev split.

    Args:
        cfg: Validated :class:`~bem.eval.config.EvalConfig` instance.

    Raises:
        FileNotFoundError: If the baseline scores master file is missing.
        ValueError: If test-split rows are found (no test leakage).
    """
    result = run_tuning(cfg)
    print_tuning_summary(result["results"], result["manifest_path"])
