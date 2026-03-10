"""e5_evaluation.py — Stage E5 thin wrapper."""

from __future__ import annotations

from bem.eval.config import EvalConfig
from bem.eval.evaluation.runner import run_evaluation, print_evaluation_summary


def run(cfg: EvalConfig) -> None:
    """Execute Stage E5: held-out evaluation with manuscript-ready outputs.

    Args:
        cfg: Validated :class:`~bem.eval.config.EvalConfig`.

    Raises:
        FileNotFoundError: If required stage inputs are missing.
        AssertionError:    If smoke checks fail.
    """
    result = run_evaluation(cfg)
    print_evaluation_summary(result["metrics_df"], result["manifest_path"])
