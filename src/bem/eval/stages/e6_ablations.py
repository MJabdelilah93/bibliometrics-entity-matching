"""e6_ablations.py — Stage E6 thin wrapper.

Delegates entirely to :mod:`bem.eval.ablations.runner`.
"""

from __future__ import annotations

from bem.eval.config import EvalConfig
from bem.eval.ablations.runner import run_ablations, print_ablation_summary


def run(cfg: EvalConfig) -> None:
    """Execute Stage E6: ablation study.

    Args:
        cfg: Validated :class:`~bem.eval.config.EvalConfig`.

    Raises:
        FileNotFoundError: If required stage inputs are missing.
        EnvironmentError:  If an API ablation is enabled but ANTHROPIC_API_KEY unset.
        RuntimeError:      If confirmation is required but running non-interactively.
    """
    result = run_ablations(cfg)
    print_ablation_summary(result["results"], result["manifest_path"])
