"""e2_metrics.py — Stage E2: compute evaluation metrics (NOT YET IMPLEMENTED).

Planned outputs (one JSON per task):
    outputs/eval/<eval_run_id>/results/metrics_{task}.json

Planned metrics:
    - Precision, Recall, F1 at BEM routing threshold
    - Precision-Recall AUC (sklearn)
    - Confusion matrix (TP/FP/TN/FN) per split (dev / test)
    - Coverage: fraction of gold pairs with a BEM decision
"""

from __future__ import annotations

from pathlib import Path

from bem.eval.config import EvalConfig


def run_metrics(cfg: EvalConfig, eval_run_dir: Path) -> None:
    raise NotImplementedError(
        "[E2] Metrics stage not yet implemented. "
        "Implement run_metrics() in src/bem/eval/stages/e2_metrics.py."
    )
