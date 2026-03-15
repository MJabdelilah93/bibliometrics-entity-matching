"""e3_baselines.py — Stage E3: classical baseline scoring.

Delegates to :mod:`bem.eval.baselines.runner`.

Baselines implemented
---------------------
  A. deterministic  — exact name/affil match (discrete {0, 0.5, 1})
  B. fuzzy          — weighted rapidfuzz lexical model (continuous [0, 1])
  C. tfidf          — TF-IDF cosine similarity (requires scikit-learn)
  D. embedding      — sentence-embedding cosine (requires sentence-transformers;
                       gated by confirmation checkpoint)

Auxiliary (not a fair baseline)
--------------------------------
  aux_scopus_id     — AND Scopus Author ID equality pass from C4 candidate pool

Fairness constraints (enforced in bem.eval.baselines.features)
--------------------------------------------------------------
  AND: Author(s) ID is never used in any fair baseline.
  AIN: title / year / source are never used.
"""

from __future__ import annotations

from pathlib import Path

from bem.eval.config import EvalConfig
from bem.eval.baselines.runner import run_baselines, print_baselines_summary


def run(cfg: EvalConfig) -> None:
    """Execute Stage E3 for all configured tasks and baselines.

    Args:
        cfg: Validated :class:`~bem.eval.config.EvalConfig` instance.

    Raises:
        FileNotFoundError: If required input artefacts are missing.
        AssertionError:    If smoke checks fail on a baseline output.
    """
    report = run_baselines(cfg)
    print_baselines_summary(report)
