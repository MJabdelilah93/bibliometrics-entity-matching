"""tfidf.py — Baseline C: TF-IDF cosine similarity.

A transparent, reproducible vector baseline.  The TF-IDF vectorizer is
fit on either the full corpus of instances or only the eval pairs (see
``tfidf_fit_corpus`` in config).  Pairwise cosine similarity is computed
for all benchmark pairs.

AND vectorisation
-----------------
  Text = ID-stripped author_norm.
  When fit_corpus = 'all_instances': fit on ALL author_instances (approx 183k texts).
  When fit_corpus = 'eval_pairs_only': fit on the 2 x N unique eval texts only.

AIN vectorisation
-----------------
  Text = affil_norm (normalised affiliation string).
  When fit_corpus = 'all_instances': fit on ALL affil_instances (approx 173k texts).

Fairness
--------
  AND: Author ID is stripped before vectorisation.
  AIN: year_int is not used.

Requires: scikit-learn (add ``scikit-learn`` to requirements.txt).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bem.eval.baselines.features import strip_author_id

BASELINE_NAME = "tfidf"


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run(
    task: str,
    features_df: pd.DataFrame,
    *,
    max_features: int,
    ngram_range: tuple[int, int],
    sublinear_tf: bool,
    fit_corpus: str,
    author_instances_path: Path | None = None,
    affil_instances_path: Path | None = None,
) -> pd.Series:
    """Fit a TF-IDF vectorizer and return cosine similarity scores.

    Args:
        task:                  ``"and"`` or ``"ain"``.
        features_df:           Features DataFrame for that task.
        max_features:          Maximum vocabulary size.
        ngram_range:           Tuple ``(min_n, max_n)`` for n-gram extraction.
        sublinear_tf:          Apply sublinear TF scaling (log(1 + tf)).
        fit_corpus:            ``'all_instances'`` or ``'eval_pairs_only'``.
        author_instances_path: Required when fit_corpus='all_instances' and task='and'.
        affil_instances_path:  Required when fit_corpus='all_instances' and task='ain'.

    Returns:
        Float64 Series with cosine similarity scores in [0, 1].

    Raises:
        ImportError: If scikit-learn is not installed.
        ValueError:  If *task* is not recognised.
    """
    _check_sklearn()

    if task == "and":
        return _run_and(
            features_df,
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            fit_corpus=fit_corpus,
            author_instances_path=author_instances_path,
        )
    if task == "ain":
        return _run_ain(
            features_df,
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            fit_corpus=fit_corpus,
            affil_instances_path=affil_instances_path,
        )
    raise ValueError(f"[tfidf] Unknown task: {task!r}")


# ---------------------------------------------------------------------------
# Per-task implementations
# ---------------------------------------------------------------------------

def _run_and(
    features_df: pd.DataFrame,
    max_features: int,
    ngram_range: tuple[int, int],
    sublinear_tf: bool,
    fit_corpus: str,
    author_instances_path: Path | None,
) -> pd.Series:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        analyzer="word",
    )

    if fit_corpus == "all_instances":
        if author_instances_path is None or not author_instances_path.exists():
            raise FileNotFoundError(
                "[tfidf] author_instances_path required when fit_corpus='all_instances'. "
                "Check eval_config.yaml -> eval_inputs.candidates_and path is set "
                "and data/interim/author_instances.parquet exists."
            )
        print(f"[tfidf] AND: loading corpus from {author_instances_path.name} ...")
        corpus_df = pd.read_parquet(author_instances_path, columns=["author_norm"])
        corpus = corpus_df["author_norm"].fillna("").apply(strip_author_id).tolist()
        print(f"[tfidf] AND: fitting on {len(corpus):,} corpus texts ...")
    else:
        # eval_pairs_only — fit on anchor+candidate texts from the eval set
        corpus = (
            features_df["name_clean_anchor"].fillna("").tolist()
            + features_df["name_clean_candidate"].fillna("").tolist()
        )
        print(f"[tfidf] AND: fitting on {len(corpus):,} eval-pair texts (eval_pairs_only mode) ...")

    vec.fit(corpus)

    anchor_texts = features_df["name_clean_anchor"].fillna("").tolist()
    cand_texts   = features_df["name_clean_candidate"].fillna("").tolist()

    scores = _paired_cosine(vec, anchor_texts, cand_texts)
    return pd.Series(scores, index=features_df.index, dtype="float64")


def _run_ain(
    features_df: pd.DataFrame,
    max_features: int,
    ngram_range: tuple[int, int],
    sublinear_tf: bool,
    fit_corpus: str,
    affil_instances_path: Path | None,
) -> pd.Series:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        analyzer="word",
    )

    if fit_corpus == "all_instances":
        if affil_instances_path is None or not affil_instances_path.exists():
            raise FileNotFoundError(
                "[tfidf] affil_instances_path required when fit_corpus='all_instances'. "
                "Ensure data/interim/affil_instances.parquet exists."
            )
        print(f"[tfidf] AIN: loading corpus from {affil_instances_path.name} ...")
        corpus_df = pd.read_parquet(affil_instances_path, columns=["affil_norm"])
        corpus = corpus_df["affil_norm"].fillna("").tolist()
        print(f"[tfidf] AIN: fitting on {len(corpus):,} corpus texts ...")
    else:
        corpus = (
            features_df["affil_norm_anchor"].fillna("").tolist()
            + features_df["affil_norm_candidate"].fillna("").tolist()
        )
        print(f"[tfidf] AIN: fitting on {len(corpus):,} eval-pair texts (eval_pairs_only mode) ...")

    vec.fit(corpus)

    anchor_texts = features_df["affil_norm_anchor"].fillna("").tolist()
    cand_texts   = features_df["affil_norm_candidate"].fillna("").tolist()

    scores = _paired_cosine(vec, anchor_texts, cand_texts)
    return pd.Series(scores, index=features_df.index, dtype="float64")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _paired_cosine(vec: Any, texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    """Vectorise texts_a and texts_b and return pairwise cosine similarities."""
    from sklearn.metrics.pairwise import paired_cosine_distances

    A = vec.transform(texts_a)
    B = vec.transform(texts_b)
    sims = 1.0 - paired_cosine_distances(A, B)
    return np.clip(sims, 0.0, 1.0)


def _check_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise ImportError(
            "[tfidf] scikit-learn is required for the TF-IDF baseline. "
            "Install with:  pip install scikit-learn"
        )
