"""embedding.py — Baseline D: sentence-embedding cosine similarity.

Encodes anchor and candidate strings with a sentence-transformer model,
then computes pairwise cosine similarity.

Features used
-------------
  AND: ID-stripped author_norm (``name_clean_*``).
  AIN: affil_norm (``affil_norm_*``).

Confirmation checkpoint
-----------------------
When ``require_confirmation=True`` (the default):
  - If stdin is a TTY: the function prints the embedding plan
    (model name, number of unique texts to encode) and waits for ``y/N``.
  - If stdin is NOT a TTY (CI / batch): embedding is SKIPPED automatically
    and a warning is printed.  Use ``require_confirmation=False`` in the
    config to bypass this check in batch mode.

Requires: sentence-transformers  (optional dependency; NOT in requirements.txt).
Install: pip install sentence-transformers
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_NAME = "embedding"


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run(
    task: str,
    features_df: pd.DataFrame,
    *,
    model_name: str,
    batch_size: int,
    device: str,
    require_confirmation: bool,
    random_seed: int,
) -> pd.Series | None:
    """Embed texts and return cosine similarity scores, or None if skipped.

    Args:
        task:                 ``"and"`` or ``"ain"``.
        features_df:          Features DataFrame for that task.
        model_name:           Sentence-Transformers model identifier.
        batch_size:           Encoding batch size.
        device:               ``"cpu"`` or ``"cuda"``.
        require_confirmation: If True, prompt before encoding (see module docs).
        random_seed:          Seed passed to numpy for reproducibility.

    Returns:
        Float64 Series with cosine similarity in [0, 1], or ``None`` if the
        user declined / the step was skipped in non-interactive mode.

    Raises:
        ImportError: If sentence-transformers is not installed.
        ValueError:  If *task* is not recognised.
    """
    _check_sentence_transformers()

    if task == "and":
        texts_a = features_df["name_clean_anchor"].fillna("").astype(str).tolist()
        texts_b = features_df["name_clean_candidate"].fillna("").astype(str).tolist()
    elif task == "ain":
        texts_a = features_df["affil_norm_anchor"].fillna("").astype(str).tolist()
        texts_b = features_df["affil_norm_candidate"].fillna("").astype(str).tolist()
    else:
        raise ValueError(f"[embedding] Unknown task: {task!r}")

    unique_texts = list(dict.fromkeys(texts_a + texts_b))  # preserve insertion order
    n_unique = len(unique_texts)

    if require_confirmation:
        if not _should_proceed(task, n_unique, model_name):
            return None  # user declined or non-interactive

    np.random.seed(random_seed)

    print(f"[embedding] {task.upper()}: encoding {n_unique:,} unique texts with '{model_name}' ...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        unique_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    text_to_idx = {t: i for i, t in enumerate(unique_texts)}
    idx_a = [text_to_idx[t] for t in texts_a]
    idx_b = [text_to_idx[t] for t in texts_b]

    emb_a = embeddings[idx_a]
    emb_b = embeddings[idx_b]

    # Cosine similarity: normalise then dot product
    norm_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-9)
    norm_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-9)
    cosine = (norm_a * norm_b).sum(axis=1)           # range [-1, 1]
    scores = np.clip((cosine + 1.0) / 2.0, 0.0, 1.0) # rescale to [0, 1]

    print(f"[embedding] {task.upper()}: done. Score range [{scores.min():.3f}, {scores.max():.3f}]")
    return pd.Series(scores.astype(float), index=features_df.index, dtype="float64")


# ---------------------------------------------------------------------------
# Confirmation logic
# ---------------------------------------------------------------------------

def _should_proceed(task: str, n_unique: int, model_name: str) -> bool:
    """Return True if the user confirms (or confirmation is not needed)."""
    if not sys.stdin.isatty():
        print(
            f"[embedding] {task.upper()}: SKIPPED — stdin is not a TTY (non-interactive "
            "environment) and embedding.require_confirmation=true. "
            "To run embedding in batch mode, set require_confirmation: false in "
            "eval_config.yaml or pass --no-embed-confirm on the CLI."
        )
        return False

    print()
    print(f"  [embedding] {task.upper()} embedding plan:")
    print(f"    model         : {model_name}")
    print(f"    unique texts  : {n_unique:,}")
    print(f"    note          : model will be downloaded if not cached (~90 MB for MiniLM)")
    resp = input("  Proceed? [y/N]: ").strip().lower()
    if resp != "y":
        print(f"[embedding] {task.upper()}: SKIPPED by user.")
        return False
    return True


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

def _check_sentence_transformers() -> None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        raise ImportError(
            "[embedding] sentence-transformers is required for the embedding baseline. "
            "Install with:  pip install sentence-transformers\n"
            "Alternatively, set baselines.run.embedding: false in eval_config.yaml "
            "to skip this baseline."
        )
