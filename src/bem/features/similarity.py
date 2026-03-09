"""similarity.py — Lexical similarity feature computation.

Evidence boundary: all features are derived from Scopus CSV field values only.
No embeddings (use_embeddings: false in run_config.yaml).

Functions are pure and stateless: they accept plain Python dicts representing
author or affiliation instance rows, and return a float score in [0, 1].
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_set(v: object) -> set:
    """Coerce a value to a set, handling None, float NaN, and non-iterables."""
    if v is None:
        return set()
    if isinstance(v, float):
        # Covers float('nan') which is truthy in Python but not iterable
        return set()
    try:
        return set(v)
    except TypeError:
        return set()


def _tokenize(s: str) -> list[str]:
    """Split a normalised string on non-alphanumeric; return non-empty tokens."""
    return [t for t in re.split(r"\W+", s) if t]


# ---------------------------------------------------------------------------
# C) Public similarity functions
# ---------------------------------------------------------------------------

def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|.

    Returns 0.0 when both sets are empty (avoids division by zero).

    Args:
        set_a: First set.
        set_b: Second set.

    Returns:
        Float in [0.0, 1.0].
    """
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def and_similarity(anchor: dict, cand: dict) -> float:
    """Compute AND similarity between two author-instance dicts.

    Formula:
        score = 0.55 * name_sim + 0.35 * co_sim + 0.10 * aff_sim

    Components:
        name_sim : rapidfuzz.fuzz.ratio on normalised author name strings.
        co_sim   : Jaccard on co-author name_key sets.
        aff_sim  : 1.0 if both have the same non-empty record_affil_prefix;
                   0.0 otherwise.

    Args:
        anchor: Dict with keys author_norm, coauthor_keys, record_affil_prefix.
        cand:   Dict with same keys for the candidate instance.

    Returns:
        Float similarity score in [0.0, 1.0].

    Evidence boundary: uses only normalised field values derived from the
    Scopus CSV; no external lookup is performed.
    """
    name_sim = fuzz.ratio(
        anchor.get("author_norm") or "",
        cand.get("author_norm") or "",
    ) / 100.0

    co_sim = jaccard(
        _to_set(anchor.get("coauthor_keys")),
        _to_set(cand.get("coauthor_keys")),
    )

    aff_a = anchor.get("record_affil_prefix") or ""
    aff_c = cand.get("record_affil_prefix") or ""
    aff_sim = 1.0 if (aff_a and aff_a == aff_c) else 0.0

    return 0.55 * name_sim + 0.35 * co_sim + 0.10 * aff_sim


def ain_similarity(anchor: dict, cand: dict) -> float:
    """Compute AIN similarity between two affiliation-instance dicts.

    Formula:
        score = 0.50 * str_sim + 0.35 * tok_sim + 0.15 * acro_sim

    Components:
        str_sim  : rapidfuzz.fuzz.token_set_ratio on normalised affil strings.
        tok_sim  : Jaccard on sets of alphanumeric tokens.
        acro_sim : 1.0 if the two instances share at least one acronym;
                   0.0 otherwise.

    Args:
        anchor: Dict with keys affil_norm, affil_acronyms.
        cand:   Dict with same keys for the candidate instance.

    Returns:
        Float similarity score in [0.0, 1.0].

    Evidence boundary: uses only normalised field values derived from the
    Scopus CSV; no external lookup is performed.
    """
    a_norm = anchor.get("affil_norm") or ""
    c_norm = cand.get("affil_norm") or ""

    tok_sim = jaccard(set(_tokenize(a_norm)), set(_tokenize(c_norm)))
    str_sim = fuzz.token_set_ratio(a_norm, c_norm) / 100.0

    acros_a = _to_set(anchor.get("affil_acronyms"))
    acros_c = _to_set(cand.get("affil_acronyms"))
    acro_sim = 1.0 if (acros_a & acros_c) else 0.0

    return 0.50 * str_sim + 0.35 * tok_sim + 0.15 * acro_sim
