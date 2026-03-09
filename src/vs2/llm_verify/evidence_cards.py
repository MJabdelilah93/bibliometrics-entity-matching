"""evidence_cards.py — Task-specific evidence-card builders for LLM verification.

Evidence boundary: all fields are derived solely from Scopus UI CSV field values
already present in the normalised records and instance tables.  No external data
sources are introduced.

AND evidence card:
  Includes author name, co-authors, record metadata (year, source_title, title),
  affiliations.  Author(s) ID is NEVER included.

AIN evidence card:
  Includes affiliation strings, acronyms, and linked author names derived from
  "Authors with affiliations".  Title, source, and year are NEVER included to
  prevent topical-inference bias.
"""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from vs2.normalise.normalise import strip_scopus_author_id


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_list(v: object) -> list:
    """Convert None, float NaN, numpy arrays, or any iterable to a plain list."""
    if v is None:
        return []
    if isinstance(v, float) and math.isnan(v):
        return []
    try:
        return list(v)
    except TypeError:
        return []


def _safe_str(v: object) -> str:
    """Return str(v), or '' for None / float NaN."""
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return str(v)


def _safe_int(v: object) -> int | None:
    """Return int(v) or None for missing / non-numeric values."""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v):
            return None
        return int(v)
    try:
        import pandas as pd_
        if pd_.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _tokenize_overlap(s: str) -> set[str]:
    """Return a set of lowercase alphanumeric tokens (length >= 2) for overlap scoring."""
    return {t for t in re.split(r"\W+", s.lower()) if len(t) >= 2}


def _get_coauthors_norm(
    record_id: str,
    anchor_pos: int,
    author_instances_df: pd.DataFrame,
) -> list[str]:
    """Return author_norm values for all co-authors in the same record (anchor excluded).

    Args:
        record_id: record_id of the anchor instance.
        anchor_pos: author_pos of the anchor (excluded from result).
        author_instances_df: Full author_instances DataFrame.

    Returns:
        List of author_norm strings for every other author in the same record.
    """
    same_rec = author_instances_df[author_instances_df["record_id"] == record_id]
    return [
        strip_scopus_author_id(_safe_str(row["author_norm"]))
        for _, row in same_rec.iterrows()
        if int(row["author_pos"]) != anchor_pos
    ]


def _extract_linked_authors(
    affil_norm: str,
    authors_with_affiliations_norm: str,
) -> tuple[list[str], bool]:
    """Heuristically extract author names linked to a specific affiliation.

    Strategy:
      1. Split ``authors_with_affiliations_norm`` on ';' into segments.
      2. For each segment, compute the ratio of affil_norm tokens present in
         the segment.  If ratio >= 0.30, the segment is considered associated
         with this affiliation.
      3. The author name is extracted as the comma-separated prefix of the
         segment before any affil_norm tokens appear.
      4. If heuristic yields no results (fallback), return all authors_norm
         (comma-split) and set fallback=True.

    Args:
        affil_norm: Normalised affiliation string for this instance.
        authors_with_affiliations_norm: Normalised "Authors with affiliations"
            field from the corresponding record.

    Returns:
        (linked_names, fallback_used):
          linked_names — list of author-name strings.
          fallback_used — True when the heuristic failed; caller should log this.
    """
    if not affil_norm or not authors_with_affiliations_norm:
        return [], True

    affil_tokens = _tokenize_overlap(affil_norm)
    if not affil_tokens:
        return [], True

    segments = [s.strip() for s in authors_with_affiliations_norm.split(";") if s.strip()]
    linked: list[str] = []

    for seg in segments:
        seg_tokens = _tokenize_overlap(seg)
        overlap_ratio = len(affil_tokens & seg_tokens) / len(affil_tokens)
        if overlap_ratio < 0.30:
            continue
        # Extract the author-name prefix: comma-parts before affil tokens appear
        parts = [p.strip() for p in seg.split(",")]
        name_parts: list[str] = []
        for part in parts:
            if not part:
                continue
            part_tokens = _tokenize_overlap(part)
            if part_tokens and (part_tokens & affil_tokens):
                break  # reached the affiliation portion
            name_parts.append(part)
        author_name = ", ".join(name_parts).strip()
        if author_name:
            linked.append(author_name)

    if linked:
        return linked, False
    return [], True  # heuristic produced no results — caller uses fallback


# ---------------------------------------------------------------------------
# Public evidence-card builders
# ---------------------------------------------------------------------------

def build_and_evidence(
    anchor_id: str,
    candidate_id: str,
    author_instances_df: pd.DataFrame,
    records_norm_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build an AND evidence card for an (anchor, candidate) author-instance pair.

    The card includes author name, co-authors, record metadata (year, source_title,
    title), and affiliations.  Author(s) ID is explicitly excluded.

    Args:
        anchor_id: author_instance_id of the anchor.
        candidate_id: author_instance_id of the candidate.
        author_instances_df: Full author_instances DataFrame.
        records_norm_df: Full records_normalised DataFrame (indexed by record_id).

    Returns:
        Dict with keys: task, anchor, candidate.
        Each side has: author_norm, author_pos, record, coauthors_norm,
        affiliations_norm, authors_with_affiliations_norm.

    Raises:
        KeyError: if either instance ID is not found.

    Evidence boundary: no Author(s) ID is included anywhere in the returned dict.
    """
    anch_rows = author_instances_df[
        author_instances_df["author_instance_id"] == anchor_id
    ]
    cand_rows = author_instances_df[
        author_instances_df["author_instance_id"] == candidate_id
    ]
    if anch_rows.empty:
        raise KeyError(f"AND anchor not found: {anchor_id}")
    if cand_rows.empty:
        raise KeyError(f"AND candidate not found: {candidate_id}")

    anch = anch_rows.iloc[0]
    cand = cand_rows.iloc[0]

    def _rec(record_id: str) -> dict:
        rows = records_norm_df[records_norm_df["record_id"] == record_id]
        return {} if rows.empty else rows.iloc[0].to_dict()

    anch_rec = _rec(_safe_str(anch["record_id"]))
    cand_rec = _rec(_safe_str(cand["record_id"]))

    def _side(inst: Any, rec: dict) -> dict:
        record_id = _safe_str(inst["record_id"])
        author_pos = int(inst["author_pos"])
        coauthors = _get_coauthors_norm(record_id, author_pos, author_instances_df)
        return {
            "author_norm": strip_scopus_author_id(_safe_str(inst.get("author_norm"))),
            "author_pos": author_pos,
            "record": {
                "eid": _safe_str(rec.get("EID")),
                "year": _safe_int(rec.get("year_int")),
                "source_title": _safe_str(rec.get("Source title")),
                "title": _safe_str(rec.get("Title")),
            },
            "coauthors_norm": coauthors,
            "affiliations_norm": _safe_str(rec.get("affiliations_norm")),
            "authors_with_affiliations_norm": _safe_str(
                rec.get("authors_with_affiliations_norm")
            ),
        }

    return {
        "task": "AND",
        "anchor": _side(anch, anch_rec),
        "candidate": _side(cand, cand_rec),
    }


def build_ain_evidence(
    anchor_id: str,
    candidate_id: str,
    affil_instances_df: pd.DataFrame,
    records_norm_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build an AIN evidence card for an (anchor, candidate) affiliation-instance pair.

    The card includes affiliation strings, acronyms, and author names linked via
    the "Authors with affiliations" heuristic.  Title, source title, and year are
    explicitly excluded to prevent topical-inference bias.

    Args:
        anchor_id: affil_instance_id of the anchor.
        candidate_id: affil_instance_id of the candidate.
        affil_instances_df: Full affil_instances DataFrame.
        records_norm_df: Full records_normalised DataFrame.

    Returns:
        Dict with keys: task, anchor, candidate.
        Each side has: affil_raw, affil_norm, affil_acronyms, linked_authors_norm,
        linked_authors_fallback.

    Raises:
        KeyError: if either instance ID is not found.

    Evidence boundary: no title / source / year is included.
    """
    anch_rows = affil_instances_df[
        affil_instances_df["affil_instance_id"] == anchor_id
    ]
    cand_rows = affil_instances_df[
        affil_instances_df["affil_instance_id"] == candidate_id
    ]
    if anch_rows.empty:
        raise KeyError(f"AIN anchor not found: {anchor_id}")
    if cand_rows.empty:
        raise KeyError(f"AIN candidate not found: {candidate_id}")

    anch = anch_rows.iloc[0]
    cand = cand_rows.iloc[0]

    def _rec(record_id: str) -> dict:
        rows = records_norm_df[records_norm_df["record_id"] == record_id]
        return {} if rows.empty else rows.iloc[0].to_dict()

    anch_rec = _rec(_safe_str(anch["record_id"]))
    cand_rec = _rec(_safe_str(cand["record_id"]))

    def _side(inst: Any, rec: dict) -> dict:
        affil_norm = _safe_str(inst.get("affil_norm"))
        awa_norm = _safe_str(rec.get("authors_with_affiliations_norm"))
        linked_names, fallback = _extract_linked_authors(affil_norm, awa_norm)
        if fallback:
            # Fallback: split authors_norm on commas (crude but deterministic)
            authors_norm_raw = _safe_str(rec.get("authors_norm"))
            linked_names = [p.strip() for p in authors_norm_raw.split(",") if p.strip()]
        return {
            "affil_raw": _safe_str(inst.get("affil_raw")),
            "affil_norm": affil_norm,
            "affil_acronyms": _to_list(inst.get("affil_acronyms")),
            "linked_authors_norm": linked_names,
            "linked_authors_fallback": fallback,
        }

    return {
        "task": "AIN",
        "anchor": _side(anch, anch_rec),
        "candidate": _side(cand, cand_rec),
    }
