"""generate.py — Candidate pair generation for AND and AIN (C4 checkpoint).

Evidence boundary: instances and candidates are derived solely from
data/interim/records_normalised.parquet.  No external entity registries are
consulted and no match decisions are made here.

Pipeline stages implemented in this module:
  A) Build per-(record, author) and per-(record, affiliation) instance tables.
  B) Build deterministic blocking indexes (inverted indexes).
  D) Generate, score, and rank candidate pairs per anchor; keep top-K=50.

Similarity scoring is delegated to vs2.features.similarity.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from vs2.normalise.normalise import extract_acronyms, normalise_text_basic
from vs2.features.similarity import and_similarity, ain_similarity

# ---------------------------------------------------------------------------
# Module-level constants (exported for use in manifests)
# ---------------------------------------------------------------------------

YEAR_WINDOW: int = 5
TOKEN_PREFIX_LEN: int = 3
PREFIX_CHARS: int = 12
TOP_K_DEFAULT: int = 50
MAX_BLOCK_SIZE_AIN: int = 50   # cap for tok3/pre12 AIN blocks; equals top_k, prevents O(N²) pairing on generic keys

AND_PASS_DEFS: list[dict[str, Any]] = [
    {
        "pass_id": "pass1_author_id",
        "priority": 1,
        "description": (
            "Author(s) ID equality — candidate generation only; "
            "the raw ID is never shown to the LLM verifier"
        ),
    },
    {
        "pass_id": "pass2_name_key",
        "priority": 2,
        "description": f"name_key match within ±{YEAR_WINDOW} years",
    },
    {
        "pass_id": "pass3_name_key_affil",
        "priority": 3,
        "description": "name_key + record-level affiliation prefix refinement",
    },
]

AIN_PASS_DEFS: list[dict[str, Any]] = [
    {"pass_id": "ain_acro",  "priority": 1, "description": "Acronym blocking"},
    {
        "pass_id": "ain_tok3",
        "priority": 2,
        "description": f"Token-prefix blocking (first {TOKEN_PREFIX_LEN} tokens)",
    },
    {
        "pass_id": "ain_pre12",
        "priority": 3,
        "description": f"String-prefix blocking (first {PREFIX_CHARS} chars)",
    },
]


# ---------------------------------------------------------------------------
# A) Helper parsing utilities
# ---------------------------------------------------------------------------

def split_semicolon_field(s: object) -> list[str]:
    """Split a semicolon-delimited field; strip whitespace and drop empties.

    Args:
        s: Raw cell value (str, NaN, or None).

    Returns:
        List of non-empty stripped strings.
    """
    if s is None:
        return []
    if isinstance(s, float):
        return []
    text = str(s).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def safe_int(s: object) -> Optional[int]:
    """Parse *s* as int; return None if not possible (NaN, None, non-numeric).

    Args:
        s: Any scalar value.

    Returns:
        Python int or None.
    """
    if s is None:
        return None
    if isinstance(s, float) and s != s:  # NaN check without import
        return None
    try:
        import pandas as pd
        if pd.isna(s):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def tokenize(s: str) -> list[str]:
    """Split a normalised string on non-alphanumeric; return non-empty tokens.

    Used for affiliation token-set operations in AIN similarity.

    Args:
        s: Normalised affiliation string.

    Returns:
        List of alphanumeric token strings.
    """
    return [t for t in re.split(r"\W+", s) if t]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sha256_str(s: str) -> str:
    """Return hex SHA-256 of a UTF-8 encoded string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _name_key(author_norm: str) -> str:
    """Compute name_key = '{surname}|{first_initial}' from a normalised author string.

    surname       = last whitespace-separated token of author_norm
                    (fallback: the full string when only one token).
    first_initial = first character of the first whitespace-separated token
                    (empty string when the string is empty).

    name_key = f"{surname}|{first_initial}"

    Args:
        author_norm: Normalised single-author string (e.g. "smith, j.").

    Returns:
        Blocking key string, or "" when author_norm is empty.
    """
    if not author_norm:
        return ""
    tokens = author_norm.split()
    if not tokens:
        return ""
    surname = tokens[-1]
    first_initial = tokens[0][0] if tokens[0] else ""
    return f"{surname}|{first_initial}"


def _record_affil_prefix(affil_norm: str, n: int = TOKEN_PREFIX_LEN) -> str:
    """Return the first *n* whitespace tokens of *affil_norm* joined by space."""
    if not affil_norm:
        return ""
    tokens = affil_norm.split()
    return " ".join(tokens[:n]) if tokens else ""


# ---------------------------------------------------------------------------
# A) Instance table builders
# ---------------------------------------------------------------------------

def build_author_instances(
    records_df: pd.DataFrame,
    truncation_author_count: int = 100,
) -> pd.DataFrame:
    """Build one row per (record, author position) from the normalised records.

    Prefers 'author_full_names_norm' for splitting; falls back to
    'authors_norm' when 'author_full_names_norm' is empty.

    All instances are written — including truncated ones — so that exclusions
    are auditable.  The 'truncation_flag' column marks records where the
    number of authors meets or exceeds *truncation_author_count*.

    Args:
        records_df: The normalised records DataFrame
            (data/interim/records_normalised.parquet).
        truncation_author_count: Author-count boundary for truncation flag.

    Returns:
        DataFrame with columns:
          author_instance_id, record_id, EID, query_frame, year_int,
          author_pos, author_raw, author_norm, author_id,
          name_key, coauthor_keys, record_affil_prefix, truncation_flag.

    Evidence boundary: derived entirely from the normalised Scopus CSV fields;
    no external data are introduced.
    """
    rows: list[dict[str, Any]] = []

    for _, rec in records_df.iterrows():
        eid = str(rec.get("EID") or "")
        record_id = str(rec.get("record_id") or "")
        query_frame = str(rec.get("query_frame") or "")
        year_int = safe_int(rec.get("year_int"))
        affiliations_norm = str(rec.get("affiliations_norm") or "")

        # --- Author list: prefer full names ---
        full_names_norm = str(rec.get("author_full_names_norm") or "")
        full_names_raw = str(rec.get("Author full names") or "")
        authors_norm_str = str(rec.get("authors_norm") or "")
        authors_raw_str = str(rec.get("Authors") or "")

        if full_names_norm.strip():
            author_raw_list = split_semicolon_field(full_names_raw)
            author_norm_list = split_semicolon_field(full_names_norm)
        else:
            author_raw_list = split_semicolon_field(authors_raw_str)
            author_norm_list = split_semicolon_field(authors_norm_str)

        n_authors = len(author_norm_list)
        truncation_flag = n_authors >= truncation_author_count

        # --- Author IDs (may be shorter or empty) ---
        raw_ids_cell = rec.get("author_ids_list")
        author_ids: list[str] = []
        if raw_ids_cell is not None and not isinstance(raw_ids_cell, float):
            try:
                author_ids = [str(x).strip() for x in raw_ids_cell if str(x).strip()]
            except TypeError:
                author_ids = []

        # --- Record-level affiliation prefix (blocking signal) ---
        rec_affil_pfx = _record_affil_prefix(affiliations_norm)

        # --- Compute name_keys for all authors (needed for coauthor_keys) ---
        all_nk = [_name_key(a) for a in author_norm_list]

        # --- Build one instance per author position ---
        for i, author_norm in enumerate(author_norm_list):
            author_raw = author_raw_list[i] if i < len(author_raw_list) else author_norm
            author_id = author_ids[i] if i < len(author_ids) else ""
            nk = all_nk[i]

            # Co-author keys: all OTHER authors' name_keys (exclude self)
            coauthor_keys: list[str] = sorted(
                {nk_j for j, nk_j in enumerate(all_nk) if j != i and nk_j}
            )

            inst_id = _sha256_str(f"{eid}|{i}|{author_raw}")

            rows.append({
                "author_instance_id": inst_id,
                "record_id": record_id,
                "EID": eid,
                "query_frame": query_frame,
                "year_int": year_int,
                "author_pos": i,
                "author_raw": author_raw,
                "author_norm": author_norm,
                "author_id": author_id,
                "name_key": nk,
                "coauthor_keys": coauthor_keys,
                "record_affil_prefix": rec_affil_pfx,
                "truncation_flag": truncation_flag,
            })

    return pd.DataFrame(rows)


def build_affil_instances(records_df: pd.DataFrame) -> pd.DataFrame:
    """Build one row per (record, affiliation element) from the normalised records.

    Splits the raw 'Affiliations' field on ';' to obtain individual affiliation
    strings.  For each element, computes normalised form, acronyms, and
    blocking keys.

    Args:
        records_df: The normalised records DataFrame.

    Returns:
        DataFrame with columns:
          affil_instance_id, record_id, EID, query_frame, year_int,
          affil_pos, affil_raw, affil_norm, affil_acronyms,
          token_prefix_key, prefix_key.

    Evidence boundary: derived entirely from the normalised Scopus CSV fields.
    """
    rows: list[dict[str, Any]] = []

    for _, rec in records_df.iterrows():
        eid = str(rec.get("EID") or "")
        record_id = str(rec.get("record_id") or "")
        query_frame = str(rec.get("query_frame") or "")
        year_int = safe_int(rec.get("year_int"))

        affil_raw_full = str(rec.get("Affiliations") or "")
        affil_parts = split_semicolon_field(affil_raw_full)

        for j, affil_raw in enumerate(affil_parts):
            affil_norm = normalise_text_basic(affil_raw)
            acronyms: list[str] = extract_acronyms(affil_raw)
            token_pfx = _record_affil_prefix(affil_norm, TOKEN_PREFIX_LEN)
            pfx = affil_norm[:PREFIX_CHARS] if affil_norm else ""

            inst_id = _sha256_str(f"{eid}|{j}|{affil_raw}")

            rows.append({
                "affil_instance_id": inst_id,
                "record_id": record_id,
                "EID": eid,
                "query_frame": query_frame,
                "year_int": year_int,
                "affil_pos": j,
                "affil_raw": affil_raw,
                "affil_norm": affil_norm,
                "affil_acronyms": acronyms,
                "token_prefix_key": token_pfx,
                "prefix_key": pfx,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B) Blocking index builders
# ---------------------------------------------------------------------------

def _build_and_blocking_indexes(
    inst_df: pd.DataFrame,
) -> dict[str, dict[str, list[str]]]:
    """Build AND inverted blocking indexes from eligible (non-truncated) instances.

    Pass 1 key  : "author_id:<id>"              — non-empty author_id
    Pass 2 key  : "name_key:<name_key>"          — non-empty name_key
    Pass 3 key  : "name_key_affil:<nk>|<afp>"   — non-empty name_key AND affil prefix

    Args:
        inst_df: Full author_instances DataFrame (including truncated rows).

    Returns:
        Dict with keys "pass1", "pass2", "pass3"; each maps block-key → [instance_id, ...].
    """
    pass1: dict[str, list[str]] = defaultdict(list)
    pass2: dict[str, list[str]] = defaultdict(list)
    pass3: dict[str, list[str]] = defaultdict(list)

    eligible = inst_df[~inst_df["truncation_flag"]]

    for _, row in eligible.iterrows():
        iid = row["author_instance_id"]
        aid = str(row["author_id"]) if row.get("author_id") else ""
        nk  = str(row["name_key"])  if row.get("name_key")  else ""
        afp = str(row["record_affil_prefix"]) if row.get("record_affil_prefix") else ""

        if aid:
            pass1[f"author_id:{aid}"].append(iid)
        if nk:
            pass2[f"name_key:{nk}"].append(iid)
        if nk and afp:
            pass3[f"name_key_affil:{nk}|{afp}"].append(iid)

    return {
        "pass1": dict(pass1),
        "pass2": dict(pass2),
        "pass3": dict(pass3),
    }


def _build_ain_blocking_indexes(
    inst_df: pd.DataFrame,
    max_block_size: int = MAX_BLOCK_SIZE_AIN,
) -> dict[str, dict[str, list[str]]]:
    """Build AIN inverted blocking indexes from all affiliation instances.

    Acronym key      : "acro:<A>"      for each A in affil_acronyms
    Token-prefix key : "tok3:<tok3>"   when token_prefix_key non-empty
    String-prefix key: "pre12:<pfx>"  when prefix_key non-empty

    Blocks exceeding *max_block_size* are truncated to the first
    *max_block_size* entries (parquet row order — deterministic).  This
    prevents O(N²) pair explosion on generic keys such as "department o"
    (46 k entries in the test corpus) while leaving small, informative blocks
    (acronym blocks, narrow tok3 keys) untouched.

    Args:
        inst_df: Full affil_instances DataFrame.
        max_block_size: Maximum number of instance IDs retained per block.

    Returns:
        Dict with keys "ain_acro", "ain_tok3", "ain_pre12"; each maps
        block-key → [instance_id, ...].
    """
    acro_idx: dict[str, list[str]] = defaultdict(list)
    tok3_idx: dict[str, list[str]] = defaultdict(list)
    pre12_idx: dict[str, list[str]] = defaultdict(list)

    for _, row in inst_df.iterrows():
        iid = row["affil_instance_id"]
        acronyms = row.get("affil_acronyms")
        tok_pfx  = str(row["token_prefix_key"]) if row.get("token_prefix_key") else ""
        pfx      = str(row["prefix_key"])       if row.get("prefix_key")       else ""

        # Acronym blocks
        if acronyms is not None and not isinstance(acronyms, float):
            try:
                for a in acronyms:
                    if a:
                        acro_idx[f"acro:{a}"].append(iid)
            except TypeError:
                pass

        if tok_pfx:
            tok3_idx[f"tok3:{tok_pfx}"].append(iid)
        if pfx:
            pre12_idx[f"pre12:{pfx}"].append(iid)

    # Cap oversized blocks to avoid O(N²) pair explosion on generic keys
    def _cap(idx: dict[str, list[str]]) -> dict[str, list[str]]:
        return {k: v[:max_block_size] for k, v in idx.items()}

    return {
        "ain_acro":  _cap(dict(acro_idx)),
        "ain_tok3":  _cap(dict(tok3_idx)),
        "ain_pre12": _cap(dict(pre12_idx)),
    }


# ---------------------------------------------------------------------------
# D) Candidate generation
# ---------------------------------------------------------------------------

def _top20_block_sizes(
    indexes: dict[str, dict[str, list[str]]],
) -> dict[str, list[dict[str, Any]]]:
    """Return top-20 largest blocks per pass index, for the manifest."""
    result: dict[str, list[dict[str, Any]]] = {}
    for pass_name, block_dict in indexes.items():
        top20 = sorted(
            ((k, len(v)) for k, v in block_dict.items()),
            key=lambda x: -x[1],
        )[:20]
        result[pass_name] = [{"key": k, "size": s} for k, s in top20]
    return result


_CAND_COLS = [
    "task", "anchor_id", "candidate_id", "best_pass_id", "best_block_key",
    "pass_priority", "similarity_score", "rank", "year_diff",
    "query_frame_anchor", "query_frame_candidate",
]


def generate_and_candidates(
    inst_df: pd.DataFrame,
    blocking_indexes: dict[str, dict[str, list[str]]],
    top_k: int = TOP_K_DEFAULT,
    year_window: int = YEAR_WINDOW,
) -> pd.DataFrame:
    """Generate AND candidate pairs for all eligible anchor instances.

    For each eligible (non-truncated) anchor:
      - Collects candidate IDs from pass1 (author_id), pass2 (name_key ±year),
        and pass3 (name_key + affil prefix) blocking indexes.
      - Unions candidates across passes; tracks best_pass (lowest priority
        number = most important pass that produced the candidate).
      - Pass-2 year filter: if both anchor and candidate have a year, keep
        only candidates where |year_a - year_c| ≤ year_window; when either
        year is missing, include without filtering.
      - Scores each candidate with and_similarity().
      - Ranks by (pass_priority ASC, similarity_score DESC, candidate_id ASC).
      - Retains the first top_k ranked candidates.

    Args:
        inst_df: Full author_instances DataFrame.
        blocking_indexes: Dict from _build_and_blocking_indexes().
        top_k: Maximum candidates to retain per anchor.
        year_window: Maximum |year difference| for pass-2 candidates.

    Returns:
        DataFrame with columns in _CAND_COLS.

    Evidence boundary: scoring uses only normalised field values; no external
    data are introduced.
    """
    pass1 = blocking_indexes.get("pass1", {})
    pass2 = blocking_indexes.get("pass2", {})
    pass3 = blocking_indexes.get("pass3", {})

    # Build fast lookup over eligible instances only
    eligible_records = inst_df[~inst_df["truncation_flag"]].to_dict(orient="records")
    inst_lookup: dict[str, dict] = {
        r["author_instance_id"]: r for r in eligible_records
    }

    output_rows: list[tuple] = []

    for anchor_id, anchor in inst_lookup.items():
        aid = anchor.get("author_id") or ""
        nk  = anchor.get("name_key")  or ""
        afp = anchor.get("record_affil_prefix") or ""
        year_a = safe_int(anchor.get("year_int"))

        # candidate_id -> (priority, block_key, pass_id)
        cand_best: dict[str, tuple[int, str, str]] = {}

        # ---- Pass 1: Author(s) ID equality ----
        if aid:
            bk = f"author_id:{aid}"
            for cid in pass1.get(bk, []):
                if cid != anchor_id:
                    if cid not in cand_best or 1 < cand_best[cid][0]:
                        cand_best[cid] = (1, bk, "pass1_author_id")

        # ---- Pass 2: name_key ± year_window ----
        if nk:
            bk = f"name_key:{nk}"
            for cid in pass2.get(bk, []):
                if cid == anchor_id:
                    continue
                cand = inst_lookup.get(cid)
                if cand is None:
                    continue
                year_c = safe_int(cand.get("year_int"))
                # Apply year filter only when both years are present
                if year_a is not None and year_c is not None:
                    if abs(year_a - year_c) > year_window:
                        continue
                if cid not in cand_best or 2 < cand_best[cid][0]:
                    cand_best[cid] = (2, bk, "pass2_name_key")

        # ---- Pass 3: name_key + affil prefix ----
        if nk and afp:
            bk = f"name_key_affil:{nk}|{afp}"
            for cid in pass3.get(bk, []):
                if cid != anchor_id:
                    if cid not in cand_best or 3 < cand_best[cid][0]:
                        cand_best[cid] = (3, bk, "pass3_name_key_affil")

        if not cand_best:
            continue

        # ---- Score and rank ----
        scored: list[tuple[int, float, str, str, str, Optional[int]]] = []
        for cid, (priority, block_key, pass_id) in cand_best.items():
            cand = inst_lookup.get(cid)
            if cand is None:
                continue
            score = and_similarity(anchor, cand)
            year_c = safe_int(cand.get("year_int"))
            yr_diff: Optional[int] = (
                int(abs(year_a - year_c))
                if year_a is not None and year_c is not None
                else None
            )
            scored.append((priority, score, cid, block_key, pass_id, yr_diff))

        # Sort: priority ASC, score DESC, candidate_id ASC (stable tiebreaker)
        scored.sort(key=lambda x: (x[0], -x[1], x[2]))

        for rank, (priority, score, cid, block_key, pass_id, yr_diff) in enumerate(
            scored[:top_k], start=1
        ):
            cand = inst_lookup[cid]
            output_rows.append((
                "AND",
                anchor_id,
                cid,
                pass_id,
                block_key,
                priority,
                round(score, 6),
                rank,
                yr_diff,
                anchor.get("query_frame", ""),
                cand.get("query_frame", ""),
            ))

    return (
        pd.DataFrame(output_rows, columns=_CAND_COLS)
        if output_rows
        else pd.DataFrame(columns=_CAND_COLS)
    )


def generate_ain_candidates(
    inst_df: pd.DataFrame,
    blocking_indexes: dict[str, dict[str, list[str]]],
    top_k: int = TOP_K_DEFAULT,
) -> pd.DataFrame:
    """Generate AIN candidate pairs for all affiliation anchor instances.

    AIN blocking order (priority): acronym (1) → token_prefix (2) → prefix (3).
    No year-window filter.  All affiliation instances are eligible anchors.

    Args:
        inst_df: Full affil_instances DataFrame.
        blocking_indexes: Dict from _build_ain_blocking_indexes().
        top_k: Maximum candidates to retain per anchor.

    Returns:
        DataFrame with columns in _CAND_COLS (year_diff is always null for AIN).
    """
    acro_idx  = blocking_indexes.get("ain_acro",  {})
    tok3_idx  = blocking_indexes.get("ain_tok3",  {})
    pre12_idx = blocking_indexes.get("ain_pre12", {})

    inst_lookup: dict[str, dict] = {
        r["affil_instance_id"]: r for r in inst_df.to_dict(orient="records")
    }

    output_rows: list[tuple] = []

    for anchor_id, anchor in inst_lookup.items():
        acronyms = anchor.get("affil_acronyms")
        tok_pfx  = anchor.get("token_prefix_key") or ""
        pfx      = anchor.get("prefix_key")       or ""

        cand_best: dict[str, tuple[int, str, str]] = {}

        # ---- Acronym blocks (priority 1) ----
        if acronyms is not None and not isinstance(acronyms, float):
            try:
                for a in acronyms:
                    if a:
                        bk = f"acro:{a}"
                        for cid in acro_idx.get(bk, []):
                            if cid != anchor_id:
                                if cid not in cand_best or 1 < cand_best[cid][0]:
                                    cand_best[cid] = (1, bk, "ain_acro")
            except TypeError:
                pass

        # ---- Token-prefix block (priority 2) ----
        if tok_pfx:
            bk = f"tok3:{tok_pfx}"
            for cid in tok3_idx.get(bk, []):
                if cid != anchor_id:
                    if cid not in cand_best or 2 < cand_best[cid][0]:
                        cand_best[cid] = (2, bk, "ain_tok3")

        # ---- String-prefix block (priority 3) ----
        if pfx:
            bk = f"pre12:{pfx}"
            for cid in pre12_idx.get(bk, []):
                if cid != anchor_id:
                    if cid not in cand_best or 3 < cand_best[cid][0]:
                        cand_best[cid] = (3, bk, "ain_pre12")

        if not cand_best:
            continue

        scored: list[tuple[int, float, str, str, str]] = []
        for cid, (priority, block_key, pass_id) in cand_best.items():
            cand = inst_lookup.get(cid)
            if cand is None:
                continue
            score = ain_similarity(anchor, cand)
            scored.append((priority, score, cid, block_key, pass_id))

        scored.sort(key=lambda x: (x[0], -x[1], x[2]))

        for rank, (priority, score, cid, block_key, pass_id) in enumerate(
            scored[:top_k], start=1
        ):
            cand = inst_lookup[cid]
            output_rows.append((
                "AIN",
                anchor_id,
                cid,
                pass_id,
                block_key,
                priority,
                round(score, 6),
                rank,
                None,  # year_diff not applicable for AIN
                anchor.get("query_frame", ""),
                cand.get("query_frame", ""),
            ))

    return (
        pd.DataFrame(output_rows, columns=_CAND_COLS)
        if output_rows
        else pd.DataFrame(columns=_CAND_COLS)
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_candidate_generation(
    normalised_df: pd.DataFrame,
    interim_dir: Path,
    run_dir: Path,
    top_k: int = TOP_K_DEFAULT,
    truncation_author_count: int = 100,
    max_block_size_ain: int = MAX_BLOCK_SIZE_AIN,
) -> dict[str, Any]:
    """Orchestrate C4: build instances, build indexes, generate candidates.

    Writes:
      - data/interim/author_instances.parquet
      - data/interim/affil_instances.parquet
      - data/interim/candidates_and.parquet
      - data/interim/candidates_ain.parquet
      - runs/<run_id>/logs/and_instance_stats.json

    Args:
        normalised_df: The records_normalised DataFrame (from C3, already
            loaded — avoids re-reading from disk).
        interim_dir: Path to data/interim/.
        run_dir: Path to the current run directory (runs/<run_id>/).
        top_k: Maximum candidates to retain per anchor.
        truncation_author_count: Author-count boundary for truncation flag.
        max_block_size_ain: Cap on AIN blocking-index block sizes to prevent
            O(N²) pair explosion on generic tok3/pre12 keys.

    Returns:
        Stats dict consumed by run.py to build candidate_manifest.json.

    Evidence boundary: all operations are local; no external services.
    """
    interim_dir = Path(interim_dir)
    run_dir = Path(run_dir)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # A) Build instance tables
    # -----------------------------------------------------------------------
    auth_inst = build_author_instances(normalised_df, truncation_author_count)
    affil_inst = build_affil_instances(normalised_df)

    auth_inst_path  = interim_dir / "author_instances.parquet"
    affil_inst_path = interim_dir / "affil_instances.parquet"
    auth_inst.to_parquet(auth_inst_path, index=False)
    affil_inst.to_parquet(affil_inst_path, index=False)

    # -----------------------------------------------------------------------
    # AND instance stats log
    # -----------------------------------------------------------------------
    total_auth    = len(auth_inst)
    truncated_cnt = int(auth_inst["truncation_flag"].sum()) if total_auth else 0
    eligible_cnt  = total_auth - truncated_cnt

    missing_id_cnt   = int((auth_inst["author_id"] == "").sum()) if total_auth else 0
    missing_year_cnt = int(auth_inst["year_int"].isna().sum()) if total_auth else 0

    and_instance_stats: dict[str, Any] = {
        "total_author_instances": total_auth,
        "truncation_excluded": truncated_cnt,
        "eligible_for_candidate_generation": eligible_cnt,
        "missing_author_id_pct": (
            round(missing_id_cnt / total_auth * 100, 2) if total_auth else 0.0
        ),
        "missing_year_pct": (
            round(missing_year_cnt / total_auth * 100, 2) if total_auth else 0.0
        ),
    }
    (logs_dir / "and_instance_stats.json").write_text(
        json.dumps(and_instance_stats, indent=2), encoding="utf-8"
    )

    # -----------------------------------------------------------------------
    # B) Build blocking indexes
    # -----------------------------------------------------------------------
    and_indexes = _build_and_blocking_indexes(auth_inst)
    ain_indexes = _build_ain_blocking_indexes(affil_inst, max_block_size=max_block_size_ain)

    # -----------------------------------------------------------------------
    # D) Generate candidates
    # -----------------------------------------------------------------------
    cand_and = generate_and_candidates(auth_inst, and_indexes, top_k, YEAR_WINDOW)
    cand_ain = generate_ain_candidates(affil_inst, ain_indexes, top_k)

    cand_and_path  = interim_dir / "candidates_and.parquet"
    cand_ain_path  = interim_dir / "candidates_ain.parquet"
    cand_and.to_parquet(cand_and_path, index=False)
    cand_ain.to_parquet(cand_ain_path, index=False)

    # -----------------------------------------------------------------------
    # E) Block size stats
    # -----------------------------------------------------------------------
    all_indexes = {**and_indexes, **ain_indexes}
    top20_blocks = _top20_block_sizes(all_indexes)

    return {
        "and_total": total_auth,
        "and_eligible": eligible_cnt,
        "and_truncation_excluded": truncated_cnt,
        "and_candidate_rows": len(cand_and),
        "ain_total": len(affil_inst),
        "ain_anchors": len(affil_inst),
        "ain_candidate_rows": len(cand_ain),
        "max_block_size_ain": max_block_size_ain,
        "top20_block_sizes": top20_blocks,
        "output_paths": {
            "author_instances":  str(auth_inst_path),
            "affil_instances":   str(affil_inst_path),
            "candidates_and":    str(cand_and_path),
            "candidates_ain":    str(cand_ain_path),
        },
    }
