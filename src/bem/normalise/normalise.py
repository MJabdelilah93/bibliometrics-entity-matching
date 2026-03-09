"""normalise.py — Deterministic, versioned normalisation of Scopus record fields.

Evidence boundary: operates only on values already present in the ingested
Scopus CSV; no external lookups are performed.

Normalisation is non-destructive: raw columns are never overwritten.
All transformations are deterministic and version-stamped so that any future
rule change can be compared against the previous output.

Design:
  - Utility functions (normalise_text_basic, extract_acronyms,
    parse_semicolon_list) are pure and stateless; they accept plain Python
    values and return plain Python values.
  - apply_normalisation wraps the utilities into a single DataFrame pass and
    returns both the augmented DataFrame and a summary-stats dict for auditing.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# A) Normalisation utilities
# ---------------------------------------------------------------------------

def normalise_text_basic(s: object) -> str:
    """Normalise a raw text value to a clean, lowercase, collapsed string.

    Steps applied in order:
      1. None / NaN / non-string → return empty string.
      2. Convert to str.
      3. Unicode NFKC normalisation (resolves ligatures, compatibility chars).
      4. Strip leading/trailing whitespace.
      5. Casefold (locale-independent lowercase, handles ß → ss etc.).
      6. Collapse internal whitespace sequences to a single space.
      7. Remove repeated identical punctuation (e.g. "---" → "-", "..." → ".").

    Args:
        s: Any value from a pandas cell (str, float NaN, None, etc.).

    Returns:
        Normalised string, or "" if the input was null/blank.

    Evidence boundary: no external lookup; result is a deterministic function
    of the input string only.
    """
    # Handle None and float NaN without importing pandas
    if s is None:
        return ""
    if isinstance(s, float) and math.isnan(s):
        return ""

    text = str(s)

    # 1. NFKC unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # 2. Strip leading/trailing whitespace
    text = text.strip()

    # 3. Casefold
    text = text.casefold()

    # 4. Collapse internal whitespace
    text = re.sub(r"\s+", " ", text)

    # 5. Remove repeated identical punctuation (same char 2+ times → once)
    text = re.sub(r"([^\w\s])\1+", r"\1", text)

    return text


# Roman numerals up to a generous upper bound; anchored so "IVY" is not excluded.
_ROMAN_RE = re.compile(
    r"^M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$",
    re.IGNORECASE,
)

# Valid acronym token: only A–Z and 0–9 (no accented chars, no punctuation).
_ALNUM_UPPER_RE = re.compile(r"^[A-Z0-9]+$")


def extract_acronyms(affil: object) -> list[str]:
    """Extract acronym-like tokens from a raw affiliation string.

    Extraction strategy:
      1. Prefer tokens found inside parentheses or brackets — these are the
         most reliable source of institutional acronyms in Scopus affiliations
         (e.g. "Université Mohammed VI Polytechnique (UM6P)").
      2. Also scan the rest of the string for qualifying tokens that are not
         already captured.

    A token qualifies if ALL of the following hold:
      - Composed solely of ASCII uppercase letters (A–Z) and digits (0–9).
        Accented characters, hyphens, and other punctuation disqualify it.
      - Length is 3–10 characters (2-letter tokens dropped to reduce noise).
      - At least one character is an uppercase letter (pure digit strings
        such as "2025" are excluded).
      - Not a Roman numeral (I … MMMM and all combinations thereof).

    Examples accepted : "UM6P", "CNRS", "ENSA", "UAE", "MIT", "INSERM", "IEEE".
    Examples rejected : "II", "VI", "ON", "QC", "TN" (length < 3 or Roman),
                        "the", "University", "al-Ain" (lowercase present).

    Args:
        affil: Raw affiliation cell value (str or NaN).

    Returns:
        Unique list of qualifying tokens in stable first-seen order
        (parenthesised tokens precede free tokens).

    Evidence boundary: extraction is purely lexical; no external vocabulary.
    """
    if affil is None:
        return []
    if isinstance(affil, float) and math.isnan(affil):
        return []

    text = str(affil)

    def _qualifies(tok: str) -> bool:
        """Return True iff tok passes all acronym filter rules."""
        if not (3 <= len(tok) <= 10):
            return False
        if not _ALNUM_UPPER_RE.match(tok):
            return False
        if not any(c.isupper() for c in tok):
            return False
        if _ROMAN_RE.match(tok):
            return False
        return True

    seen: set[str] = set()
    result: list[str] = []

    def _add(tok: str) -> None:
        if tok not in seen:
            seen.add(tok)
            result.append(tok)

    # Pass 1: tokens inside parentheses or square brackets (highest confidence)
    for bracketed in re.findall(r"[(\[](.*?)[)\]]", text):
        for raw in re.split(r"[\s,;./\\|]+", bracketed):
            tok = raw.strip("-–—")
            if _qualifies(tok):
                _add(tok)

    # Pass 2: all other tokens in the full string
    for raw in re.split(r"[\s,;.()\[\]{}/\\|]+", text):
        tok = raw.strip("-–—")
        if _qualifies(tok):
            _add(tok)

    return result


_SCOPUS_AUTHOR_ID_RE = re.compile(r"\s*\(\d{6,20}\)")


def strip_scopus_author_id(s: str) -> str:
    """Remove Scopus numeric author-ID suffixes embedded in name strings.

    Removes every occurrence of a parenthesised token that contains ONLY
    digits with a length of 6–20 characters, e.g. " (7004513174)".
    Parenthesised tokens that contain non-digit characters (e.g. department
    acronyms like "(LMCE)") are left untouched.

    Args:
        s: A name string, possibly containing " (digits)" suffixes.

    Returns:
        The string with all matching suffixes removed and outer whitespace
        stripped.  An empty input returns an empty string.

    Examples:
        "ez-zahraouy, hamid (7004513174)" → "ez-zahraouy, hamid"
        "lazfi, souad (55998718300)"       → "lazfi, souad"
        "university (LMCE)"                → "university (LMCE)"
    """
    return _SCOPUS_AUTHOR_ID_RE.sub("", s).strip()


def parse_semicolon_list(s: object) -> list[str]:
    """Split a semicolon-delimited field into a clean list of strings.

    Args:
        s: Raw cell value (typically "Author(s) ID").

    Returns:
        List of stripped, non-empty strings; empty list if input is null/blank.

    Evidence boundary: pure string splitting; no validation or lookup.
    """
    if s is None:
        return []
    if isinstance(s, float) and math.isnan(s):
        return []

    text = str(s).strip()
    if not text:
        return []

    return [part.strip() for part in text.split(";") if part.strip()]


# ---------------------------------------------------------------------------
# B) Apply normalisation to the canonical records DataFrame
# ---------------------------------------------------------------------------

def apply_normalisation(
    records_df: pd.DataFrame,
    rules_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply deterministic normalisation to the canonical records DataFrame.

    Adds the following new columns (raw columns are never modified):
      - authors_norm               (str) casefold + collapsed "Authors"
      - author_full_names_norm     (str) casefold + collapsed "Author full names"
      - author_ids_list            (list[str]) semicolon-split "Author(s) ID"
      - year_int                   (nullable Int64) safe parse of "Year"
      - affiliations_norm          (str) casefold + collapsed "Affiliations"
      - affiliations_acronyms      (list[str]) acronyms from raw "Affiliations"
      - authors_with_affiliations_norm (str) "Authors with affiliations"

    Args:
        records_df: The canonical records DataFrame (output of
            build_canonical_records). Must contain the 33 Scopus columns.
        rules_config: Parsed normalisation_rules.yaml content. Used to stamp
            the rules versions in the returned stats dict.

    Returns:
        (df_normalised, stats):
          df_normalised — records_df extended with the 7 new columns.
          stats — dict with:
            column_nonnull_counts  : non-null/non-empty count per new column.
            changed_counts         : rows where norm != raw for text columns.
            top_20_acronyms        : list of {acronym, count} dicts.
            normalisation_rules_versions : version strings from rules_config.

    Evidence boundary: all transformations are derived from field values in
    the canonical DataFrame; no external data sources are consulted.
    """
    df = records_df.copy()

    # ------------------------------------------------------------------
    # Text normalisation columns
    # ------------------------------------------------------------------
    df["authors_norm"] = df["Authors"].apply(normalise_text_basic)
    df["author_full_names_norm"] = df["Author full names"].apply(normalise_text_basic)
    df["affiliations_norm"] = df["Affiliations"].apply(normalise_text_basic)
    df["authors_with_affiliations_norm"] = df["Authors with affiliations"].apply(
        normalise_text_basic
    )

    # ------------------------------------------------------------------
    # List columns
    # ------------------------------------------------------------------
    df["author_ids_list"] = df["Author(s) ID"].apply(parse_semicolon_list)

    # Acronyms extracted from the RAW field (before casefolding) to preserve
    # original capitalisation, which is the signal we rely on.
    df["affiliations_acronyms"] = df["Affiliations"].apply(extract_acronyms)

    # ------------------------------------------------------------------
    # Numeric column
    # ------------------------------------------------------------------
    df["year_int"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    # Non-null / non-empty counts per new column
    column_nonnull_counts: dict[str, int] = {
        "authors_norm": int((df["authors_norm"] != "").sum()),
        "author_full_names_norm": int((df["author_full_names_norm"] != "").sum()),
        "author_ids_list": int(df["author_ids_list"].apply(bool).sum()),
        "year_int": int(df["year_int"].notna().sum()),
        "affiliations_norm": int((df["affiliations_norm"] != "").sum()),
        "affiliations_acronyms": int(df["affiliations_acronyms"].apply(bool).sum()),
        "authors_with_affiliations_norm": int(
            (df["authors_with_affiliations_norm"] != "").sum()
        ),
    }

    # Changed counts: rows where the normalised value differs from the raw value
    # (comparison is done only on non-null raw rows)
    text_col_pairs: list[tuple[str, str]] = [
        ("authors_norm", "Authors"),
        ("author_full_names_norm", "Author full names"),
        ("affiliations_norm", "Affiliations"),
        ("authors_with_affiliations_norm", "Authors with affiliations"),
    ]
    changed_counts: dict[str, int] = {}
    for norm_col, raw_col in text_col_pairs:
        raw_notnull = df[raw_col].notna()
        changed = int(
            (df.loc[raw_notnull, norm_col] != df.loc[raw_notnull, raw_col]).sum()
        )
        changed_counts[norm_col] = changed

    # Top-20 most frequent acronyms across all records
    all_acronyms: list[str] = []
    for acr_list in df["affiliations_acronyms"]:
        all_acronyms.extend(acr_list)
    top_20_acronyms = [
        {"acronym": a, "count": c}
        for a, c in Counter(all_acronyms).most_common(20)
    ]

    # Rules version stamps from config
    normalisation_rules_versions: dict[str, str] = {
        "and_name_normalisation": (
            rules_config.get("and_name_normalisation", {}).get("version", "unknown")
        ),
        "ain_affiliation_normalisation": (
            rules_config.get("ain_affiliation_normalisation", {}).get("version", "unknown")
        ),
    }

    stats: dict[str, Any] = {
        "column_nonnull_counts": column_nonnull_counts,
        "changed_counts": changed_counts,
        "top_20_acronyms": top_20_acronyms,
        "normalisation_rules_versions": normalisation_rules_versions,
    }

    return df, stats
