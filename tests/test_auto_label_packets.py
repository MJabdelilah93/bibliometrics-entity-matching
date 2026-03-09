"""tests/test_auto_label_packets.py — Unit tests for auto_label_packets helpers.

Tests are self-contained: no real CSV files are required.
Run with:  python -m pytest tests/test_auto_label_packets.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from vs2.benchmark.auto_label_packets import (
    _and_label,
    _ain_label,
    _and_signals,
    _ain_signals,
    _validate_schema,
    norm_basic,
    parse_acronyms_pipe,
    split_pipe_list,
    strip_numeric_parens,
    tokenise,
)

# ---------------------------------------------------------------------------
# Default threshold dicts matching run_config.yaml values
# ---------------------------------------------------------------------------

AND_THR = {
    "name_sim_match":    0.98,
    "affil_sim_match":   0.95,
    "name_sim_nonmatch": 0.60,
}

AIN_THR = {
    "affil_str_sim_match":    0.95,
    "token_jaccard_match":    0.55,
    "affil_str_sim_nonmatch": 0.45,
}


# ---------------------------------------------------------------------------
# Helper: strip_numeric_parens
# ---------------------------------------------------------------------------

class TestStripNumericParens:
    def test_removes_6digit_id(self):
        assert strip_numeric_parens("smith, john (123456)") == "smith, john"

    def test_removes_10digit_scopus_id(self):
        assert strip_numeric_parens("ez-zahraouy, hamid (7004513174)") == "ez-zahraouy, hamid"

    def test_removes_11digit_id(self):
        assert strip_numeric_parens("lazfi, souad (55998718300)") == "lazfi, souad"

    def test_preserves_dept_acronym(self):
        assert strip_numeric_parens("university (LMCE)") == "university (LMCE)"

    def test_preserves_alphanumeric_paren(self):
        assert strip_numeric_parens("um6p (UM6P)") == "um6p (UM6P)"

    def test_preserves_short_digits(self):
        # Only 5 digits — below the 6-digit minimum
        assert strip_numeric_parens("algo (12345)") == "algo (12345)"

    def test_empty_string(self):
        assert strip_numeric_parens("") == ""

    def test_no_parens(self):
        assert strip_numeric_parens("plain name") == "plain name"

    def test_multiple_ids_removed(self):
        result = strip_numeric_parens("a (1234567) b (9876543)")
        assert "1234567" not in result
        assert "9876543" not in result


# ---------------------------------------------------------------------------
# Helper: norm_basic
# ---------------------------------------------------------------------------

class TestNormBasic:
    def test_casefolded(self):
        assert norm_basic("SMITH") == "smith"

    def test_collapses_whitespace(self):
        assert norm_basic("  a   b  ") == "a b"

    def test_none_returns_empty(self):
        assert norm_basic(None) == ""

    def test_nan_returns_empty(self):
        import math
        assert norm_basic(float("nan")) == ""

    def test_nfkc_ligature(self):
        # 'ﬁ' (U+FB01) should become 'fi'
        assert norm_basic("ﬁeld") == "field"


# ---------------------------------------------------------------------------
# Helper: split_pipe_list
# ---------------------------------------------------------------------------

class TestSplitPipeList:
    def test_basic_split(self):
        result = split_pipe_list("smith, j | jones, p")
        assert "smith, j" in result
        assert "jones, p" in result

    def test_drops_truncation_marker(self):
        result = split_pipe_list("smith, j | …")
        assert "…" not in result

    def test_strips_numeric_id(self):
        result = split_pipe_list("smith, john (7004513174) | jones, paul (55998718300)")
        assert any("7004513174" not in r for r in result)

    def test_empty_returns_empty_set(self):
        assert split_pipe_list("") == set()

    def test_none_returns_empty_set(self):
        assert split_pipe_list(None) == set()


# ---------------------------------------------------------------------------
# Helper: tokenise
# ---------------------------------------------------------------------------

class TestTokenise:
    def test_basic(self):
        assert tokenise("Mohammed VI University") == {"mohammed", "vi", "university"}

    def test_drops_short_tokens(self):
        result = tokenise("a b university")
        assert "a" not in result
        assert "b" not in result
        assert "university" in result

    def test_empty(self):
        assert tokenise("") == set()


# ---------------------------------------------------------------------------
# Helper: parse_acronyms_pipe
# ---------------------------------------------------------------------------

class TestParseAcronymsPipe:
    def test_basic(self):
        result = parse_acronyms_pipe("UM6P | CNRS | ENSA")
        assert result == {"UM6P", "CNRS", "ENSA"}

    def test_lowercased_input_uppercased_output(self):
        result = parse_acronyms_pipe("um6p | cnrs")
        assert "UM6P" in result
        assert "CNRS" in result

    def test_drops_truncation(self):
        result = parse_acronyms_pipe("UM6P | …")
        assert "…" not in result
        assert "UM6P" in result

    def test_empty(self):
        assert parse_acronyms_pipe("") == set()


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestValidateSchema:
    def test_passes_when_all_columns_present(self):
        from vs2.benchmark.auto_label_packets import AND_REQUIRED_COLS
        df = pd.DataFrame(columns=AND_REQUIRED_COLS)
        _validate_schema(df, AND_REQUIRED_COLS, "AND")  # should not raise

    def test_raises_on_missing_columns(self):
        df = pd.DataFrame(columns=["task", "split"])
        with pytest.raises(ValueError, match="Missing columns"):
            _validate_schema(df, ["task", "split", "anchor_id"], "AND")


# ---------------------------------------------------------------------------
# AND signal computation
# ---------------------------------------------------------------------------

def _and_row(**kwargs) -> pd.Series:
    defaults = {
        "anchor_author_norm":      "",
        "candidate_author_norm":   "",
        "anchor_coauthors_norm":   "",
        "candidate_coauthors_norm": "",
        "anchor_affiliations_norm": "",
        "candidate_affiliations_norm": "",
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


class TestAndSignals:
    def test_identical_name_gives_high_name_sim(self):
        row = _and_row(
            anchor_author_norm="smith, john",
            candidate_author_norm="smith, john",
        )
        sig = _and_signals(row)
        assert sig["name_sim"] >= 0.99

    def test_coauthor_overlap_counted(self):
        row = _and_row(
            anchor_author_norm="smith, john",
            candidate_author_norm="smith, john",
            anchor_coauthors_norm="jones, p | miller, k",
            candidate_coauthors_norm="jones, p | brown, t",
        )
        sig = _and_signals(row)
        assert sig["coauthor_overlap_count"] == 1

    def test_empty_names_gives_zero_sim(self):
        row = _and_row()
        sig = _and_signals(row)
        assert sig["name_sim"] == 0.0

    def test_different_names_low_sim(self):
        row = _and_row(
            anchor_author_norm="smith, john",
            candidate_author_norm="zhang, wei",
        )
        sig = _and_signals(row)
        assert sig["name_sim"] < 0.60


# ---------------------------------------------------------------------------
# AND label rules
# ---------------------------------------------------------------------------

class TestAndLabel:
    def test_match_identical_name_with_coauthor(self):
        sig = {"name_sim": 0.99, "coauthor_overlap_count": 2, "affil_sim": 0.50}
        assert _and_label(sig, AND_THR) == "match"

    def test_match_identical_name_with_high_affil(self):
        sig = {"name_sim": 0.99, "coauthor_overlap_count": 0, "affil_sim": 0.97}
        assert _and_label(sig, AND_THR) == "match"

    def test_nonmatch_very_low_name_sim(self):
        sig = {"name_sim": 0.30, "coauthor_overlap_count": 0, "affil_sim": 0.20}
        assert _and_label(sig, AND_THR) == "non-match"

    def test_uncertain_high_name_no_corroboration(self):
        # name_sim high but below match threshold, no coauthor overlap, low affil
        sig = {"name_sim": 0.95, "coauthor_overlap_count": 0, "affil_sim": 0.50}
        assert _and_label(sig, AND_THR) == "uncertain"

    def test_uncertain_correct(self):
        sig = {"name_sim": 0.98, "coauthor_overlap_count": 0, "affil_sim": 0.80}
        # 0.98 >= 0.98 AND (co=0 OR affil=0.80 >= 0.95) → second part False → uncertain
        assert _and_label(sig, AND_THR) == "uncertain"

    def test_nonmatch_at_boundary(self):
        sig = {"name_sim": 0.60, "coauthor_overlap_count": 0, "affil_sim": 0.10}
        assert _and_label(sig, AND_THR) == "non-match"

    def test_uncertain_mid_sim(self):
        sig = {"name_sim": 0.75, "coauthor_overlap_count": 0, "affil_sim": 0.50}
        assert _and_label(sig, AND_THR) == "uncertain"


# ---------------------------------------------------------------------------
# AIN signal computation
# ---------------------------------------------------------------------------

def _ain_row(**kwargs) -> pd.Series:
    defaults = {
        "anchor_affil_norm":       "",
        "candidate_affil_norm":    "",
        "anchor_affil_acronyms":   "",
        "candidate_affil_acronyms": "",
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


class TestAinSignals:
    def test_identical_affil_high_sim(self):
        row = _ain_row(
            anchor_affil_norm="université mohammed vi polytechnique",
            candidate_affil_norm="université mohammed vi polytechnique",
        )
        sig = _ain_signals(row)
        assert sig["affil_str_sim"] >= 0.99

    def test_acronym_overlap_detected(self):
        row = _ain_row(
            anchor_affil_norm="um6p faculty",
            candidate_affil_norm="um6p ben guerir",
            anchor_affil_acronyms="UM6P | ENSA",
            candidate_affil_acronyms="UM6P | CNRS",
        )
        sig = _ain_signals(row)
        assert sig["acronym_overlap"] is True

    def test_no_acronym_overlap(self):
        row = _ain_row(
            anchor_affil_acronyms="CNRS",
            candidate_affil_acronyms="MIT",
        )
        sig = _ain_signals(row)
        assert sig["acronym_overlap"] is False

    def test_empty_affils_zero_sim(self):
        row = _ain_row()
        sig = _ain_signals(row)
        assert sig["affil_str_sim"] == 0.0


# ---------------------------------------------------------------------------
# AIN label rules
# ---------------------------------------------------------------------------

class TestAinLabel:
    def test_match_high_sim_and_acronym_overlap(self):
        sig = {"affil_str_sim": 0.97, "token_jaccard": 0.40, "acronym_overlap": True}
        assert _ain_label(sig, AIN_THR) == "match"

    def test_match_high_sim_and_high_jaccard(self):
        sig = {"affil_str_sim": 0.97, "token_jaccard": 0.60, "acronym_overlap": False}
        assert _ain_label(sig, AIN_THR) == "match"

    def test_nonmatch_very_low_sim(self):
        sig = {"affil_str_sim": 0.20, "token_jaccard": 0.05, "acronym_overlap": False}
        assert _ain_label(sig, AIN_THR) == "non-match"

    def test_uncertain_mid_sim(self):
        sig = {"affil_str_sim": 0.70, "token_jaccard": 0.30, "acronym_overlap": False}
        assert _ain_label(sig, AIN_THR) == "uncertain"

    def test_uncertain_high_sim_but_no_corroboration(self):
        # sim >= t_m but jaccard low and no acronym overlap
        sig = {"affil_str_sim": 0.96, "token_jaccard": 0.40, "acronym_overlap": False}
        assert _ain_label(sig, AIN_THR) == "uncertain"

    def test_nonmatch_at_boundary(self):
        sig = {"affil_str_sim": 0.45, "token_jaccard": 0.10, "acronym_overlap": False}
        assert _ain_label(sig, AIN_THR) == "non-match"
