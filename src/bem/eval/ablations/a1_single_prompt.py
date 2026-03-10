"""a1_single_prompt.py -- Ablation A1: full DAG BEM vs single-prompt LLM.

Calls the LLM with a flat, unstructured representation of each pair instead
of the BEM structured evidence card.  The flat prompt concatenates all
relevant Scopus record fields as plain prose.  No blocking structure, no
signal-category guards, no two-threshold routing -- the LLM's raw decision
is used directly (with a confidence threshold for match acceptance).

This ablation isolates the contribution of the full BEM DAG architecture
relative to a naive "ask the LLM" baseline on the same benchmark pairs.

Differences from full BEM:
  - Evidence: flat field concatenation vs. structured evidence card
  - No co-author / affiliation breakdown
  - No C6 signal-count guard (confidence-only acceptance)

Fairness constraints (same as full BEM):
  - AND: Author(s) ID never included in flat prompt
  - AIN: title / year / source never included in flat prompt

Requires:
  - ANTHROPIC_API_KEY environment variable
  - Confirmation checkpoint (printed cost estimate + y/N prompt)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bem.eval.evaluation.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Flat prompt templates
# ---------------------------------------------------------------------------

_AND_FLAT_PROMPT = """\
You are comparing two author records from Scopus to decide whether they refer to the same person.

RECORD A:
  Author name   : {a_name}
  Co-authors    : {a_coauthors}
  Affiliation   : {a_affiliation}
  Publication   : {a_source}, {a_year}

RECORD B:
  Author name   : {b_name}
  Co-authors    : {b_coauthors}
  Affiliation   : {b_affiliation}
  Publication   : {b_source}, {b_year}

Task: Are RECORD A and RECORD B the same person?

Respond with ONLY a JSON object on a single line (no markdown, no commentary):
{{"label": "match" | "non-match" | "uncertain", "confidence": <float 0.0–1.0>, \
"reason": "<one sentence>"}}
"""

_AIN_FLAT_PROMPT = """\
You are comparing two affiliation records from Scopus to decide whether they refer to the same institution.

AFFILIATION A:
  Name (raw)    : {a_raw}
  Name (normalised): {a_norm}
  Acronyms      : {a_acronyms}
  Linked authors: {a_linked}

AFFILIATION B:
  Name (raw)    : {b_raw}
  Name (normalised): {b_norm}
  Acronyms      : {b_acronyms}
  Linked authors: {b_linked}

Task: Are AFFILIATION A and AFFILIATION B the same institution?

Respond with ONLY a JSON object on a single line (no markdown, no commentary):
{{"label": "match" | "non-match" | "uncertain", "confidence": <float 0.0–1.0>, \
"reason": "<one sentence>"}}
"""


def run_a1(
    routing_log_and: Path,
    routing_log_ain: Path,
    author_instances: Path,
    affil_instances: Path,
    records_normalised: Path,
    tasks: list[str],
    model: str,
    max_pairs_per_task: int,
    require_confirmation: bool,
    rng: np.random.Generator,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    output_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run ablation A1 (single-prompt LLM) for the requested tasks.

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
        RuntimeError: If the user declines the confirmation prompt.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "[A1] ANTHROPIC_API_KEY is not set. "
            "Export it before running ablation A1."
        )

    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "[A1] The 'anthropic' package is required. "
            "Install with: pip install anthropic"
        )

    log_paths  = {"and": routing_log_and, "ain": routing_log_ain}
    results: dict[str, list[dict]] = {}

    for task in tasks:
        log_path = log_paths[task]
        if not log_path.exists():
            print(f"  [A1/{task.upper()}] routing log not found -- skipping: {log_path}")
            continue

        cache_path = (output_dir / f"metrics_a1_{task}.json") if output_dir else None
        decisions_path = (output_dir / f"a1_decisions_{task}.jsonl") if output_dir else None

        if cache_path and cache_path.exists() and not force:
            print(f"  [A1/{task.upper()}] loading cached results from {cache_path}")
            results[task] = json.loads(cache_path.read_text(encoding="utf-8"))
            continue

        bm_df  = pd.read_parquet(log_path)
        # Sample benchmark pairs
        n_sample = min(max_pairs_per_task, len(bm_df))
        sample   = bm_df.sample(n=n_sample, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)

        print(f"\n  [A1/{task.upper()}] {n_sample} pairs selected for single-prompt LLM ablation")
        print(f"    Model : {model}")
        print(f"    Est.  : ~{n_sample} API calls")

        if require_confirmation and sys.stdin.isatty():
            answer = input(f"  Continue with A1/{task.upper()}? [y/N] ").strip().lower()
            if answer != "y":
                print(f"  [A1/{task.upper()}] skipped by user.")
                continue
        elif require_confirmation and not sys.stdin.isatty():
            print(f"  [A1/{task.upper()}] non-interactive mode -- skipping (require_confirmation=true).")
            continue

        # Load instance data for flat prompt construction
        flat_prompts = _build_flat_prompts(
            task=task,
            sample_df=sample,
            author_instances_path=author_instances,
            affil_instances_path=affil_instances,
            records_path=records_normalised,
        )

        # Call API + collect decisions
        decisions = _call_api_batch(
            task=task,
            prompts_df=flat_prompts,
            model=model,
            client=anthropic.Anthropic(api_key=api_key),
            decisions_path=decisions_path,
            force=force,
        )

        # Compute metrics (confidence-only gate, threshold = 0.85)
        gold = decisions["gold_label"].astype(str)
        pred = decisions["predicted"].astype(str)
        m = compute_all_metrics(gold, pred, rng, n_bootstrap, alpha)

        task_result = [{
            "ablation":    "a1_single_prompt",
            "variant":     "single_prompt",
            "task":        task,
            "model":       model,
            "n_pairs":     n_sample,
            "description": "Single flat-text LLM prompt; no DAG structure; confidence-only gate",
            "metrics":     _serialise(m),
        }]
        _print_result(task, m)
        results[task] = task_result

        if output_dir and cache_path:
            output_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(task_result, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    return results


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_flat_prompts(
    task: str,
    sample_df: pd.DataFrame,
    author_instances_path: Path,
    affil_instances_path: Path,
    records_path: Path,
) -> pd.DataFrame:
    """Build flat prompt strings for all pairs in sample_df."""
    records_df = pd.read_parquet(records_path) if records_path.exists() else pd.DataFrame()

    rows = []
    if task == "and":
        inst_df = pd.read_parquet(author_instances_path) if author_instances_path.exists() else pd.DataFrame()
        for _, pair in sample_df.iterrows():
            prompt = _and_flat_prompt(pair, inst_df, records_df)
            rows.append({
                "anchor_id":    pair["anchor_id"],
                "candidate_id": pair["candidate_id"],
                "gold_label":   pair.get("gold_label", ""),
                "prompt":       prompt,
            })
    else:  # ain
        inst_df = pd.read_parquet(affil_instances_path) if affil_instances_path.exists() else pd.DataFrame()
        for _, pair in sample_df.iterrows():
            prompt = _ain_flat_prompt(pair, inst_df)
            rows.append({
                "anchor_id":    pair["anchor_id"],
                "candidate_id": pair["candidate_id"],
                "gold_label":   pair.get("gold_label", ""),
                "prompt":       prompt,
            })

    return pd.DataFrame(rows)


def _and_flat_prompt(pair: pd.Series, inst_df: pd.DataFrame, records_df: pd.DataFrame) -> str:
    def _get_inst(iid: str) -> pd.Series | None:
        rows = inst_df[inst_df["author_instance_id"] == iid] if not inst_df.empty else pd.DataFrame()
        return rows.iloc[0] if len(rows) > 0 else None

    def _get_rec(eid: str) -> pd.Series | None:
        rows = records_df[records_df["EID"] == eid] if not records_df.empty else pd.DataFrame()
        return rows.iloc[0] if len(rows) > 0 else None

    a = _get_inst(pair["anchor_id"])
    b = _get_inst(pair["candidate_id"])

    def _field(inst, fld, fallback="N/A") -> str:
        if inst is None:
            return fallback
        v = inst.get(fld, None)
        return str(v).strip() if v and str(v).strip() not in ("", "nan") else fallback

    # Strip Scopus ID from author name (fairness -- never expose Author(s) ID)
    def _strip_id(name: str) -> str:
        import re
        return re.sub(r"\s*\(\d+\)\s*$", "", name).strip()

    a_name = _strip_id(_field(a, "author_norm", _field(a, "author_raw")))
    b_name = _strip_id(_field(b, "author_norm", _field(b, "author_raw")))

    a_rec  = _get_rec(_field(a, "eid")) if a is not None else None
    b_rec  = _get_rec(_field(b, "eid")) if b is not None else None

    return _AND_FLAT_PROMPT.format(
        a_name=a_name,
        a_coauthors=_field(a_rec, "Authors_norm", "N/A")[:300] if a_rec is not None else "N/A",
        a_affiliation=_field(a_rec, "affiliations_norm", "N/A")[:300] if a_rec is not None else "N/A",
        a_source=_field(a_rec, "Source title", "N/A") if a_rec is not None else "N/A",
        a_year=_field(a_rec, "Year", "N/A") if a_rec is not None else "N/A",
        b_name=b_name,
        b_coauthors=_field(b_rec, "Authors_norm", "N/A")[:300] if b_rec is not None else "N/A",
        b_affiliation=_field(b_rec, "affiliations_norm", "N/A")[:300] if b_rec is not None else "N/A",
        b_source=_field(b_rec, "Source title", "N/A") if b_rec is not None else "N/A",
        b_year=_field(b_rec, "Year", "N/A") if b_rec is not None else "N/A",
    )


def _ain_flat_prompt(pair: pd.Series, inst_df: pd.DataFrame) -> str:
    def _get_inst(iid: str) -> pd.Series | None:
        rows = inst_df[inst_df["affil_instance_id"] == iid] if not inst_df.empty else pd.DataFrame()
        return rows.iloc[0] if len(rows) > 0 else None

    def _field(inst, fld, fallback="N/A") -> str:
        if inst is None:
            return fallback
        v = inst.get(fld, None)
        if isinstance(v, (list, np.ndarray)):
            return ", ".join(str(x) for x in v) if len(v) > 0 else fallback
        return str(v).strip() if v and str(v).strip() not in ("", "nan") else fallback

    a = _get_inst(pair["anchor_id"])
    b = _get_inst(pair["candidate_id"])

    return _AIN_FLAT_PROMPT.format(
        a_raw=_field(a, "affil_raw"),
        a_norm=_field(a, "affil_norm"),
        a_acronyms=_field(a, "affil_acronyms"),
        a_linked=_field(a, "linked_authors_norm", "N/A"),
        b_raw=_field(b, "affil_raw"),
        b_norm=_field(b, "affil_norm"),
        b_acronyms=_field(b, "affil_acronyms"),
        b_linked=_field(b, "linked_authors_norm", "N/A"),
    )


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

_T_MATCH_DEFAULT    = 0.85
_T_NONMATCH_DEFAULT = 0.85


def _call_api_batch(
    task: str,
    prompts_df: pd.DataFrame,
    model: str,
    client,
    decisions_path: Path | None,
    force: bool,
    rpm_limit: int = 40,
) -> pd.DataFrame:
    """Call the Claude API for each prompt and collect three-class predictions."""
    min_interval = 60.0 / rpm_limit
    results = []
    last_call = 0.0

    # Resume from existing decisions file if present
    existing: dict[tuple, dict] = {}
    if decisions_path and decisions_path.exists() and not force:
        with open(decisions_path, encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                existing[(obj["anchor_id"], obj["candidate_id"])] = obj

    dec_fh = open(decisions_path, "a", encoding="utf-8") if decisions_path else None

    try:
        for _, row in prompts_df.iterrows():
            key = (row["anchor_id"], row["candidate_id"])
            if key in existing:
                results.append(existing[key])
                continue

            # Rate-limit
            elapsed = time.monotonic() - last_call
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": row["prompt"]}],
                    temperature=0,
                )
                last_call = time.monotonic()
                raw_text  = response.content[0].text.strip()
                parsed    = _parse_flat_response(raw_text)
            except Exception as exc:
                print(f"    [A1] API error for pair {key}: {exc}")
                parsed = {"label": "uncertain", "confidence": 0.0, "reason": f"api_error:{exc}"}

            label = parsed.get("label", "uncertain")
            conf  = float(parsed.get("confidence", 0.0))
            # Confidence gate
            if label == "match" and conf < _T_MATCH_DEFAULT:
                label = "uncertain"
            elif label == "non-match" and conf < _T_NONMATCH_DEFAULT:
                label = "uncertain"

            record = {
                "anchor_id":    row["anchor_id"],
                "candidate_id": row["candidate_id"],
                "gold_label":   row["gold_label"],
                "raw_response": parsed,
                "predicted":    label,
            }
            results.append(record)
            if dec_fh:
                dec_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                dec_fh.flush()

    finally:
        if dec_fh:
            dec_fh.close()

    return pd.DataFrame(results)


def _parse_flat_response(text: str) -> dict:
    """Parse the flat LLM response -- expects a JSON object."""
    import re
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting first {...} block
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Fallback: keyword scan
    lbl = "uncertain"
    for kw in ("match", "non-match", "uncertain"):
        if kw in text.lower():
            lbl = kw
            break
    return {"label": lbl, "confidence": 0.5, "reason": "parse_fallback"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialise(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if isinstance(v, tuple):
            out[k] = [float(x) if isinstance(x, (float, np.floating)) else x for x in v]
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            out[k] = None if np.isnan(v) else float(v)
        else:
            out[k] = v
    return out


def _print_result(task: str, m: dict) -> None:
    print(
        f"    {task.upper()} / single_prompt  "
        f"Prec={m.get('precision_match', float('nan')):.3f}  "
        f"Rec={m.get('recall_match', float('nan')):.3f}  "
        f"F1={m.get('f1_match', float('nan')):.3f}  "
        f"Cov={m.get('coverage', float('nan')):.3f}"
    )
