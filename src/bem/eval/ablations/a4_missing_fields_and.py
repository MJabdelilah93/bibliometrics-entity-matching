"""a4_missing_fields_and.py -- Ablation A4: AND missing-field ablations.

Measures the contribution of individual evidence components to BEM AND
matching performance by calling the LLM with a modified evidence card that
has one field category zeroed out.

Variants:
  remove_coauthor    -- co-author list set to empty; all other fields intact
  remove_affiliation -- affiliations_norm and authors_with_affiliations_norm set
                       to empty; all other fields intact

Fairness:
  Author(s) ID is NEVER included (same constraint as full BEM production).

Each variant re-runs LLM inference on a sample of benchmark AND pairs.
Existing decisions are resumed rather than re-computed (skip/force semantics).
C6 guards are applied with the same default thresholds as production.
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
from bem.guards.apply_guards import apply_guards, count_signals
from bem.llm_verify.evidence_cards import build_and_evidence


_VARIANTS = {
    "remove_coauthor":    "No co-author evidence",
    "remove_affiliation": "No affiliation evidence",
}

_DEFAULT_THRESHOLDS = {"t_match": 0.85, "t_nonmatch": 0.85}
_M_SIGNALS_DEFAULT  = 2


def run_a4(
    routing_log_and: Path,
    author_instances: Path,
    affil_instances: Path,
    records_normalised: Path,
    benchmark_and: Path,
    and_prompt_path: Path,
    thresholds_manifest: Path,
    model: str,
    max_pairs: int,
    require_confirmation: bool,
    variants: list[str],
    rng: np.random.Generator,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    output_dir: Path | None = None,
    force: bool = False,
) -> dict[str, list[dict]]:
    """Run ablation A4 for each requested variant.

    Returns:
        Dict keyed by variant name -> list of result dicts.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError("[A4] ANTHROPIC_API_KEY is not set.")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        raise ImportError("[A4] Install the 'anthropic' package: pip install anthropic")

    thresholds = _load_thresholds(thresholds_manifest, "and")

    # Load shared data once
    bm_df    = pd.read_parquet(benchmark_and)
    inst_df  = pd.read_parquet(author_instances)
    rec_df   = pd.read_parquet(records_normalised)
    prompt_template = _load_prompt(and_prompt_path)

    # Sample pairs
    n_sample = min(max_pairs, len(bm_df))
    sample   = bm_df.sample(n=n_sample, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)

    results: dict[str, list[dict]] = {}

    for variant in variants:
        if variant not in _VARIANTS:
            print(f"  [A4] unknown variant '{variant}' -- skipping")
            continue

        cache_path    = (output_dir / f"metrics_a4_{variant}.json") if output_dir else None
        decisions_path = (output_dir / f"a4_decisions_{variant}.jsonl") if output_dir else None

        if cache_path and cache_path.exists() and not force:
            print(f"  [A4/{variant}] loading cached results")
            results[variant] = json.loads(cache_path.read_text(encoding="utf-8"))
            continue

        print(f"\n  [A4/{variant}] {_VARIANTS[variant]}")
        print(f"    Pairs  : {n_sample}")
        print(f"    Model  : {model}")

        if require_confirmation and sys.stdin.isatty():
            ans = input(f"  Continue with A4/{variant}? [y/N] ").strip().lower()
            if ans != "y":
                print(f"  [A4/{variant}] skipped by user.")
                continue
        elif require_confirmation and not sys.stdin.isatty():
            print(f"  [A4/{variant}] non-interactive -- skipping.")
            continue

        decisions = _run_variant(
            variant=variant,
            sample=sample,
            inst_df=inst_df,
            rec_df=rec_df,
            prompt_template=prompt_template,
            model=model,
            client=client,
            decisions_path=decisions_path,
            force=force,
        )

        # Apply C6 guards
        guarded = apply_guards(decisions, task="AND", thresholds=thresholds, m_signals=_M_SIGNALS_DEFAULT)
        gold  = guarded["gold_label"].astype(str)
        pred  = guarded["label_final"].astype(str).apply(
            lambda x: x if x in ("match", "non-match") else "uncertain"
        )
        m = compute_all_metrics(gold, pred, rng, n_bootstrap, alpha)

        variant_result = [{
            "ablation":    "a4_missing_fields_and",
            "variant":     variant,
            "task":        "and",
            "description": _VARIANTS[variant],
            "n_pairs":     n_sample,
            "metrics":     _serialise(m),
        }]
        _print_result("and", variant, m)
        results[variant] = variant_result

        if output_dir and cache_path:
            output_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(variant_result, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    return results


# ---------------------------------------------------------------------------
# Variant execution
# ---------------------------------------------------------------------------

def _run_variant(
    variant: str,
    sample: pd.DataFrame,
    inst_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    prompt_template: str,
    model: str,
    client,
    decisions_path: Path | None,
    force: bool,
    rpm_limit: int = 40,
) -> pd.DataFrame:
    """Build modified evidence cards, call LLM, return decisions DataFrame."""
    min_interval = 60.0 / rpm_limit
    existing = _load_existing_decisions(decisions_path, force)
    records_out = []
    last_call   = 0.0

    dec_fh = open(decisions_path, "a", encoding="utf-8") if decisions_path else None
    try:
        for _, pair in sample.iterrows():
            key = (str(pair["anchor_id"]), str(pair["candidate_id"]))
            if key in existing:
                records_out.append(existing[key])
                continue

            evidence = _build_modified_evidence(
                variant=variant,
                anchor_id=str(pair["anchor_id"]),
                candidate_id=str(pair["candidate_id"]),
                inst_df=inst_df,
                rec_df=rec_df,
            )
            if evidence is None:
                records_out.append(_stub_decision(pair, "build_error"))
                continue

            filled_prompt = prompt_template.replace("{{EVIDENCE_CARD}}", json.dumps(evidence, ensure_ascii=False))

            elapsed = time.monotonic() - last_call
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": filled_prompt}],
                    temperature=0,
                )
                last_call = time.monotonic()
                raw_text  = response.content[0].text.strip()
                parsed    = _parse_json_decision(raw_text)
            except Exception as exc:
                print(f"    [A4/{variant}] API error: {exc}")
                parsed = {"label": "uncertain", "confidence": 0.0,
                          "evidence_used": [], "reason_code": f"api_error:{exc}"}

            record = {
                "anchor_id":    str(pair["anchor_id"]),
                "candidate_id": str(pair["candidate_id"]),
                "gold_label":   str(pair.get("gold_label", "")),
                "label":        parsed.get("label", "uncertain"),
                "confidence":   float(parsed.get("confidence", 0.0)),
                "evidence_used": parsed.get("evidence_used", []),
                "reason_code":  parsed.get("reason_code", ""),
                "task":         "and",
            }
            records_out.append(record)
            if dec_fh:
                dec_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                dec_fh.flush()
    finally:
        if dec_fh:
            dec_fh.close()

    return pd.DataFrame(records_out)


def _build_modified_evidence(
    variant: str,
    anchor_id: str,
    candidate_id: str,
    inst_df: pd.DataFrame,
    rec_df: pd.DataFrame,
) -> dict | None:
    """Build a BEM AND evidence card with the specified field(s) removed."""
    try:
        evidence = build_and_evidence(
            anchor_id=anchor_id,
            candidate_id=candidate_id,
            inst_df=inst_df,
            rec_df=rec_df,
        )
    except Exception as exc:
        print(f"    [A4] evidence build error for {anchor_id}: {exc}")
        return None

    if variant == "remove_coauthor":
        for side in ("anchor", "candidate"):
            evidence[side]["coauthors_norm"] = []

    elif variant == "remove_affiliation":
        for side in ("anchor", "candidate"):
            evidence[side]["affiliations_norm"]              = []
            evidence[side]["authors_with_affiliations_norm"] = []

    return evidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"AND prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def _load_thresholds(manifest_path: Path, task: str) -> dict:
    if not manifest_path.exists():
        return _DEFAULT_THRESHOLDS
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "t_match":    float(raw.get(f"t_match_{task}",    raw.get("t_match",    0.85))),
        "t_nonmatch": float(raw.get(f"t_nonmatch_{task}", raw.get("t_nonmatch", 0.85))),
    }


def _load_existing_decisions(decisions_path: Path | None, force: bool) -> dict:
    if decisions_path is None or not decisions_path.exists() or force:
        return {}
    existing = {}
    with open(decisions_path, encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            existing[(obj["anchor_id"], obj["candidate_id"])] = obj
    return existing


def _stub_decision(pair: pd.Series, reason: str) -> dict:
    return {
        "anchor_id":    str(pair["anchor_id"]),
        "candidate_id": str(pair["candidate_id"]),
        "gold_label":   str(pair.get("gold_label", "")),
        "label":        "uncertain",
        "confidence":   0.0,
        "evidence_used": [],
        "reason_code":  reason,
        "task":         "and",
    }


def _parse_json_decision(text: str) -> dict:
    import re
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"label": "uncertain", "confidence": 0.5, "evidence_used": [], "reason_code": "parse_fallback"}


def _serialise(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if isinstance(v, tuple):
            out[k] = [float(x) if isinstance(x, (float, np.floating)) else x for x in v]
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            out[k] = None if np.isnan(v) else float(v)
        else:
            out[k] = v
    return out


def _print_result(task: str, variant: str, m: dict) -> None:
    print(
        f"    {task.upper()} / {variant:25s}  "
        f"Prec={m.get('precision_match', float('nan')):.3f}  "
        f"Rec={m.get('recall_match', float('nan')):.3f}  "
        f"F1={m.get('f1_match', float('nan')):.3f}  "
        f"Cov={m.get('coverage', float('nan')):.3f}"
    )
