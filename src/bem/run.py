"""run.py — CLI runner for the BEM pipeline (C1 + C2 + C3 checkpoints).

Usage:
    python -m bem --config configs/run_config.yaml

This runner:
  1. Loads and hashes the run config.
  2. Loads and hashes the schema headers file.
  3. Creates a unique run directory under runs/<bem_run_id>/.
  4. Writes runs/<bem_run_id>/manifests/run_manifest.json              (C1).
  5. Ingests Q1 and Q2 CSV batches with strict schema validation       (C2).
  6. Writes data/interim/records_canonical.parquet.
  7. Writes runs/<bem_run_id>/manifests/export_manifest.json.
  8. Applies deterministic normalisation to the canonical records      (C3).
  9. Writes data/interim/records_normalised.parquet.
 10. Writes runs/<bem_run_id>/manifests/normalisation_manifest.json.
 11. Writes runs/<bem_run_id>/logs/normalisation_log.jsonl.
 12. Prints a short summary per stage.

No candidate generation, LLM verification, or matching logic is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import string
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from bem.artefacts.manifests import (
    RunManifest,
    compute_dataframe_stats,
    sha256_file,
    write_candidate_manifest,
    write_export_manifest,
    write_normalisation_log,
    write_normalisation_manifest,
    write_run_manifest,
)
from bem.candidates.generate import (
    AND_PASS_DEFS,
    AIN_PASS_DEFS,
    MAX_BLOCK_SIZE_AIN,
    PREFIX_CHARS,
    TOKEN_PREFIX_LEN,
    YEAR_WINDOW,
    run_candidate_generation,
)
from bem.ingest.ingest import (
    build_canonical_records,
    ingest_scopus_exports,
    load_expected_headers,
)
from bem.normalise.normalise import apply_normalisation
from bem.llm_verify.evidence_cards import build_and_evidence, build_ain_evidence
from bem.llm_verify.pair_loader import get_pairs_for_task
from bem.llm_verify.verifier import run_verification, write_verification_diagnostics
from bem.guards.apply_guards import load_llm_decisions_jsonl, apply_guards, filter_routing_to_benchmark
from bem.guards.tune_thresholds import join_with_gold, tune_t_match, write_thresholds_manifest
from bem.aggregate.cluster import run_c7_task, write_aggregation_manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand6() -> str:
    """Return a 6-character alphanumeric random string."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=6))


def _print_top_errors(err_log_path: Path, n: int = 3) -> None:
    """Print the top N distinct error classifications from an errors JSONL file."""
    import json as _json
    from collections import Counter

    if not err_log_path.exists():
        print("  (error log not found)")
        return

    counts: Counter = Counter()
    samples: dict[str, str] = {}
    with open(err_log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
            except Exception:
                continue
            cls = obj.get("classification", "UNKNOWN")
            msg = obj.get("error_message", "")
            counts[cls] += 1
            if cls not in samples:
                # Redact any string that looks like an API key
                samples[cls] = _redact_key(msg)

    print(f"  Top {n} error classifications:")
    for cls, cnt in counts.most_common(n):
        msg_sample = samples.get(cls, "")[:200]
        print(f"    [{cls}] x{cnt} — {msg_sample}")
    _print_recommended_action(counts)


def _redact_key(text: str) -> str:
    """Replace patterns that look like API keys with [REDACTED]."""
    import re
    return re.sub(r"sk-ant-[A-Za-z0-9\-_]{10,}", "[REDACTED]", text)


def _print_recommended_action(counts: Any) -> None:
    """Print a one-line recommended action based on the error classification."""
    if not counts:
        return
    top_cls = counts.most_common(1)[0][0]
    actions = {
        "AUTH":           "  -> Action: check ANTHROPIC_API_KEY in bem/.env (wrong or revoked key)",
        "PERMISSION":     "  -> Action: key exists but lacks permission — check model access in Anthropic console",
        "BILLING":        "  -> Action: account has no credits — add funds at console.anthropic.com",
        "RATE_LIMIT":     "  -> Action: reduce llm.rate_limit.max_requests_per_minute in run_config.yaml",
        "MODEL_NOT_FOUND":"  -> Action: check llm.model_id in run_config.yaml (current value may be invalid)",
        "INVALID_REQUEST":"  -> Action: request payload was rejected — check prompt format or reduce max_tokens",
        "NETWORK":        "  -> Action: network connectivity issue — check internet access / proxy settings",
        "UNKNOWN":        "  -> Action: inspect the full error message in llm_errors_*.jsonl",
    }
    print(actions.get(top_cls, f"  -> Action: inspect llm_errors_*.jsonl for {top_cls} details"))


def _make_run_id() -> str:
    """Return a run ID of the form YYYYMMDD_HHMMSS_<rand6>."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{_rand6()}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="BEM runner — C1 manifest + C2 ingestion + C3 normalisation (no matching)."
    )
    parser.add_argument(
        "--config",
        default="configs/run_config.yaml",
        help="Path to run_config.yaml (default: configs/run_config.yaml)",
    )
    parser.add_argument(
        "--resume_run_id",
        default=None,
        help=(
            "Resume C5 verification from an existing run. "
            "Provide a prior bem_run_id (e.g. 20260308_172313_2coxjo). "
            "Pairs already present in that run's llm_decisions_*.jsonl are skipped. "
            "C1-C4 are still executed (idempotent). "
            "New manifests are written under the resumed run directory."
        ),
    )
    args = parser.parse_args(argv)

    # Load .env from repo root (no-op if file absent; never overrides existing env vars)
    _repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=_repo_root / ".env", override=False)

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"ERROR: config file not found: {config_path}")

    # 1. Load and hash config
    config_raw = config_path.read_bytes()
    config_hash = hashlib.sha256(config_raw).hexdigest()
    config: dict[str, Any] = yaml.safe_load(config_raw)

    # 2. Load and hash schema headers
    schema_headers_path = Path(config["inputs"]["schema_headers_path"])
    if not schema_headers_path.exists():
        sys.exit(f"ERROR: schema_headers file not found: {schema_headers_path}")
    schema_headers_hash = sha256_file(schema_headers_path)

    # 3. Fail-fast: verify API key when backend requires it
    _backend_early: str = config.get("llm", {}).get("backend", "requests_only")
    if _backend_early == "anthropic_api" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ERROR: llm.backend is 'anthropic_api' but ANTHROPIC_API_KEY is not set.\n"
            "Create a file  bem/.env  containing:\n"
            "  ANTHROPIC_API_KEY=sk-ant-api03-...\n"
            "Then re-run the pipeline."
        )

    # 4. Resolve pipeline parameters
    top_k: int = config["candidate_generation"]["top_k"]
    truncation_author_count: int = config["candidate_generation"]["truncation_author_count"]
    model_id: str = config["llm"]["model_id"]
    timezone: str = config["project"]["timezone"]

    # 5. Create run directory structure
    # When --resume_run_id is set, C5 logs are written into that run's directory so
    # the verifier's skip logic can find and reuse already-decided pairs.
    resume_run_id: str | None = args.resume_run_id
    if resume_run_id:
        resume_logs_dir = Path("runs") / resume_run_id / "logs"
        if not resume_logs_dir.exists():
            sys.exit(
                f"ERROR: --resume_run_id '{resume_run_id}' not found. "
                f"Expected directory: {resume_logs_dir}"
            )
        print(f"Resuming run: {resume_run_id}")

    bem_run_id = _make_run_id()
    run_root = Path("runs") / bem_run_id
    manifests_dir = run_root / "manifests"
    # For C5, use the resumed run's logs dir if resuming; otherwise use the new one.
    logs_dir = resume_logs_dir if resume_run_id else run_root / "logs"
    outputs_dir = run_root / "outputs"
    for d in (manifests_dir, logs_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 5. Write run manifest (C1)
    timestamp_iso = datetime.now().isoformat(timespec="seconds")
    run_manifest = RunManifest(
        bem_run_id=bem_run_id,
        timestamp_iso=timestamp_iso,
        timezone=timezone,
        config_path=str(config_path),
        config_hash=config_hash,
        schema_headers_path=str(schema_headers_path),
        schema_headers_hash=schema_headers_hash,
        model_id=model_id,
        top_k=top_k,
        truncation_author_count=truncation_author_count,
    )
    run_manifest_path = write_run_manifest(run_manifest, manifests_dir)
    print("C1 scaffold complete.")
    print(f"  run_id   : {bem_run_id}")
    print(f"  manifest : {run_manifest_path}")

    # 6. C2 ingestion
    q1_csv_paths: list[str] = config["inputs"].get("q1_csv_paths", [])
    q2_csv_paths: list[str] = config["inputs"].get("q2_csv_paths", [])

    # Warn on missing files before attempting ingestion
    all_declared = [("Q1", p) for p in q1_csv_paths] + [("Q2", p) for p in q2_csv_paths]
    for frame, p in all_declared:
        if not Path(p).exists():
            warnings.warn(
                f"[C2] {frame} CSV not found (expected after data placement): {p}",
                stacklevel=1,
            )

    q1_existing = [p for p in q1_csv_paths if Path(p).exists()]
    q2_existing = [p for p in q2_csv_paths if Path(p).exists()]

    if not q1_existing and not q2_existing:
        print("\nC2 ingestion skipped: no input CSVs found.")
        return

    import pandas as pd

    q1_df = (
        ingest_scopus_exports(q1_existing, schema_headers_path, query_frame="Q1")
        if q1_existing
        else pd.DataFrame()
    )
    q2_df = (
        ingest_scopus_exports(q2_existing, schema_headers_path, query_frame="Q2")
        if q2_existing
        else pd.DataFrame()
    )

    # Build canonical records table and write parquet
    expected_headers = load_expected_headers(schema_headers_path)
    canonical_df = build_canonical_records(q1_df, q2_df, expected_headers)

    interim_dir = Path("data") / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    canonical_parquet = interim_dir / "records_canonical.parquet"
    canonical_df.to_parquet(canonical_parquet, index=False)

    # Write export manifest
    df_stats = compute_dataframe_stats(q1_df, q2_df)
    export_manifest_path = write_export_manifest(
        run_dir=run_root,
        q1_paths=q1_existing,
        q2_paths=q2_existing,
        schema_headers_path=schema_headers_path,
        dataframe_stats=df_stats,
    )

    print()
    print("C2 ingestion complete.")
    print(f"  total rows Q1      : {len(q1_df)}")
    print(f"  total rows Q2      : {len(q2_df)}")
    print(f"  canonical parquet  : {canonical_parquet}")
    print(f"  export manifest    : {export_manifest_path}")

    # -----------------------------------------------------------------------
    # C3 normalisation
    # -----------------------------------------------------------------------
    norm_rules_path = Path("configs/normalisation_rules.yaml")
    if not norm_rules_path.exists():
        sys.exit(f"ERROR: normalisation_rules not found: {norm_rules_path}")

    norm_rules_hash = sha256_file(norm_rules_path)
    norm_rules_config: dict[str, Any] = yaml.safe_load(norm_rules_path.read_bytes())

    normalised_df, norm_stats = apply_normalisation(canonical_df, norm_rules_config)

    # Write normalised parquet
    normalised_parquet = interim_dir / "records_normalised.parquet"
    normalised_df.to_parquet(normalised_parquet, index=False)

    # Write normalisation manifest + log
    norm_manifest_path = write_normalisation_manifest(
        run_dir=run_root,
        normalisation_rules_path=norm_rules_path,
        rules_hash=norm_rules_hash,
        summary_stats=norm_stats,
    )
    norm_log_path = write_normalisation_log(run_root, norm_stats)

    # Print C3 summary
    top10 = [(e["acronym"], e["count"]) for e in norm_stats["top_20_acronyms"][:10]]
    print()
    print("C3 normalisation complete.")
    print(f"  normalised parquet : {normalised_parquet}")
    print(f"  changed (authors)  : {norm_stats['changed_counts'].get('authors_norm', 0)}")
    print(f"  changed (affils)   : {norm_stats['changed_counts'].get('affiliations_norm', 0)}")
    print(f"  top 10 acronyms    : {top10}")
    print(f"  norm manifest      : {norm_manifest_path}")
    print(f"  norm log           : {norm_log_path}")

    # -----------------------------------------------------------------------
    # C4 candidate generation
    # -----------------------------------------------------------------------
    c4_stats = run_candidate_generation(
        normalised_df=normalised_df,
        interim_dir=interim_dir,
        run_dir=run_root,
        top_k=top_k,
        truncation_author_count=truncation_author_count,
    )

    # Build and write candidate_manifest.json
    output_paths = c4_stats["output_paths"]
    cand_manifest: dict[str, Any] = {
        "parameters": {
            "top_k": top_k,
            "truncation_author_count": truncation_author_count,
            "year_window": YEAR_WINDOW,
            "token_prefix_len": TOKEN_PREFIX_LEN,
            "prefix_chars": PREFIX_CHARS,
            "max_block_size_ain": c4_stats["max_block_size_ain"],
        },
        "and_pass_definitions": AND_PASS_DEFS,
        "ain_pass_definitions": AIN_PASS_DEFS,
        "counts": {
            "and_total_instances":       c4_stats["and_total"],
            "and_eligible_anchors":      c4_stats["and_eligible"],
            "and_truncation_excluded":   c4_stats["and_truncation_excluded"],
            "and_candidate_rows":        c4_stats["and_candidate_rows"],
            "ain_total_instances":       c4_stats["ain_total"],
            "ain_anchors":               c4_stats["ain_anchors"],
            "ain_candidate_rows":        c4_stats["ain_candidate_rows"],
        },
        "top20_block_sizes": c4_stats["top20_block_sizes"],
        "output_file_hashes": {
            name: sha256_file(path)
            for name, path in output_paths.items()
        },
        "output_paths": output_paths,
    }
    cand_manifest_path = write_candidate_manifest(run_root, cand_manifest)

    print()
    print("C4 candidate generation complete.")
    print(f"  AND author instances : {c4_stats['and_total']}")
    print(f"  AND eligible anchors : {c4_stats['and_eligible']}")
    print(f"  AND truncation excl. : {c4_stats['and_truncation_excluded']}")
    print(f"  AND candidate rows   : {c4_stats['and_candidate_rows']}")
    print(f"  AND output           : {output_paths['candidates_and']}")
    print(f"  AIN affil instances  : {c4_stats['ain_total']}")
    print(f"  AIN candidate rows   : {c4_stats['ain_candidate_rows']}")
    print(f"  AIN output           : {output_paths['candidates_ain']}")
    print(f"  candidate manifest   : {cand_manifest_path}")

    # -----------------------------------------------------------------------
    # C5 LLM verification
    # -----------------------------------------------------------------------
    verification_cfg: dict[str, Any] = config.get("verification", {})
    if not verification_cfg.get("enabled", False):
        print()
        print("C5 verification skipped (verification.enabled: false in config).")
    else:
        # Load instance tables (written by C4)
        auth_inst_df = pd.read_parquet(interim_dir / "author_instances.parquet")
        affil_inst_df = pd.read_parquet(interim_dir / "affil_instances.parquet")

        backend: str = config.get("llm", {}).get("backend", "requests_only")
        scope: str = verification_cfg.get("scope", "benchmark")
        smoke_test: bool = bool(verification_cfg.get("smoke_test", False))
        smoke_n: int = int(verification_cfg.get("smoke_pairs_per_task", 5))
        print()
        if smoke_test:
            print(
                f"*** SMOKE TEST MODE — capped at {smoke_n} pairs per task "
                f"(set verification.smoke_test: false for full run) ***"
            )
        print(f"C5 LLM verification  (backend={backend}, scope={scope})")
        write_verification_diagnostics(
            manifests_dir=manifests_dir,
            llm_cfg=config.get("llm", {}),
            smoke_test=smoke_test,
            smoke_n=smoke_n,
        )

        for task, instance_df, evidence_fn in [
            (
                "AND",
                auth_inst_df,
                lambda a, c: build_and_evidence(a, c, auth_inst_df, normalised_df),
            ),
            (
                "AIN",
                affil_inst_df,
                lambda a, c: build_ain_evidence(a, c, affil_inst_df, normalised_df),
            ),
        ]:
            pairs_df = get_pairs_for_task(task, verification_cfg, interim_dir)
            if pairs_df is None:
                continue

            if smoke_test and len(pairs_df) > smoke_n:
                pairs_df = pairs_df.head(smoke_n).reset_index(drop=True)
                print(f"  [{task}] Smoke test: using first {smoke_n} pairs.")

            _, v_stats = run_verification(
                pairs_df=pairs_df,
                task=task,
                evidence_fn=evidence_fn,
                config=config,
                run_id=bem_run_id,
                logs_dir=logs_dir,
            )

            if v_stats.get("skipped_already_decided"):
                print(f"  [{task}] pairs skipped (resume): {v_stats['skipped_already_decided']}")
            print(f"  [{task}] pairs attempted  : {v_stats['pairs_attempted']}")
            print(f"  [{task}] decisions written : {v_stats['decisions_written']}")
            if backend == "requests_only":
                print(f"  [{task}] requests written  : {v_stats['requests_written']}")
            if v_stats["errors"]:
                print(f"  [{task}] errors            : {v_stats['errors']}")
            if v_stats["fallback_count"]:
                print(f"  [{task}] linked_auth fallbacks: {v_stats['fallback_count']}")
            dec_log = logs_dir / f"llm_decisions_{task.lower()}.jsonl"
            req_log = logs_dir / f"llm_requests_{task.lower()}.jsonl"
            err_log = logs_dir / f"llm_errors_{task.lower()}.jsonl"
            print(f"  [{task}] decisions log    : {dec_log}")
            if backend == "requests_only":
                print(f"  [{task}] requests log     : {req_log}")
            if v_stats["errors"]:
                print(f"  [{task}] errors log       : {err_log}")

            # Fail-fast: if triggered, print top error classifications and abort
            if v_stats.get("fail_fast_triggered"):
                print()
                if v_stats.get("billing_limit_hit"):
                    # API usage/billing limit reached — remaining pairs skipped.
                    # C6 can still run on decisions already written to the log.
                    print(
                        f"C5 paused on task={task}: API usage limit reached. "
                        f"Remaining pairs will be retried on next resume. "
                        f"See {err_log}"
                    )
                    _print_top_errors(err_log, n=3)
                    # Do NOT sys.exit — allow C6 to run on existing decisions.
                else:
                    print(
                        f"C5 failed fast on task={task}: "
                        f"{v_stats['errors']} consecutive errors. "
                        f"See {err_log}"
                    )
                    _print_top_errors(err_log, n=3)
                    sys.exit(1)


    # -----------------------------------------------------------------------
    # C6 guards / thresholds
    # -----------------------------------------------------------------------
    if not verification_cfg.get("enabled", False):
        print()
        print("C6 guards skipped (verification.enabled: false in config).")
    else:
        guards_cfg: dict[str, Any] = config.get("guards_and_routing", {})
        m_signals: int = int(guards_cfg.get("m_independent_signals", 2))
        initial_thresholds: dict[str, dict[str, float]] = guards_cfg.get(
            "thresholds_initial", {}
        )
        benchmark_cfg: dict[str, Any] = verification_cfg.get("benchmark_pairs", {})

        print()
        print(f"C6 guards  (m_signals={m_signals})")

        and_tune_result: dict[str, Any] | None = None
        ain_tune_result: dict[str, Any] | None = None

        for task in ("AND", "AIN"):
            task_lower = task.lower()
            dec_path = logs_dir / f"llm_decisions_{task_lower}.jsonl"

            if not dec_path.exists():
                print(f"  [{task}] Decisions log not found — skipping guards.")
                continue

            # Load decisions
            decisions_df = load_llm_decisions_jsonl(dec_path)
            if decisions_df.empty:
                print(f"  [{task}] No decisions found — skipping guards.")
                continue

            # DEV-set threshold tuning (only if benchmark parquet is available)
            bench_key = f"{task_lower}_path"
            bench_path_str: str = benchmark_cfg.get(bench_key, "")
            bench_path = Path(bench_path_str) if bench_path_str else None

            task_initial = initial_thresholds.get(task_lower, {})
            t_match_init    = float(task_initial.get("t_match",    0.85))
            t_nonmatch_init = float(task_initial.get("t_nonmatch", 0.85))
            tuned_thresholds: dict[str, float] = {
                "t_match":    t_match_init,
                "t_nonmatch": t_nonmatch_init,
            }

            if bench_path and bench_path.exists():
                try:
                    dev_df = join_with_gold(decisions_df, bench_path, split="dev")
                    if not dev_df.empty:
                        precision_floor: float = float(
                            guards_cfg.get("precision_floor_target", 0.98)
                        )
                        result = tune_t_match(
                            dev_df=dev_df,
                            task=task,
                            initial_thresholds=tuned_thresholds,
                            m_signals=m_signals,
                            precision_floor=precision_floor,
                        )
                        tuned_thresholds["t_match"] = result["tuned_t_match"]
                        if task == "AND":
                            and_tune_result = result
                        else:
                            ain_tune_result = result
                    else:
                        print(f"  [{task}] DEV split empty — using initial T_match={t_match_init}.")
                except Exception as exc:
                    print(f"  [{task}] WARN: threshold tuning failed ({exc}). "
                          f"Using initial thresholds.")
            else:
                print(
                    f"  [{task}] Benchmark parquet not found "
                    f"({bench_path or bench_key!r}) — using initial thresholds."
                )

            # Apply guards
            guarded_df = apply_guards(
                df=decisions_df,
                task=task,
                thresholds=tuned_thresholds,
                m_signals=m_signals,
            )

            # Write routing log parquet (full — all decided pairs)
            routing_log_path = logs_dir / f"routing_log_{task_lower}.parquet"
            guarded_df.to_parquet(routing_log_path, index=False)

            # Write benchmark-filtered routing log (*_bm.parquet — C7 input)
            bm_log_path = logs_dir / f"routing_log_{task_lower}_bm.parquet"
            if bench_path and bench_path.exists():
                try:
                    bm_stats = filter_routing_to_benchmark(
                        routing_log_path=routing_log_path,
                        benchmark_path=bench_path,
                        out_path=bm_log_path,
                    )
                    print(
                        f"  [{task}] BM routing log  : {bm_log_path.name}  "
                        f"({bm_stats['bm_rows']} rows)"
                    )
                    print(
                        f"  [{task}] BM labels       : {bm_stats['label_final_dist']}"
                    )
                    print(
                        f"  [{task}] BM human queue  : "
                        f"{bm_stats['routed_to_human']} / {bm_stats['bm_rows']}"
                    )
                except Exception as exc:
                    print(f"  [{task}] WARN: BM routing log failed ({exc}). "
                          "C7 must filter manually.")
            else:
                print(f"  [{task}] WARN: benchmark parquet not found — "
                      "skipping *_bm.parquet write.")

            # Summary
            label_counts = guarded_df["label_final"].value_counts().to_dict()
            n_human = int(guarded_df["routed_to_human"].sum())
            print(
                f"  [{task}] T_match={tuned_thresholds['t_match']:.2f}  "
                f"T_nonmatch={tuned_thresholds['t_nonmatch']:.2f}"
            )
            print(f"  [{task}] Labels  : {label_counts}")
            print(f"  [{task}] Routed to human : {n_human} / {len(guarded_df)}")
            print(f"  [{task}] Routing log     : {routing_log_path}")

        # Write thresholds manifest
        thresh_manifest_path = write_thresholds_manifest(
            run_dir=run_root,
            and_result=and_tune_result,
            ain_result=ain_tune_result,
            m_signals=m_signals,
        )
        print(f"  Thresholds manifest : {thresh_manifest_path}")

    # -----------------------------------------------------------------------
    # C7 entity aggregation (union-find over MATCH edges, benchmark-filtered)
    # -----------------------------------------------------------------------
    print()
    print("C7 entity aggregation  (benchmark-filtered routing logs)")

    outputs_dir   = logs_dir.parent / "outputs"
    manifests_dir = run_root / "manifests"

    and_c7_result: dict[str, Any] | None = None
    ain_c7_result: dict[str, Any] | None = None

    for task in ("AND", "AIN"):
        task_lower   = task.lower()
        bm_log_path  = logs_dir / f"routing_log_{task_lower}_bm.parquet"

        if not bm_log_path.exists():
            print(
                f"  [{task}] BM routing log not found ({bm_log_path.name}) — "
                "run C6 first to generate *_bm.parquet."
            )
            continue

        try:
            result = run_c7_task(
                task=task,
                bm_routing_path=bm_log_path,
                outputs_dir=outputs_dir,
                manifests_dir=manifests_dir,
            )
        except Exception as exc:
            print(f"  [{task}] ERROR in C7 aggregation: {exc}")
            continue

        cs = result["cluster_stats"]
        print(
            f"  [{task}] nodes={cs['n_nodes']}  "
            f"match_edges={cs['n_match_edges']}  "
            f"clusters={cs['n_clusters']}  "
            f"largest={cs['largest_cluster_size']}  "
            f"singletons={cs['n_singleton_clusters']}  "
            f"conflicted={result['n_conflicted_entities']}"
        )
        print(f"  [{task}] Clusters CSV  : {result['clusters_path']}")
        if result["n_conflicted_entities"] > 0:
            print(
                f"  [{task}] WARN: {result['n_conflicted_entities']} "
                f"conflicted entities — see {result['conf_summary_path']}"
            )

        if task == "AND":
            and_c7_result = result
        else:
            ain_c7_result = result

    if and_c7_result is not None or ain_c7_result is not None:
        agg_manifest = write_aggregation_manifest(
            manifests_dir=manifests_dir,
            and_result=and_c7_result,
            ain_result=ain_c7_result,
        )
        print(f"  Aggregation manifest: {agg_manifest}")


if __name__ == "__main__":
    main()
