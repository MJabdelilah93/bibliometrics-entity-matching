"""python -m bem.eval — master entrypoint for the BEM evaluation pipeline.

Usage
-----
    # Dry-run (default): print plan, check artefacts, write eval_manifest only
    python -m bem.eval --config configs/eval_config.yaml

    # Execute all stages
    python -m bem.eval --config configs/eval_config.yaml --no-dry-run

    # Execute only one stage
    python -m bem.eval --config configs/eval_config.yaml --no-dry-run --stage materialise-inputs
    python -m bem.eval --config configs/eval_config.yaml --no-dry-run --stage baselines

    # Force re-computation of existing outputs
    python -m bem.eval --config configs/eval_config.yaml --no-dry-run --force

    # Run embedding baseline without confirmation prompt
    python -m bem.eval --config configs/eval_config.yaml --no-dry-run --stage baselines --no-embed-confirm

Stages
------
    E0   Config load + artefact check       (always runs; gate for all other stages)
    E1   Materialise split-specific inputs  (joins routing log to gold pairs; splits by dev/test)
    E2   Metrics                            [NOT YET IMPLEMENTED]
    E3   Classical baselines               (deterministic, fuzzy, TF-IDF, embedding)
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

_VALID_STAGES = ("all", "materialise-inputs", "metrics", "baselines", "tune-thresholds", "evaluate", "ablations", "robustness-efficiency", "export")


def _make_eval_run_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_eval")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bem.eval",
        description="BEM evaluation pipeline",
    )
    parser.add_argument(
        "--config",
        default="configs/eval_config.yaml",
        help="Path to eval_config.yaml (default: configs/eval_config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=None,
        help="Print plan only; do not write any stage outputs (overrides config)",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Execute stages (overrides config dry_run setting)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=None,
        help="Re-compute outputs even if they already exist (overrides config)",
    )
    parser.add_argument(
        "--stage",
        choices=_VALID_STAGES,
        default="all",
        help=(
            "Run only this stage: all | materialise-inputs | metrics | baselines | "
            "tune-thresholds  (default: all)"
        ),
    )
    parser.add_argument(
        "--no-embed-confirm",
        dest="no_embed_confirm",
        action="store_true",
        default=False,
        help=(
            "Skip the embedding confirmation prompt (allows embedding to run in "
            "non-interactive / batch environments)"
        ),
    )
    args = parser.parse_args(argv)

    # --- E0: Load config ---
    from bem.eval.config import load_eval_config, ConfigError
    from bem.eval.io_check import (
        check_artefacts,
        print_artefact_report,
        print_dry_run_plan,
        ArtefactCheckError,
    )
    from bem.eval.manifest import write_eval_manifest

    try:
        cfg = load_eval_config(args.config)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"[E0] CONFIG ERROR: {exc}", file=sys.stderr)
        return 2

    # CLI flags override config values
    if args.dry_run is not None:
        cfg.dry_run = args.dry_run
    if args.force is not None:
        cfg.force = args.force
    if args.no_embed_confirm:
        cfg.baselines.embedding_require_confirmation = False

    print(f"\n[bem.eval] config      : {cfg.config_path}")
    print(f"[bem.eval] dry_run     : {cfg.dry_run}")
    print(f"[bem.eval] force       : {cfg.force}")
    print(f"[bem.eval] tasks       : {cfg.tasks}")
    print(f"[bem.eval] stage       : {args.stage}")
    print(f"[bem.eval] random_seed : {cfg.random_seed}")

    # --- E0: Artefact check ---
    try:
        report = check_artefacts(cfg)
    except ArtefactCheckError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    print_artefact_report(report)

    # Establish eval run output directory (for eval manifests)
    eval_run_id = _make_eval_run_id()
    eval_run_dir = cfg.output_dir / eval_run_id
    eval_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bem.eval] eval run dir: {eval_run_dir}")

    manifest_path = write_eval_manifest(
        eval_run_dir=eval_run_dir,
        config_path=cfg.config_path,
        artefact_entries=report.entries,
        dry_run=cfg.dry_run,
        tasks=cfg.tasks,
        random_seed=cfg.random_seed,
    )
    print(f"[bem.eval] eval manifest: {manifest_path}")

    if cfg.dry_run:
        print_dry_run_plan(cfg)
        print(f"[bem.eval] Eval inputs will be written to: {cfg.eval_inputs_output_dir}")
        print(f"[bem.eval] Baseline outputs will be written to: {cfg.baselines.output_dir}")
        print(f"[bem.eval] Tuning outputs will be written to: {cfg.tuning.output_dir}")
        print(f"[bem.eval] Ablation outputs will be written to: {cfg.ablation.output_dir}")
        print(f"[bem.eval] Robustness outputs will be written to: {cfg.rob_eff.robustness_output_dir}")
        print(f"[bem.eval] Efficiency outputs will be written to: {cfg.rob_eff.efficiency_output_dir}")
        _print_ablation_plan(cfg)
        _print_robeff_plan(cfg)
        print(f"[bem.eval] Manuscript export will be written to: {cfg.evaluation.manuscript_dir}")
        return 0

    run_all = (args.stage == "all")

    # --- E1: Materialise split-specific inputs ---
    if run_all or args.stage == "materialise-inputs":
        _t = time.time()
        try:
            from bem.eval.stages.e1_materialise_inputs import (
                run_materialise_inputs,
                print_materialise_summary,
            )
            mat_report = run_materialise_inputs(cfg)
            print_materialise_summary(mat_report)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[E1] ERROR: {exc}", file=sys.stderr)
            return 1
        except AssertionError as exc:
            print(f"[E1] SMOKE CHECK FAILURE: {exc}", file=sys.stderr)
            return 1
        except Exception:
            print("[E1] Unexpected error in materialise-inputs stage:", file=sys.stderr)
            traceback.print_exc()
            return 1
        print(f"[E1] Wall-clock elapsed: {time.time()-_t:.1f}s")

    # --- E2: Metrics (stub) ---
    if run_all or args.stage == "metrics":
        print("[E2] Metrics stage not yet implemented — skipping.")

    # --- E3: Classical baselines ---
    if run_all or args.stage == "baselines":
        _t = time.time()
        try:
            from bem.eval.stages.e3_baselines import run as run_e3
            run_e3(cfg)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[E3] ERROR: {exc}", file=sys.stderr)
            return 1
        except AssertionError as exc:
            print(f"[E3] SMOKE CHECK FAILURE: {exc}", file=sys.stderr)
            return 1
        except Exception:
            print("[E3] Unexpected error in baselines stage:", file=sys.stderr)
            traceback.print_exc()
            return 1
        print(f"[E3] Wall-clock elapsed: {time.time()-_t:.1f}s")

    # --- E4: Threshold tuning ---
    if run_all or args.stage == "tune-thresholds":
        _t = time.time()
        try:
            from bem.eval.stages.e4_tune_thresholds import run as run_e4
            run_e4(cfg)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[E4] ERROR: {exc}", file=sys.stderr)
            return 1
        except AssertionError as exc:
            print(f"[E4] SMOKE CHECK FAILURE: {exc}", file=sys.stderr)
            return 1
        except Exception:
            print("[E4] Unexpected error in tune-thresholds stage:", file=sys.stderr)
            traceback.print_exc()
            return 1
        print(f"[E4] Wall-clock elapsed: {time.time()-_t:.1f}s")

    # --- E5: Evaluation ---
    if run_all or args.stage == "evaluate":
        _t = time.time()
        try:
            from bem.eval.stages.e5_evaluation import run as run_e5
            run_e5(cfg)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[E5] ERROR: {exc}", file=sys.stderr)
            return 1
        except AssertionError as exc:
            print(f"[E5] SMOKE CHECK FAILURE: {exc}", file=sys.stderr)
            return 1
        except Exception:
            print("[E5] Unexpected error in evaluate stage:", file=sys.stderr)
            traceback.print_exc()
            return 1
        print(f"[E5] Wall-clock elapsed: {time.time()-_t:.1f}s")

    # --- E6: Ablations ---
    if run_all or args.stage == "ablations":
        _t = time.time()
        try:
            from bem.eval.stages.e6_ablations import run as run_e6
            run_e6(cfg)
        except (FileNotFoundError, ValueError, EnvironmentError, RuntimeError) as exc:
            print(f"[E6] ERROR: {exc}", file=sys.stderr)
            return 1
        except AssertionError as exc:
            print(f"[E6] SMOKE CHECK FAILURE: {exc}", file=sys.stderr)
            return 1
        except Exception:
            print("[E6] Unexpected error in ablations stage:", file=sys.stderr)
            traceback.print_exc()
            return 1
        print(f"[E6] Wall-clock elapsed: {time.time()-_t:.1f}s")

    # --- E7: Robustness & Efficiency ---
    if run_all or args.stage == "robustness-efficiency":
        _t = time.time()
        try:
            from bem.eval.stages.e7_robustness_efficiency import run as run_e7
            run_e7(cfg)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[E7] ERROR: {exc}", file=sys.stderr)
            return 1
        except AssertionError as exc:
            print(f"[E7] SMOKE CHECK FAILURE: {exc}", file=sys.stderr)
            return 1
        except Exception:
            print("[E7] Unexpected error in robustness-efficiency stage:", file=sys.stderr)
            traceback.print_exc()
            return 1
        print(f"[E7] Wall-clock elapsed: {time.time()-_t:.1f}s")

    # --- E8: Manuscript export + smoke checks ---
    if run_all or args.stage == "export":
        _t = time.time()
        try:
            from bem.eval.stages.e8_export import run as run_e8
            run_e8(cfg)
        except Exception:
            print("[E8] Unexpected error in export stage:", file=sys.stderr)
            traceback.print_exc()
            return 1
        print(f"[E8] Wall-clock elapsed: {time.time()-_t:.1f}s")

    print("\n[bem.eval] Done.")
    return 0


def _print_ablation_plan(cfg) -> None:
    """Print the ablation stage dry-run plan."""
    abl = cfg.ablation
    flags = {
        "A1 single-prompt LLM":      abl.run_a1_single_prompt,
        "A2 no C6 guards":           abl.run_a2_no_guards,
        "A3 K sensitivity":          abl.run_a3_k_sensitivity,
        "A4 AND missing fields":      abl.run_a4_missing_fields_and,
        "A5 AIN missing fields":      abl.run_a5_missing_fields_ain,
        "A6 threshold sweep":        abl.run_a6_threshold_sweep,
    }
    enabled  = [k for k, v in flags.items() if v]
    disabled = [k for k, v in flags.items() if not v]
    print("\n[bem.eval] Ablation plan:")
    for label in enabled:
        api_note = " [REQUIRES API KEY + CONFIRMATION]" if "prompt" in label.lower() or "fields" in label.lower() else ""
        print(f"  ENABLED  {label}{api_note}")
    for label in disabled:
        print(f"  skipped  {label}")


def _print_robeff_plan(cfg) -> None:
    """Print the robustness-efficiency stage dry-run plan."""
    re = cfg.rob_eff
    print("\n[bem.eval] Robustness & Efficiency plan (E7):")
    print(f"  compute_q1_q2      : {re.compute_q1_q2}")
    print(f"  strongest_baseline : {re.strongest_baseline}")
    print(f"  pricing_note       : {re.pricing_model_note}")
    print(f"  robustness_output  : {re.robustness_output_dir}")
    print(f"  efficiency_output  : {re.efficiency_output_dir}")


if __name__ == "__main__":
    sys.exit(main())
