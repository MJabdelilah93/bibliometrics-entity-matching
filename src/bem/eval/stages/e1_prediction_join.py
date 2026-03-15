"""e1_prediction_join.py — Stage E1: join gold pairs to BEM routing decisions.

For each task (AND, AIN) this stage:
  1. Loads the dev2 benchmark parquet (anchor_id, candidate_id, gold_label, split).
  2. Loads the benchmark-filtered routing log parquet (label_final, confidence,
     signals_count, fired_categories, routed_to_human, …).
  3. Inner-joins on (anchor_id, candidate_id).
  4. Writes one predictions parquet per task under::

       outputs/eval/<eval_run_id>/predictions_{task}.parquet

  5. Writes a stage manifest: stage_E1_prediction_join_manifest.json.

Skip behaviour
--------------
If the output already exists and cfg.force is False, the stage is skipped
(prints a notice and returns immediately).
"""

from __future__ import annotations

from pathlib import Path

from bem.eval.config import EvalConfig
from bem.eval.manifest import write_stage_manifest


def run_prediction_join(cfg: EvalConfig, eval_run_dir: Path) -> None:
    """Execute Stage E1 for all configured tasks.

    Args:
        cfg:          Validated evaluation configuration.
        eval_run_dir: Root output directory for this evaluation run.
    """
    import pandas as pd

    results_dir = eval_run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    task_inputs = {
        "and": (cfg.benchmark_and, cfg.routing_and_bm),
        "ain": (cfg.benchmark_ain, cfg.routing_ain_bm),
    }

    stage_inputs: dict[str, Path] = {}
    stage_outputs: dict[str, Path] = {}

    for task in cfg.tasks:
        bm_path, routing_path = task_inputs[task]
        out_path = results_dir / f"predictions_{task}.parquet"

        stage_inputs[f"benchmark_{task}"] = bm_path
        stage_inputs[f"routing_{task}_bm"] = routing_path
        stage_outputs[f"predictions_{task}"] = out_path

        if out_path.exists() and not cfg.force:
            print(f"[E1] {task.upper()} predictions already exist — skipping "
                  f"(use --force to overwrite): {out_path}")
            continue

        print(f"[E1] Joining {task.upper()} gold pairs with routing decisions ...")

        gold = pd.read_parquet(bm_path)
        routing = pd.read_parquet(routing_path)

        # Normalise join keys to str to guard against type mismatches.
        for df, name in ((gold, "benchmark"), (routing, "routing")):
            for col in ("anchor_id", "candidate_id"):
                if col not in df.columns:
                    raise ValueError(
                        f"[E1] {task.upper()} {name} parquet is missing column '{col}'. "
                        f"Found columns: {list(df.columns)}"
                    )
            df["anchor_id"] = df["anchor_id"].astype(str)
            df["candidate_id"] = df["candidate_id"].astype(str)

        joined = gold.merge(
            routing,
            on=["anchor_id", "candidate_id"],
            how="inner",
            suffixes=("_gold", "_routing"),
        )

        n_gold = len(gold)
        n_routing = len(routing)
        n_joined = len(joined)
        n_unmatched = n_gold - n_joined
        print(
            f"[E1] {task.upper()}: gold={n_gold:,}  routing={n_routing:,}  "
            f"joined={n_joined:,}  unmatched_gold={n_unmatched:,}"
        )
        if n_unmatched > 0:
            print(
                f"[E1] WARNING: {n_unmatched:,} gold pairs have no BEM routing decision. "
                "These will be absent from metrics — verify routing log coverage."
            )

        joined.to_parquet(out_path, index=False)
        print(f"[E1] {task.upper()} predictions written: {out_path}")

    write_stage_manifest(
        eval_run_dir=eval_run_dir,
        stage_id="E1_prediction_join",
        params={"tasks": cfg.tasks, "random_seed": cfg.random_seed},
        inputs=stage_inputs,
        outputs=stage_outputs,
    )
    print("[E1] Stage manifest written.")
