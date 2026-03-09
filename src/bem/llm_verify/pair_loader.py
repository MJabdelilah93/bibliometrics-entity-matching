"""pair_loader.py — Pair-list loading for LLM verification (C5).

Two modes:
  benchmark       — reads a pre-built parquet of labelled pairs supplied by the
                    researcher (gold-standard annotation is done externally, never
                    by the LLM).  Missing benchmark file → skip verification for
                    that task (with informative message).
  full_candidates — reads data/interim/candidates_{and,ain}.parquet and takes a
                    deterministic sample of up to max_pairs_per_task rows.

Evidence boundary: no external sources; pair IDs are derived from the C4 parquets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# Expected columns produced by both loaders
_REQUIRED_COLS = {"anchor_id", "candidate_id", "task"}


def load_pairs_from_parquet(path: str | Path) -> pd.DataFrame:
    """Load a pair list from a parquet file.

    The file must contain at minimum the columns ``anchor_id``, ``candidate_id``,
    and ``task``.  Optional columns ``gold_label`` and ``split`` are preserved if
    present; otherwise they are added with value ``None``.

    Args:
        path: Path to the parquet file.

    Returns:
        DataFrame with at least: anchor_id, candidate_id, task, gold_label, split.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pair file not found: {path}")

    df = pd.read_parquet(path)

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Pair file {path} is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    if "gold_label" not in df.columns:
        df = df.assign(gold_label=None)
    if "split" not in df.columns:
        df = df.assign(split=None)

    return df[["anchor_id", "candidate_id", "task", "gold_label", "split"]].copy()


def get_pairs_for_task(
    task: str,
    verification_config: dict[str, Any],
    interim_dir: str | Path,
) -> pd.DataFrame | None:
    """Return the pair DataFrame for a given task, or None if unavailable.

    Args:
        task: "AND" or "AIN".
        verification_config: The ``verification`` sub-dict from run_config.yaml.
        interim_dir: Path to data/interim/ (used for full_candidates mode).

    Returns:
        DataFrame with columns anchor_id, candidate_id, task, gold_label, split;
        or None when the source is unavailable (with an informative message printed).
    """
    scope: str = verification_config.get("scope", "benchmark")
    task_key = task.lower()  # "and" or "ain"

    if scope == "benchmark":
        bench_cfg = verification_config.get("benchmark_pairs", {})
        raw_path = bench_cfg.get(f"{task_key}_path", "")
        bench_path = Path(raw_path) if raw_path else None

        if bench_path is None or not bench_path.exists():
            print(
                f"  [C5/{task}] benchmark file not found: {bench_path or '(not configured)'}. "
                f"Skipping {task} verification.\n"
                f"  To create it, export annotations from Label Studio and run:\n"
                f"    python -m bem.benchmark.convert_labelstudio \\\n"
                f"        --input <path/to/labelstudio_export.csv> \\\n"
                f"        --format auto\n"
                f"  Then re-run the pipeline."
            )
            return None

        try:
            df = load_pairs_from_parquet(bench_path)
        except (ValueError, Exception) as exc:
            print(f"  [C5/{task}] Failed to load benchmark file: {exc}. Skipping.")
            return None

        # Filter to this task in case the file mixes tasks
        df = df[df["task"] == task].copy()
        if df.empty:
            print(
                f"  [C5/{task}] Benchmark file contains no rows for task={task!r}. Skipping."
            )
            return None

        return df.reset_index(drop=True)

    elif scope == "full_candidates":
        fname = "candidates_and.parquet" if task == "AND" else "candidates_ain.parquet"
        cand_path = Path(interim_dir) / fname
        if not cand_path.exists():
            print(
                f"  [C5/{task}] Candidates file not found: {cand_path}. "
                "Run C4 first. Skipping."
            )
            return None

        max_pairs: int = int(verification_config.get("max_pairs_per_task", 5000))
        df = pd.read_parquet(cand_path)

        # Keep only the columns we need; add optional gold_label / split
        df = df[["anchor_id", "candidate_id", "task"]].copy()
        df["gold_label"] = None
        df["split"] = None

        if len(df) > max_pairs:
            df = (
                df.sample(n=max_pairs, random_state=42)
                .reset_index(drop=True)
            )

        return df

    else:
        print(
            f"  [C5/{task}] Unknown verification scope: {scope!r}. "
            "Expected 'benchmark' or 'full_candidates'. Skipping."
        )
        return None
