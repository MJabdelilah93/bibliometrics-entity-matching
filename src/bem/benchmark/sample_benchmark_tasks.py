"""sample_benchmark_tasks.py — Draw stratified annotation task CSVs from C4 candidate parquets.

Two modes
---------
Default mode (--dev_only false):
    Samples 5 000 pairs per task across five similarity-score quintile bands
    (1 000 pairs per band).  The first 1 000 rows become split="dev", the rest
    split="test".  Outputs annotation_tasks_and.csv / annotation_tasks_ain.csv.

Dev-quota mode (--dev_only true):
    Samples exactly --dev_size pairs per task using a three-stratum quota:
      matchlike   = top 20 % of similarity_score  (proxy for true matches)
      nonmatchlike = bottom 20 %                   (proxy for non-matches)
      hard        = middle 60 %                    (ambiguous / hard cases)
    Quotas are set per-task via --and_dev_quota / --ain_dev_quota.
    All rows get split="dev".  Outputs {prefix}_annotation_tasks_and.csv
    where --prefix defaults to "dev2" in dev-quota mode.

Usage
-----
    # Default mode (backward-compatible):
    python -m bem.benchmark.sample_benchmark_tasks --out_dir data/derived --seed 42

    # Dev-quota mode:
    python -m bem.benchmark.sample_benchmark_tasks \\
        --dev_only true \\
        --dev_size 1000 \\
        --seed 42 \\
        --and_dev_quota "matchlike=250,nonmatchlike=250,hard=500" \\
        --ain_dev_quota "matchlike=250,nonmatchlike=250,hard=500" \\
        --out_dir data/derived

Outputs (dev-quota mode)
------------------------
    <out_dir>/dev2_annotation_tasks_and.csv
    <out_dir>/dev2_annotation_tasks_ain.csv

Each CSV columns:
    task, anchor_id, candidate_id, similarity_score, best_pass_id,
    stratum, split, gold_label, notes

Determinism
-----------
All random operations use --seed. Re-running with the same seed and
the same input parquets produces identical CSVs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BINS        = 5
PER_BIN       = 1_000
TOTAL_PAIRS   = N_BINS * PER_BIN   # 5 000

DEV_SIZE      = 1_000              # first N rows → "dev" (default mode)

OUTPUT_COLS = [
    "task",
    "anchor_id",
    "candidate_id",
    "similarity_score",
    "best_pass_id",
    "split",
    "gold_label",
    "notes",
]

# Dev-quota mode adds a stratum column
OUTPUT_COLS_QUOTA = [
    "task",
    "anchor_id",
    "candidate_id",
    "similarity_score",
    "best_pass_id",
    "stratum",
    "split",
    "gold_label",
    "notes",
]

CANDIDATE_PATHS = {
    "AND": "data/interim/candidates_and.parquet",
    "AIN": "data/interim/candidates_ain.parquet",
}

OUT_NAMES = {
    "AND": "annotation_tasks_and.csv",
    "AIN": "annotation_tasks_ain.csv",
}

# Stratum boundaries: top/bottom N% by similarity_score
_STRATUM_LO_PCT = 20   # nonmatchlike: bottom 20 %
_STRATUM_HI_PCT = 80   # matchlike:    top 20 %


# ---------------------------------------------------------------------------
# Quota string parser
# ---------------------------------------------------------------------------

def _parse_quota(quota_str: str) -> dict[str, int]:
    """Parse "matchlike=250,nonmatchlike=250,hard=500" → {"matchlike": 250, ...}.

    Raises ValueError on malformed input.
    """
    result: dict[str, int] = {}
    for part in quota_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad quota token (expected key=value): {part!r}")
        key, val = part.split("=", 1)
        key = key.strip().lower()
        if key not in ("matchlike", "nonmatchlike", "hard"):
            raise ValueError(f"Unknown stratum name {key!r}. Use: matchlike, nonmatchlike, hard")
        result[key] = int(val.strip())
    for required in ("matchlike", "nonmatchlike", "hard"):
        if required not in result:
            raise ValueError(f"Quota missing stratum {required!r}")
    return result


# ---------------------------------------------------------------------------
# Default mode: quintile-based (original logic)
# ---------------------------------------------------------------------------

def _sample_task_quintile(df: pd.DataFrame, task: str, seed: int) -> pd.DataFrame:
    """Return a stratified sample of TOTAL_PAIRS rows from df (original quintile mode).

    Args:
        df:   Candidate parquet loaded as a DataFrame.
        task: "AND" or "AIN" (used only to set the task column).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with OUTPUT_COLS (gold_label and notes are empty strings).
    """
    rng = np.random.default_rng(seed)

    try:
        df = df.copy()
        df["_bin"] = pd.qcut(df["similarity_score"], q=N_BINS, labels=False, duplicates="drop")
    except ValueError:
        lo, hi = df["similarity_score"].min(), df["similarity_score"].max()
        edges = np.linspace(lo, hi + 1e-9, N_BINS + 1)
        df["_bin"] = pd.cut(df["similarity_score"], bins=edges, labels=False, include_lowest=True)

    bins_present = sorted(df["_bin"].dropna().unique())

    sampled_parts: list[pd.DataFrame] = []
    shortfall = 0

    for b in bins_present:
        bin_df = df[df["_bin"] == b]
        n_available = len(bin_df)
        n_take = min(PER_BIN, n_available)
        shortfall += PER_BIN - n_take
        if n_take > 0:
            idx = rng.choice(n_available, size=n_take, replace=False)
            sampled_parts.append(bin_df.iloc[idx])

    sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame(columns=df.columns)

    if shortfall > 0 and len(sampled) < TOTAL_PAIRS:
        sampled_ids = set(zip(sampled["anchor_id"], sampled["candidate_id"]))
        remainder = df[
            ~df.apply(lambda r: (r["anchor_id"], r["candidate_id"]) in sampled_ids, axis=1)
        ]
        if not remainder.empty:
            n_extra = min(shortfall, len(remainder))
            idx = rng.choice(len(remainder), size=n_extra, replace=False)
            sampled = pd.concat([sampled, remainder.iloc[idx]], ignore_index=True)

    sampled = sampled.sort_values(["_bin", "anchor_id", "candidate_id"]).reset_index(drop=True)

    sampled["split"] = "test"
    sampled.loc[sampled.index < DEV_SIZE, "split"] = "dev"

    sampled["task"]       = task
    sampled["gold_label"] = ""
    sampled["notes"]      = ""

    if "best_pass_id" not in sampled.columns:
        sampled["best_pass_id"] = ""

    return sampled[OUTPUT_COLS].copy()


# ---------------------------------------------------------------------------
# Dev-quota mode: three-stratum sampling
# ---------------------------------------------------------------------------

def _assign_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'stratum' column to df based on similarity_score percentiles.

    Boundaries are computed from the *full* candidate set (df), so that the
    percentile cut-points reflect the global distribution.

    Strata:
        matchlike    >= 80th percentile of similarity_score
        nonmatchlike <= 20th percentile
        hard         between 20th and 80th percentile (exclusive)
    """
    df = df.copy()
    lo = df["similarity_score"].quantile(_STRATUM_LO_PCT / 100)
    hi = df["similarity_score"].quantile(_STRATUM_HI_PCT / 100)

    conditions = [
        df["similarity_score"] >= hi,
        df["similarity_score"] <  lo,   # strict: pairs strictly below p20
    ]
    choices = ["matchlike", "nonmatchlike"]
    df["stratum"] = np.select(conditions, choices, default="hard")
    return df, lo, hi


def _sample_task_dev_quota(
    df: pd.DataFrame,
    task: str,
    seed: int,
    dev_size: int,
    quota: dict[str, int],
) -> pd.DataFrame:
    """Sample dev_size pairs using three-stratum quotas.

    Args:
        df:       Candidate parquet as DataFrame.
        task:     "AND" or "AIN".
        seed:     Random seed.
        dev_size: Total pairs to sample (should equal sum(quota.values())).
        quota:    {"matchlike": N, "nonmatchlike": N, "hard": N}.

    Returns:
        DataFrame with OUTPUT_COLS_QUOTA; all rows have split="dev".
    """
    rng = np.random.default_rng(seed)

    df, lo_threshold, hi_threshold = _assign_strata(df)

    strata_order = ["matchlike", "nonmatchlike", "hard"]
    sampled_parts: list[pd.DataFrame] = []

    print(f"\n  [{task}] Stratum boundaries:")
    print(f"    nonmatchlike : similarity_score <  {lo_threshold:.4f}  "
          f"(strictly below p{_STRATUM_LO_PCT})")
    print(f"    matchlike    : similarity_score >= {hi_threshold:.4f}  "
          f"(top {100 - _STRATUM_HI_PCT}%)")
    print(f"    hard         : {lo_threshold:.4f} <= similarity_score < {hi_threshold:.4f}")
    print()
    print(f"  [{task}] Stratum pool sizes and quotas:")
    print(f"    {'Stratum':<14}  {'Pool':>10}  {'Quota':>7}  {'Score range'}")
    print(f"    {'-'*14}  {'-'*10}  {'-'*7}  {'-'*30}")

    shortfall = 0
    shortfall_donors: list[str] = []   # strata with surplus

    for stratum in strata_order:
        stratum_df = df[df["stratum"] == stratum]
        pool_n     = len(stratum_df)
        want       = quota.get(stratum, 0)
        n_take     = min(want, pool_n)
        gap        = want - n_take
        shortfall += gap

        s_min = stratum_df["similarity_score"].min() if pool_n else float("nan")
        s_max = stratum_df["similarity_score"].max() if pool_n else float("nan")
        flag  = f"  *** SHORT by {gap} ***" if gap else ""

        print(f"    {stratum:<14}  {pool_n:>10,}  {want:>7}  "
              f"[{s_min:.4f}, {s_max:.4f}]{flag}")

        if n_take > 0:
            idx = rng.choice(pool_n, size=n_take, replace=False)
            sampled_parts.append(stratum_df.iloc[idx])

        if pool_n > n_take:
            shortfall_donors.append(stratum)

    # Fill any shortfall from strata with surplus (largest first)
    if shortfall > 0:
        print(f"\n  [{task}] Filling shortfall of {shortfall} from surplus strata: "
              f"{shortfall_donors}")
        sampled_ids: set[tuple] = set()
        for part in sampled_parts:
            sampled_ids.update(zip(part["anchor_id"], part["candidate_id"]))

        remainder = df[
            (~df.apply(lambda r: (r["anchor_id"], r["candidate_id"]) in sampled_ids, axis=1))
        ]
        if not remainder.empty:
            n_extra = min(shortfall, len(remainder))
            idx = rng.choice(len(remainder), size=n_extra, replace=False)
            extra = remainder.iloc[idx].copy()
            sampled_parts.append(extra)
            print(f"  [{task}] Drew {n_extra} extra rows.")

    sampled = (
        pd.concat(sampled_parts, ignore_index=True)
        if sampled_parts
        else pd.DataFrame(columns=df.columns)
    )

    # Stable sort: stratum order → similarity_score desc → (anchor_id, candidate_id)
    stratum_order_map = {s: i for i, s in enumerate(strata_order)}
    sampled["_stratum_ord"] = sampled["stratum"].map(stratum_order_map).fillna(99)
    sampled = sampled.sort_values(
        ["_stratum_ord", "similarity_score", "anchor_id", "candidate_id"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)

    sampled["split"]      = "dev"
    sampled["task"]       = task
    sampled["gold_label"] = ""
    sampled["notes"]      = ""

    if "best_pass_id" not in sampled.columns:
        sampled["best_pass_id"] = ""

    return sampled[OUTPUT_COLS_QUOTA].copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sample stratified annotation task CSVs from C4 candidate parquets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out_dir",
        default="data/derived",
        help="Directory to write annotation task CSVs (default: data/derived).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--candidates_and",
        default=CANDIDATE_PATHS["AND"],
        help=f"Path to candidates_and.parquet (default: {CANDIDATE_PATHS['AND']}).",
    )
    parser.add_argument(
        "--candidates_ain",
        default=CANDIDATE_PATHS["AIN"],
        help=f"Path to candidates_ain.parquet (default: {CANDIDATE_PATHS['AIN']}).",
    )
    # -- Dev-quota mode flags --
    parser.add_argument(
        "--dev_only",
        default="false",
        help="'true' → dev-quota mode (3-stratum quota sampling); "
             "'false' → default quintile mode (default: false).",
    )
    parser.add_argument(
        "--dev_size",
        type=int,
        default=1000,
        help="Total pairs per task in dev-quota mode (default: 1000).",
    )
    parser.add_argument(
        "--and_dev_quota",
        default="matchlike=250,nonmatchlike=250,hard=500",
        help='AND quota per stratum, e.g. "matchlike=250,nonmatchlike=250,hard=500".',
    )
    parser.add_argument(
        "--ain_dev_quota",
        default="matchlike=250,nonmatchlike=250,hard=500",
        help='AIN quota per stratum, e.g. "matchlike=250,nonmatchlike=250,hard=500".',
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix.  Defaults to 'dev2' in dev-quota mode, "
             "empty in default mode.  E.g. 'dev2' → dev2_annotation_tasks_and.csv.",
    )
    args = parser.parse_args(argv)

    dev_only = args.dev_only.strip().lower() == "true"
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve output prefix
    if args.prefix is not None:
        prefix = args.prefix.rstrip("_") + "_" if args.prefix else ""
    else:
        prefix = "dev2_" if dev_only else ""

    # Parse quotas (only used in dev-quota mode but validate early)
    quotas: dict[str, dict[str, int]] = {}
    if dev_only:
        try:
            quotas["AND"] = _parse_quota(args.and_dev_quota)
            quotas["AIN"] = _parse_quota(args.ain_dev_quota)
        except ValueError as exc:
            print(f"ERROR: Invalid quota specification: {exc}")
            sys.exit(1)

        total_and = sum(quotas["AND"].values())
        total_ain = sum(quotas["AIN"].values())
        if total_and != args.dev_size:
            print(f"WARNING: AND quota sums to {total_and}, not --dev_size={args.dev_size}.")
        if total_ain != args.dev_size:
            print(f"WARNING: AIN quota sums to {total_ain}, not --dev_size={args.dev_size}.")

    mode_label = "dev-quota (3-stratum)" if dev_only else "default (5-quintile)"
    print(f"sample_benchmark_tasks  [mode={mode_label}]")
    if dev_only:
        print(f"  AND quota : {args.and_dev_quota}")
        print(f"  AIN quota : {args.ain_dev_quota}")
        print(f"  dev_size  : {args.dev_size}")
    print(f"  seed      : {args.seed}")
    print(f"  out_dir   : {out_dir}")
    print(f"  prefix    : {prefix!r}")
    print()

    paths = {
        "AND": Path(args.candidates_and),
        "AIN": Path(args.candidates_ain),
    }

    for task, cand_path in paths.items():
        if not cand_path.exists():
            print(f"  [{task}] ERROR: candidate parquet not found: {cand_path}")
            print(f"  [{task}] Run C4 first:  python -m bem --config configs/run_config.yaml")
            sys.exit(1)

        print(f"  [{task}] Loading {cand_path} …", end="", flush=True)
        df = pd.read_parquet(cand_path)
        print(f" {len(df):,} rows")

        if dev_only:
            sampled = _sample_task_dev_quota(
                df, task, seed=args.seed,
                dev_size=args.dev_size,
                quota=quotas[task],
            )
        else:
            sampled = _sample_task_quintile(df, task, seed=args.seed)

        out_name = f"{prefix}{OUT_NAMES[task]}"
        out_path = out_dir / out_name
        sampled.to_csv(out_path, index=False)

        # Summary
        if dev_only:
            print()
            print(f"  [{task}] Sampled {len(sampled):,} pairs  (all split=dev)")
            if "stratum" in sampled.columns:
                for s, cnt in sampled["stratum"].value_counts().sort_index().items():
                    s_min = sampled.loc[sampled["stratum"] == s, "similarity_score"].min()
                    s_max = sampled.loc[sampled["stratum"] == s, "similarity_score"].max()
                    print(f"    {s:<14}  {cnt:>5}  score=[{float(s_min):.4f}, {float(s_max):.4f}]")
        else:
            n_dev  = int((sampled["split"] == "dev").sum())
            n_test = int((sampled["split"] == "test").sum())
            desc   = sampled["similarity_score"].describe()[["min", "25%", "50%", "75%", "max"]]
            print(f"  [{task}] Sampled {len(sampled):,} pairs  (dev={n_dev}, test={n_test})")
            print(f"  [{task}] Score range: min={desc['min']:.3f}  p25={desc['25%']:.3f}  "
                  f"median={desc['50%']:.3f}  p75={desc['75%']:.3f}  max={desc['max']:.3f}")

        print(f"  [{task}] Written -> {out_path}")
        print()

    print("Next step:")
    if dev_only:
        print(f"  Build evidence packets:")
        print(f"    python -m bem.benchmark.build_annotation_packets \\")
        print(f"        --prefix {prefix.rstrip('_')} \\")
        print(f"        --only_dev true")
        print()
        print(f"  Then open {prefix}annotation_packets_and.csv / "
              f"{prefix}annotation_packets_ain.csv,")
        print(f"  fill the 'gold_label' column (match / non-match / uncertain),")
        print(f"  and run pack_benchmark_pairs with --in_prefix {prefix.rstrip('_')}.")
    else:
        print("  Open the annotation CSVs, fill the 'gold_label' column")
        print("  (match / non-match / uncertain) for every row — without LLM assistance.")
        print("  Then run:")
        print("    python -m bem.benchmark.pack_benchmark_pairs --in_dir data/derived")


if __name__ == "__main__":
    main()
