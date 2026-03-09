"""dev_diagnostics.py — DEV-set diagnostics for benchmark composition and LLM behaviour.

Joins benchmark gold labels, LLM decisions, and C6 routing outputs; prints
distribution tables and exports CSV samples for manual inspection.

Usage
-----
    python -m bem.analysis.dev_diagnostics \\
        --run_id 20260308_173145_sueeqr \\
        --out runs/20260308_173145_sueeqr/analysis

    # Override default benchmark paths:
    python -m bem.analysis.dev_diagnostics \\
        --run_id 20260308_173145_sueeqr \\
        --out runs/20260308_173145_sueeqr/analysis \\
        --benchmark_and data/derived/benchmark_pairs_and.parquet \\
        --benchmark_ain data/derived/benchmark_pairs_ain.parquet

No API calls are made.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALL_LABELS = ["match", "non-match", "uncertain"]
_SAMPLE_N   = 20


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_decisions_jsonl(path: Path) -> pd.DataFrame:
    """Load llm_decisions_*.jsonl into a flat DataFrame.

    Returns columns: anchor_id, candidate_id, llm_label, llm_confidence,
    llm_reason_code, llm_retry_count.
    """
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            dec = obj.get("decision") or {}
            records.append({
                "anchor_id":        obj.get("anchor_id", ""),
                "candidate_id":     obj.get("candidate_id", ""),
                "llm_label":        dec.get("label", "error") if dec else "error",
                "llm_confidence":   float(dec.get("confidence", 0.0)) if dec else 0.0,
                "llm_reason_code":  dec.get("reason_code", "") if dec else "",
                "llm_retry_count":  int(obj.get("retry_count", 0)),
            })

    if not records:
        return pd.DataFrame(columns=[
            "anchor_id", "candidate_id", "llm_label",
            "llm_confidence", "llm_reason_code", "llm_retry_count",
        ])

    df = pd.DataFrame(records)
    # Keep last non-error per pair (mirrors apply_guards dedup logic)
    non_err = df[df["llm_label"] != "error"]
    if non_err.empty:
        return df.groupby(["anchor_id", "candidate_id"], sort=False).last().reset_index()
    return non_err.groupby(["anchor_id", "candidate_id"], sort=False).last().reset_index()


def _load_routing_parquet(path: Path) -> pd.DataFrame:
    """Load routing_log_*.parquet, keeping only the columns needed for diagnostics."""
    df = pd.read_parquet(path)
    keep = [c for c in [
        "anchor_id", "candidate_id",
        "label_final", "override_reason", "signals_count",
        "fired_categories", "routed_to_human",
    ] if c in df.columns]
    df = df[keep].copy()
    # Deduplicate by pair, keeping last row (matches apply_guards dedup)
    df = df.groupby(["anchor_id", "candidate_id"], sort=False).last().reset_index()
    return df


def _load_benchmark(path: Path, split: str = "dev") -> pd.DataFrame:
    """Load benchmark parquet filtered to a single split."""
    df = pd.read_parquet(path)
    return df[df["split"] == split][["anchor_id", "candidate_id", "gold_label"]].copy()


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------

def build_joined(
    benchmark: pd.DataFrame,
    decisions: pd.DataFrame,
    routing: pd.DataFrame,
) -> pd.DataFrame:
    """Join benchmark gold labels with LLM decisions and C6 routing on (anchor_id, candidate_id).

    Pairs present in the benchmark but absent from decisions/routing get NaN
    for the LLM/routing columns.
    """
    df = benchmark.copy()
    df = df.merge(decisions, on=["anchor_id", "candidate_id"], how="left")
    df = df.merge(routing,   on=["anchor_id", "candidate_id"], how="left")
    return df


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _bar(value: int, total: int, width: int = 20) -> str:
    filled = round(width * value / max(total, 1))
    return "#" * filled + "." * (width - filled)


def _print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _print_distribution(df: pd.DataFrame, col: str, title: str) -> None:
    total = len(df)
    counts = df[col].value_counts(dropna=False)
    # Ensure all standard labels appear
    all_keys = _ALL_LABELS + [k for k in counts.index if k not in _ALL_LABELS and pd.notna(k)]
    print(f"\n  {title}  (n={total})")
    print(f"  {'Label':<14}  {'Count':>6}  {'%':>6}  {'Bar'}")
    print(f"  {'-'*14}  {'-'*6}  {'-'*6}  {'-'*20}")
    for label in all_keys:
        n = int(counts.get(label, 0))
        pct = 100.0 * n / max(total, 1)
        print(f"  {str(label):<14}  {n:>6}  {pct:>5.1f}%  {_bar(n, total)}")
    na_count = int(df[col].isna().sum())
    if na_count:
        print(f"  {'<missing>':<14}  {na_count:>6}  {100.*na_count/max(total,1):>5.1f}%  {_bar(na_count, total)}")


def _print_confusion(df: pd.DataFrame, row_col: str, col_col: str, title: str) -> None:
    """Print a crosstab confusion matrix."""
    row_labels = [l for l in _ALL_LABELS if l in df[row_col].values or l in df[col_col].values]
    col_labels = [l for l in _ALL_LABELS if l in df[row_col].values or l in df[col_col].values]
    ct = pd.crosstab(
        df[row_col].fillna("<missing>"),
        df[col_col].fillna("<missing>"),
        margins=True,
    )
    print(f"\n  {title}")
    # Header
    all_cols = [c for c in ct.columns]
    col_w = max(10, *(len(str(c)) for c in all_cols))
    row_label_w = max(10, len(row_col))
    header = f"  {row_col:<{row_label_w}}  " + "  ".join(f"{str(c):>{col_w}}" for c in all_cols)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row_key in ct.index:
        row_str = f"  {str(row_key):<{row_label_w}}  "
        row_str += "  ".join(f"{int(ct.loc[row_key, c]):>{col_w}}" for c in all_cols)
        print(row_str)


# ---------------------------------------------------------------------------
# CSV sample export
# ---------------------------------------------------------------------------

_SAMPLE_COLS = [
    "anchor_id", "candidate_id",
    "gold_label", "llm_label", "llm_confidence",
    "label_final", "signals_count", "override_reason",
]


def _export_sample(
    df: pd.DataFrame,
    mask: pd.Series,
    out_path: Path,
    description: str,
) -> int:
    """Export up to _SAMPLE_N rows matching mask to out_path.  Returns row count."""
    subset = df[mask].copy()
    n = len(subset)
    if n == 0:
        return 0
    cols = [c for c in _SAMPLE_COLS if c in subset.columns]
    subset[cols].head(_SAMPLE_N).to_csv(out_path, index=False)
    return n


# ---------------------------------------------------------------------------
# Per-task diagnostics
# ---------------------------------------------------------------------------

def run_task_diagnostics(
    task: str,
    benchmark_path: Path,
    decisions_path: Path,
    routing_path: Path,
    out_dir: Path,
) -> None:
    """Run full diagnostics for one task (AND or AIN)."""
    # -- Load --
    benchmark = _load_benchmark(benchmark_path, split="dev")
    decisions = _load_decisions_jsonl(decisions_path)
    routing   = _load_routing_parquet(routing_path)

    joined = build_joined(benchmark, decisions, routing)
    n_total = len(joined)
    n_matched = joined["llm_label"].notna().sum()
    n_missing = n_total - n_matched

    _print_section(f"TASK: {task.upper()}  ({n_total} DEV pairs)")

    if n_missing:
        print(f"\n  WARNING: {n_missing} benchmark pairs have no LLM decision (join miss).")

    # -- Distributions --
    _print_distribution(joined, "gold_label",  "Gold label distribution")
    _print_distribution(joined, "llm_label",   "LLM label distribution")
    _print_distribution(joined, "label_final", "Final label (after C6 guards)")

    # -- Confusion matrices --
    joined_dec = joined.dropna(subset=["llm_label"])
    _print_confusion(joined_dec, "gold_label", "llm_label",   "Confusion: gold x llm")
    joined_fin = joined.dropna(subset=["label_final"])
    _print_confusion(joined_fin, "gold_label", "label_final", "Confusion: gold x final")

    # -- Guard statistics (auto-routed vs adjudication) --
    if "routed_to_human" in joined.columns:
        n_auto   = int((~joined["routed_to_human"].fillna(True)).sum())
        n_human  = int(joined["routed_to_human"].fillna(True).sum())
        print(f"\n  Guard routing:")
        print(f"    auto-routed  : {n_auto:>5}  ({100.*n_auto/max(n_total,1):.1f}%)")
        print(f"    -> human     : {n_human:>5}  ({100.*n_human/max(n_total,1):.1f}%)")

    # -- Override reasons summary --
    if "override_reason" in joined.columns:
        overrides = joined["override_reason"].replace("", pd.NA).dropna()
        if not overrides.empty:
            print(f"\n  Override reasons:")
            for reason, cnt in overrides.value_counts().items():
                print(f"    {reason:<50}  {cnt:>5}")

    # -- CSV samples --
    print(f"\n  Sample exports -> {out_dir}")

    samples: list[tuple[pd.Series, str, str]] = [
        (
            joined["gold_label"].eq("uncertain") & joined["label_final"].isin(["match", "non-match"]),
            f"sample_{task.lower()}_gold_uncertain_but_decided.csv",
            "gold=uncertain & final in {match, non-match}",
        ),
        (
            joined["gold_label"].eq("match") & joined["label_final"].ne("match").fillna(True),
            f"sample_{task.lower()}_gold_match_missed.csv",
            "gold=match & final != match",
        ),
        (
            joined["gold_label"].eq("non-match") & joined["label_final"].ne("non-match").fillna(True),
            f"sample_{task.lower()}_gold_nonmatch_missed.csv",
            "gold=non-match & final != non-match",
        ),
    ]

    for mask, fname, desc in samples:
        n = _export_sample(joined, mask, out_dir / fname, desc)
        if n:
            print(f"    {fname}  ({min(n, _SAMPLE_N)} rows / {n} total)  [{desc}]")
        else:
            print(f"    {fname}  (0 rows — no cases)  [{desc}]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DEV-set diagnostics: benchmark composition + LLM behaviour.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Example:
              python -m bem.analysis.dev_diagnostics \\
                  --run_id 20260308_173145_sueeqr \\
                  --out runs/20260308_173145_sueeqr/analysis
        """),
    )
    parser.add_argument("--run_id",        required=True, help="Run ID (subfolder of runs/)")
    parser.add_argument("--out",           required=True, help="Output directory for CSV samples")
    parser.add_argument("--benchmark_and", default="data/derived/benchmark_pairs_and.parquet")
    parser.add_argument("--benchmark_ain", default="data/derived/benchmark_pairs_ain.parquet")
    args = parser.parse_args()

    run_dir = Path("runs") / args.run_id
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"bem dev diagnostics")
    print(f"  run_id : {args.run_id}")
    print(f"  out    : {out_dir}")

    for task in ["AND", "AIN"]:
        t = task.lower()
        run_task_diagnostics(
            task=task,
            benchmark_path=Path(args.benchmark_and if task == "AND" else args.benchmark_ain),
            decisions_path=run_dir / "logs" / f"llm_decisions_{t}.jsonl",
            routing_path=run_dir  / "logs" / f"routing_log_{t}.parquet",
            out_dir=out_dir,
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
