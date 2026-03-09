# Reproducibility Guide for BEM

This document explains how to reproduce the BEM entity-matching pipeline
results using your own Scopus UI exports.

---

## 1. Prerequisites

- Python ≥ 3.11
- A Scopus institutional licence (for export access)
- Optional: an Anthropic API key (only for LLM verification; `requests_only`
  mode works without one)

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 2. Scopus UI Exports (33-column schema)

BEM ingests **Scopus UI CSV exports only** — no Scopus API, no external data.

### How to export from Scopus

1. Run your search query in [Scopus](https://www.scopus.com).
2. Select all results → **Export** → **CSV** → tick **All available columns**.
3. Save the file using this naming convention:

```
scopus_<Q>_<QUERY_LABEL>_batch<NNN>_<YYYYMMDD>.csv
```

Example: `scopus_Q1_CORE_MOROCCO_2020_2021_batch001_20260227.csv`

4. Place the file(s) in `data/raw/` (this directory is git-ignored and must be
   created locally).

### Required 33-column schema

The pipeline validates column headers against `configs/schema_headers.txt`.
Scopus exports all 33 columns by default when "All available columns" is
selected. If you get a schema validation error, compare your export headers to
`configs/schema_headers.txt`.

---

## 3. Running the Pipeline

Copy `configs/example_run_config.yaml` to `configs/run_config.yaml` and edit
the `inputs` section to point at your CSV files:

```yaml
inputs:
  q1_csv_paths:
    - "data/raw/scopus_Q1_YOUR_QUERY_batch001_YYYYMMDD.csv"
  q2_csv_paths:
    - "data/raw/scopus_Q2_YOUR_QUERY_batch001_YYYYMMDD.csv"
```

Then run:

```bash
python -m bem --config configs/run_config.yaml
```

Each run writes a timestamped manifest under `runs/<run_id>/manifests/`.

---

## 4. LLM Backend Modes

BEM supports two LLM backends, controlled by `llm.backend` in the config:

### `requests_only` (default, no API key needed)

- Writes filled prompt envelopes to `runs/<run_id>/logs/llm_requests_*.jsonl`.
- Produces stub decisions (`uncertain / 0.0`) — no real LLM calls are made.
- Use this mode to inspect prompts, test the pipeline end-to-end, and review
  what evidence the LLM would receive.

### `anthropic_api` (real LLM calls)

- Requires `ANTHROPIC_API_KEY` set in the environment (e.g., in a `.env` file
  that is **never committed**).
- Sends each evidence card to Claude and records the decision + reasoning.
- Enable by setting `llm.backend: "anthropic_api"` in your config.

To switch:

```yaml
llm:
  backend: "anthropic_api"   # or "requests_only"
```

And set the key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## 5. Benchmark Annotation Workflow (no LLM labelling)

Gold-standard labels are assigned **by the human researcher only** — no LLM
is ever used to suggest or decide match/non-match labels.

### Step 1 — Run C1–C4 to generate candidate pairs

```bash
python -m bem --config configs/run_config.yaml
```

This produces `data/interim/candidates_and.parquet` and
`data/interim/candidates_ain.parquet`.

### Step 2 — Sample pairs for annotation

```bash
python -m bem.benchmark.sample_benchmark_tasks --out_dir data/derived --seed 42
```

Writes `data/derived/annotation_tasks_and.csv` and `_ain.csv` with 5 000
pairs per task across five similarity-quintile bands.

### Step 3 — Build evidence packets

```bash
python -m bem.benchmark.build_annotation_packets \
    --in_dir data/derived \
    --out_dir data/derived \
    --only_dev false
```

Writes enriched CSVs (`annotation_packets_and.csv`, `_ain.csv`) with all
Scopus-derived evidence needed to label each pair.

### Step 4 — Assign gold labels manually

Open the CSV files in Excel or LibreOffice Calc.
For each row, enter one of the following in the `gold_label` column:

| Value | Meaning |
|-------|---------|
| `match` | Same real-world entity |
| `non-match` | Different entities |
| `uncertain` | Insufficient evidence (treated as NO MERGE) |

**No LLM assistance at this step.**

### Step 5 — Pack to parquet and run C5

```bash
python -m bem.benchmark.pack_benchmark_pairs --in_dir data/derived
python -m bem --config configs/run_config.yaml
```

---

## 6. Optional: Dev2 Minimal Annotation (numeric codes)

For a compact Excel workflow with numeric codes (0 = uncertain, 1 = non-match,
2 = match):

```bash
# Generate minimal files
python -m bem.benchmark.make_min_annotation_files --in_dir data/derived --prefix dev2

# After annotating in Excel:
python -m bem.benchmark.apply_min_labels_to_dev2 --in_dir data/derived --prefix dev2
```

---

## 7. Deterministic Auto-fill (optional QC aid)

Before manual annotation, you may pre-fill obvious cases deterministically:

```bash
python -m bem.benchmark.autofill_gold_labels \
    --and_in  data/derived/annotation_packets_and.csv \
    --ain_in  data/derived/annotation_packets_ain.csv \
    --and_out data/derived/annotation_packets_and.csv \
    --ain_out data/derived/annotation_packets_ain.csv \
    --overwrite false
```

This uses conservative thresholds on evidence columns only — no LLM involved.
Ambiguous cases are left blank for human judgement.

---

## 8. Evidence Boundary

- **Input source:** Scopus UI CSV exports (33-column schema) only.
- **Truncation rule:** Records at the 100-author boundary are excluded from LLM
  verification and logged as `insufficient_evidence` (default: NO MERGE).
- **Author(s) ID** is used only for candidate generation and is never shown
  to the LLM during verification.
- **Gold labels** are always assigned by the human researcher; LLM is used only
  for code generation, QA, and documentation — never for labelling decisions.

---

## 9. Run Manifests and Auditability

Every pipeline run writes:

```
runs/<run_id>/
  manifests/
    run_manifest.json          # config + schema hashes (C1)
    export_manifest.json       # input file hashes, row counts (C2)
    normalisation_manifest.json # rules hash, column stats (C3)
  logs/
    normalisation_log.jsonl
    llm_requests_and.jsonl     # filled prompt envelopes (C5)
    llm_decisions_and.jsonl    # LLM verdicts (C5)
```

All outputs are deterministic given the same inputs, config, and seed.
