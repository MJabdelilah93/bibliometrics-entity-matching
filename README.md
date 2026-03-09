# Bibliometrics Entity Matching (BEM)

## What is BEM?

BEM is a reproducible entity-matching pipeline for bibliometric data exported from the
Scopus UI. It resolves two entity types:

- **AND** (Author Name Disambiguation): assigns each (author-name, paper) occurrence
  to a unique real-world researcher.
- **AIN** (Affiliation Name Normalisation): clusters raw affiliation strings to canonical
  institution records.

The pipeline is fully audit-ready: every decision is traceable to a run manifest, a
config hash, and explicit evidence records.

---

## Evidence boundary & gold-annotation independence

**Input source:** Scopus UI CSV exports only (33-column schema). No Scopus API. No
external data sources are used to make entity decisions.

**Truncation rule:** Records at the observed 100-author boundary are flagged as
truncation cases and excluded from LLM verification. They are logged as
`insufficient_evidence` and default to NO MERGE.

**Pairwise labels:** `{match, non-match, uncertain}`.
`uncertain` is an abstention and defaults to NO MERGE.

**Gold-standard annotation independence:** LLM usage is restricted to code generation,
QA, and documentation. No LLM is used to suggest or decide match/non-match labels
for gold-standard annotation. All annotation decisions are made by the human annotator
from primary evidence only.

---

## Where to place raw data

Copy Scopus UI CSV exports into `bem/data/raw/`. **Do not commit them to git** —
`data/raw/` and all `*.csv` files are git-ignored.

The current batch files are:

```
data/raw/scopus_Q1_CORE_MOROCCO_2020_2021_batch001_20260227.csv
data/raw/scopus_Q2_STRESS_MAGHREB_AI_2018_2025_batch001_20260227.csv
```

Naming convention for future batches:

```
scopus_<Q>_<QUERY_LABEL>_batch<NNN>_<YYYYMMDD>.csv
```

---

## Processed outputs

Canonical intermediate outputs are written to `data/interim/` (git-ignored).
Final derived outputs are written to `data/derived/`.

### Interim tables

| File | Columns | Description |
|---|---|---|
| `data/interim/records_canonical.parquet` | 33 Scopus + 4 provenance | Raw field values as strings; no modification. Provenance: `query_frame`, `source_file`, `row_id_in_file`, `record_id` (sha256 of EID). |
| `data/interim/records_normalised.parquet` | 33 + 4 + 7 normalised | Canonical table extended with deterministically normalised columns (see below). Raw columns are never overwritten. |

Normalised columns added by C3:

| Column | Type | Source field |
|---|---|---|
| `authors_norm` | str | `Authors` — NFKC, casefold, collapsed whitespace |
| `author_full_names_norm` | str | `Author full names` — same |
| `author_ids_list` | list[str] | `Author(s) ID` — semicolon-split |
| `year_int` | nullable int | `Year` — safe integer parse |
| `affiliations_norm` | str | `Affiliations` — same as above |
| `affiliations_acronyms` | list[str] | `Affiliations` — all-caps tokens of length 2–10 |
| `authors_with_affiliations_norm` | str | `Authors with affiliations` — same |

### Normalisation artefacts

Each pipeline run writes its manifest and logs under:

```
runs/<bem_run_id>/
  manifests/
    run_manifest.json               C1: config + schema hashes
    export_manifest.json            C2: input file hashes, row counts, missingness
    normalisation_manifest.json     C3: rules hash, changed-value counts, top acronyms
  logs/
    normalisation_log.jsonl         C3: one JSON object per stat item
  outputs/
```

The normalisation manifest records:

- SHA-256 of `configs/normalisation_rules.yaml` (version-stamps the rule set).
- Per-column count of non-null/non-empty values for all 7 new columns.
- Per-column count of values that changed after normalisation (raw ≠ norm).
- Top-20 most frequent affiliation acronyms across the full corpus.
- Version strings from the rules config (`v0.1` placeholder; extend as rules grow).

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

The editable install (`-e .`) makes the `bem` package importable from `src/` without
path manipulation.

### 2. Run the scaffold

```bash
python -m bem --config configs/run_config.yaml
```

This writes `runs/<run_id>/manifests/run_manifest.json` and confirms the C1 scaffold
is wired correctly. No matching logic is executed yet.

### 3. Verify the config loads correctly

```bash
python tests/test_config_load.py
```

---

## One-time setup — Label Studio export placement

Before running C5 verification for the first time, place your Label Studio
annotation export in `data/derived/`:

| Export format | Target path |
|---|---|
| CSV  | `data/derived/labelstudio_export.csv`  |
| JSON | `data/derived/labelstudio_export.json` |

**Do not modify the content of the export file.**

A placeholder file already exists at `data/derived/labelstudio_export.csv`.
Overwrite it with your real export, or add a `.json` file alongside it.

Once the file is in place, run the converter:

```bash
python -m bem.benchmark.convert_labelstudio \
    --input data/derived/labelstudio_export.csv \
    --format auto \
    --task auto
```

This writes `data/derived/benchmark_pairs_and.parquet` and
`data/derived/benchmark_pairs_ain.parquet`, which C5 reads during the next
pipeline run.

---

## C5 Benchmark-first verification workflow

### Overview

C5 runs LLM-based pairwise verification against a human-annotated benchmark.
Gold labels are assigned by the researcher from primary evidence; the LLM never
suggests or decides labels.

With `verification.enabled: true` and `llm.backend: "requests_only"` (the defaults
after PLAN A setup), the pipeline writes:

- `runs/<run_id>/logs/llm_requests_and.jsonl` — filled prompt envelopes (no API call)
- `runs/<run_id>/logs/llm_decisions_and.jsonl` — stub decisions (`uncertain / 0.0`)
- same for `ain`

Switch to `llm.backend: "anthropic_api"` and set `ANTHROPIC_API_KEY` to make real
API calls.

### Step 1 — Annotate pairs in Label Studio

Export your Label Studio project as **CSV** or **JSON** (list-of-tasks format).

Required columns (CSV) or data fields (JSON):

| Field | Description |
|---|---|
| `anchor_id` | `author_instance_id` (AND) or `affil_instance_id` (AIN) |
| `candidate_id` | same type as anchor |
| `task` | `AND` or `AIN` |
| `gold_label` | `match` \| `non-match` \| `uncertain` |
| `split` *(optional)* | `train` \| `dev` \| `test` |
| `stratum` *(optional)* | annotation stratum tag |

### Step 2 — Convert the export

```bash
python -m bem.benchmark.convert_labelstudio \
    --input path/to/labelstudio_export.csv \
    --format auto \
    --task auto
```

This writes:

```
data/derived/benchmark_pairs_and.parquet
data/derived/benchmark_pairs_ain.parquet
```

The converter validates all IDs against `data/interim/author_instances.parquet`
and `data/interim/affil_instances.parquet`. Unknown IDs are reported and dropped.

Full options:

```
--input           Path to Label Studio CSV or JSON export (required)
--format          auto | csv | json  (default: auto, detects from file extension)
--task            auto | AND | AIN   (default: auto, reads 'task' column)
--out_and         Output path for AND parquet (default: data/derived/benchmark_pairs_and.parquet)
--out_ain         Output path for AIN parquet (default: data/derived/benchmark_pairs_ain.parquet)
--author_instances  Path to author_instances.parquet (default: data/interim/author_instances.parquet)
--affil_instances   Path to affil_instances.parquet  (default: data/interim/affil_instances.parquet)
```

### Step 3 — Run the pipeline

```bash
python -m bem --config configs/run_config.yaml
```

With `verification.enabled: true`, C5 will load the benchmark parquets and
produce decision logs. If a benchmark file is missing, the pipeline prints the
exact converter command and skips that task gracefully.

### Benchmark parquet schema

| Column | Type | Description |
|---|---|---|
| `anchor_id` | str | SHA-256 instance ID |
| `candidate_id` | str | SHA-256 instance ID |
| `task` | str | `AND` or `AIN` |
| `gold_label` | str | `match` \| `non-match` \| `uncertain` |
| `split` | str or None | `train` / `dev` / `test` |
| `stratum` | str or None | annotation stratum tag |

---

## Benchmark without Label Studio

Gold labels are assigned by the human researcher from primary evidence only —
**no LLM assistance at any step**.  The three-step workflow below produces the
benchmark parquets that C5 reads.

### Step 1 — Sample pairs

```bash
python -m bem.benchmark.sample_benchmark_tasks --out_dir data/derived --seed 42
```

Reads the C4 candidate parquets and samples **5 000 pairs per task** (AND + AIN)
across five similarity-score quintile bands (1 000 pairs per band).  Writes:

```
data/derived/annotation_tasks_and.csv
data/derived/annotation_tasks_ain.csv
```

Columns: `task`, `anchor_id`, `candidate_id`, `similarity_score`, `best_pass_id`,
`split` (`dev` = first 1 000 rows / `test` = remaining 4 000), `gold_label` (blank),
`notes` (blank).  Re-running with the same `--seed` produces identical output.

### Step 2 — Build evidence packets

```bash
python -m bem.benchmark.build_annotation_packets \
    --in_dir data/derived \
    --out_dir data/derived \
    --only_dev false
```

Joins each pair to the instance and record tables and writes enriched CSVs
containing all Scopus-derived evidence needed to assign a label:

```
data/derived/annotation_packets_and.csv
data/derived/annotation_packets_ain.csv
```

**AND evidence columns** (Author(s) ID is never included):

| Column | Description |
|---|---|
| `anchor_author_norm` / `candidate_author_norm` | Normalised author name |
| `anchor_coauthors_norm` / `candidate_coauthors_norm` | Co-authors, pipe-separated (first 30) |
| `anchor_affiliations_norm` / `candidate_affiliations_norm` | Affiliation string (≤ 300 chars) |
| `anchor_title` / `candidate_title` | Paper title (≤ 300 chars) |
| `anchor_source_title` / `candidate_source_title` | Journal / venue |
| `anchor_year` / `candidate_year` | Publication year |

**AIN evidence columns** (title, source, year are never included):

| Column | Description |
|---|---|
| `anchor_affil_raw` / `candidate_affil_raw` | Raw affiliation string (≤ 300 chars) |
| `anchor_affil_norm` / `candidate_affil_norm` | Normalised affiliation (≤ 300 chars) |
| `anchor_affil_acronyms` / `candidate_affil_acronyms` | All-caps tokens, pipe-separated |
| `anchor_linked_authors_norm` / `candidate_linked_authors_norm` | Authors linked to this affiliation (pipe-separated, first 30) |
| `anchor_linked_authors_fallback` / `candidate_linked_authors_fallback` | `true` if heuristic failed |

Use `--only_dev true` to build packets for the dev split only (faster for initial
annotation rounds).

### Step 3 — Assign gold labels and pack

Open `annotation_packets_and.csv` and `annotation_packets_ain.csv` in a spreadsheet
editor (Excel, LibreOffice Calc, etc.).

For each row, review the evidence columns and enter one of the following in the
`gold_label` column:

- `match` — the two instances refer to the same real-world entity
- `non-match` — they refer to different entities
- `uncertain` — insufficient evidence to decide (treated as NO MERGE downstream)

**Do not use any LLM to suggest or verify labels.**  Save the file in-place.

Then validate and convert to benchmark parquets:

```bash
python -m bem.benchmark.pack_benchmark_pairs --in_dir data/derived
```

The packer resolves its input files in this priority order:

| Priority | File | When used |
|---|---|---|
| 1 (preferred) | `annotation_packets_{and,ain}.csv` | Default — annotation is done directly in the evidence packets |
| 2 (fallback)  | `annotation_tasks_{and,ain}.csv`   | Used only when packet files are absent |

Pass `--use_packets false` to force the old task-template behaviour.

The packer normalises labels, rejects unlabelled rows, validates IDs against the
instance parquets (unknown IDs → `data/derived/benchmark_id_mismatches.csv` + exit
with error), and writes whatever valid labelled rows exist (e.g., 1 000 dev rows
if only the dev split has been annotated so far):

```
data/derived/benchmark_pairs_and.parquet
data/derived/benchmark_pairs_ain.parquet
```

Run the full pipeline after packing:

```bash
python -m bem --config configs/run_config.yaml
```

C5 loads the benchmark parquets and writes filled-prompt envelopes + stub decisions
to `runs/<run_id>/logs/` (`backend: requests_only`).  Switch to `backend: anthropic_api`
(and set `ANTHROPIC_API_KEY`) when you are ready to make real API calls.

### Optional — Auto-fill clear cases (deterministic)

After building the evidence packets (Step 2 above), you can pre-fill the obvious
`match` / `non-match` rows before opening the spreadsheet:

```bash
python -m bem.benchmark.autofill_gold_labels \
    --and_in  data/derived/annotation_packets_and.csv \
    --ain_in  data/derived/annotation_packets_ain.csv \
    --and_out data/derived/annotation_packets_and.csv \
    --ain_out data/derived/annotation_packets_ain.csv \
    --overwrite false
```

**What this does:**

- Applies conservative, deterministic rules derived *solely* from the evidence
  columns already present in the CSV (author names, co-authors, affiliations,
  affiliation acronyms).
- Fills `gold_label` **only** for cases where the signal is unambiguous:

  | Task | Label filled | Condition |
  |---|---|---|
  | AND | `match` | name_sim ≥ 0.97 **and** ≥ 1 shared co-author *or* affil_sim ≥ 0.95 **and** surname agrees |
  | AND | `non-match` | surname mismatch with name_sim ≤ 0.85, *or* name_sim ≤ 0.70 |
  | AIN | `match` | str_sim ≥ 0.95 **and** acronym overlap *or* token-Jaccard ≥ 0.60 |
  | AIN | `non-match` | str_sim ≤ 0.45 **and** token-Jaccard ≤ 0.20 **and** no acronym overlap |

- **No LLM is involved.** All rules are deterministic functions of the CSV fields.
- **Conservative by design:** the script fills only obvious cases and leaves all
  ambiguous rows blank for human judgement.
- Rows that already have a `gold_label` are **not overwritten** unless you pass
  `--overwrite true`.
- Three audit columns are added to every row: `auto_filled` (true/false),
  `auto_rule` (rule tag), `auto_metrics` (raw signal values).

After reviewing the auto-filled rows, proceed with Step 3 (assign remaining labels
manually, then run `pack_benchmark_pairs`).

---

## Auto-labelled code packets (QC only)

The auto-labeller produces `*_code.csv` files with a machine-assigned label in
`gold_label_auto`.  These are intended for quality-checking and comparison
against manual labels — **they are not gold-standard annotations**.

```bash
python -m bem.benchmark.auto_label_packets \
    --in_dir  data/derived \
    --out_dir data/derived \
    --config  configs/run_config.yaml
```

**Outputs**

| File | Description |
|---|---|
| `data/derived/annotation_packets_and_code.csv` | AND packets with auto label + signals |
| `data/derived/annotation_packets_ain_code.csv` | AIN packets with auto label + signals |
| `data/derived/auto_label_manifest.json` | Thresholds, input/output hashes, label counts |

**Added columns in `*_code.csv`**

| Column | Description |
|---|---|
| `gold_label` | Auto-assigned label (also written to `gold_label_auto`) |
| `gold_label_auto` | Machine label: `match` / `non-match` / `uncertain` |
| `gold_label_source` | Always `"auto"` in these files |
| AND: `name_sim`, `coauthor_overlap_count`, `affil_sim` | Raw signal values |
| AIN: `affil_str_sim`, `token_jaccard`, `acronym_overlap` | Raw signal values |

**Important caveats**

- The original `annotation_packets_and.csv` / `_ain.csv` files are **never
  modified**.
- The rules are conservative: `uncertain` is the default when evidence is
  ambiguous.  Only clear-cut cases receive `match` or `non-match`.
- Auto labels are **not** a substitute for human annotation.  Human labels
  remain authoritative for all downstream evaluation and model training.

---

## Dev2 minimal annotation workflow

This workflow creates a compact, Excel-friendly annotation file for each task
so you can assign gold labels manually — without LLM assistance — using a
simple numeric code, then write the labels back into the full packet files and
build the benchmark parquets.

**Label codes**

| Code | Meaning |
|------|---------|
| `0`  | uncertain |
| `1`  | non-match |
| `2`  | match |

### Step 1 — Generate minimal evidence files

```bash
python -m bem.benchmark.make_min_annotation_files \
    --in_dir data/derived \
    --prefix dev2
```

Produces:
- `data/derived/dev2_min_annotate_and.csv` — AND evidence (author names, co-authors, affiliations, titles, years) + empty `gold_label_code`
- `data/derived/dev2_min_annotate_ain.csv` — AIN evidence (affiliation strings, acronyms, linked authors) + empty `gold_label_code`

### Step 2 — Annotate in Excel

Open each file in Excel and fill the `gold_label_code` column for every row.
The `notes` column is available for free-text remarks.
Save as CSV (keep original filename, do not change format).

### Step 3 — Apply labels and pack parquets

```bash
python -m bem.benchmark.apply_min_labels_to_dev2 \
    --in_dir data/derived \
    --prefix dev2
```

This script:
1. Validates that every row has a valid code (0, 1, or 2).
2. Creates a timestamped backup of the dev2 packet files in
   `data/derived/_backup_dev2_labels_<YYYYMMDD_HHMMSS>/`.
3. Merges the mapped string labels (`uncertain` / `non-match` / `match`)
   back into `dev2_annotation_packets_and.csv` and `_ain.csv`.
4. Calls `pack_benchmark_pairs` automatically, producing:
   - `data/derived/dev2_benchmark_pairs_and.parquet`
   - `data/derived/dev2_benchmark_pairs_ain.parquet`

### Step 4 — Verify

```bash
python -m bem.benchmark.pack_benchmark_pairs \
    --in_dir data/derived \
    --prefix dev2
```

(This is called automatically by step 3; re-run manually if needed.)

### Notes

- `apply_min_labels_to_dev2` will **refuse to run** if any `gold_label_code`
  cell is blank or contains a value other than 0, 1, or 2.
- Dev1 files (`annotation_packets_and.csv` / `_ain.csv`) are **never touched**.
- Re-run step 1 at any time to regenerate the minimal files from the latest
  packet state; existing annotation is in the backup, not overwritten.
