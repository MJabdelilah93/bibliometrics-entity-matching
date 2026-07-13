# Paper artifacts

Frozen artifacts supporting the manuscript *"Governing Uncertainty in Bibliometric Entity
Matching: An Auditable, Abstention-Aware LLM Workflow for Author and Affiliation Resolution"*.
These files document the exact configurations, tuned thresholds, benchmark labels, and run
provenance behind the reported results. They are provided for inspection and reuse; the
pipeline source code in this repository can be rerun on equivalent Scopus exports.

## Contents

| File | Description |
|---|---|
| `run_config.yaml` | Pipeline configuration used for the reported runs (LLM settings, candidate generation, guards and routing). |
| `eval_config.yaml` | Evaluation configuration: baseline definitions (deterministic, fuzzy, TF-IDF, embedding), threshold-tuning stages, bootstrap settings. |
| `baselines/deterministic.py` | Deterministic matcher: AND = exact normalised-name match plus affiliation-prefix agreement; AIN = exact normalised affiliation-string match. |
| `baselines/fuzzy.py` | Weighted lexical matcher (rapidfuzz): AND = 0.60 name token-sort-ratio + 0.20 affiliation-prefix agreement + 0.20 year closeness; AIN = 0.50 token-set-ratio + 0.30 token-sort-ratio + 0.20 acronym Jaccard. |
| `baselines/tfidf.py` | TF-IDF cosine matcher (word 1–2-grams, 20,000 max features, sublinear tf, fitted on the full instance corpus). |
| `baselines/embedding.py` | Embedding cosine matcher (sentence-transformers/all-MiniLM-L6-v2, CPU). |
| `thresholds/tuner.py` | Threshold tuning: precision-floor (primary), F1-optimal, and two-threshold (abstention band) methods over a 0.00–1.00 grid (step 0.01). |
| `benchmark/benchmark_pairs_and.parquet` | Gold-standard AND benchmark: 1,000 pairs. Columns: `anchor_id`, `candidate_id` (SHA-256 instance identifiers), `task`, `gold_label` (match / non-match / uncertain), `split` (dev / test), `stratum`. |
| `benchmark/benchmark_pairs_ain.parquet` | Gold-standard AIN benchmark: 1,000 pairs, same schema. |
| `manifests/thresholds_tuned_dev.json` | Tuned production thresholds with the full dev-split tuning grid. |
| `manifests/candidate_manifest.json` | Candidate-generation parameters and full-corpus candidate counts (949,584 AND; 6,882,645 AIN). Block keys are redacted in this public copy; sizes preserved. |
| `manifests/export_manifest.json` | Input file SHA-256 hashes, row counts, and per-column missingness for the two Scopus export batches. |
| `examples/llm_request_example_and.json` | One verification request envelope (AND), illustrating the evidence card and prompt structure. Field values are synthetic to respect Scopus licensing; the structure is identical to the real run logs. |

## Notes on licensing and identifiers

Raw Scopus exports and record-level intermediates are not redistributed. Benchmark pair
tables use SHA-256 instance identifiers rather than raw field values. Readers with
equivalent Scopus access can rerun the pipeline on their own exports using the documented
query frames (scope, search fields, document types, year ranges) and verify integrity
against the export manifests.
