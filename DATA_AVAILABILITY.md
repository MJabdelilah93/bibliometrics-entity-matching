# Data Availability Statement

## Raw data

The raw data used in this study consist of Scopus UI CSV exports obtained
under an institutional Scopus licence.  Redistribution of Scopus exports is
not permitted under the licence terms.  Accordingly, **no raw data files are
included in this repository**.

Researchers with access to a Scopus institutional licence may reproduce the
dataset by executing the same queries described in the paper (query frame,
date range, field set) and placing the resulting CSV files in `data/raw/`
following the naming convention documented in `README.md`.

## Gold-standard benchmark

The human-annotated benchmark used in the paper comprises **1 000 pairs per
task** (AND and AIN), split into **800 development pairs** and **200 held-out
test pairs** per task.  All labels were assigned by the authors from primary
Scopus evidence without LLM assistance at any step.

The benchmark parquets are not redistributed in this repository because they
contain bibliometric identifiers (author instance IDs, affiliation instance
IDs) that are derived from the Scopus exports.

The annotation methodology — sampling strategy, split assignment, evidence
packet construction, label codes, and quality-control procedures — is fully
described in the paper and in `README.md`.  The public repository documents
the complete benchmark-generation procedure so that researchers with access to
the same Scopus data can reproduce the benchmark from scratch.

## Code

All pipeline code, evaluation scripts, and analysis utilities are provided in
this repository under the MIT licence (see `LICENSE`).

## Reproducibility

A researcher with access to the same Scopus data and an Anthropic API key can
reproduce the full pipeline by following the steps in `README.md`:

1. Place Scopus CSV exports in `data/raw/`.
2. Run `python -m bem --config configs/run_config.yaml` through stages C1–C7.
3. Annotate pairs using the annotation workflow described in `README.md`.
4. Run the evaluation pipeline: `python -m bem.eval --config configs/eval_config.yaml`.

All hyperparameters, threshold values, model IDs, and random seeds are
recorded in the run manifests written to `runs/<run_id>/manifests/`.

## Contact

For questions about data access or reproducibility, contact the corresponding
author via the repository issue tracker.
