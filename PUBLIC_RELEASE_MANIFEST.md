# Public Release Manifest

This document records what is and is not included in this public release of
the BEM (Bibliometrics Entity Matching) codebase.

## Included

| Path | Description |
|---|---|
| `src/bem/` | Full Python source for the BEM pipeline (C1–C7) and evaluation pipeline (E1–E8) |
| `configs/` | Example configuration files (`run_config.yaml`, `eval_config.yaml`, `normalisation_rules.yaml`) |
| `tests/` | Unit and integration tests |
| `scripts/run_smoke.ps1` | PowerShell smoke-test helper |
| `docs/` | Additional documentation |
| `requirements.txt` | Python dependency list |
| `pyproject.toml` | Package build configuration |
| `LICENSE` | MIT licence |
| `CITATION.cff` | Citation metadata |
| `README.md` | Setup and usage guide |
| `DATA_AVAILABILITY.md` | Data availability statement |

## Excluded

| Category | Reason |
|---|---|
| `data/` | Contains Scopus UI exports and derived parquets; redistribution not permitted under Scopus licence |
| `runs/` | Pipeline run outputs (large binary artefacts; reproducible from code + data) |
| `outputs/` | Evaluation outputs (reproducible from code + data + benchmark) |
| `.env` / `*.env` | API keys and secrets |
| `.venv/` | Virtual environment |
| `__pycache__/` | Compiled Python bytecode |
| Gold-standard benchmark parquets | Derived from Scopus exports; not redistributable |

## Import namespace

All Python imports use the `bem` namespace (`from bem.eval.evaluation.metrics import ...`).
The internal development namespace `vs2` is not present in this release.

## Evaluation pipeline stages

| Stage | Module | Description |
|---|---|---|
| E1 | `bem.eval.config` | Configuration and path resolution |
| E2 | `bem.eval.baselines` | Classical non-LLM baselines (deterministic, fuzzy, TF-IDF, embedding) |
| E3 | `bem.eval.tuning` | Threshold tuning (f1_optimal, precision_floor_match, two_threshold) |
| E4 | `bem.eval.evaluation` | Main evaluation metrics with bootstrap 95% CI |
| E5 | `bem.eval.evaluation` | Manuscript-ready tables and figures |
| E6 | `bem.eval.ablations` | Ablation studies (A1–A6) |
| E7 | `bem.eval.robustness`, `bem.eval.efficiency` | Robustness slices and efficiency accounting |
| E8 | `bem.eval.export` | Final manuscript export and smoke checks |

## Release date

2026-03-10
