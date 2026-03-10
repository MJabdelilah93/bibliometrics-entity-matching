"""config.py — Load and validate the evaluation configuration.

Usage
-----
    from bem.eval.config import load_eval_config
    cfg = load_eval_config("configs/eval_config.yaml")

Validation rules
----------------
- benchmark.and and benchmark.ain MUST contain the substring "dev2" unless
  cfg.allow_non_dev2 is True.  Violating this raises ConfigError (fail fast).
- All mandatory input paths are resolved relative to the project root inferred
  from the config file location (parent of configs/).
- output.dir is created if it does not exist.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Sentinel that must appear in benchmark file names unless explicitly overridden.
_DEV2_SENTINEL = "dev2"


class ConfigError(RuntimeError):
    """Raised when the evaluation config is invalid or unsafe."""


# ---------------------------------------------------------------------------
# Nested config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BaselineConfig:
    """Settings for Stage E3 classical baselines."""

    output_dir: Path

    # Which baselines to run
    run_deterministic: bool
    run_fuzzy: bool
    run_tfidf: bool
    run_embedding: bool

    # TF-IDF settings
    tfidf_max_features: int
    tfidf_ngram_range: tuple[int, int]
    tfidf_sublinear_tf: bool
    tfidf_fit_corpus: str  # 'all_instances' | 'eval_pairs_only'

    # Embedding settings
    embedding_model: str
    embedding_batch_size: int
    embedding_device: str
    embedding_require_confirmation: bool

    # Auxiliary Scopus-ID comparator
    aux_scopus_id_in_master: bool
    aux_scopus_id_source: Path


@dataclass
class TuningConfig:
    """Settings for Stage E4 threshold tuning."""

    output_dir: Path

    # Precision floors — separate for match and non-match classes, per task
    precision_floor_and: float
    precision_floor_ain: float
    precision_floor_nonmatch_and: float
    precision_floor_nonmatch_ain: float

    # F-beta metric (1.0 = standard F1)
    f1_beta: float

    # Threshold grid
    grid_start: float
    grid_stop: float
    grid_step: float

    # Two-threshold uncertain-band tuning
    two_threshold_enabled: bool


@dataclass
class AblationConfig:
    """Settings for Stage E6 ablation experiments."""

    output_dir: Path

    # Which ablations to run
    run_a1_single_prompt:      bool
    run_a2_no_guards:          bool
    run_a3_k_sensitivity:      bool
    run_a4_missing_fields_and: bool
    run_a5_missing_fields_ain: bool
    run_a6_threshold_sweep:    bool

    # A1: single-prompt LLM
    a1_model:                  str
    a1_max_pairs_per_task:     int
    a1_require_confirmation:   bool

    # A3: K sensitivity
    a3_k_values:               list[int]

    # A4: AND missing fields
    a4_remove_coauthor:        bool
    a4_remove_affiliation:     bool
    a4_max_pairs:              int
    a4_require_confirmation:   bool

    # A5: AIN missing fields
    a5_raw_affil_only:         bool
    a5_remove_author_link:     bool
    a5_max_pairs:              int
    a5_require_confirmation:   bool

    # A6: threshold sweep
    a6_t_match_values:         list[float]
    a6_m_signals_values:       list[int]
    a6_t_nonmatch_fixed:       float


@dataclass
class RobustnessEfficiencyConfig:
    """Settings for Stage E7 robustness and efficiency summary."""

    robustness_output_dir: Path
    efficiency_output_dir: Path

    # Pricing assumptions — snapshot- and environment-dependent; document explicitly.
    pricing_input_per_million: float   # USD per 1M input tokens
    pricing_output_per_million: float  # USD per 1M output tokens
    pricing_model_note: str            # free-text: model name + pricing date

    # 'auto' = pick highest F1_M non-LLM system from E5 metrics_master.parquet;
    # or a specific display_name string (e.g. 'Tfidf').
    strongest_baseline: str

    # Q1/Q2 slicing requires a join from anchor_id to instance tables.
    compute_q1_q2: bool
    instances_and: Path
    instances_ain: Path


@dataclass
class EvaluationConfig:
    """Settings for Stage E5 held-out evaluation."""

    output_dir: Path
    manuscript_dir: Path

    # Which threshold methods to evaluate (e.g. 'precision_floor_match', 'f1_optimal')
    methods: list[str]

    # System inclusion flags
    include_bem: bool
    include_aux_scopus_id: bool

    # Bootstrap CI
    bootstrap_n_samples: int
    bootstrap_alpha: float

    # Figure output
    figure_dpi: int
    figure_format: str  # 'png' | 'pdf' | 'svg'


# ---------------------------------------------------------------------------
# Top-level EvalConfig
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Parsed and validated evaluation configuration."""

    # --- benchmark inputs ---
    benchmark_and: Path
    benchmark_ain: Path
    allow_non_dev2: bool

    # --- BEM routing log inputs ---
    routing_and_bm: Path
    routing_ain_bm: Path
    bem_run_dir: Path  # informational; used for manifest provenance

    # --- threshold manifest ---
    thresholds_manifest: Path

    # --- outputs ---
    output_dir: Path

    # --- eval_inputs stage ---
    eval_inputs_output_dir: Path
    candidates_and: Path
    produce_scopus_id_comparator: bool

    # --- baselines stage ---
    baselines: BaselineConfig

    # --- tuning stage ---
    tuning: TuningConfig

    # --- evaluation stage ---
    evaluation: EvaluationConfig

    # --- ablation stage ---
    ablation: AblationConfig

    # --- robustness & efficiency stage ---
    rob_eff: RobustnessEfficiencyConfig

    # --- run control ---
    tasks: list[str]
    random_seed: int
    dry_run: bool
    force: bool

    # --- provenance ---
    config_path: Path = field(repr=False)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_eval_config(path: str | Path) -> EvalConfig:
    """Load *path* as YAML, validate, and return an :class:`EvalConfig`.

    Args:
        path: Path to ``eval_config.yaml``.

    Returns:
        Populated :class:`EvalConfig` with all paths resolved and validated.

    Raises:
        ConfigError: If a mandatory field is missing or a safety check fails.
        FileNotFoundError: If *path* does not exist.
    """
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Eval config not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    # Infer project root: parent of configs/, or parent of config file if placed elsewhere.
    project_root = config_path.parent
    if project_root.name == "configs":
        project_root = project_root.parent

    def _resolve(section: str, key: str) -> Path:
        try:
            val = raw[section][key]
        except KeyError:
            raise ConfigError(f"Missing required config key: {section}.{key}")
        return (project_root / val).resolve()

    def _get(section: str, key: str, default: Any = None) -> Any:
        """Get raw[section][key] where section is a mapping."""
        sec = raw.get(section)
        if not isinstance(sec, dict):
            return default
        return sec.get(key, default)

    def _get2(section: str, subsection: str, key: str, default: Any = None) -> Any:
        """Get raw[section][subsection][key]."""
        sec = raw.get(section, {})
        if not isinstance(sec, dict):
            return default
        sub = sec.get(subsection, {})
        if not isinstance(sub, dict):
            return default
        return sub.get(key, default)

    def _top(key: str, default: Any = None) -> Any:
        """Get a top-level key directly from raw."""
        return raw.get(key, default)

    # --- benchmark ---
    bm_and = _resolve("benchmark", "and")
    bm_ain = _resolve("benchmark", "ain")
    allow_non_dev2: bool = bool(_get("benchmark", "allow_non_dev2", False))

    _check_dev2(bm_and, allow_non_dev2, "benchmark.and")
    _check_dev2(bm_ain, allow_non_dev2, "benchmark.ain")

    # --- routing logs ---
    rt_and = _resolve("routing", "and_bm_log")
    rt_ain = _resolve("routing", "ain_bm_log")
    run_dir_raw = _get("routing", "bem_run_dir", "")
    run_dir = (project_root / run_dir_raw).resolve()

    # --- thresholds ---
    thresh = _resolve("thresholds", "manifest")

    # --- output ---
    out_dir = (project_root / _get("output", "dir", "outputs/eval")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- eval_inputs ---
    ei_out_dir = (
        project_root / _get("eval_inputs", "output_dir", "outputs/eval_inputs")
    ).resolve()
    candidates_and_path = (
        project_root / _get("eval_inputs", "candidates_and", "data/interim/candidates_and.parquet")
    ).resolve()
    produce_scopus = bool(_get("eval_inputs", "produce_scopus_id_comparator", True))

    # --- baselines ---
    bl_out_dir = (
        project_root / _get("baselines", "output_dir", "outputs/baselines")
    ).resolve()

    ngram_raw = _get2("baselines", "tfidf", "ngram_range", [1, 2])
    ngram: tuple[int, int] = (int(ngram_raw[0]), int(ngram_raw[1]))

    aux_src_raw = _get2("baselines", "aux_scopus_id", "source",
                         "outputs/eval_inputs/aux_scopus_id_comparator_and.parquet")
    aux_src = (project_root / aux_src_raw).resolve()

    baselines = BaselineConfig(
        output_dir=bl_out_dir,
        run_deterministic=bool(_get2("baselines", "run", "deterministic", True)),
        run_fuzzy=bool(_get2("baselines", "run", "fuzzy", True)),
        run_tfidf=bool(_get2("baselines", "run", "tfidf", True)),
        run_embedding=bool(_get2("baselines", "run", "embedding", False)),
        tfidf_max_features=int(_get2("baselines", "tfidf", "max_features", 20000)),
        tfidf_ngram_range=ngram,
        tfidf_sublinear_tf=bool(_get2("baselines", "tfidf", "sublinear_tf", True)),
        tfidf_fit_corpus=str(_get2("baselines", "tfidf", "fit_corpus", "all_instances")),
        embedding_model=str(_get2("baselines", "embedding", "model",
                                   "sentence-transformers/all-MiniLM-L6-v2")),
        embedding_batch_size=int(_get2("baselines", "embedding", "batch_size", 64)),
        embedding_device=str(_get2("baselines", "embedding", "device", "cpu")),
        embedding_require_confirmation=bool(
            _get2("baselines", "embedding", "require_confirmation", True)
        ),
        aux_scopus_id_in_master=bool(
            _get2("baselines", "aux_scopus_id", "include_in_master", True)
        ),
        aux_scopus_id_source=aux_src,
    )

    # --- tuning ---
    tu_out_dir = (
        project_root / _get("tuning", "output_dir", "outputs/thresholds")
    ).resolve()

    def _floor(subsection: str, task: str, default: float) -> float:
        return float(_get2("tuning", subsection, task, default))

    grid_raw = raw.get("tuning", {}).get("threshold_grid", {}) or {}
    two_raw  = raw.get("tuning", {}).get("two_threshold", {}) or {}

    tuning = TuningConfig(
        output_dir=tu_out_dir,
        precision_floor_and=_floor("precision_floor", "and", 0.90),
        precision_floor_ain=_floor("precision_floor", "ain", 0.90),
        precision_floor_nonmatch_and=_floor("precision_floor_nonmatch", "and", 0.90),
        precision_floor_nonmatch_ain=_floor("precision_floor_nonmatch", "ain", 0.90),
        f1_beta=float(_get("tuning", "f1_beta", 1.0)),
        grid_start=float(grid_raw.get("start", 0.00)),
        grid_stop=float(grid_raw.get("stop",  1.00)),
        grid_step=float(grid_raw.get("step",  0.01)),
        two_threshold_enabled=bool(two_raw.get("enabled", True)),
    )

    # --- evaluation ---
    ev_out_dir  = (project_root / _get("evaluation", "output_dir",    "outputs/evaluation")).resolve()
    ev_ms_dir   = (project_root / _get("evaluation", "manuscript_dir", "outputs/manuscript")).resolve()
    ev_methods_raw  = raw.get("evaluation", {}).get("methods") or ["precision_floor_match", "f1_optimal"]
    ev_bootstrap    = raw.get("evaluation", {}).get("bootstrap", {}) or {}
    ev_figure       = raw.get("evaluation", {}).get("figure", {}) or {}

    evaluation = EvaluationConfig(
        output_dir=ev_out_dir,
        manuscript_dir=ev_ms_dir,
        methods=[str(m) for m in ev_methods_raw],
        include_bem=bool(_get("evaluation", "include_bem", True)),
        include_aux_scopus_id=bool(_get("evaluation", "include_aux_scopus_id", True)),
        bootstrap_n_samples=int(ev_bootstrap.get("n_samples", 1000)),
        bootstrap_alpha=float(ev_bootstrap.get("alpha", 0.05)),
        figure_dpi=int(ev_figure.get("dpi", 300)),
        figure_format=str(ev_figure.get("format", "png")),
    )

    # --- ablation ---
    abl_out_dir = (
        project_root / _get("ablations", "output_dir", "outputs/ablations")
    ).resolve()
    _abl_run = raw.get("ablations", {}).get("run", {}) or {}
    _a1_raw  = raw.get("ablations", {}).get("a1_single_prompt", {}) or {}
    _a3_raw  = raw.get("ablations", {}).get("a3_k_sensitivity", {}) or {}
    _a4_raw  = raw.get("ablations", {}).get("a4_missing_fields_and", {}) or {}
    _a5_raw  = raw.get("ablations", {}).get("a5_missing_fields_ain", {}) or {}
    _a6_raw  = raw.get("ablations", {}).get("a6_threshold_sweep", {}) or {}

    _a6_t_match_raw = _a6_raw.get("t_match_values", [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    _a6_m_raw       = _a6_raw.get("m_signals_values", [0, 1, 2, 3])
    _a3_k_raw       = _a3_raw.get("k_values", [10, 25, 50])

    ablation = AblationConfig(
        output_dir=abl_out_dir,
        run_a1_single_prompt      =bool(_abl_run.get("a1_single_prompt",      False)),
        run_a2_no_guards          =bool(_abl_run.get("a2_no_guards",          True)),
        run_a3_k_sensitivity      =bool(_abl_run.get("a3_k_sensitivity",      True)),
        run_a4_missing_fields_and =bool(_abl_run.get("a4_missing_fields_and", False)),
        run_a5_missing_fields_ain =bool(_abl_run.get("a5_missing_fields_ain", False)),
        run_a6_threshold_sweep    =bool(_abl_run.get("a6_threshold_sweep",    True)),
        a1_model                  =str(_a1_raw.get("model", "claude-opus-4-6")),
        a1_max_pairs_per_task     =int(_a1_raw.get("max_pairs_per_task", 200)),
        a1_require_confirmation   =bool(_a1_raw.get("require_confirmation", True)),
        a3_k_values               =[int(k) for k in _a3_k_raw],
        a4_remove_coauthor        =bool(_a4_raw.get("remove_coauthor", True)),
        a4_remove_affiliation     =bool(_a4_raw.get("remove_affiliation", True)),
        a4_max_pairs              =int(_a4_raw.get("max_pairs", 300)),
        a4_require_confirmation   =bool(_a4_raw.get("require_confirmation", True)),
        a5_raw_affil_only         =bool(_a5_raw.get("raw_affil_only", True)),
        a5_remove_author_link     =bool(_a5_raw.get("remove_author_link", True)),
        a5_max_pairs              =int(_a5_raw.get("max_pairs", 300)),
        a5_require_confirmation   =bool(_a5_raw.get("require_confirmation", True)),
        a6_t_match_values         =[float(t) for t in _a6_t_match_raw],
        a6_m_signals_values       =[int(m) for m in _a6_m_raw],
        a6_t_nonmatch_fixed       =float(_a6_raw.get("t_nonmatch_fixed", 0.85)),
    )

    # --- robustness & efficiency ---
    _re_raw     = raw.get("robustness_efficiency", {}) or {}
    _re_pricing = _re_raw.get("pricing", {}) or {}

    rob_eff = RobustnessEfficiencyConfig(
        robustness_output_dir=(
            project_root / _re_raw.get("robustness_output_dir", "outputs/robustness")
        ).resolve(),
        efficiency_output_dir=(
            project_root / _re_raw.get("efficiency_output_dir", "outputs/efficiency")
        ).resolve(),
        pricing_input_per_million=float(
            _re_pricing.get("input_cost_per_million", 3.00)
        ),
        pricing_output_per_million=float(
            _re_pricing.get("output_cost_per_million", 15.00)
        ),
        pricing_model_note=str(
            _re_pricing.get("model_note",
                             "claude-sonnet-4-6 (pricing snapshot 2025-08; verify before citing)")
        ),
        strongest_baseline=str(_re_raw.get("strongest_baseline", "auto")),
        compute_q1_q2=bool(_re_raw.get("compute_q1_q2", True)),
        instances_and=(
            project_root / _re_raw.get("instances_and", "data/interim/author_instances.parquet")
        ).resolve(),
        instances_ain=(
            project_root / _re_raw.get("instances_ain", "data/interim/affil_instances.parquet")
        ).resolve(),
    )

    # --- run control (top-level keys) ---
    tasks_raw = _top("tasks") or ["and", "ain"]
    tasks = [str(t).lower() for t in tasks_raw]
    bad_tasks = [t for t in tasks if t not in ("and", "ain")]
    if bad_tasks:
        raise ConfigError(
            f"Unknown task(s) in config: {bad_tasks}. Must be 'and' and/or 'ain'."
        )

    seed: int = int(_top("random_seed") or 42)
    _dry = _top("dry_run")
    dry_run: bool = bool(_dry) if _dry is not None else True
    _force = _top("force")
    force: bool = bool(_force) if _force is not None else False

    return EvalConfig(
        benchmark_and=bm_and,
        benchmark_ain=bm_ain,
        allow_non_dev2=allow_non_dev2,
        routing_and_bm=rt_and,
        routing_ain_bm=rt_ain,
        bem_run_dir=run_dir,
        thresholds_manifest=thresh,
        output_dir=out_dir,
        eval_inputs_output_dir=ei_out_dir,
        candidates_and=candidates_and_path,
        produce_scopus_id_comparator=produce_scopus,
        baselines=baselines,
        tuning=tuning,
        evaluation=evaluation,
        ablation=ablation,
        rob_eff=rob_eff,
        tasks=tasks,
        random_seed=seed,
        dry_run=dry_run,
        force=force,
        config_path=config_path,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_dev2(path: Path, allow_non_dev2: bool, field_name: str) -> None:
    """Warn or raise if *path* does not contain the dev2 sentinel."""
    if _DEV2_SENTINEL not in path.name:
        msg = (
            f"[SAFETY] {field_name} does not contain '{_DEV2_SENTINEL}' in its filename: "
            f"{path.name!r}. "
            "Non-dev2 benchmark files are legacy/superseded. "
            "Set benchmark.allow_non_dev2: true to suppress this error."
        )
        if allow_non_dev2:
            warnings.warn(msg, stacklevel=3)
        else:
            raise ConfigError(msg)
