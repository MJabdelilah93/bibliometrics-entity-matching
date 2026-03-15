"""manifest.py — Write evaluation manifests.

Every stage writes a JSON manifest into::

    outputs/eval/<eval_run_id>/manifests/

Manifests record inputs (paths + SHA-256), parameters, and output paths so
results are fully reproducible and auditable.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    """Return hex SHA-256 of *path* if it exists, else 'missing'."""
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _env_info() -> dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
    }


# ---------------------------------------------------------------------------
# Eval-run manifest (written once at startup, after E0)
# ---------------------------------------------------------------------------

def write_eval_manifest(
    eval_run_dir: Path,
    config_path: Path,
    artefact_entries: list[Any],  # list[ArtefactEntry]
    dry_run: bool,
    tasks: list[str],
    random_seed: int,
) -> Path:
    """Write ``eval_manifest.json`` for the full evaluation run.

    Args:
        eval_run_dir: Root output directory for this evaluation run.
        config_path:  Path to the eval_config.yaml that was loaded.
        artefact_entries: List of ArtefactEntry objects from io_check.
        dry_run: Whether this is a dry-run execution.
        tasks: Task list from config.
        random_seed: Random seed from config.

    Returns:
        Path to the written manifest file.
    """
    manifests_dir = eval_run_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    inputs = []
    for e in artefact_entries:
        inputs.append({
            "label": e.label,
            "path": str(e.path),
            "required": e.required,
            "present": e.present,
            "size_bytes": e.size_bytes,
            "sha256": _sha256(e.path) if e.present else "missing",
        })

    manifest: dict[str, Any] = {
        "eval_run_dir": str(eval_run_dir),
        "timestamp_iso": _now_iso(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "dry_run": dry_run,
        "tasks": tasks,
        "random_seed": random_seed,
        "inputs": inputs,
        "environment": _env_info(),
    }

    out_path = manifests_dir / "eval_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Per-stage manifest
# ---------------------------------------------------------------------------

def write_stage_manifest(
    eval_run_dir: Path,
    stage_id: str,
    params: dict[str, Any],
    inputs: dict[str, Path],
    outputs: dict[str, Path],
    status: str = "completed",
) -> Path:
    """Write a stage-level manifest JSON file.

    Args:
        eval_run_dir: Root output directory for this evaluation run.
        stage_id:     Short identifier, e.g. ``"E1_prediction_join"``.
        params:       Stage parameters (thresholds, seeds, etc.).
        inputs:       Mapping of label → input path.
        outputs:      Mapping of label → output path.

    Returns:
        Path to the written manifest file.
    """
    manifests_dir = eval_run_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "stage_id": stage_id,
        "timestamp_iso": _now_iso(),
        "status": status,
        "params": params,
        "inputs": {
            label: {
                "path": str(p),
                "sha256": _sha256(p),
                "size_bytes": p.stat().st_size if p.exists() else 0,
            }
            for label, p in inputs.items()
        },
        "outputs": {
            label: {
                "path": str(p),
                "sha256": _sha256(p),
                "size_bytes": p.stat().st_size if p.exists() else 0,
            }
            for label, p in outputs.items()
        },
    }

    fname = f"stage_{stage_id}_manifest.json"
    out_path = manifests_dir / fname
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path
