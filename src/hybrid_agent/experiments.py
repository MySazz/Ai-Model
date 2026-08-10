"""Integrity gates and run metadata for reproducible model experiments."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot safely or reproducibly start."""


@dataclass(frozen=True)
class VerifiedExperiment:
    config_path: Path
    config: dict[str, Any]
    manifest_path: Path
    evaluation_path: Path
    manifest_sha256: str
    evaluation_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_experiment(config_path: Path, *, workspace: Path) -> VerifiedExperiment:
    workspace = workspace.resolve()
    config_path = _inside(config_path, workspace, "configuration")
    try:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ExperimentError(f"Could not read experiment configuration: {exc}") from exc

    manifest_path = _configured_path(config, "dataset_manifest", workspace)
    evaluation_path = _configured_path(config, "evaluation_suite", workspace)
    manifest_hash = _require_hash(config, "dataset_manifest_sha256")
    evaluation_hash = _require_hash(config, "evaluation_suite_sha256")
    _match_hash(manifest_path, manifest_hash, "dataset manifest")
    _match_hash(evaluation_path, evaluation_hash, "evaluation suite")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Could not read dataset manifest: {exc}") from exc
    if manifest.get("validation", {}).get("valid") is not True:
        raise ExperimentError("Dataset manifest does not record successful validation.")
    for split, expected in manifest.get("output_sha256", {}).items():
        _match_hash(manifest_path.parent / f"{split}.jsonl", expected, f"{split} split")
    for source in manifest.get("input_files", []):
        source_path = _inside(workspace / source["path"], workspace, "dataset source")
        _match_hash(source_path, source["sha256"], "dataset source")
    return VerifiedExperiment(
        config_path, config, manifest_path, evaluation_path, manifest_hash, evaluation_hash
    )


def write_blocked_baseline(
    verified: VerifiedExperiment, output: Path, *, reason: str, evidence: list[str]
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "experiment_id": verified.config["experiment_id"],
        "kind": "base-model-baseline",
        "status": "blocked",
        "recorded_at": datetime.now(UTC).isoformat(),
        "model_id": verified.config["model_id"],
        "model_revision": verified.config["model_revision"],
        "dataset_manifest_sha256": verified.manifest_sha256,
        "evaluation_suite_sha256": verified.evaluation_sha256,
        "reason": reason,
        "evidence": evidence,
        "cases_scored": 0,
        "score": None,
    }
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExperimentError(f"Failed to create baseline output directory: {exc}") from exc
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def score_evaluation(
    cases: list[dict[str, Any]], scores: dict[str, int | None], *, require_complete: bool = False
) -> dict[str, Any]:
    """Validate human scores and summarize a frozen evaluation suite."""
    case_ids = {case.get("id") for case in cases}
    if not all(isinstance(case_id, str) and case_id for case_id in case_ids):
        raise ExperimentError("Every evaluation case must have a non-empty string ID.")
    if len(case_ids) != len(cases):
        raise ExperimentError("Evaluation case IDs must be unique.")
    unknown = sorted(set(scores) - case_ids)
    if unknown:
        raise ExperimentError(f"Scores contain unknown evaluation IDs: {', '.join(unknown)}")

    results: list[dict[str, Any]] = []
    points = 0
    possible = 0
    critical_failures: list[str] = []
    for case in cases:
        case_id = case["id"]
        maximum = case.get("max_score")
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
            raise ExperimentError(f"Evaluation case {case_id} has an invalid max_score.")
        value = scores.get(case_id)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= maximum
        ):
            raise ExperimentError(f"Score for {case_id} must be an integer from 0 to {maximum}.")
        if value is not None:
            points += value
            possible += maximum
            if case.get("critical_failure") is True and value < maximum:
                critical_failures.append(case_id)
        results.append({"id": case_id, "score": value, "max_score": maximum})

    reviewed = sum(result["score"] is not None for result in results)
    if require_complete and reviewed != len(results):
        raise ExperimentError(
            f"Evaluation scoring is incomplete: reviewed {reviewed} of {len(results)} cases."
        )
    return {
        "cases": results,
        "reviewed": reviewed,
        "total": len(results),
        "points": points,
        "possible_points_reviewed": possible,
        "score": points / possible if possible else None,
        "critical_failures": critical_failures,
        "complete": reviewed == len(results),
    }


def _configured_path(config: dict[str, Any], key: str, workspace: Path) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ExperimentError(f"Missing configuration field: {key}")
    return _inside(workspace / value, workspace, key)


def _inside(path: Path, workspace: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved != workspace and workspace not in resolved.parents:
        raise ExperimentError(f"{label.capitalize()} escapes workspace: {path}")
    if not resolved.is_file():
        raise ExperimentError(f"Missing {label}: {resolved}")
    return resolved


def _require_hash(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ExperimentError(f"Missing or invalid frozen hash: {key}")
    return value


def _match_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ExperimentError(f"Missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ExperimentError(
            f"{label.capitalize()} hash mismatch: expected {expected}, observed {observed}"
        )
