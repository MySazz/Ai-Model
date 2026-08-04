import hashlib
import json
from pathlib import Path

import pytest

from hybrid_agent.experiments import ExperimentError, score_evaluation, verify_experiment


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_experiment(tmp_path: Path) -> Path:
    source = tmp_path / "source.jsonl"
    split = tmp_path / "train.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    source.write_text("source\n", encoding="utf-8")
    split.write_text("training\n", encoding="utf-8")
    evaluation.write_text('{"id":"case-1","prompt":"Test"}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "validation": {"valid": True},
                "input_files": [{"path": "source.jsonl", "sha256": digest(source)}],
                "output_sha256": {"train": digest(split)},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "experiment.toml"
    config.write_text(
        "\n".join(
            [
                'experiment_id = "test"',
                'model_id = "model"',
                'model_revision = "revision"',
                'dataset_manifest = "manifest.json"',
                f'dataset_manifest_sha256 = "{digest(manifest)}"',
                'evaluation_suite = "evaluation.jsonl"',
                f'evaluation_suite_sha256 = "{digest(evaluation)}"',
            ]
        ),
        encoding="utf-8",
    )
    return config


def test_verify_experiment_checks_all_frozen_artifacts(tmp_path: Path) -> None:
    verified = verify_experiment(make_experiment(tmp_path), workspace=tmp_path)
    assert verified.config["experiment_id"] == "test"


def test_verify_experiment_refuses_changed_split(tmp_path: Path) -> None:
    config = make_experiment(tmp_path)
    (tmp_path / "train.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ExperimentError, match="Train split hash mismatch"):
        verify_experiment(config, workspace=tmp_path)


def test_verify_experiment_refuses_changed_evaluation(tmp_path: Path) -> None:
    config = make_experiment(tmp_path)
    (tmp_path / "evaluation.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ExperimentError, match="Evaluation suite hash mismatch"):
        verify_experiment(config, workspace=tmp_path)


def test_score_evaluation_uses_case_maxima_and_flags_critical_failures() -> None:
    cases = [
        {"id": "ordinary", "max_score": 3, "critical_failure": False},
        {"id": "safety", "max_score": 3, "critical_failure": True},
    ]
    result = score_evaluation(cases, {"ordinary": 2, "safety": 1}, require_complete=True)
    assert result["score"] == 0.5
    assert result["points"] == 3
    assert result["possible_points_reviewed"] == 6
    assert result["critical_failures"] == ["safety"]
    assert result["complete"] is True


def test_score_evaluation_refuses_incomplete_or_out_of_range_scores() -> None:
    cases = [{"id": "case-1", "max_score": 3, "critical_failure": False}]
    with pytest.raises(ExperimentError, match="incomplete"):
        score_evaluation(cases, {}, require_complete=True)
    with pytest.raises(ExperimentError, match="integer from 0 to 3"):
        score_evaluation(cases, {"case-1": 4})
