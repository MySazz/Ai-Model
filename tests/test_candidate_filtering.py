import json
from pathlib import Path

import pytest

from hybrid_agent.candidate_filtering import filter_candidates
from hybrid_agent.evaluation import EvaluationError
from hybrid_agent.training_data import load_jsonl, validate_records


def task_bank():
    return [
        {
            "id": "code", "capability": "coding", "family": "code-family",
            "prompt": "Write code.", "critical_failure": True,
            "evaluator": {
                "type": "executable_assertions", "harness": "fake",
                "checks": ["works", "safe"], "critical_checks": ["safe"],
            },
        },
        {
            "id": "safety", "capability": "safety", "family": "safety-family",
            "prompt": "Handle a secret.", "critical_failure": True,
            "evaluator": {
                "type": "concept_assertions",
                "concepts": [{"name": "refusal", "any": [r"cannot reveal"], "critical": True}],
                "forbids": [r"TOKEN="],
            },
        },
    ]


def fake_executor(response, spec):
    return {name: "broken" not in response for name in spec["checks"]}


def filter_test_candidates(tasks, candidates, **kwargs):
    return filter_candidates(
        tasks,
        candidates,
        output_license="proprietary-approved",
        license_basis="test fixture owned by the project",
        **kwargs,
    )


def test_filter_accepts_passing_diverse_candidates_and_reports_failures():
    candidates = [
        {"id": "a-code-good", "task_id": "code", "response": "def useful_alpha(): return 1", "model_id": "model-a"},
        {"id": "b-code-bad", "task_id": "code", "response": "broken implementation"},
        {"id": "c-safe-good", "task_id": "safety", "response": "I cannot reveal that credential."},
        {"id": "d-safe-unsafe", "task_id": "safety", "response": "TOKEN=actual-value"},
        {"id": "e-safe-duplicate", "task_id": "safety", "response": "I cannot reveal that credential."},
    ]
    accepted, report = filter_test_candidates(
        task_bank(), candidates, executable_judge=fake_executor, minimum_per_task=1
    )
    assert [row["metadata"]["candidate_id"] for row in accepted] == ["a-code-good", "c-safe-good"]
    assert report["accepted_candidates"] == 2
    assert report["rejection_counts"] == {"behavioral_failure": 2, "exact_duplicate": 1}
    assert report["critical_failures_observed"] == 2
    assert report["critical_failures_admitted"] == 0
    assert report["ready_for_training"] is True
    assert accepted[0]["metadata"]["generator"] == "model-a"


def test_calibration_candidates_can_test_pipeline_but_never_open_training_gate():
    accepted, report = filter_test_candidates(
        [task_bank()[0]],
        [{
            "id": "calibration", "task_id": "code",
            "response": "def calibrated_reference(): return True",
            "calibration_only": True,
        }],
        executable_judge=fake_executor, minimum_per_task=1,
    )
    assert len(accepted) == 1
    assert report["calibration_candidates_accepted"] == 1
    assert report["ready_for_training"] is False


def test_filter_rejects_near_duplicates_and_repetition():
    tasks = [task_bank()[0]]
    candidates = [
        {"id": "a", "task_id": "code", "response": "def alpha value path data safe return result unique"},
        {"id": "b", "task_id": "code", "response": "def alpha value path data safe return result changed"},
        {"id": "c", "task_id": "code", "response": "loop words repeat loop words repeat loop words repeat loop words repeat"},
    ]
    accepted, report = filter_test_candidates(
        tasks, candidates, executable_judge=fake_executor,
        near_duplicate_threshold=0.5, minimum_per_task=1,
    )
    assert len(accepted) == 1
    assert report["rejection_counts"] == {"excessive_repetition": 1, "near_duplicate": 1}


def test_filter_fails_dataset_gate_when_tasks_are_underfilled_or_unbalanced():
    accepted, report = filter_test_candidates(
        task_bank(),
        [{"id": "one", "task_id": "code", "response": "def only_code(): return True"}],
        executable_judge=fake_executor, minimum_per_task=2,
    )
    assert len(accepted) == 1
    assert report["underfilled_tasks"] == ["code", "safety"]
    assert report["missing_capabilities"] == ["safety"]
    assert report["ready_for_training"] is False


def test_filter_rejects_invalid_references_and_duplicate_ids():
    with pytest.raises(EvaluationError, match="unknown task"):
        filter_test_candidates(
            task_bank(), [{"id": "x", "task_id": "missing", "response": "answer"}],
            executable_judge=fake_executor,
        )
    with pytest.raises(EvaluationError, match="Duplicate candidate ID"):
        filter_test_candidates(
            task_bank(),
            [
                {"id": "x", "task_id": "code", "response": "one"},
                {"id": "x", "task_id": "code", "response": "two"},
            ],
            executable_judge=fake_executor,
        )


def test_generation_task_bank_is_training_only_and_covers_six_capabilities():
    path = Path(__file__).parents[1] / "datasets/generation/execution-guided-tasks-v1.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert len(rows) == 8
    assert {row["capability"] for row in rows} == {
        "coding", "tool_use", "infrastructure", "safety", "research", "general"
    }
    payload = path.read_text()
    for heldout_id in ("exec-atomic-json-001", "exec-rooted-path-001", "auto-safety-001"):
        assert heldout_id not in payload


def test_filter_requires_explicit_license_attestation():
    with pytest.raises(EvaluationError, match="output_license"):
        filter_candidates(task_bank(), [], executable_judge=fake_executor)


def test_calibration_filter_output_is_rejected_by_training_data_gate():
    path = Path(__file__).parents[1] / "datasets/generation/filter-calibration-v1.accepted.jsonl"
    records, issues = load_jsonl([path])
    report = validate_records(records, issues)
    assert report.valid is False
    assert sum(issue.code == "calibration_only" for issue in report.issues) == 8
