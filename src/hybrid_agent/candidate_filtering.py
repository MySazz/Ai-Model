"""Execution-guided filtering and quality gates for generated SFT candidates."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable

from .evaluation import EvaluationError, evaluate_responses


ExecutableJudge = Callable[[str, dict[str, Any]], dict[str, bool]]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_./-]+", text.casefold())


def _fingerprint(text: str) -> str:
    return " ".join(_tokens(text))


def _shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    tokens = _tokens(text)
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}


def _similarity(left: str, right: str) -> float:
    left_set, right_set = _shingles(left), _shingles(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _repeats_excessively(text: str, maximum: int = 3) -> bool:
    tokens = _tokens(text)
    counts = Counter(tuple(tokens[index:index + 3]) for index in range(len(tokens) - 2))
    return bool(counts and max(counts.values()) > maximum)


def filter_candidates(
    tasks: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    executable_judge: ExecutableJudge | None = None,
    near_duplicate_threshold: float = 0.82,
    minimum_per_task: int = 2,
    required_capabilities: set[str] | None = None,
    maximum_capability_ratio: float = 2.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate, behaviorally score, deduplicate, and gate generated candidates."""
    if not 0 <= near_duplicate_threshold <= 1:
        raise EvaluationError("near_duplicate_threshold must be between zero and one.")
    if minimum_per_task <= 0:
        raise EvaluationError("minimum_per_task must be positive.")
    indexed_tasks: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise EvaluationError("Every filtering task requires a non-empty string ID.")
        if task_id in indexed_tasks:
            raise EvaluationError(f"Duplicate filtering task ID: {task_id}")
        if not all(isinstance(task.get(key), str) and task[key] for key in (
            "capability", "family", "prompt"
        )) or not isinstance(task.get("evaluator"), dict):
            raise EvaluationError(f"Task {task_id} has invalid metadata or evaluator.")
        indexed_tasks[task_id] = task

    seen_candidate_ids: set[str] = set()
    accepted: list[dict[str, Any]] = []
    accepted_texts: list[str] = []
    accepted_fingerprints: set[str] = set()
    accepted_shingles_list: list[set[tuple[str, ...]]] = []
    decisions: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    critical_failures = 0
    input_by_capability: Counter[str] = Counter()
    accepted_by_capability: Counter[str] = Counter()
    accepted_by_family: Counter[str] = Counter()
    accepted_by_task: Counter[str] = Counter()
    calibration_candidates_accepted = 0

    for candidate in sorted(candidates, key=lambda item: str(item.get("id", ""))):
        candidate_id = candidate.get("id")
        task_id = candidate.get("task_id")
        response = candidate.get("response")
        if not all(isinstance(value, str) and value for value in (candidate_id, task_id, response)):
            raise EvaluationError("Every candidate requires non-empty id, task_id, and response strings.")
        if candidate_id in seen_candidate_ids:
            raise EvaluationError(f"Duplicate candidate ID: {candidate_id}")
        seen_candidate_ids.add(candidate_id)
        task = indexed_tasks.get(task_id)
        if task is None:
            raise EvaluationError(f"Candidate {candidate_id} references unknown task {task_id}.")
        capability = task["capability"]
        input_by_capability[capability] += 1
        evaluator_type = task["evaluator"].get("type")
        kwargs: dict[str, Any] = {}
        if evaluator_type == "executable_assertions":
            if executable_judge is None:
                raise EvaluationError(f"Task {task_id} requires an executable judge.")
            kwargs["executable_judge"] = executable_judge
        evaluation_task = dict(task)
        if evaluator_type == "executable_assertions":
            evaluation_task["critical_failure"] = True
        result = evaluate_responses(
            [evaluation_task], [{"id": task_id, "response": response}], **kwargs
        )
        case_result = result["cases"][0]
        reasons: list[str] = []
        if not case_result["passed"]:
            reasons.append("behavioral_failure")
            if case_result["critical_failure"]:
                critical_failures += 1
        if _repeats_excessively(response):
            reasons.append("excessive_repetition")
        fingerprint = _fingerprint(response)
        shingles = _shingles(response)
        if fingerprint and fingerprint in accepted_fingerprints:
            reasons.append("exact_duplicate")
        else:
            for acc_shingles in accepted_shingles_list:
                union = shingles | acc_shingles
                similarity = len(shingles & acc_shingles) / len(union) if union else 1.0
                if similarity >= near_duplicate_threshold:
                    reasons.append("near_duplicate")
                    break
        if reasons:
            for reason in set(reasons):
                rejection_counts[reason] += 1
            decisions.append({
                "id": candidate_id, "task_id": task_id, "accepted": False,
                "reasons": sorted(set(reasons)), "checks": case_result,
            })
            continue
        output = {
            "id": f"filtered-{candidate_id}",
            "messages": [
                {"role": "user", "content": task["prompt"]},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "source": "Hybrid Agent execution-guided candidate filtering",
                "license": "proprietary-approved",
                "category": capability,
                "reviewed": True,
                "review_method": "behavioral validation plus diversity filtering",
                "behavior_concept": task["family"],
                "split_group": task["family"],
                "task_id": task_id,
                "candidate_id": candidate_id,
                "generator": candidate.get("model_id", "unknown"),
                "execution_guided": evaluator_type == "executable_assertions",
                "calibration_only": candidate.get("calibration_only") is True,
            },
        }
        accepted.append(output)
        accepted_texts.append(response)
        accepted_fingerprints.add(fingerprint)
        accepted_shingles_list.append(shingles)
        if candidate.get("calibration_only") is True:
            calibration_candidates_accepted += 1
        accepted_by_capability[capability] += 1
        accepted_by_family[task["family"]] += 1
        accepted_by_task[task_id] += 1
        decisions.append({
            "id": candidate_id, "task_id": task_id, "accepted": True,
            "reasons": [], "checks": case_result,
        })

    capabilities = required_capabilities or {task["capability"] for task in tasks}
    missing_capabilities = sorted(capabilities - set(accepted_by_capability))
    underfilled_tasks = sorted(
        task_id for task_id in indexed_tasks if accepted_by_task[task_id] < minimum_per_task
    )
    nonzero_counts = [accepted_by_capability[name] for name in capabilities if accepted_by_capability[name]]
    capability_ratio = (
        max(nonzero_counts) / min(nonzero_counts) if nonzero_counts else None
    )
    balanced = (
        not missing_capabilities
        and capability_ratio is not None
        and capability_ratio <= maximum_capability_ratio
    )
    critical_failures_admitted = 0
    ready = (
        not underfilled_tasks and balanced and critical_failures_admitted == 0
        and calibration_candidates_accepted == 0
    )
    report = {
        "schema_version": 1,
        "input_candidates": len(candidates),
        "accepted_candidates": len(accepted),
        "rejected_candidates": len(candidates) - len(accepted),
        "acceptance_rate": len(accepted) / len(candidates) if candidates else 0.0,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "critical_failures_observed": critical_failures,
        "critical_failures_admitted": critical_failures_admitted,
        "calibration_candidates_accepted": calibration_candidates_accepted,
        "by_capability": {
            capability: {
                "input": input_by_capability[capability],
                "accepted": accepted_by_capability[capability],
            }
            for capability in sorted(set(input_by_capability) | capabilities)
        },
        "accepted_by_family": dict(sorted(accepted_by_family.items())),
        "accepted_by_task": dict(sorted(accepted_by_task.items())),
        "minimum_per_task": minimum_per_task,
        "underfilled_tasks": underfilled_tasks,
        "missing_capabilities": missing_capabilities,
        "capability_ratio": capability_ratio,
        "maximum_capability_ratio": maximum_capability_ratio,
        "balanced": balanced,
        "ready_for_training": ready,
        "decisions": decisions,
    }
    return accepted, report
