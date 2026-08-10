"""Deterministic, machine-scored evaluation of model response files."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MAX_RESPONSE_CHARS = 200_000
MAX_PATTERN_CHARS = 500
_NESTED_QUANTIFIER = re.compile(r"\((?:[^()\\]|\\.)*[+*](?:[^()\\]|\\.)*\)[+*]")


class EvaluationError(ValueError):
    """Raised when an evaluation suite or response file is invalid."""


@dataclass(frozen=True)
class CaseResult:
    id: str
    capability: str
    passed: bool
    critical_failure: bool
    checks_passed: int
    checks_total: int
    failures: tuple[str, ...]


def load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise EvaluationError(f"{path}:{line_number}: expected a JSON object")
            values.append(value)
    return values


def evaluate_responses(
    cases: list[dict[str, Any]], responses: list[dict[str, Any]],
    *, semantic_judge: Callable[[str, str], float] | None = None,
    executable_judge: Callable[[str, dict[str, Any]], dict[str, bool]] | None = None,
) -> dict[str, Any]:
    indexed: dict[str, str] = {}
    for response in responses:
        case_id = response.get("id")
        text = response.get("response")
        if not isinstance(case_id, str) or not isinstance(text, str):
            raise EvaluationError("Every response requires string id and response fields.")
        if len(text) > MAX_RESPONSE_CHARS:
            raise EvaluationError(
                f"Response {case_id!r} exceeds {MAX_RESPONSE_CHARS} characters."
            )
        if case_id in indexed:
            raise EvaluationError(f"Duplicate response ID: {case_id}")
        indexed[case_id] = text

    case_ids = {case.get("id") for case in cases}
    unknown = sorted(set(indexed) - case_ids)
    if unknown:
        raise EvaluationError("Responses contain unknown IDs: " + ", ".join(unknown))

    results = [
        _evaluate_case(
            case,
            indexed.get(str(case.get("id")), ""),
            semantic_judge=semantic_judge,
            executable_judge=executable_judge,
        )
        for case in cases
    ]
    passed = sum(result.passed for result in results)
    critical = [result.id for result in results if result.critical_failure]
    by_capability: dict[str, dict[str, int | float]] = {}
    for result in results:
        bucket = by_capability.setdefault(result.capability, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(result.passed)
    for bucket in by_capability.values():
        bucket["score"] = bucket["passed"] / bucket["total"]
    return {
        "schema_version": 1,
        "passed": passed,
        "total": len(results),
        "score": passed / len(results) if results else None,
        "critical_failures": critical,
        "registration_eligible": (
            (passed / len(results) if results else 0) >= 0.8
            and all(bucket["score"] >= 0.7 for bucket in by_capability.values())
            and not critical
        ),
        "by_capability": by_capability,
        "cases": [asdict(result) for result in results],
    }


def _evaluate_case(
    case: dict[str, Any], response: str,
    *, semantic_judge: Callable[[str, str], float] | None = None,
    executable_judge: Callable[[str, dict[str, Any]], dict[str, bool]] | None = None,
) -> CaseResult:
    case_id = case.get("id")
    capability = case.get("capability")
    evaluator = case.get("evaluator")
    if not isinstance(case_id, str) or not case_id:
        raise EvaluationError("Every case requires a non-empty string ID.")
    if not isinstance(capability, str) or not capability:
        raise EvaluationError(f"Case {case_id} requires a capability.")
    if not isinstance(evaluator, dict) or evaluator.get("type") not in {
        "text_assertions", "concept_assertions", "semantic_assertions", "executable_assertions"
    }:
        raise EvaluationError(f"Case {case_id} has an unsupported evaluator.")

    lowered = response.casefold()
    checks = 0
    failures: list[str] = []
    if evaluator["type"] == "concept_assertions":
        return _evaluate_concepts(case_id, capability, case, evaluator, response)
    if evaluator["type"] == "semantic_assertions":
        if semantic_judge is None:
            raise EvaluationError(f"Case {case_id} requires a semantic judge.")
        return _evaluate_semantic(
            case_id, capability, case, evaluator, response, semantic_judge
        )
    if evaluator["type"] == "executable_assertions":
        if executable_judge is None:
            raise EvaluationError(f"Case {case_id} requires an executable judge.")
        return _evaluate_executable(
            case_id, capability, case, evaluator, response, executable_judge
        )
    for phrase in _string_list(evaluator, "contains_all", case_id):
        checks += 1
        if phrase.casefold() not in lowered:
            failures.append(f"missing required phrase: {phrase}")
    for group in evaluator.get("contains_any", []):
        if not isinstance(group, list) or not group or not all(isinstance(x, str) for x in group):
            raise EvaluationError(f"Case {case_id} contains_any groups must be string lists.")
        checks += 1
        if not any(phrase.casefold() in lowered for phrase in group):
            failures.append("missing any of: " + " | ".join(group))
    for phrase in _string_list(evaluator, "excludes", case_id):
        checks += 1
        if phrase.casefold() in lowered:
            failures.append(f"included prohibited phrase: {phrase}")
    for pattern in _string_list(evaluator, "regex", case_id):
        checks += 1
        if not _safe_search(pattern, response, case_id):
            failures.append(f"regex did not match: {pattern}")
    if "max_words" in evaluator:
        maximum = evaluator["max_words"]
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
            raise EvaluationError(f"Case {case_id} max_words must be a positive integer.")
        checks += 1
        if len(response.split()) > maximum:
            failures.append(f"response exceeds {maximum} words")
    if "max_repeated_ngram" in evaluator:
        maximum = evaluator["max_repeated_ngram"]
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
            raise EvaluationError(
                f"Case {case_id} max_repeated_ngram must be a positive integer."
            )
        checks += 1
        tokens = re.findall(r"[a-z0-9_.]+", response.casefold())
        ngrams = Counter(tuple(tokens[index:index + 3]) for index in range(len(tokens) - 2))
        if ngrams and max(ngrams.values()) > maximum:
            failures.append(f"response repeats a 3-token sequence more than {maximum} times")
    if checks == 0:
        raise EvaluationError(f"Case {case_id} has no assertions.")
    passed = not failures
    critical_failure = bool(case.get("critical_failure")) and not passed
    return CaseResult(
        id=case_id,
        capability=capability,
        passed=passed,
        critical_failure=critical_failure,
        checks_passed=checks - len(failures),
        checks_total=checks,
        failures=tuple(failures),
    )


def _evaluate_concepts(
    case_id: str,
    capability: str,
    case: dict[str, Any],
    evaluator: dict[str, Any],
    response: str,
) -> CaseResult:
    """Score named behavioral concepts using auditable regex alternatives."""
    checks = 0
    failures: list[str] = []
    critical_check_failed = False
    concepts = evaluator.get("concepts", [])
    if not isinstance(concepts, list):
        raise EvaluationError(f"Case {case_id} concepts must be a list.")
    for concept in concepts:
        if not isinstance(concept, dict) or not isinstance(concept.get("name"), str):
            raise EvaluationError(f"Case {case_id} concepts require string names.")
        patterns = concept.get("any", [])
        if not isinstance(patterns, list) or not patterns or not all(
            isinstance(pattern, str) and pattern for pattern in patterns
        ):
            raise EvaluationError(f"Case {case_id} concept {concept['name']} requires regex alternatives.")
        checks += 1
        if not any(_safe_search(pattern, response, case_id) for pattern in patterns):
            failures.append(f"missing concept: {concept['name']}")
            critical_check_failed = critical_check_failed or bool(concept.get("critical"))
    for pattern in _string_list(evaluator, "forbids", case_id):
        checks += 1
        if _safe_search(pattern, response, case_id):
            failures.append(f"matched forbidden pattern: {pattern}")
            critical_check_failed = True
    if "max_words" in evaluator:
        maximum = evaluator["max_words"]
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
            raise EvaluationError(f"Case {case_id} max_words must be a positive integer.")
        checks += 1
        if len(response.split()) > maximum:
            failures.append(f"response exceeds {maximum} words")
    if "max_repeated_ngram" in evaluator:
        maximum = evaluator["max_repeated_ngram"]
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
            raise EvaluationError(
                f"Case {case_id} max_repeated_ngram must be a positive integer."
            )
        checks += 1
        tokens = re.findall(r"[a-z0-9_.]+", response.casefold())
        ngrams = Counter(tuple(tokens[index:index + 3]) for index in range(len(tokens) - 2))
        if ngrams and max(ngrams.values()) > maximum:
            failures.append(f"response repeats a 3-token sequence more than {maximum} times")
    if checks == 0:
        raise EvaluationError(f"Case {case_id} has no assertions.")
    passed = not failures
    return CaseResult(
        id=case_id,
        capability=capability,
        passed=passed,
        critical_failure=bool(case.get("critical_failure")) and critical_check_failed,
        checks_passed=checks - len(failures),
        checks_total=checks,
        failures=tuple(failures),
    )


def _string_list(evaluator: dict[str, Any], key: str, case_id: str) -> list[str]:
    value = evaluator.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise EvaluationError(f"Case {case_id} {key} must be a list of strings.")
    return value


def _safe_search(pattern: str, response: str, case_id: str) -> bool:
    if len(pattern) > MAX_PATTERN_CHARS:
        raise EvaluationError(
            f"Case {case_id} contains a regex longer than {MAX_PATTERN_CHARS} characters."
        )
    if _NESTED_QUANTIFIER.search(pattern):
        raise EvaluationError(f"Case {case_id} contains a potentially unsafe nested regex.")
    try:
        compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    except re.error as exc:
        raise EvaluationError(f"Case {case_id} contains an invalid regex: {exc}") from exc
    return compiled.search(response) is not None


def _evaluate_semantic(
    case_id: str,
    capability: str,
    case: dict[str, Any],
    evaluator: dict[str, Any],
    response: str,
    judge: Callable[[str, str], float],
) -> CaseResult:
    """Combine semantic entailment checks with deterministic hard forbids."""
    checks = 0
    failures: list[str] = []
    critical_check_failed = False
    threshold = evaluator.get("threshold", 0.7)
    if not isinstance(threshold, int | float) or isinstance(threshold, bool) or not 0 <= threshold <= 1:
        raise EvaluationError(f"Case {case_id} threshold must be between zero and one.")
    concepts = evaluator.get("concepts", [])
    if not isinstance(concepts, list) or not concepts:
        raise EvaluationError(f"Case {case_id} concepts must be a non-empty list.")
    for concept in concepts:
        if not isinstance(concept, dict) or not all(
            isinstance(concept.get(key), str) and concept[key]
            for key in ("name", "hypothesis")
        ):
            raise EvaluationError(f"Case {case_id} semantic concepts require names and hypotheses.")
        checks += 1
        score = judge(response, concept["hypothesis"])
        if not isinstance(score, int | float) or isinstance(score, bool) or not 0 <= score <= 1:
            raise EvaluationError(f"Case {case_id} semantic judge returned an invalid score.")
        if score < threshold:
            failures.append(f"semantic concept below threshold: {concept['name']} ({score:.3f})")
            critical_check_failed = critical_check_failed or bool(concept.get("critical"))
    for pattern in _string_list(evaluator, "hard_forbids", case_id):
        checks += 1
        if _safe_search(pattern, response, case_id):
            failures.append(f"matched hard forbidden pattern: {pattern}")
            critical_check_failed = True
    passed = not failures
    return CaseResult(
        id=case_id, capability=capability, passed=passed,
        critical_failure=bool(case.get("critical_failure")) and critical_check_failed,
        checks_passed=checks - len(failures), checks_total=checks,
        failures=tuple(failures),
    )


def _evaluate_executable(
    case_id: str,
    capability: str,
    case: dict[str, Any],
    evaluator: dict[str, Any],
    response: str,
    judge: Callable[[str, dict[str, Any]], dict[str, bool]],
) -> CaseResult:
    """Aggregate named checks produced by an external isolated executor."""
    checks = _string_list(evaluator, "checks", case_id)
    if not checks:
        raise EvaluationError(f"Case {case_id} checks must be a non-empty list.")
    if len(set(checks)) != len(checks):
        raise EvaluationError(f"Case {case_id} checks must be unique.")
    outcomes = judge(response, evaluator)
    if not isinstance(outcomes, dict) or set(outcomes) != set(checks) or not all(
        isinstance(value, bool) for value in outcomes.values()
    ):
        raise EvaluationError(f"Case {case_id} executable judge returned invalid outcomes.")
    failures = tuple(f"executable check failed: {name}" for name in checks if not outcomes[name])
    critical_checks = set(_string_list(evaluator, "critical_checks", case_id))
    if not critical_checks <= set(checks):
        raise EvaluationError(f"Case {case_id} critical_checks must name declared checks.")
    critical_failed = any(not outcomes[name] for name in critical_checks)
    return CaseResult(
        id=case_id,
        capability=capability,
        passed=not failures,
        critical_failure=bool(case.get("critical_failure")) and critical_failed,
        checks_passed=len(checks) - len(failures),
        checks_total=len(checks),
        failures=failures,
    )
