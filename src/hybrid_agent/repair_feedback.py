"""Bounded, non-leaking feedback for execution-guided candidate repair."""

from __future__ import annotations

from typing import Any

from .evaluation import EvaluationError


def named_failures(case_result: dict[str, Any]) -> list[str]:
    """Reduce evaluator failures to stable names without exposing hidden assertions."""
    failures = case_result.get("failures")
    if not isinstance(failures, (list, tuple)) or not all(
        isinstance(item, str) for item in failures
    ):
        raise EvaluationError("Case result requires a string failure list.")
    names: list[str] = []
    for failure in failures:
        if failure.startswith("executable check failed: "):
            name = failure.removeprefix("executable check failed: ")
        elif failure.startswith("missing concept: "):
            name = failure.removeprefix("missing concept: ")
        elif failure.startswith("response exceeds "):
            name = "response_length"
        elif "repeats a 3-token sequence" in failure:
            name = "excessive_repetition"
        elif failure.startswith(("matched forbidden pattern:", "included prohibited phrase:")):
            name = "prohibited_content"
        else:
            name = "behavioral_assertion"
        if name and name not in names:
            names.append(name)
    return names


def repair_prompt(original_prompt: str, response: str, failures: list[str], attempt: int) -> str:
    """Build a focused repair request containing only public task text and check names."""
    if not original_prompt or not response:
        raise EvaluationError("Repair prompts require non-empty task and response text.")
    if attempt not in (1, 2):
        raise EvaluationError("Repair attempt must be one or two.")
    if not failures or not all(isinstance(item, str) and item for item in failures):
        raise EvaluationError("Repair prompts require named failures.")
    checks = "\n".join(f"- {name}" for name in failures)
    return (
        f"Original task:\n{original_prompt}\n\n"
        f"Your previous answer:\n{response}\n\n"
        f"Automated checks that failed:\n{checks}\n\n"
        "Return a complete replacement answer that fixes every named failure while preserving "
        "the exact requested interface. Do not discuss the repair or the checks."
    )
