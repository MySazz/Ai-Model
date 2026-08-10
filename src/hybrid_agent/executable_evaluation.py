"""Disposable subprocess execution for authored Python capability checks."""

from __future__ import annotations

import json
import os
import re
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .evaluation import EvaluationError


HARNESS_PATH = Path(__file__).with_name("executable_harness.py")


def extract_python(response: str) -> str:
    """Extract the largest Python fenced block, or accept an unfenced response."""
    blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", response, re.IGNORECASE | re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    if "```" in response:
        raise EvaluationError("Response has code fences but no Python fenced block.")
    if not response.strip():
        raise EvaluationError("Response contains no Python code.")
    return response.strip() + "\n"


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))


def run_python_checks(response: str, evaluator: dict[str, Any]) -> dict[str, bool]:
    """Run one authored harness in a constrained temporary subprocess.

    The surrounding disposable VM is the network/security boundary. Resource
    limits and the temporary working directory limit accidental damage; they are
    not a container or a substitute for executing untrusted code on an ephemeral VM.
    """
    checks = evaluator.get("checks")
    harness = evaluator.get("harness")
    if not isinstance(checks, list) or not all(isinstance(item, str) for item in checks):
        raise EvaluationError("Executable evaluator requires string checks.")
    if not isinstance(harness, str) or not harness:
        raise EvaluationError("Executable evaluator requires a harness name.")
    try:
        code = extract_python(response)
    except EvaluationError:
        return {name: False for name in checks}
    with tempfile.TemporaryDirectory(prefix="hybrid-exec-") as directory:
        submission = Path(directory) / "submission.py"
        submission.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(HARNESS_PATH), harness, str(submission)],
                cwd=directory,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
                preexec_fn=_limits if os.name == "posix" else None,
            )
        except (subprocess.TimeoutExpired, OSError):
            return {name: False for name in checks}
    if completed.returncode != 0:
        return {name: False for name in checks}
    try:
        outcomes = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {name: False for name in checks}
    if not isinstance(outcomes, dict):
        return {name: False for name in checks}
    return {name: outcomes.get(name) is True for name in checks}
