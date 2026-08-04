"""Score generated Python responses in a disposable Colab VM."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path


ROOT = Path("/content/hybrid-agent")
CASES = ROOT / "datasets/evaluations/executable-development-v1.jsonl"
RESPONSES = Path("/content/executable-development-v1-responses.jsonl")
RUN_DIR = Path("/content/executable-development-v1")
RESULT = Path("/content/executable-development-v1-results.zip")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from hybrid_agent.evaluation import evaluate_responses, load_jsonl_objects
    from hybrid_agent.executable_evaluation import run_python_checks

    cases = load_jsonl_objects(CASES)
    responses = load_jsonl_objects(RESPONSES)
    report = evaluate_responses(cases, responses, executable_judge=run_python_checks)
    summary = {
        "schema_version": 1,
        "suite": "executable-development-v1",
        "suite_sha256": digest(CASES),
        "responses_sha256": digest(RESPONSES),
        "execution_boundary": "disposable-colab-vm",
        **report,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    shutil.copy2(RESPONSES, RUN_DIR / "responses.jsonl")
    shutil.make_archive(str(RESULT.with_suffix("")), "zip", root_dir=RUN_DIR)
    print(json.dumps({"phase": "complete", "result": summary, "archive": str(RESULT)}), flush=True)


if __name__ == "__main__":
    main()
