import importlib.util
from pathlib import Path

import pytest

from hybrid_agent.evaluation import EvaluationError
from hybrid_agent.repair_feedback import named_failures, repair_prompt

ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "training/colab/run_qwen3_4b_execution_feedback_repair_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("repair_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_named_failures_sanitizes_evaluator_details():
    result = {"failures": [
        "executable check failed: leaves_no_staging_file",
        "missing concept: safe comparison",
        "matched forbidden pattern: SECRET=.*",
        "response exceeds 100 words",
    ]}

    assert named_failures(result) == [
        "leaves_no_staging_file", "safe comparison", "prohibited_content", "response_length"
    ]


def test_repair_prompt_contains_names_but_not_evaluator_internals():
    prompt = repair_prompt("Implement f().", "def f(): pass", ["returns_true"], 1)

    assert "returns_true" in prompt
    assert "Implement f()." in prompt
    assert "def f(): pass" in prompt
    assert "harness" not in prompt.casefold()
    assert "expected solution" not in prompt.casefold()


def test_repair_prompt_rejects_attempt_outside_bound():
    with pytest.raises(EvaluationError, match="one or two"):
        repair_prompt("task", "answer", ["check"], 3)


def test_runner_is_pinned_and_bounded():
    runner = load_runner()

    assert runner.MODEL_ID == "Qwen/Qwen3-4B-Instruct-2507"
    assert runner.MODEL_REVISION == "cdbee75f17c01a7cc42f958dc650907174af0554"
    assert runner.SAMPLES_PER_TASK == 4
    assert runner.MAX_REPAIR_ATTEMPTS == 2


def test_colab_bundle_contains_repair_module(tmp_path):
    import subprocess
    import zipfile

    bundle = tmp_path / "bundle.zip"
    subprocess.run(["bash", str(ROOT / "scripts/create_colab_bundle.sh"), str(bundle)], check=True)
    with zipfile.ZipFile(bundle) as archive:
        assert "src/hybrid_agent/repair_feedback.py" in archive.namelist()
