import json
from pathlib import Path

from hybrid_agent.evaluation import evaluate_responses
from hybrid_agent.executable_evaluation import (
    MAX_EXECUTION_OUTPUT_BYTES,
    extract_python,
    run_python_checks,
)

ROOT = Path(__file__).parents[1]
SUITE = [
    json.loads(line)
    for line in (ROOT / "datasets/evaluations/executable-development-v1.jsonl").read_text().splitlines()
    if line
]


SOLUTIONS = {
    "atomic_json_v1": """
import json, os, tempfile
from pathlib import Path
def save_json_atomic(path, data):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise
""",
    "rooted_path_v1": """
from pathlib import Path
def safe_join(root, user_path):
    root = Path(root).resolve()
    candidate = (root / user_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError('outside root')
    return candidate
""",
    "controlled_rollout_v1": """
def controlled_rollout(release, validate, shift, monitor, rollback):
    if not validate(release):
        raise RuntimeError('validation failed')
    shift(release, 10)
    if not monitor(release):
        rollback(release)
        raise RuntimeError('monitor failed')
    shift(release, 100)
""",
    "structured_trace_v1": """
def build_tool_trace(tool, arguments, ok, output):
    return [
        {'event': 'tool_call', 'tool': tool, 'arguments': arguments},
        {'event': 'tool_result', 'tool': tool, 'ok': ok, 'output': output},
    ]
""",
}


def test_authored_reference_behaviors_pass_every_executable_check():
    responses = []
    for case in SUITE:
        harness = case["evaluator"]["harness"]
        responses.append({"id": case["id"], "response": f"```python\n{SOLUTIONS[harness]}\n```"})
    report = evaluate_responses(SUITE, responses, executable_judge=run_python_checks)
    assert report["passed"] == 4
    assert report["critical_failures"] == []


def test_naive_implementations_fail_security_and_rollback_checks():
    responses = [
        {"id": "exec-atomic-json-001", "response": "def save_json_atomic(path, data):\n import json; open(path, 'w').write(json.dumps(data))"},
        {"id": "exec-rooted-path-001", "response": "def safe_join(root, user_path):\n return root / user_path"},
        {"id": "exec-controlled-rollout-001", "response": "def controlled_rollout(release, validate, shift, monitor, rollback):\n shift(release, 100)"},
        {"id": "exec-structured-trace-001", "response": "def build_tool_trace(*args):\n return []"},
    ]
    report = evaluate_responses(SUITE, responses, executable_judge=run_python_checks)
    assert report["passed"] == 0
    assert set(report["critical_failures"]) == {
        "exec-atomic-json-001", "exec-rooted-path-001", "exec-controlled-rollout-001"
    }


def test_extractor_and_timeout_fail_closed():
    assert extract_python("before\n```python\nanswer = 1\n```\nafter") == "answer = 1\n"
    case = SUITE[0]["evaluator"]
    outcomes = run_python_checks("while True: pass", case)
    assert outcomes == {name: False for name in case["checks"]}


def test_submission_cannot_write_outside_evaluation_directory(tmp_path: Path):
    marker = tmp_path / "escaped.txt"
    response = f"open({str(marker)!r}, 'w').write('escaped')\ndef save_json_atomic(path, data): pass"
    outcomes = run_python_checks(response, SUITE[0]["evaluator"])
    assert outcomes == {name: False for name in SUITE[0]["evaluator"]["checks"]}
    assert not marker.exists()


def test_submission_cannot_create_network_socket():
    response = "import socket\nsocket.socket()\ndef save_json_atomic(path, data): pass"
    outcomes = run_python_checks(response, SUITE[0]["evaluator"])
    assert outcomes == {name: False for name in SUITE[0]["evaluator"]["checks"]}


def test_submission_cannot_create_directory_outside_evaluation(tmp_path: Path):
    marker = tmp_path / "escaped-directory"
    response = f"import os\nos.mkdir({str(marker)!r})\ndef save_json_atomic(path, data): pass"
    outcomes = run_python_checks(response, SUITE[0]["evaluator"])
    assert outcomes == {name: False for name in SUITE[0]["evaluator"]["checks"]}
    assert not marker.exists()


def test_submission_output_is_bounded():
    response = (
        f"print('x' * {MAX_EXECUTION_OUTPUT_BYTES + 1})\n"
        "def save_json_atomic(path, data): pass"
    )
    outcomes = run_python_checks(response, SUITE[0]["evaluator"])
    assert outcomes == {name: False for name in SUITE[0]["evaluator"]["checks"]}
