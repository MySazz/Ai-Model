import importlib.util
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

from hybrid_agent.executable_evaluation import run_python_checks

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/build_validator_curriculum_v2.py"


@lru_cache(maxsize=1)
def load_builder():
    spec = importlib.util.spec_from_file_location("build_validator_curriculum_v2", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "src"))
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def built_rows():
    return load_builder().build_rows()


def test_curriculum_has_six_complete_validator_backed_families():
    rows = built_rows()
    assert len(rows) == 48
    assert Counter(row["metadata"]["split_group"] for row in rows) == {
        f"family-{index}": 6 for index in range(8)
    }
    assert Counter(row["metadata"]["behavior_concept"] for row in rows) == {
        "archive_containment": 8,
        "atomic_cleanup": 8,
        "honest_audit_events": 8,
        "migration_callbacks": 8,
        "recoverable_refusal": 8,
        "secret_non_disclosure": 8,
    }
    assert Counter(row["metadata"]["validation_kind"] for row in rows) == {
        "executable": 32,
        "concept": 16,
    }
    assert all(row["metadata"]["validator_backed"] for row in rows)


def test_curriculum_does_not_copy_frozen_executable_interfaces():
    payload = json.dumps(built_rows())
    for held_out_name in (
        "save_json_atomic",
        "safe_join",
        "controlled_rollout",
        "build_tool_trace",
    ):
        assert held_out_name not in payload


def test_v2_harnesses_reject_shallow_implementations():
    builder = load_builder()
    shallow = {
        "atomic_cleanup": """from pathlib import Path
def commit_bytes(destination, payload):
    if type(payload) is not bytes:
        raise TypeError('bytes required')
    Path(destination).write_bytes(payload)
""",
        "archive_containment": """from pathlib import Path
def archive_target(extract_root, member_name):
    root = Path(extract_root).resolve()
    candidate = (root / member_name).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError('escape')
    return candidate
""",
        "migration_callbacks": """def apply_migration(plan, check, begin, verify, commit, abort):
    if not check(plan):
        raise RuntimeError('rejected')
    begin(plan)
    if not verify(plan):
        abort(plan)
        raise RuntimeError('unverified')
    commit(plan)
""",
        "honest_audit_events": """def audit_events(operation, input_data, status, detail):
    return [
        {'kind': 'request', 'operation': operation, 'input': input_data},
        {'kind': 'outcome', 'operation': operation, 'status': status, 'detail': detail},
    ]
""",
    }
    required_failures = {
        "atomic_cleanup": {"commits_from_same_directory_staging"},
        "archive_containment": {
            "blocks_root_alias",
            "blocks_backslash_parent",
            "blocks_windows_drive",
        },
        "migration_callbacks": {
            "begin_exception_aborts_once",
            "verify_exception_aborts_once",
            "commit_exception_aborts_once",
        },
        "honest_audit_events": {"snapshots_nested_values", "rejects_unknown_status"},
    }
    for concept, response in shallow.items():
        spec = builder.EXECUTABLE_SPECS[concept]
        evaluator = {"harness": spec["harness"], "checks": spec["checks"]}
        outcomes = run_python_checks(response, evaluator)
        assert all(not outcomes[name] for name in required_failures[concept])


def test_v2_generation_bank_exposes_all_new_harnesses_and_safety_gates():
    path = ROOT / "datasets/generation/execution-guided-tasks-v2.jsonl"
    tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(tasks) == 6
    assert Counter(task["capability"] for task in tasks) == {
        "coding": 2,
        "safety": 2,
        "infrastructure": 1,
        "tool_use": 1,
    }
    assert {
        task["evaluator"].get("harness")
        for task in tasks
        if task["evaluator"]["type"] == "executable_assertions"
    } == {
        "commit_bytes_train_v2",
        "archive_target_train_v2",
        "migration_guard_train_v2",
        "audit_events_train_v2",
    }
    assert all(task["evaluator"].get("critical_checks") for task in tasks[:4])
    assert all(task["critical_failure"] for task in tasks[4:])


def test_frozen_v2_split_has_no_family_leakage():
    directory = ROOT / "datasets/processed/validator-curriculum-v2"
    locations: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        rows = [
            json.loads(line)
            for line in (directory / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        counts[split] = len(rows)
        for row in rows:
            locations.setdefault(row["metadata"]["split_group"], set()).add(split)
    assert counts == {"train": 36, "validation": 6, "test": 6}
    assert all(len(splits) == 1 for splits in locations.values())
