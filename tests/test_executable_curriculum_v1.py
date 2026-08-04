import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/build_executable_curriculum_v1.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_executable_curriculum_v1", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parents[1] / "src"))
    spec.loader.exec_module(module)
    return module


def test_curriculum_has_eight_complete_validator_backed_families():
    rows = load_builder().build_rows()
    assert len(rows) == 32
    assert Counter(row["metadata"]["split_group"] for row in rows) == {
        f"family-{index}": 4 for index in range(8)
    }
    assert Counter(row["metadata"]["behavior_concept"] for row in rows) == {
        "byte_commit": 8, "archive_containment": 8,
        "migration_state": 8, "audit_trace": 8,
    }
    assert all(row["metadata"]["executable_validated"] for row in rows)


def test_curriculum_does_not_copy_executable_evaluation_interfaces():
    payload = json.dumps(load_builder().build_rows())
    for held_out_name in (
        "save_json_atomic", "safe_join", "controlled_rollout", "build_tool_trace"
    ):
        assert held_out_name not in payload


def test_frozen_split_has_no_family_leakage():
    directory = SCRIPT.parents[1] / "datasets/processed/executable-curriculum-v1"
    locations = {}
    counts = {}
    for split in ("train", "validation", "test"):
        rows = [json.loads(line) for line in (directory / f"{split}.jsonl").read_text().splitlines()]
        counts[split] = len(rows)
        for row in rows:
            locations.setdefault(row["metadata"]["split_group"], set()).add(split)
    assert counts == {"train": 24, "validation": 4, "test": 4}
    assert all(len(splits) == 1 for splits in locations.values())
