import importlib.util
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

from hybrid_agent.executable_evaluation import run_python_checks

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/build_validator_curriculum_v3.py"


@lru_cache(maxsize=1)
def load_builder():
    spec = importlib.util.spec_from_file_location("build_validator_curriculum_v3", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "src"))
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def built_rows():
    return load_builder().build_rows()


def test_curriculum_has_two_complete_validator_backed_families():
    rows = built_rows()
    assert len(rows) == 32
    assert Counter(row["metadata"]["split_group"] for row in rows) == {
        f"family-{index}": 4 for index in range(8)
    }
    assert Counter(row["metadata"]["behavior_concept"] for row in rows) == {
        "atomic_install": 16,
        "archive_batch_plan": 16,
    }
    # Two diverse passes per task: every family holds both variants of both concepts.
    assert Counter(
        (row["metadata"]["split_group"], row["metadata"]["behavior_concept"])
        for row in rows
    ) == {(f"family-{index}", concept): 2 for index in range(8) for concept in (
        "atomic_install",
        "archive_batch_plan",
    )}
    assert Counter(row["metadata"]["variant"] for row in rows) == {
        "variant-0": 16,
        "variant-1": 16,
    }
    assert all(row["metadata"]["validator_backed"] for row in rows)
    assert all(row["metadata"]["validation_kind"] == "executable" for row in rows)


def test_curriculum_does_not_copy_frozen_evaluation_suite():
    # v3 teaches the interface SHAPES that failed holdout-v2 but must not copy
    # the frozen evaluation suite's prompts, ids, or fixtures.
    payload = json.dumps(built_rows())
    assert "validator-holdout-v2" not in payload
    assert "Write Python defining" not in payload
    assert "Consume an iterable of exact bytes chunks" not in payload
    assert "Return a list of resolved Paths for a batch" not in payload


def test_v3_harnesses_reject_shallow_implementations():
    builder = load_builder()
    shallow = {
        "atomic_install": """def install_payload(destination, chunks):
    with open(destination, 'wb') as stream:
        for chunk in chunks:
            stream.write(chunk)
""",
        "archive_batch_plan": """from pathlib import Path
def plan_archive(extract_root, names):
    root = Path(extract_root).resolve()
    return [(root / name).resolve() for name in names]
""",
    }
    required_failures = {
        "atomic_install": {
            "preserves_on_late_invalid_chunk",
            "preserves_on_source_exception",
            "commits_from_sibling_staging",
        },
        "archive_batch_plan": {
            "rejects_parent_in_batch",
            "rejects_absolute_in_batch",
            "rejects_portable_aliases",
            "rejects_linked_escape",
            "rejects_colliding_destinations",
        },
    }
    for concept, response in shallow.items():
        spec = builder.EXECUTABLE_SPECS[concept]
        evaluator = {"harness": spec["harness"], "checks": spec["checks"]}
        outcomes = run_python_checks(response, evaluator)
        assert all(not outcomes[name] for name in required_failures[concept])


def test_frozen_v3_split_has_no_family_leakage():
    directory = ROOT / "datasets/processed/validator-curriculum-v3"
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
    assert counts == {"train": 24, "validation": 4, "test": 4}
    assert all(len(splits) == 1 for splits in locations.values())
