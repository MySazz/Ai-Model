import importlib.util
import sys
from collections import Counter
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_transfer_curriculum_v2.py"


def load_builder():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("build_transfer_curriculum_v2", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transfer_curriculum_has_complete_distinct_families() -> None:
    rows = load_builder().build_rows()
    concepts = {row["metadata"]["behavior_concept"] for row in rows}
    families = Counter(row["metadata"]["paraphrase_family"] for row in rows)

    assert len(rows) == 48
    assert len(concepts) == 12
    assert len(families) == 4
    assert set(families.values()) == {12}
    assert all(
        row["metadata"]["paraphrase_family"] == row["metadata"]["split_group"]
        for row in rows
    )


def test_each_concept_has_one_example_per_family() -> None:
    rows = load_builder().build_rows()
    concept_families: dict[str, set[str]] = {}
    for row in rows:
        metadata = row["metadata"]
        concept_families.setdefault(metadata["behavior_concept"], set()).add(
            metadata["paraphrase_family"]
        )

    assert all(len(families) == 4 for families in concept_families.values())
