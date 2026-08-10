import importlib.util
import sys
from collections import Counter
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_transfer_curriculum_v4.py"


def load_builder():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("build_transfer_curriculum_v4", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v4_is_balanced_by_complete_family() -> None:
    rows = load_builder().build_rows()
    families = Counter(row["metadata"]["split_group"] for row in rows)
    assert len(rows) == 120
    assert families == {f"family-{index}": 15 for index in range(8)}


def test_v4_targeted_replay_is_validated_and_complete() -> None:
    module = load_builder()
    rows = module.build_rows()
    targeted = [row for row in rows if row["metadata"].get("targeted_replay")]
    assert len(targeted) == 24
    counts = Counter(row["metadata"]["behavior_concept"] for row in targeted)
    assert counts == {"unavailable_source": 8, "atomic_json": 8, "trusted_path": 8}


def test_frozen_v4_split_keeps_complete_families() -> None:
    root = SCRIPT.parents[1] / "datasets" / "processed" / "transfer-curriculum-v4"
    counts = {}
    locations = {}
    for split in ("train", "validation", "test"):
        rows = [__import__("json").loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()]
        counts[split] = len(rows)
        for row in rows:
            locations.setdefault(row["metadata"]["split_group"], set()).add(split)
    assert counts == {"train": 90, "validation": 15, "test": 15}
    assert all(len(splits) == 1 for splits in locations.values())
