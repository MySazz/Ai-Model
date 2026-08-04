import importlib.util
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_automated_holdout_v3.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_automated_holdout_v3", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_holdout_has_five_unique_cases_per_frozen_rubric() -> None:
    cases = load_builder().build_cases()
    assert len(cases) == 60
    assert len({case["id"] for case in cases}) == 60
    assert Counter(case["rubric_source"] for case in cases) == {
        source_id: 5 for source_id in load_builder().PROMPTS
    }


def test_holdout_has_no_development_prompts() -> None:
    module = load_builder()
    development = {
        __import__("json").loads(line)["prompt"]
        for line in module.SOURCE.read_text().splitlines() if line
    }
    assert not development & {case["prompt"] for case in module.build_cases()}
