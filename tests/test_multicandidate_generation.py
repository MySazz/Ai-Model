import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "training/colab/run_qwen3_4b_multicandidate_generation_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("multicandidate_runner", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_freezes_pinned_model_and_candidate_budget():
    runner = load_runner()
    assert runner.MODEL_ID == "Qwen/Qwen3-4B-Instruct-2507"
    assert runner.MODEL_REVISION == "cdbee75f17c01a7cc42f958dc650907174af0554"
    assert runner.SAMPLES_PER_TASK == 4
    assert runner.SEED == 20260804


def test_task_bank_produces_32_candidates_with_unique_seeds_and_ids():
    runner = load_runner()
    tasks = [
        json.loads(line)
        for line in (ROOT / runner.TASKS).read_text().splitlines()
        if line
    ]
    pairs = [
        (f"{task['id']}-sample-{sample}", runner.SEED + index * 100 + sample)
        for index, task in enumerate(tasks)
        for sample in range(runner.SAMPLES_PER_TASK)
    ]
    assert len(pairs) == 32
    assert len(set(pairs)) == 32


def test_colab_bundle_contains_filter_and_training_only_tasks(tmp_path):
    import subprocess

    output = tmp_path / "bundle.zip"
    subprocess.run(["bash", str(ROOT / "scripts/create_colab_bundle.sh"), str(output)], check=True)
    with ZipFile(output) as archive:
        names = set(archive.namelist())
    assert "src/hybrid_agent/candidate_filtering.py" in names
    assert "datasets/generation/execution-guided-tasks-v1.jsonl" in names
