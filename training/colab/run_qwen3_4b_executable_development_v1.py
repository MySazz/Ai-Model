"""Run the frozen Qwen3-4B configuration against executable development tasks."""

from pathlib import Path

import run_qwen3_8b_baseline_v1 as baseline


baseline.RUN_DIR = baseline.ROOT / "training/runs/qwen3-4b-executable-development-v1"
baseline.RESULT = Path("/content/qwen3-4b-executable-development-v1-results.zip")
baseline.MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
baseline.MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
baseline.EVALUATION_SUITE = "executable-development-v1"
baseline.SYSTEM_PROMPT = (
    "You are a precise Python software engineer. Follow the requested function signature exactly. "
    "Return one complete Python code block with all imports. Prefer standard-library, portable, "
    "failure-safe implementations. Do not claim execution results."
)
baseline.RETRIEVAL_DATASET_NAME = ""
baseline.RETRIEVAL_TOP_K = 0
baseline.MAX_NEW_TOKENS = 512
baseline.EXECUTABLE_EVALUATION = True


if __name__ == "__main__":
    baseline.main()
