"""QLoRA Qwen3-4B on family-held-out, validator-backed executable targets."""

from pathlib import Path

import run_assistant_mask_diagnostic_v1 as runner


runner.DATASET_NAME = "executable-curriculum-v1"
runner.MANIFEST_SHA256 = "a1a43b270965cbccc757b793d3338c8e2072f8383912938437ac5bf109ca39e0"
runner.EVALUATION_SUITE_NAME = "executable-development-v1"
runner.EVALUATION_EXECUTABLE = True
runner.SECONDARY_EVALUATION_SUITE_NAME = "automated-development-v2"
runner.RUN_DIR = runner.ROOT / "training/runs/qwen3-4b-executable-transfer-v1"
runner.RESULT = Path("/content/qwen3-4b-executable-transfer-v1-results.zip")
runner.MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
runner.MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
runner.SYSTEM_PROMPT = (
    "You are a precise, evidence-grounded Python software agent. Follow requested signatures "
    "exactly. Return complete code with imports. Preserve prior state on failure, validate before "
    "side effects, enforce canonical path containment, clean temporary artifacts, record outcomes "
    "honestly, and never claim unobserved execution."
)
runner.EPOCHS = 6
runner.LEARNING_RATE = 0.0001
runner.MAX_LENGTH = 1536
runner.MAX_NEW_TOKENS = 512
runner.REQUIRED_RECALL = 0.40
runner.REQUIRED_IMPROVEMENT = 0.05
runner.REQUIRED_HELDOUT_RECALL = 0.30
runner.REQUIRED_HELDOUT_IMPROVEMENT = 0.03
runner.REQUIRED_TRANSFER_SCORE = 0.75
runner.MAX_CRITICAL_FAILURES = 0


if __name__ == "__main__":
    runner.main()
