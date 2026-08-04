"""Second assistant-mask diagnostic with concise, concept-balanced answers."""

from pathlib import Path

import run_assistant_mask_diagnostic_v1 as runner


runner.DATASET_NAME = "assistant-mask-diagnostic-v2"
runner.MANIFEST_SHA256 = "22fba533a0db0b6d4737e991a44fb7de862585c94dd847cf33bf43b1548d9293"
runner.RUN_DIR = runner.ROOT / "training/runs/assistant-mask-diagnostic-v2"
runner.RESULT = Path("/content/assistant-mask-diagnostic-v2-results.zip")
runner.EPOCHS = 6

if __name__ == "__main__":
    runner.main()
