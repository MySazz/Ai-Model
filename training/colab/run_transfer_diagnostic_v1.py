"""Transfer diagnostic using independently authored paraphrase families."""

from pathlib import Path

import run_assistant_mask_diagnostic_v1 as runner


runner.DATASET_NAME = "transfer-curriculum-v1"
runner.MANIFEST_SHA256 = "66f8cf9b1e1d0cd6f01fefb1d457f6b4531c01b53b022e30e53a92f417a99cb0"
runner.RUN_DIR = runner.ROOT / "training/runs/transfer-diagnostic-v1"
runner.RESULT = Path("/content/transfer-diagnostic-v1-results.zip")
runner.EPOCHS = 6
runner.REQUIRED_TRANSFER_SCORE = 0.50
runner.MAX_CRITICAL_FAILURES = 0

if __name__ == "__main__":
    runner.main()
