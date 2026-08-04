"""Eight-family, family-held-out transfer diagnostic."""

from pathlib import Path

import run_assistant_mask_diagnostic_v1 as runner


runner.DATASET_NAME = "transfer-curriculum-v3"
runner.MANIFEST_SHA256 = "256daa252b0644b0ba9ed99f2d099bf502ad9a64a0ed9c38619178e86bb4e70f"
runner.RUN_DIR = runner.ROOT / "training/runs/transfer-diagnostic-v4"
runner.RESULT = Path("/content/transfer-diagnostic-v4-results.zip")
runner.EPOCHS = 8
runner.REQUIRED_TRANSFER_SCORE = 0.50
runner.MAX_CRITICAL_FAILURES = 0

if __name__ == "__main__":
    runner.main()
