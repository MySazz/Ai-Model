"""Assistant-masked diagnostic that requires held-out behavioral transfer."""

from pathlib import Path

import run_assistant_mask_diagnostic_v1 as runner


runner.DATASET_NAME = "generalization-diagnostic-v1"
runner.MANIFEST_SHA256 = "c697cd7be3175e67e39c62a44ed89554e98010c0f00b0e4afb3609557ed41169"
runner.RUN_DIR = runner.ROOT / "training/runs/generalization-diagnostic-v1"
runner.RESULT = Path("/content/generalization-diagnostic-v1-results.zip")
runner.EPOCHS = 2
runner.REQUIRED_TRANSFER_SCORE = 0.50
runner.MAX_CRITICAL_FAILURES = 0

if __name__ == "__main__":
    runner.main()
