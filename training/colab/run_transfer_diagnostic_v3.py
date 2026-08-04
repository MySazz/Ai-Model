"""Transfer diagnostic with targeted weak-area paraphrase boosters."""

from pathlib import Path

import run_assistant_mask_diagnostic_v1 as runner


runner.DATASET_NAME = "transfer-curriculum-v2"
runner.MANIFEST_SHA256 = "c2b79ea45424a23c8122a7c72192d82e4e6505598fc0777509f332bcb64c208f"
runner.RUN_DIR = runner.ROOT / "training/runs/transfer-diagnostic-v3"
runner.RESULT = Path("/content/transfer-diagnostic-v3-results.zip")
runner.EPOCHS = 20
runner.REQUIRED_TRANSFER_SCORE = 0.50
runner.MAX_CRITICAL_FAILURES = 0

if __name__ == "__main__":
    runner.main()
