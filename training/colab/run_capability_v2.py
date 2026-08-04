"""Non-interactive Colab entry point for the capability-v2 experiment."""

from pathlib import Path

import run_capability_v1 as runner


runner.RESULT_ARCHIVE = Path("/content/qwen3-1.7b-capability-v2-results.zip")
runner.CONFIG_PATH = Path("training/configs/qwen3-1.7b-capability-v2.toml")

if __name__ == "__main__":
    runner.main()
