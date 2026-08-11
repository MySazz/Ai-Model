#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_path="${1:-$project_root/hybrid-agent-colab-input.zip}"

cd "$project_root"
python3 - "$output_path" <<'PY'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import sys

output = Path(sys.argv[1]).resolve()
files = [
    "pyproject.toml",
    "README.md",
    "src/hybrid_agent/__init__.py",
    "src/hybrid_agent/cli.py",
    "src/hybrid_agent/evaluation.py",
    "src/hybrid_agent/candidate_filtering.py",
    "src/hybrid_agent/repair_feedback.py",
    "src/hybrid_agent/executable_evaluation.py",
    "src/hybrid_agent/executable_harness.py",
    "src/hybrid_agent/experiments.py",
    "src/hybrid_agent/privacy.py",
    "training/configs/qwen3-1.7b-capability-v1.toml",
    "training/configs/qwen3-1.7b-capability-v2.toml",
    "training/requirements-qlora-v1.txt",
    "datasets/candidates/capability-v1.jsonl",
    "datasets/candidates/capability-v2.jsonl",
    "datasets/manifests/capability-v1.sources.json",
    "datasets/manifests/capability-v2.sources.json",
    "datasets/evaluations/automated-development-v1.jsonl",
    "datasets/evaluations/automated-development-v2.jsonl",
    "datasets/evaluations/automated-holdout-v3.jsonl",
    "datasets/evaluations/executable-development-v1.jsonl",
    "datasets/evaluations/validator-holdout-v2.jsonl",
    "datasets/evaluations/validator-holdout-v2.manifest.json",
    "datasets/generation/execution-guided-tasks-v1.jsonl",
    "datasets/generation/execution-guided-tasks-v2.jsonl",
    "datasets/processed/capability-v1/manifest.json",
    "datasets/processed/capability-v1/train.jsonl",
    "datasets/processed/capability-v1/validation.jsonl",
    "datasets/processed/capability-v1/test.jsonl",
    "datasets/processed/capability-v2/manifest.json",
    "datasets/processed/capability-v2/train.jsonl",
    "datasets/processed/capability-v2/validation.jsonl",
    "datasets/processed/capability-v2/test.jsonl",
    "datasets/candidates/assistant-mask-diagnostic-v1.jsonl",
    "datasets/processed/assistant-mask-diagnostic-v1/manifest.json",
    "datasets/processed/assistant-mask-diagnostic-v1/train.jsonl",
    "datasets/processed/assistant-mask-diagnostic-v1/validation.jsonl",
    "datasets/processed/assistant-mask-diagnostic-v1/test.jsonl",
    "datasets/candidates/assistant-mask-diagnostic-v2.jsonl",
    "datasets/processed/assistant-mask-diagnostic-v2/manifest.json",
    "datasets/processed/assistant-mask-diagnostic-v2/train.jsonl",
    "datasets/processed/assistant-mask-diagnostic-v2/validation.jsonl",
    "datasets/processed/assistant-mask-diagnostic-v2/test.jsonl",
    "datasets/candidates/generalization-diagnostic-v1.jsonl",
    "datasets/processed/generalization-diagnostic-v1/manifest.json",
    "datasets/processed/generalization-diagnostic-v1/train.jsonl",
    "datasets/processed/generalization-diagnostic-v1/validation.jsonl",
    "datasets/processed/generalization-diagnostic-v1/test.jsonl",
    "datasets/candidates/transfer-curriculum-v1.jsonl",
    "datasets/processed/transfer-curriculum-v1/manifest.json",
    "datasets/processed/transfer-curriculum-v1/train.jsonl",
    "datasets/processed/transfer-curriculum-v1/validation.jsonl",
    "datasets/processed/transfer-curriculum-v1/test.jsonl",
    "datasets/candidates/transfer-curriculum-v2.jsonl",
    "datasets/processed/transfer-curriculum-v2/manifest.json",
    "datasets/processed/transfer-curriculum-v2/train.jsonl",
    "datasets/processed/transfer-curriculum-v2/validation.jsonl",
    "datasets/processed/transfer-curriculum-v2/test.jsonl",
    "datasets/candidates/transfer-curriculum-v3.jsonl",
    "datasets/processed/transfer-curriculum-v3/manifest.json",
    "datasets/processed/transfer-curriculum-v3/train.jsonl",
    "datasets/processed/transfer-curriculum-v3/validation.jsonl",
    "datasets/processed/transfer-curriculum-v3/test.jsonl",
    "datasets/candidates/transfer-curriculum-v4.jsonl",
    "datasets/processed/transfer-curriculum-v4/manifest.json",
    "datasets/processed/transfer-curriculum-v4/train.jsonl",
    "datasets/processed/transfer-curriculum-v4/validation.jsonl",
    "datasets/processed/transfer-curriculum-v4/test.jsonl",
    "datasets/candidates/executable-curriculum-v1.jsonl",
    "datasets/processed/executable-curriculum-v1/manifest.json",
    "datasets/processed/executable-curriculum-v1/train.jsonl",
    "datasets/processed/executable-curriculum-v1/validation.jsonl",
    "datasets/processed/executable-curriculum-v1/test.jsonl",
    "scripts/build_validator_curriculum_v2.py",
    "scripts/build_validator_holdout_v2.py",
    "datasets/candidates/validator-curriculum-v2.jsonl",
    "datasets/processed/validator-curriculum-v2/manifest.json",
    "datasets/processed/validator-curriculum-v2/train.jsonl",
    "datasets/processed/validator-curriculum-v2/validation.jsonl",
    "datasets/processed/validator-curriculum-v2/test.jsonl",
    "training/colab/run_qwen3_8b_baseline_v1.py",
    "training/colab/run_qwen3_4b_validator_holdout_v2.py",
]
with ZipFile(output, "w", ZIP_DEFLATED) as archive:
    for name in files:
        path = Path(name)
        if not path.is_file():
            raise SystemExit(f"Missing required Colab input: {name}")
        archive.write(path, name)
print(output)
PY
