# Training data pipeline

This directory contains reproducible training configuration and data contracts.
It does not contain model weights or a training run yet.

## First QLoRA experiment

The accepted model decision is documented in
`docs/decisions/0001-first-training-model.md`. The frozen configuration is
`training/configs/qwen3-1.7b-qlora-v1.toml`, and the held-out suite is
`datasets/evaluations/flagship-v1.jsonl`.

The checked-in example dataset is a pipeline demonstration, not adequate
training data. Do not start training until the reviewed coding/tool-use dataset
has been prepared at `datasets/processed/coding-tool-v1` and its manifest exists.

Reference cloud environment:

1. Use an ephemeral Linux instance with Python 3.11 and one NVIDIA GPU (16 GiB
   VRAM minimum, 24 GiB preferred).
2. Install a PyTorch build matching the instance CUDA driver.
3. Install `training/requirements-qlora-v1.txt` without changing versions.
4. Verify the pinned model revision and dataset/evaluation hashes.
5. Run the base evaluation before training, then train, then run the identical
   evaluation against the adapter.
6. Copy run metadata, logs, metrics, and adapter hashes into the ignored
   `training/runs/qwen3-1.7b-qlora-v1` directory. Commit a redacted run manifest,
   not weights, credentials, caches, or raw logs containing private data.

The experiment integrity entry point is enabled. It refuses to proceed when the
configured dataset manifest, its source/split files, or the held-out evaluation
suite is missing or differs from its frozen SHA-256 value:

```bash
PYTHONPATH=src python3 -m hybrid_agent.cli experiment verify
```

Batch 001 is intentionally only a 12-record quality-calibration dataset. It is
too small to justify training or a capability claim. A model execution entry
point must preserve the same verified hashes in every run result.

The first baseline preflight result is stored at
`training/results/qwen3-1.7b-base-baseline.json`. Its `blocked` status and null
score are intentional: the pinned model was not locally available, the ML
dependencies were absent, and the NVIDIA driver was unavailable. Do not replace
that baseline with a score from a different locally installed model.

## Capability-v1 candidate

The first candidate-scale workflow uses
`training/configs/qwen3-1.7b-capability-v1.toml`. It contains 3,200 accepted
records: 1,500 coding, 1,000 tool-use, and 700 infrastructure conversations.
Deterministic preparation produces 2,890 training, 145 validation, and 165 test
records. The source revisions, licenses, downloaded hashes, selection procedure,
and output hash are recorded in `datasets/manifests/capability-v1.sources.json`.

Rebuild the ignored candidate source from pinned upstream revisions:

```bash
PYTHONPATH=src python3 scripts/build_capability_corpus.py --download
```

This is the first candidate-scale specialization dataset, not a claim that every
general capability is covered. Its dominant signals are coding, tool selection,
structured calls, tool-result grounding, and DevOps knowledge. Registration
still requires an automated improvement over the pinned base model with no
critical safety failures.

## Google Colab

The runnable Colab workflow is documented in `training/colab/README.md` and
implemented by `training/colab/qwen3-1.7b-qlora-v1.ipynb`. If the repository is
not yet available through GitHub, create its upload bundle with:

```bash
bash scripts/create_colab_bundle.sh
```

The generated ZIP is ignored because it is reproducibly assembled from the
frozen repository inputs.

## Capability-v2 candidate

Capability-v2 replaces the rejected v1 mixture. It retains 1,500 coding, 700
infrastructure, and 400 structured tool-use records, then adds 900 authored
safety and reliability conversations. The added curriculum covers prohibited
operations, secret handling, prompt injection, evidence discipline, safe
deployments, and secure file handling. Its deterministic split contains 3,138
training, 173 validation, and 189 held-out test records with zero validation
errors or warnings.

The v2 QLoRA is deliberately less aggressive: rank 8, attention projection
targets only, and a 2e-5 learning rate. Run it with
`training/colab/run_capability_v2.py`; registration still requires a strict
automated improvement over the base model and zero critical failures.

## Validate source data

```bash
PYTHONPATH=src python3 -m hybrid_agent.cli dataset validate \
  datasets/examples/sft.sample.jsonl
```

The command exits nonzero when it finds malformed JSON, missing metadata,
unapproved licenses, unreviewed records, secrets, duplicate IDs, exact duplicate
conversations, near-duplicate conversations, or other blocking errors. Possible
PII is reported as a warning for human review.

## Prepare deterministic splits

```bash
PYTHONPATH=src python3 -m hybrid_agent.cli dataset prepare \
  datasets/examples/sft.sample.jsonl \
  --output datasets/processed/example-v1 \
  --seed hybrid-agent-v1
```

Preparation validates the complete input, normalizes records, assigns splits
from a stable hash of the seed and record ID, sorts output by record ID, and
writes SHA-256 hashes and source hashes to `manifest.json`.

Changing input content, record IDs, ratios, or the seed creates an auditable
change in the generated outputs. Evaluation datasets remain in
`datasets/evaluations/` and must never be passed to this preparation command for
training.
