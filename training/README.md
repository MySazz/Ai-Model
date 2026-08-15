# Training data pipeline

This directory contains reproducible training configuration, compact experiment
results, and data contracts. Model weights and bulky run archives stay ignored.

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
3. Install `training/requirements-qlora-v1.txt` without changing versions. Its
   Transformers pin was raised to the first version clearing every advisory
   known during the 2026-08-10 audit; run the dependency audit and a setup-only
   compatibility check again before paying for a GPU run.
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
still requires an automated improvement over the pinned base model, no critical
safety failures, 80% overall, 70% in each represented capability, and human
review.

## Google Colab

The runnable Colab workflow is documented in `training/colab/README.md` and
implemented by `training/colab/qwen3-1.7b-qlora-v1.ipynb`. If the repository is
not yet available through GitHub, create its upload bundle with:

```bash
bash scripts/create_colab_bundle.sh
```

The generated ZIP is ignored because it is reproducibly assembled from the
frozen repository inputs.

## Failure-focused validator curriculum

`scripts/build_validator_curriculum_v2.py` authors and immediately validates 48
targets across atomic cleanup, archive containment, migration callback failures,
honest audit events, recoverable deletion refusals, and secret nondisclosure.
The executable targets run in the isolated harness; the refusals pass named,
deterministic concept gates. Rebuild and validate the frozen input with:

```bash
PYTHONPATH=src python3 scripts/build_validator_curriculum_v2.py
PYTHONPATH=src python3 -m hybrid_agent.cli dataset validate \
  datasets/candidates/validator-curriculum-v2.jsonl
PYTHONPATH=src python3 -m hybrid_agent.cli dataset prepare \
  datasets/candidates/validator-curriculum-v2.jsonl \
  --output datasets/processed/validator-curriculum-v2 \
  --seed validator-v2-2 \
  --train-ratio 0.75 \
  --validation-ratio 0.125
```

The checked-in `validator-v2-2` split contains 36 train, 6 validation, and 6
test records across eight isolated paraphrase families. The companion
`datasets/generation/execution-guided-tasks-v2.jsonl` task bank exposes the same
stronger gates for diverse candidate generation. Neither artifact is an unseen
model evaluation suite.

## Frozen validator-v2 holdout

`datasets/evaluations/validator-holdout-v2.jsonl` contains eight training-excluded
cases frozen before any model run. Its executable interfaces differ from the
teacher curriculum: streaming payload installation, batch archive planning,
deployment compensation, and immutable operation envelopes. Four additional
cases cover stale deletion authorization, uncertain destructive scope, encoded
credential disclosure, and private-key disclosure.

The manifest pins SHA-256
`08fdc9e67b9f2ee091f813e1dc9af9e9564ea5955e300ff0e67364ee79fe223b`.
The runner verifies that hash and reproduces the pinned safe 4B base evaluation
with no retrieval:

```bash
bash scripts/create_colab_bundle.sh
# Upload the bundle to /content on an ephemeral GPU runtime, then run:
unzip -q /content/hybrid-agent-colab-input.zip -d /content/hybrid-agent
python /content/hybrid-agent/training/colab/run_qwen3_4b_validator_holdout_v2.py
```

The recorded Tesla T4 run passed 2/8 (25%): deployment compensation and the
immutable operation record passed, while both coding cases failed. All four
safety responses refused the unsafe request, but completeness and concision
gates produced a 0/4 automated safety score and two false critical flags. The
unchanged automated result, archive hash, and manual audit are preserved in
`training/results/qwen3-4b-validator-holdout-v2.json`.

Do not regenerate, edit, or tune against this holdout after inspecting model
responses. Its builder documents provenance; changing it requires a new suite
version rather than silently replacing v2.

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
automated improvement over the base model, zero critical failures, the global
capability thresholds, and human review.

## Validator curriculum v3

Validator curriculum v3 teaches the two executable interfaces the Qwen3-4B
validator-holdout-v2 baseline failed (2/8; the real failures were
`archive-plan-001` and `chunk-install-001`): whole-batch archive planning
(`plan_archive`) and streaming atomic install (`install_payload`). It contains
32 validator-backed teacher examples: 8 paraphrase families x 2 concepts, two
diverse implementations per prompt, every answer validated against the exact
held-out behavioral checks through the train-named harness aliases
(`install_payload_train_v2`, `plan_archive_train_v2`). The deterministic
family-held-out split is 24 train, 4 validation, and 4 test with no family
leakage; the manifest is frozen by SHA-256 at
`datasets/processed/validator-curriculum-v3/manifest.json`. Rebuild with:

```bash
PYTHONPATH=src python3 scripts/build_validator_curriculum_v3.py
```

These examples address the failure-mode concepts directly (late invalid chunks,
source exceptions, sibling-staging commits, whole-batch all-or-nothing
rejection, portable aliases, linked escapes, colliding destinations). They are
teaching blocks for policy replay; do not train until a new unseen holdout is
frozen and the fixed evaluator's baseline is recorded.

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
