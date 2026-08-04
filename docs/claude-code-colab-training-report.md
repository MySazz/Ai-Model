# Claude Code Handoff: Training the Hybrid Agent in Google Colab

**Status date:** August 3, 2026
**Audience:** Claude Code or another coding agent continuing work in this repository
**Repository root:** `/home/j-alien/Documents/GitRepositories/Ai Model`

## Technical summary

This project fine-tunes the pinned `Qwen/Qwen3-1.7B` base model with QLoRA in
Google Colab. The end-to-end Colab mechanism works: the repository can build a
self-contained upload bundle, verify frozen inputs, train on a Tesla T4,
evaluate the base model and adapter, and export hash-verifiable results.

The current blocker is **behavioral transfer**, not training execution. Two
candidate-scale runs reduced token-level validation loss but scored `0/12` on
the automated behavioral suite and introduced critical failures. A later
assistant-only masking diagnostic proved that the model can learn the intended
response tokens, reaching `1.000` held-in recall, but it transferred to only
`2/12` held-out cases. A larger templated dataset made transfer worse (`0/12`).

Claude Code should **not rerun capability-v1 or capability-v2 unchanged** and
should not launch another candidate-scale GPU job yet. The next experiment must
test independently authored paraphrase families at small diagnostic scale,
retain assistant-only loss masking, strengthen behavioral evaluation, and pass
the transfer gate before scaling.

## What “training with Colab” means in this project

Google Colab is the disposable GPU execution environment. The repository—not
the notebook—is the source of truth for:

- model identity and immutable revision;
- QLoRA hyperparameters;
- reviewed training records and deterministic splits;
- evaluation prompts and scoring rules;
- dataset, suite, adapter, and archive hashes;
- compact run results and the accept/reject decision.

Colab may download model files and produce temporary weights, but credentials,
model caches, raw private logs, and adapter weights must not be committed. A
redacted result summary, configuration, manifests, hashes, and reproduction
instructions belong in Git.

## Current training architecture

| Component | Current implementation |
| --- | --- |
| Base model | `Qwen/Qwen3-1.7B` |
| Pinned revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Fine-tuning method | 4-bit NF4 QLoRA with PEFT |
| Reference GPU | One NVIDIA GPU; T4 works, L4/A10G/A100 preferred |
| Precision | BF16 when supported, FP16 on T4 |
| Current v2 LoRA | Rank 8, alpha 16, dropout 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Context length | 2,048 tokens for the current experiment |
| Batch strategy | Batch size 1, 16 gradient-accumulation steps |
| Optimizer | `paged_adamw_8bit` |
| Candidate schedule | One epoch, cosine schedule, 2e-5 learning rate |
| Evaluation decoding | Non-thinking mode, temperature 0, max 384 new tokens |
| Integrity control | Frozen SHA-256 values for model inputs, data, and evaluation suite |

The accepted model decision is in
`docs/decisions/0001-first-training-model.md`. The latest candidate
configuration is `training/configs/qwen3-1.7b-capability-v2.toml`.

## End-to-end Colab workflow

### 1. Make changes locally

All required project components must remain under the repository root. Update
the dataset builder, candidate data, processed splits, manifests, runner,
configuration, tests, and documentation together. Do not make `/content` or
`/tmp` a hidden dependency.

### 2. Validate before using a GPU

Run the local test suite and validate every candidate dataset before packaging:

```bash
cd '/home/j-alien/Documents/GitRepositories/Ai Model'
PYTHONPATH=src python3 -m pytest
PYTHONPATH=src python3 -m hybrid_agent.cli dataset validate \
  datasets/candidates/NEW_DATASET.jsonl
```

Prepare deterministic splits with a fixed seed, then ensure the emitted
manifest hashes the source and every split. Evaluation cases must never be
passed through the training-data preparation path.

### 3. Freeze the experiment

Each runnable experiment needs a checked-in TOML configuration containing:

- experiment ID;
- exact model ID and revision;
- dataset manifest path and SHA-256;
- evaluation suite path and SHA-256;
- random seed;
- QLoRA and training parameters;
- deterministic generation settings.

The runner must refuse to proceed when a configured file is missing or a hash
does not match. Never silently recompute and accept a changed hash inside the
Colab run.

### 4. Build the upload bundle

When the repository is not being cloned from GitHub, build the reproducible
Colab input ZIP:

```bash
bash scripts/create_colab_bundle.sh
```

This creates `hybrid-agent-colab-input.zip`. The script uses an explicit file
allowlist. When adding a new experiment, update that allowlist so the new
config, runner inputs, candidate data, processed splits, manifest, and
evaluation suite are included. The ZIP is generated and ignored; do not commit
it.

### 5. Run in Google Colab

Use a GPU runtime. For the notebook route, upload
`training/colab/qwen3-1.7b-qlora-v1.ipynb` and the input ZIP, then select
**Runtime → Run all**. For a direct runner, upload the ZIP and the relevant
`training/colab/run_*.py` entry point to `/content` and run it there.

The runner is responsible for:

1. unpacking repository inputs;
2. installing the pinned ML dependencies;
3. verifying configuration and all frozen hashes;
4. loading the pinned base model;
5. scoring the base model on the same frozen suite;
6. training the QLoRA adapter with assistant-only labels;
7. scoring the adapter with identical deterministic decoding;
8. applying critical safety and registration gates;
9. recording environment, metrics, and artifact hashes;
10. downloading one results ZIP.

Do not treat successful optimization or a falling validation loss as proof of
capability. The behavioral comparison decides whether the adapter is useful.

### 6. Verify and record the result

Copy the downloaded archive into the ignored `training/runs/` area, verify the
archive and contained artifact hashes, and commit only a compact redacted JSON
summary under `training/results/`. The summary must state whether the adapter is
registration-eligible and why.

## Evidence from completed runs

| Experiment | Scale | Training signal | Held-out behavior | Decision |
| --- | ---: | --- | --- | --- |
| QLoRA calibration | 12 records | Completed two optimizer steps | Not a capability test | Pipeline-only |
| capability-v1 | 3,200 accepted records | Loss `1.3091`; validation loss reached `0.8662` | Base `1/12`; adapter `0/12`; five critical failures | Rejected |
| capability-v2 | 3,500 accepted records | Loss `2.7087`; validation loss improved to `2.2262` | Base `1/12`; adapter `0/12`; five critical failures | Rejected |
| assistant-mask diagnostic v1 | 101 train records | Recall `0.173 → 0.408` | `3/12`; four critical failures | Diagnostic only |
| assistant-mask diagnostic v2 | 105 train records | Recall `0.253 → 1.000` | `2/12`; five critical failures | Mechanics passed; not deployable |
| generalization diagnostic v1 | 873 train records | Held-in recall `0.740` | `0/12`; five critical failures | Rejected |

These results establish three important points:

1. **The Colab and QLoRA pipeline is operational.** Runs complete on a T4 and
   produce reproducible artifacts.
2. **Assistant-only masking is required.** It dramatically improves learning of
   intended answers and must carry forward.
3. **Templated volume does not create transfer.** Repetitive generated examples
   taught surface patterns and misplaced safety language instead of robust
   behavior.

The authoritative compact results are under `training/results/`; the detailed
project interpretation is in `CHECKPOINT.md`.

## Required next experiment

Build a small transfer diagnostic before a candidate-scale v3 run. Its purpose
is to answer one question: can independently authored variants of a behavior
generalize to an unseen paraphrase family?

### Dataset design

For each behavioral concept:

- write several genuinely different user requests rather than swapping nouns in
  one template;
- write concise responses with different valid organization and wording;
- vary task setting, constraints, failure mode, and requested output shape;
- hold out a complete paraphrase family from training;
- include positive behavior, bounded refusal, recovery, and tool-evidence cases
  where applicable;
- keep solutions observable; do not include hidden chain-of-thought;
- validate coding answers with executable tests where possible;
- validate tool decisions and safety outcomes with structured assertions.

Do not reuse the long shared reasoning tails from the rejected generalization
dataset. They dominated the learned signal and reduced transfer.

### Training mechanics

Use the verified assistant-only masking approach in
`training/colab/run_assistant_mask_diagnostic_v2.py` as the starting point. The
loss mask must supervise assistant answer tokens only. Retain a runtime check
that reports prompt tokens masked and assistant tokens supervised, and fail if
no assistant tokens are supervised.

Keep the diagnostic cheap. Do not increase scale to compensate for a failing
small transfer test. First establish that the dataset structure and evaluator
reward the desired generalized behavior.

### Evaluation design

The current suite contains brittle exact-phrase checks. Improve it without
weakening hard safety rules:

- use executable tests for coding outputs;
- use parsed structured assertions for tool selection and arguments;
- accept semantically equivalent safe refusals and recovery plans;
- test agreement with supplied tool or research evidence;
- keep exact hard gates for secret disclosure and prohibited destructive action;
- report overall score, capability score, and critical failures separately;
- preserve a frozen suite hash and evaluator version.

The training generator must not see held-out prompts, fixtures, expected
phrases, or hidden tests.

### Scale-up gate

Do not build or run capability-v3 at candidate scale until the small diagnostic:

- clearly improves held-out transfer over the pinned base model;
- has zero critical safety failures;
- demonstrates performance on entirely held-out paraphrase families;
- produces the same result under a repeat run or an explicitly documented seed
  sensitivity check;
- passes all repository data, privacy, provenance, license, duplicate, and
  contamination checks.

## Registration gate for any future adapter

An adapter may be registered only when all of the following are true:

1. It beats the pinned base model on the primary coding and tool-use score.
2. It has zero critical safety failures.
3. General capability regresses by no more than five percentage points.
4. Base and adapter use the same frozen suite and deterministic decoding.
5. The result records model revision, configuration, seed, package versions,
   GPU, dataset and evaluation hashes, metrics, and adapter hashes.

Anything trained on an undersized or diagnostic dataset must be labeled
`pipeline-only` or `diagnostic` and must set `registration_eligible` to `false`.

## Constraints Claude Code must preserve

- Keep the entire reproducible project inside this repository.
- Keep secrets, credentials, runtime caches, weights, and private raw logs out
  of version control.
- Treat training data and evaluation data as separate trust domains.
- Preserve immutable model revisions and manifest hashes.
- Do not replace the recorded pinned-model baseline with another local model.
- Do not deploy or register any currently recorded adapter; all are rejected or
  diagnostic-only.
- Do not infer success from training loss alone.
- Do not spend another candidate-scale GPU run until the small transfer
  diagnostic passes.

## Suggested Claude Code implementation sequence

1. Read `AGENTS.md`, `CHECKPOINT.md`, `training/README.md`, and
   `training/colab/README.md`.
2. Inspect `run_assistant_mask_diagnostic_v2.py` and preserve its verified label
   masking behavior.
3. Audit `transfer-curriculum-v1.jsonl` for true authorship diversity and
   paraphrase-family boundaries.
4. Add or repair the builder that deterministically creates the next diagnostic
   dataset and its family-level split.
5. Add tests proving no paraphrase family crosses the train/validation boundary.
6. Upgrade evaluation assertions while retaining hard critical-failure gates.
7. Run all local tests and dataset checks.
8. Update `scripts/create_colab_bundle.sh` with every new required input.
9. Run the small diagnostic in Colab and record a hash-verifiable compact result.
10. Decide from behavioral transfer—not loss—whether capability-v3 is justified.

## Open questions

- What minimum number of independently authored families per concept produces a
  stable transfer estimate?
- Should the development suite move from 12 cases to a broader intermediate
  suite before the planned 300-case acceptance suite is frozen?
- Which coding subsets can be filtered automatically with executable tests
  before human review?
- How sensitive is transfer to LoRA target modules and rank after data quality is
  fixed?
- Does Qwen3-1.7B have enough capacity for the combined behavioral curriculum,
  or should model size be revisited only after the data/evaluation experiment is
  conclusive?

## Repository sources

This report is based on the repository state dated August 3, 2026, especially:

- `AGENTS.md`
- `CHECKPOINT.md`
- `README.md`
- `docs/training-capability-plan.md`
- `docs/decisions/0001-first-training-model.md`
- `training/README.md`
- `training/colab/README.md`
- `training/configs/qwen3-1.7b-capability-v2.toml`
- `training/colab/run_capability_v2.py`
- `scripts/create_colab_bundle.sh`
- compact JSON results under `training/results/`
