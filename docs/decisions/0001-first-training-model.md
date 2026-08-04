# Decision 0001: first training model

**Status:** accepted
**Date:** 2026-08-02

## Decision

Use `Qwen/Qwen3-1.7B` as the first instruct-model baseline and QLoRA target,
pinned to Hugging Face revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.

The upstream model card identifies the model as a 1.7-billion-parameter causal
language model with a 32,768-token context window and an Apache-2.0 license.
The repository is approximately 4.08 GB. The license and model card must be
rechecked at the pinned revision before any adapter is distributed.

Sources:

- https://huggingface.co/Qwen/Qwen3-1.7B
- https://huggingface.co/Qwen/Qwen3-1.7B/commit/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e

## Why this model

- It is small enough for a cheap first experiment and quantized local inference.
- Its stated strengths include instruction following, coding, reasoning, tool use,
  and general dialogue, matching this project's initial evaluation mix.
- Apache-2.0 is compatible with the intended product experimentation.
- It supplies both thinking and non-thinking modes. Version 1 evaluates
  non-thinking mode for predictable latency and scoring.

This is an experiment choice, not a permanent architecture commitment. The
provider boundary remains model-neutral.

## Local hardware finding

Inventory recorded on 2026-08-02:

- Intel Core i7-9750H, 6 cores / 12 threads
- approximately 15 GiB RAM and 11 GiB swap
- NVIDIA GeForce GTX 1650 Mobile / Max-Q
- 650 GiB free repository filesystem space
- `nvidia-smi` unable to communicate with the installed driver

The local machine is suitable for validation and quantized inference after the
driver is repaired. It is not the reference training environment. The first
QLoRA run targets a single cloud NVIDIA GPU with at least 16 GiB VRAM; 24 GiB is
preferred. No cloud resource is provisioned by this decision.

## Registration gate

Do not register the adapter unless it:

1. beats the pinned base model on the coding and tool-use primary score;
2. has no critical safety failures;
3. loses no more than five percentage points on general capability; and
4. records the base revision, configuration, dataset manifest SHA-256, evaluation
   suite SHA-256, package versions, GPU type, seed, and output adapter hashes.
