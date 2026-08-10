# Hybrid Agent / Ai-Model

An experimental, local-first AI agent and training laboratory focused on reliable
coding, research, tool use, and infrastructure work.

## Goal

Build a meaningfully capable open agent that can:

- write and verify working code;
- use tools through explicit permission and privacy boundaries;
- research with source provenance instead of invented evidence;
- make reversible infrastructure changes with validation and rollback; and
- improve through automated Colab training without requiring manual answer grading.

“Frontier” is an aspiration, not a current claim. Every candidate must earn
progress through untouched holdouts, executable tests, and zero critical safety
failures.

## Current status

The repository currently includes:

- a provider-neutral CLI agent with local Ollama support;
- dynamic Model Context Protocol (MCP) client integration (`--mcp-server`);
- workspace-confined tools, approvals, privacy checks, audit history, memory, research snapshots, vision inputs, and reversible text editing;
- Docker container sandboxing for secure tool execution (`--sandbox docker`);
- vector-embedded semantic memory and research recall via local Ollama embeddings;
- automated schema migration engine for SQLite datastores;
- deterministic dataset validation and family-held-out splitting;
- reproducible QLoRA runners for Google Colab;
- semantic, policy, and executable evaluation harnesses; and
- comprehensive pytest suites for automated validation.

The latest Qwen3-4B executable-training adapter was rejected. Although validation
loss improved, it remained at 1/4 executable development tasks and regressed the
policy suite to 2/12 with critical failures. Failed adapters are retained as
compact evidence, not presented as successful models. See [CHECKPOINT.md](CHECKPOINT.md)
for exact hashes, scores, and experiment history.

## Roadmap

| Stage | Status |
| --- | --- |
| Safe local agent core | Complete |
| Reproducible datasets and Colab automation | Complete |
| Automated policy and executable evaluation | Complete |
| Execution-guided rejection and diversity pipeline | Complete |
| Multi-candidate Colab generation baseline | Complete; corpus gate failed |
| Execution-feedback candidate repair | Complete; corpus gate failed |
| Dynamic MCP Client Capability | Complete |
| Docker / Firecracker Tool Sandboxing | Scaffolded |
| Vector Embeddings (Semantic Recall) | Scaffolded |
| Async Architecture Refactoring | Scaffolded |
| SQLite Schema Migration Engine | Scaffolded |
| Comprehensive Pytest Suites | Next |
| Teacher-seeded executable curriculum expansion | Planned |
| Balanced capability and safety replay | Planned |
| New frozen unseen holdout: 80% overall, 70% per capability, zero safety failures | Planned |
| Broader coding, research, vision, and infrastructure benchmarks | Planned |
| Candidate model registration and packaging | Blocked until gates pass |

The first 32-candidate generation run accepted five answers. Bounded repair then
generated 49 revisions but produced only one accepted repair, on the general
feedback task. The combined run accepted eight answers, including one direct
byte-commit implementation, but still had no infrastructure, safety, or tool-use
coverage. The next experiment will expand validator-backed teacher examples for
the failed executable interfaces before another training run.

## Run locally

Python 3.12 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/hybrid-agent doctor
.venv/bin/pytest
```

Chat with an installed Ollama model:

```bash
PYTHONPATH=src python3 -m hybrid_agent.cli chat \
  --provider ollama --model YOUR_MODEL
```

Build a reproducible Colab input bundle:

```bash
bash scripts/create_colab_bundle.sh
```

## Repository policy

- Evaluation cases never enter training data.
- Model outputs are not accepted from loss metrics alone.
- Generated code runs only in disposable environments.
- Credentials, local runtime state, model weights, adapters, and large run
  archives are excluded from Git.
- Compact results, hashes, manifests, and reproduction scripts remain versioned.

## License

MIT. See [LICENSE](LICENSE).
