# Hybrid Agent

Hybrid Agent is a local-first Python agent with explicit permission boundaries,
Ollama tool calling, optional MCP stdio servers, private SQLite state, and
reproducible model-training evidence.

The project is deliberately conservative: observation tools can run
automatically, state-changing tools require one-time approval, and prohibited
commands cannot be authorized by a model.

## Implemented capabilities

- Provider-neutral agent loop with offline and Ollama providers.
- Ollama-compatible function tools and multi-turn tool-result history.
- MCP stdio initialization, discovery, calls, shutdown, approval gating, and
  explicit environment-variable forwarding.
- Workspace-confined file inspection, literal search, Git status, exclusive file
  creation, and digest-bound reversible text patches.
- Application-append-only audit history, explicit memory, and approved research
  snapshots in owner-only SQLite files.
- Optional semantic memory through Ollama `/api/embed`; research-source search is
  currently deterministic lexical search with line-level provenance.
- Bounded image inputs, privacy checks, deterministic dataset validation,
  group-held-out splitting, and frozen experiment hashes.
- Policy, semantic, and executable evaluation. Failed adapters remain recorded
  as evidence and are not presented as successful models.

## Install and validate

Python 3.12 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/hybrid-agent doctor
.venv/bin/pytest --cov=hybrid_agent --cov-report=term-missing
.venv/bin/ruff check src tests scripts
.venv/bin/mypy
```

Run the offline smoke provider:

```bash
.venv/bin/hybrid-agent chat "Check that the agent loop is working"
```

Run an installed Ollama model:

```bash
OLLAMA_MODEL=YOUR_MODEL .venv/bin/hybrid-agent chat \
  --provider ollama "List the files in this workspace"
```

`OLLAMA_URL` defaults to `http://127.0.0.1:11434`. Semantic memory uses
`OLLAMA_EMBED_MODEL`, which defaults to `nomic-embed-text` and falls back to
lexical recall when the embedding service is unavailable.

## MCP servers

Treat MCP servers as local plugins: inspect and install a pinned version before
running one. Hybrid Agent passes only `PATH` and `LANG` by default. Forward each
credential explicitly by variable name, without putting its value on the command
line:

```bash
.venv/bin/hybrid-agent chat \
  --mcp-server '/absolute/path/to/audited-mcp-server --stdio' \
  --mcp-pass-env GITHUB_TOKEN
```

Every discovered MCP tool requires approval. Starting a server still executes
that server as local code, so approval of individual tool calls is not a sandbox
for a malicious server.

## Command sandbox

Without `--sandbox docker`, the narrow allowlisted command tool runs on the host.
Build the repository-owned image before enabling Docker mode:

```bash
docker build -f sandbox/Dockerfile -t hybrid-agent-sandbox:0.1.0 .
.venv/bin/hybrid-agent chat --sandbox docker
```

Docker mode uses a digest-pinned base image, disables networking, mounts the
workspace read-only, drops capabilities, prevents privilege escalation, applies
CPU/memory/PID limits, and refuses automatic image pulls. See
[`sandbox/README.md`](sandbox/README.md) for the remaining boundary assumptions.

Executable model-output evaluation also applies resource limits, a clean
environment, temporary working directories, network/process denial, and Python
audit-hook file confinement. Python-level controls are defense in depth, not a
hostile-native-code boundary; run untrusted evaluations in a disposable VM or
the hardened container.

## Training and evaluation

The training history contains both successful pipeline checks and rejected
model experiments. A training record's `reviewed` field means its declared
review method completed; `human_reviewed` separately records human review.
Generated-candidate filtering requires an explicit output license and license
basis and never marks automated review as human review.

Verify a frozen experiment before spending GPU time:

```bash
PYTHONPATH=src python3 -m hybrid_agent.cli experiment verify
```

Registration requires no critical failures, at least 80% overall, at least 70%
within every represented capability, improvement over the pinned base model,
and human review. See [`training/README.md`](training/README.md) and
[`CHECKPOINT.md`](CHECKPOINT.md) before starting another run.

## Repository policy

- Evaluation cases never enter training data.
- Generated code runs only inside an explicitly selected disposable boundary.
- Credentials, local state, model weights, adapters, caches, and large run
  archives remain outside Git.
- Compact results, hashes, manifests, licenses, and reproduction scripts remain
  versioned.

Security assumptions and disclosure guidance are in [`SECURITY.md`](SECURITY.md).

## License

MIT. See [`LICENSE`](LICENSE).
