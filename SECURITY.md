# Security policy

## Supported version

Security fixes are applied to the latest commit on `main`. This is a pre-1.0
project; older commits and rejected training artifacts are retained for evidence
but are not supported runtime releases.

## Reporting a vulnerability

Do not open a public issue containing credentials, private data, or a working
exploit. Use GitHub's private vulnerability-reporting feature for
`Jdrexx/Ai-Model`. Include the affected commit, reproduction steps, impact, and
the smallest safe proof of concept.

## Boundary assumptions

- Workspace path checks constrain cooperative Python tools; they do not make an
  untrusted local process safe.
- MCP servers are user-installed code. Only explicitly named environment
  variables are forwarded, but the process still has the operating-system access
  of the account running it.
- Docker command mode is the preferred local boundary. Its workspace mount is
  read-only and its network and privileges are restricted, but Docker and its
  daemon remain part of the trusted computing base.
- Executable evaluation's audit hooks and resource limits are defense in depth.
  Hostile Python/native code must be evaluated in a disposable VM or similarly
  isolated worker.
- SQLite audit history is append-only through the application API, not
  cryptographically tamper-proof against an account that can edit the database.
- The privacy classifier detects common credential formats but cannot guarantee
  that arbitrary private data is safe to send to a cloud model.

Never commit `.env`, `.hybrid-agent/`, model weights, raw private datasets, or
unredacted execution logs.
