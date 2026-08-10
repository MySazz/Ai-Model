# Command sandbox

Build the image from the repository root:

```bash
docker build -f sandbox/Dockerfile -t hybrid-agent-sandbox:0.1.0 .
```

The base image is pinned to the official multi-platform digest resolved on
2026-08-10. The runtime uses `--pull=never`, so it fails closed if this local
image has not been built. Re-resolve and review the digest deliberately when
updating Python or Debian.

At runtime Hybrid Agent supplies:

- no network;
- a read-only root filesystem and read-only workspace bind mount;
- all Linux capabilities dropped and `no-new-privileges` enabled;
- a non-root caller UID/GID;
- bounded memory, CPU, and process counts; and
- a size-limited temporary filesystem.

The image contains only Python, Git, ripgrep, pytest, and pytest-asyncio. It is
intended for the command tool's narrow allowlist, not general package installation
or model execution. Docker daemon compromise, kernel vulnerabilities, and
hostile native code are outside this boundary; use a disposable VM when those
threats are in scope.
