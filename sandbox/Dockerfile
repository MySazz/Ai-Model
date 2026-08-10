FROM python:3.12-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY sandbox/requirements.txt /tmp/sandbox-requirements.txt
RUN python -m pip install --requirement /tmp/sandbox-requirements.txt \
    && rm /tmp/sandbox-requirements.txt

USER 65532:65532
WORKDIR /workspace
