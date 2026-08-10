"""Filesystem helpers for private local state."""

from __future__ import annotations

import os
from pathlib import Path


def prepare_private_parent(path: Path) -> None:
    """Create a private parent directory without changing unrelated existing parents."""
    parent = path.parent
    created = not parent.exists()
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if created or parent.name == ".hybrid-agent":
        try:
            parent.chmod(0o700)
        except OSError:
            if os.name == "posix":
                raise


def protect_private_file(path: Path) -> None:
    """Limit a local state file to its owner on POSIX systems."""
    if os.name == "posix":
        path.chmod(0o600)
