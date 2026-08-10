import struct
from pathlib import Path

import pytest

from hybrid_agent.vision import load_image


def minimal_png(width: int = 2, height: int = 3) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", width, height)


def test_png_is_validated_and_hashed(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(minimal_png())
    loaded = load_image(image, workspace=tmp_path)
    assert loaded.media_type == "image/png"
    assert loaded.name == "image.png"
    assert len(loaded.sha256) == 64


def test_non_image_is_rejected(tmp_path: Path) -> None:
    file = tmp_path / "fake.png"
    file.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_image(file, workspace=tmp_path)


def test_zero_sized_image_is_rejected(tmp_path: Path) -> None:
    image = tmp_path / "zero.png"
    image.write_bytes(minimal_png(width=0, height=3))
    with pytest.raises(ValueError, match="positive"):
        load_image(image, workspace=tmp_path)


def test_image_path_is_workspace_scoped(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = tmp_path / "outside.png"
    image.write_bytes(minimal_png())
    with pytest.raises(PermissionError):
        load_image(image, workspace=workspace)
