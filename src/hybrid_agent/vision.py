"""Validated, bounded image inputs for multimodal providers."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

from .models import ImageInput


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


def load_image(path: Path, *, workspace: Path) -> ImageInput:
    root = workspace.resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise PermissionError("Image path escapes the configured workspace.")
    data = resolved.read_bytes()
    if not data:
        raise ValueError("Image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image exceeds the {MAX_IMAGE_BYTES}-byte limit.")

    media_type, width, height = _identify_image(data)
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"Image exceeds the {MAX_IMAGE_PIXELS}-pixel limit.")
    return ImageInput(
        name=resolved.name,
        media_type=media_type,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _identify_image(data: bytes) -> tuple[str, int, int]:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return "image/png", width, height
    if data.startswith((b"GIF87a", b"GIF89a")) and len(data) >= 10:
        width, height = struct.unpack("<HH", data[6:10])
        return "image/gif", width, height
    if data.startswith(b"\xff\xd8"):
        width, height = _jpeg_dimensions(data)
        return "image/jpeg", width, height
    raise ValueError("Unsupported image type; use PNG, JPEG, or GIF.")


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    offset = 2
    start_of_frame = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        offset += 2
        if marker in {0xD8, 0xD9}:
            continue
        if offset + 2 > len(data):
            break
        length = int.from_bytes(data[offset : offset + 2], "big")
        if length < 2 or offset + length > len(data):
            break
        if marker in start_of_frame and length >= 7:
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            return width, height
        offset += length
    raise ValueError("Could not determine JPEG dimensions.")
