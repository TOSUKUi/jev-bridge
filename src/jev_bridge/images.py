"""Image reference normalization for the image-input extension.

Jev itself has no image input, so this is a jev-bridge extension: the request
may carry a top-level ``images`` array alongside ``state``. Each entry can be

- an OpenAI-style data URI: ``data:image/png;base64,iVBOR...``
- an ``http(s)://`` URL (the backend fetches it)
- a bare base64 string (MIME sniffed from magic bytes)
- a local file path — **disabled by default** (``JEVB_ALLOW_LOCAL_IMAGES=1``
  enables it; reading arbitrary server files is only safe on a trusted host)

Normalized entries are always data URIs or URLs, ready to drop into an
OpenAI-compatible ``image_url`` content part.

Images are attached to the *first* user message together with the state text
(OpenAI-compatible templates reject images in system messages), so all
questions in one request still share an identical prefix — including the
image — for backends with prefix caching.
"""

from __future__ import annotations

import base64
import binascii
import os
import re
from typing import Optional

from .schemas import JevBridgeError

DATA_URI_RE = re.compile(
    r"^data:(?P<mime>image/[a-zA-Z0-9.+-]+);base64,(?P<data>[A-Za-z0-9+/=\s]+)$"
)
URL_RE = re.compile(r"^https?://", re.IGNORECASE)

MAGIC: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # requires WEBP at offset 8
    (b"BM", "image/bmp"),
]


def sniff_mime(data: bytes) -> Optional[str]:
    for magic, mime in MAGIC:
        if data.startswith(magic):
            if mime == "image/webp" and data[8:12] != b"WEBP":
                continue
            return mime
    return None


def to_data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _check_size(data: bytes, max_bytes: int, what: str) -> None:
    if len(data) > max_bytes:
        raise JevBridgeError(
            f"{what} is {len(data) / 1024 / 1024:.1f} MiB, over the "
            f"{max_bytes / 1024 / 1024:.0f} MiB limit (raise JEVB_MAX_IMAGE_BYTES to allow it)"
        )


def normalize_image(ref: str, *, allow_local: bool = False, max_bytes: int = 20 * 1024 * 1024) -> str:
    """Return a backend-ready image URL (data URI or http(s) URL)."""
    if not isinstance(ref, str) or not ref.strip():
        raise JevBridgeError("image entries must be non-empty strings")
    ref = ref.strip()

    m = DATA_URI_RE.match(ref)
    if m:
        try:
            raw = base64.b64decode(m.group("data"), validate=True)
        except (binascii.Error, ValueError) as e:
            raise JevBridgeError(f"invalid base64 in image data URI: {e}") from e
        _check_size(raw, max_bytes, "image")
        sniffed = sniff_mime(raw)
        if sniffed is None:
            raise JevBridgeError(
                "data URI payload is not a recognized image format (png/jpeg/gif/webp/bmp)"
            )
        # Prefer the sniffed type: a mismatched declared type (jpeg header on a
        # PNG) would otherwise mislead the backend's image decoder.
        return to_data_uri(raw, sniffed)

    if URL_RE.match(ref):
        return ref

    # Order matters: JPEG base64 payloads start with "/9j/…", so try base64
    # first and only treat the string as a path when it is not valid base64.
    try:
        raw = base64.b64decode(ref, validate=True)
    except (binascii.Error, ValueError):
        raw = None
    if raw is not None and len(raw) >= 16 and sniff_mime(raw) is not None:
        _check_size(raw, max_bytes, "image")
        return to_data_uri(raw, sniff_mime(raw) or "application/octet-stream")

    path = os.path.expanduser(ref)
    if os.path.isfile(path):
        if not allow_local:
            raise JevBridgeError(
                "local image paths are disabled (security). Send a data URI, an http(s) URL, "
                "or set JEVB_ALLOW_LOCAL_IMAGES=1 on a trusted host."
            )
        try:
            with open(path, "rb") as f:
                file_bytes = f.read()
        except OSError as e:
            raise JevBridgeError(f"cannot read image file {ref}: {e}") from e
        _check_size(file_bytes, max_bytes, f"image file {ref}")
        mime = sniff_mime(file_bytes)
        if mime is None:
            raise JevBridgeError(
                f"unsupported or unrecognized image format: {ref} (png/jpeg/gif/webp/bmp)"
            )
        return to_data_uri(file_bytes, mime)
    if ref.startswith(("/", "./", "~")) or ref.endswith(".png") or ref.endswith(".jpg"):
        raise JevBridgeError(f"image file not found or unrecognized: {ref}")

    if raw is not None:
        raise JevBridgeError(
            "base64 image payload is not a recognized image format (png/jpeg/gif/webp/bmp)"
        )
    raise JevBridgeError(
        "unrecognized image reference: expected a data URI, an http(s) URL, "
        "or a base64-encoded image"
    )
