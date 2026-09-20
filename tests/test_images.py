"""Image reference normalization tests."""

from __future__ import annotations

import base64

import pytest

from jev_bridge.images import normalize_image, sniff_mime, to_data_uri
from jev_bridge.schemas import JevBridgeError

# 1x1 red PNG
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
GIF = b"GIF89a" + b"\x00" * 32
WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 16


def test_sniff_mime():
    assert sniff_mime(PNG) == "image/png"
    assert sniff_mime(JPEG) == "image/jpeg"
    assert sniff_mime(GIF) == "image/gif"
    assert sniff_mime(WEBP) == "image/webp"
    assert sniff_mime(b"not an image") is None
    # RIFF but not WEBP
    assert sniff_mime(b"RIFF....WAVE") is None


def test_normalize_data_uri_passthrough():
    src = to_data_uri(PNG, "image/png")
    out = normalize_image(src)
    assert out.startswith("data:image/png;base64,")
    assert base64.b64decode(out.split(",", 1)[1]) == PNG


def test_normalize_data_uri_resniffs_mime():
    # declared as jpeg but actually png: sniffed MIME wins
    src = "data:image/jpeg;base64," + base64.b64encode(PNG).decode()
    out = normalize_image(src)
    assert out.startswith("data:image/png;base64,")


def test_normalize_bare_base64():
    out = normalize_image(base64.b64encode(JPEG).decode())
    assert out.startswith("data:image/jpeg;base64,")


def test_normalize_http_url_passthrough():
    assert normalize_image("https://example.com/a.png") == "https://example.com/a.png"
    assert normalize_image("http://example.com/a.png") == "http://example.com/a.png"


def test_normalize_rejects_bad_base64():
    with pytest.raises(JevBridgeError):
        normalize_image("data:image/png;base64,!!!not-base64!!!")


def test_normalize_rejects_unknown_format():
    with pytest.raises(JevBridgeError):
        normalize_image(base64.b64encode(b"hello world, not an image").decode())


def test_normalize_rejects_non_image_data_uri_payload():
    with pytest.raises(JevBridgeError):
        normalize_image("data:image/png;base64," + base64.b64encode(b"plain text").decode())


def test_local_paths_disabled_by_default(tmp_path):
    p = tmp_path / "img.png"
    p.write_bytes(PNG)
    with pytest.raises(JevBridgeError) as ei:
        normalize_image(str(p))
    assert "JEVB_ALLOW_LOCAL_IMAGES" in str(ei.value)


def test_local_path_allowed_when_enabled(tmp_path):
    p = tmp_path / "img.png"
    p.write_bytes(PNG)
    out = normalize_image(str(p), allow_local=True)
    assert out.startswith("data:image/png;base64,")


def test_looks_like_path_but_missing(tmp_path):
    with pytest.raises(JevBridgeError) as ei:
        normalize_image(str(tmp_path / "nope.png"))
    assert "not found" in str(ei.value)


def test_max_bytes_enforced():
    with pytest.raises(JevBridgeError) as ei:
        normalize_image(to_data_uri(PNG, "image/png"), max_bytes=10)
    assert "limit" in str(ei.value)


def test_empty_rejected():
    with pytest.raises(JevBridgeError):
        normalize_image("")
