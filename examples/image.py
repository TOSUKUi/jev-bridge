#!/usr/bin/env python3
"""Image-input example (jev-bridge extension): classify one image with several
questions in a single request.

    python examples/image.py http://127.0.0.1:8900 [path/to/image.png]

Without arguments it generates a small red-circle test image in memory, so it
works with no local files. Requires a vision-capable backend model.
"""

from __future__ import annotations

import base64
import io
import json
import sys

import httpx


def make_test_png() -> bytes:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise SystemExit("Pillow is needed only for the generated test image: pip install pillow")
    img = Image.new("RGB", (224, 224), "white")
    ImageDraw.Draw(img).ellipse([20, 20, 204, 204], fill="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8900").rstrip("/")
    if len(sys.argv) > 2:
        raw = open(sys.argv[2], "rb").read()
    else:
        raw = make_test_png()

    payload = {
        "model": "jev-latest",
        "state": "Classify the attached image.",
        "images": ["data:image/png;base64," + base64.b64encode(raw).decode()],
        "questions": {
            "shape": {
                "type": "choice",
                "instructions": "Which shape is drawn?",
                "criteria": {"circle": "a round shape", "square": "four straight sides", "triangle": "three sides"},
            },
            "color": {
                "type": "choice",
                "instructions": "What is the dominant color?",
                "criteria": {"red": "red", "blue": "blue", "green": "green"},
            },
            "symmetric": {"type": "noul", "instructions": "Is the shape vertically symmetric?"},
        },
    }
    resp = httpx.post(f"{base}/v1/systemone", json=payload, timeout=120.0)
    resp.raise_for_status()
    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
