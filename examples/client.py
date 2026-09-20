#!/usr/bin/env python3
"""Call a jev-bridge /v1/systemone endpoint with httpx and print answers.

Usage:
    python examples/client.py http://127.0.0.1:8900
"""

from __future__ import annotations

import json
import sys

import httpx


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8900").rstrip("/")
    payload = {
        "model": "jev-latest",
        "state": {
            "from": "customer",
            "message": "My card was charged twice for order A-104 and I need this fixed today.",
        },
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Charges, invoices, payment problems",
                    "shipping": "Delivery status, delays, lost packages",
                    "returns": "Exchanges, refunds, damaged items",
                },
            },
            "is_urgent": {"type": "noul", "instructions": "Does this message express urgency?"},
            "severity": {
                "type": "score",
                "instructions": "How severe is the issue?",
                "criteria": ["Cosmetic", "Workaround exists", "Blocking, no workaround"],
            },
        },
    }
    resp = httpx.post(f"{base}/v1/systemone", json=payload, timeout=60.0)
    resp.raise_for_status()
    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))

    answers = data["answers"]
    print("\n--- summary ---")
    print("department :", answers["department"]["choice"],
          f"(confidence {answers['department']['confidence']})")
    print("is_urgent  :", answers["is_urgent"]["noul"])
    print("severity   :", answers["severity"]["score"],
          f"(confidence {answers['severity']['confidence']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
