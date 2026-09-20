#!/usr/bin/env python3
"""Black-box latency benchmark for a running jev-bridge.

    python examples/bench.py http://127.0.0.1:8900 [runs]

Reports median/p10/p90 wall time for a 1-question and a 3-question request
(noul + choice + score), mirroring the way TypeSafe's own latency claims are
usually tested.
"""

from __future__ import annotations

import json
import statistics
import sys
import time

import httpx

ONE = {
    "questions": {"is_urgent": {"type": "noul", "instructions": "Does this message express urgency?"}}
}

THREE = {
    "state": "My card was charged twice for order A-104 and I need this fixed today.",
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


def run_once(client: httpx.Client, base: str, body: dict) -> float:
    payload = {"model": "jev-latest", **body}
    t0 = time.perf_counter()
    resp = client.post(f"{base}/v1/systemone", json=payload)
    resp.raise_for_status()
    return (time.perf_counter() - t0) * 1000.0


def summarize(name: str, samples: list[float]) -> None:
    ordered = sorted(samples)
    print(
        f"{name:12s} median {statistics.median(ordered):7.1f} ms"
        f"   (p10 {ordered[len(ordered) // 10]:.0f} / p90 {ordered[-max(1, len(ordered) // 10)]:.0f})"
    )


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8900").rstrip("/")
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    with httpx.Client(timeout=60.0) as client:
        # warmup
        for _ in range(3):
            run_once(client, base, THREE)

        one = [run_once(client, base, ONE) for _ in range(runs)]
        three = [run_once(client, base, THREE) for _ in range(runs)]
        sample = client.post(f"{base}/v1/systemone", json={"model": "jev-latest", **THREE}).json()

    print(f"jev-bridge benchmark @ {base}  ({runs} runs each)")
    summarize("1 question", one)
    summarize("3 questions", three)
    print("answers:", json.dumps(sample["answers"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
