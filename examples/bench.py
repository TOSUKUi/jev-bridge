#!/usr/bin/env python3
"""Black-box latency benchmark for a running jev-bridge.

    python examples/bench.py http://127.0.0.1:8900 [runs]

Reports median/p10/p90 wall time for a 1-, 3- and 6-question request
(noul + choice + score mixes) plus two concurrent-throughput probes, mirroring
the way TypeSafe's own latency claims are usually tested.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

ONE = {
    "state": "My card was charged twice for order A-104 and I need this fixed today.",
    "questions": {"is_urgent": {"type": "noul", "instructions": "Does this message express urgency?"}},
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


SIX = {
    "state": "My card was charged twice for order A-104, and the replacement label was wrong too.",
    "questions": {
        "department": THREE["questions"]["department"],
        "is_urgent": THREE["questions"]["is_urgent"],
        "severity": THREE["questions"]["severity"],
        "needs_human": {"type": "noul", "instructions": "Should a human take over?"},
        "channel": {
            "type": "choice",
            "instructions": "Which channel did this arrive on?",
            "criteria": {
                "email": "Email thread",
                "chat": "Live chat or messaging",
                "phone": "Phone or voice call",
            },
        },
        "sentiment": {
            "type": "score",
            "instructions": "How does the customer feel?",
            "criteria": ["Delighted", "Neutral", "Furious"],
        },
    },
}

BODIES = [("1 question", ONE), ("3 questions", THREE), ("6 questions", SIX)]


def run_once(client: httpx.Client, base: str, body: dict) -> float:
    payload = {"model": "jev-latest", **body}
    t0 = time.perf_counter()
    resp = client.post(f"{base}/v1/systemone", json=payload)
    resp.raise_for_status()
    return (time.perf_counter() - t0) * 1000.0


def summarize(name: str, samples: list[float]) -> None:
    ordered = sorted(samples)
    print(
        f"{name:14s} median {statistics.median(ordered):7.1f} ms"
        f"   (p10 {ordered[len(ordered) // 10]:.0f} / p90 {ordered[-max(1, len(ordered) // 10)]:.0f})"
    )


def burst(client: httpx.Client, base: str, body: dict, n: int) -> tuple[float, float]:
    """Fire ``n`` identical requests at once; return (ms per request, wall ms)."""
    with ThreadPoolExecutor(max_workers=n) as pool:
        t0 = time.perf_counter()
        list(pool.map(lambda _: run_once(client, base, body), range(n)))
        wall = (time.perf_counter() - t0) * 1000.0
    return wall / n, wall


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8900").rstrip("/")
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    with httpx.Client(timeout=120.0) as client:
        # warmup
        for _ in range(3):
            run_once(client, base, THREE)

        samples = {
            name: [run_once(client, base, body) for _ in range(runs)]
            for name, body in BODIES
        }
        bursts = {n: burst(client, base, THREE, n) for n in (4, 8)}
        sample = client.post(f"{base}/v1/systemone", json={"model": "jev-latest", **THREE}).json()

    print(f"jev-bridge benchmark @ {base}  ({runs} runs each)")
    for name, _ in BODIES:
        summarize(name, samples[name])
    for n, (per_req, wall) in bursts.items():
        print(f"{n}x 3q at once  median {per_req:7.1f} ms/request   (wall {wall:.0f} ms for {n})")
    print("answers:", json.dumps(sample["answers"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
