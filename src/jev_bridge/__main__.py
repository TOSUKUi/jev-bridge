"""CLI entrypoint: ``jev-bridge`` (serve) and ``jev-bridge probe``."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run(
        "jev_bridge.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


async def _probe_async(args: argparse.Namespace) -> int:
    """Smoke-test the bridge (or a real Jev-style endpoint) end to end."""
    from .backend import OpenAICompatClient
    from .prompts import serialize_state
    from .scorer import QuestionScorer

    base = args.url.rstrip("/")
    state = {
        "from": "customer",
        "message": "My card was charged twice for order A-104 and I need this fixed today.",
    }
    questions: Dict[str, Any] = {
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
    }
    payload = {"model": "jev-latest", "state": state, "questions": questions}
    if base.endswith("/systemone") or "/v1/" in base:
        # probe a running bridge end to end
        import httpx

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=60) as hc:
            resp = await hc.post(base, json=payload)
        dt = (time.perf_counter() - t0) * 1000
        print(f"POST {base} -> {resp.status_code} in {dt:.0f} ms")
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
        return 0 if resp.status_code == 200 else 1

    # direct-backend mode: build prompts locally and score one question
    from .schemas import Question, SystemOneRequest

    req = SystemOneRequest.parse(payload)
    client = OpenAICompatClient(
        base_url=base,
        api_key=os.environ.get("JEVB_BACKEND_API_KEY", "dummy"),
        model=args.model,
    )
    scorer = QuestionScorer(client, prefill_assistant=not args.no_prefill)
    state_text = serialize_state(req.state)
    total = 0.0
    for q in req.questions.values():
        t0 = time.perf_counter()
        answer, _usage = await scorer.score(state_text, q)
        dt = (time.perf_counter() - t0) * 1000
        total += dt
        print(f"{q.key} ({q.type}) in {dt:.0f} ms -> {json.dumps(answer, ensure_ascii=False)}")
    await client.aclose()
    print(f"total {total:.0f} ms for {len(req.questions)} questions (sequential)")
    return 0


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="jev-bridge", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="run the /v1/systemone proxy server")
    p_serve.add_argument("--host", default=os.environ.get("JEVB_HOST", "0.0.0.0"))
    p_serve.add_argument("--port", type=int, default=int(os.environ.get("JEVB_PORT", "8900")))
    p_serve.add_argument("--log-level", default="info")
    p_serve.set_defaults(func=_serve)

    p_probe = sub.add_parser("probe", help="smoke-test a backend or a running bridge")
    p_probe.add_argument("url", help="OpenAI-compatible base URL or bridge /v1/systemone URL")
    p_probe.add_argument("--model", default=os.environ.get("JEVB_BACKEND_MODEL", "default"))
    p_probe.add_argument("--no-prefill", action="store_true", help="disable <think></think> prefill")
    p_probe.set_defaults(func=lambda a: sys.exit(asyncio.run(_probe_async(a))))

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
