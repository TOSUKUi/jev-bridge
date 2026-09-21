"""Jev-bridge: expose a Jev-style /v1/systemone API over any OpenAI-compatible
LLM server by scoring single-token answer labels with logprobs.

Public surface (mirrors TypeSafe's Jev "System One" API shape):

    POST /v1/systemone
    {
      "model": "jev-latest",             # optional, forwarded/validated
      "state": "...",                     # the situation to judge (string or JSON-serializable)
      "questions": {
        "dept":    {"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": "..."}},
        "refund":  {"type": "noul",   "instructions": "..."},
        "urgency": {"type": "score",  "instructions": "...", "criteria": ["low", "mid", "high"]}
      }
    }

    -> {"model": "...", "answers": {...}, "usage": {...}}

The implementation never generates JSON with the model. Each question becomes
a short follow-up message appended after a shared prefix; the backend's first
next-token distribution over the answer labels (A/B/C... / true/false /
level indices) is read via the OpenAI-compatible `logprobs` field and
normalized. Servers with prefix caching (SGLang radix cache, vLLM automatic
prefix caching) reuse that shared prefill, so a warm state costs about what a
short prompt does; the first request for a state still pays it, and concurrent
questions pay it once each unless the first goes out alone (`JEVB_PRIME_PREFIX`)
or concurrency is pinned to 1.
"""

__version__ = "0.1.0"
