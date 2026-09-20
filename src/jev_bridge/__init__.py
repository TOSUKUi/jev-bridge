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
prefix caching) therefore pay the state prefill once per request.
"""

__version__ = "0.1.0"
