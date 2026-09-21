"""FastAPI application exposing the Jev-style /v1/systemone endpoint.

Configuration is entirely environment-driven so the container needs no config
file:

============================  ==============================================
env                           meaning
============================  ==============================================
JEVB_BACKEND_BASE_URL         OpenAI-compatible base URL (…/v1)             *
JEVB_BACKEND_MODEL            model name sent to the backend                *
JEVB_BACKEND_API_KEY          bearer token for the backend (default dummy)
JEVB_BACKEND_TIMEOUT          seconds per backend call (default 60)
JEVB_MAX_CONCURRENCY          in-flight questions per request (default 8)
JEVB_BACKEND_EXTRA_BODY       JSON merged into every chat-completions body
JEVB_DISABLE_THINKING         1/0 — send chat_template_kwargs enable_thinking=false
                              (auto-retry without it if the backend rejects it)
JEVB_TOP_K                    minimum top_logprobs to request (default 20)
JEVB_CONFIDENCE_METHOD        linear | max_prob | entropy (default linear)
JEVB_PREFILL_ASSISTANT        1/0 — append "<think></think>" prefill
JEVB_MODEL_NAME               value reported in answers' "model" (default
                              jev-bridge-1; client-supplied model wins)
============================  ==============================================

(*) required.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__
from .backend import OpenAICompatClient
from .schemas import JevBridgeError, SystemOneRequest
from .scorer import QuestionScorer, score_all

MODEL_NAME = os.environ.get("JEVB_MODEL_NAME", "jev-bridge-1")


def _bool_env(name: str, default: bool = True) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off")


def _parse_extra_body(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def build_state() -> Dict[str, Any]:
    base = os.environ.get("JEVB_BACKEND_BASE_URL")
    if not base:
        raise RuntimeError("JEVB_BACKEND_BASE_URL is required (e.g. http://host:8000/v1)")
    client = OpenAICompatClient(
        base_url=base,
        api_key=os.environ.get("JEVB_BACKEND_API_KEY", "dummy"),
        model=os.environ.get("JEVB_BACKEND_MODEL", "default"),
        timeout=float(os.environ.get("JEVB_BACKEND_TIMEOUT", "60")),
        max_concurrency=int(os.environ.get("JEVB_MAX_CONCURRENCY", "8")),
        extra_body=_parse_extra_body(os.environ.get("JEVB_BACKEND_EXTRA_BODY", "")),
        disable_thinking=_bool_env("JEVB_DISABLE_THINKING", True),
    )
    scorer = QuestionScorer(
        client,
        prefill_assistant=_bool_env("JEVB_PREFILL_ASSISTANT", True),
        top_k=int(os.environ.get("JEVB_TOP_K", "20")),
        confidence_method=os.environ.get("JEVB_CONFIDENCE_METHOD", "linear"),
    )
    return {"client": client, "scorer": scorer}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.bridge = build_state()
    try:
        yield
    finally:
        await app.state.bridge["client"].aclose()


app = FastAPI(title="jev-bridge", version=__version__, lifespan=lifespan)


@app.exception_handler(JevBridgeError)
async def _jev_error_handler(_: Request, exc: JevBridgeError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content=exc.to_payload())


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "version": __version__,
        "model_name": MODEL_NAME,
        "backend": os.environ.get("JEVB_BACKEND_BASE_URL", ""),
        "backend_model": os.environ.get("JEVB_BACKEND_MODEL", "default"),
        "prefill_assistant": _bool_env("JEVB_PREFILL_ASSISTANT", True),
        "disable_thinking": _bool_env("JEVB_DISABLE_THINKING", True),
        "confidence_method": os.environ.get("JEVB_CONFIDENCE_METHOD", "linear"),
    }


@app.post("/v1/systemone")
async def systemone(request: Request) -> JSONResponse:
    started = time.perf_counter()
    try:
        body = await request.json()
    except Exception as e:
        raise JevBridgeError(f"invalid JSON body: {e}")
    parsed = SystemOneRequest.parse(body)
    scorer: QuestionScorer = request.app.state.bridge["scorer"]
    answers, usage = await score_all(
        scorer,
        parsed,
        max_concurrency=int(os.environ.get("JEVB_MAX_CONCURRENCY", "8")),
        allow_local_images=_bool_env("JEVB_ALLOW_LOCAL_IMAGES", False),
    )
    usage["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    return JSONResponse(
        {
            "model": body.get("model") or MODEL_NAME,
            "answers": answers,
            "usage": usage,
        }
    )
