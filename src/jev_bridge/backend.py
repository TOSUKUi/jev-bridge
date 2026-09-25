"""Async OpenAI-compatible chat-completions client with logprob capture.

Only needs ``POST {base_url}/chat/completions`` with ``logprobs=True`` and
``top_logprobs`` — supported by SGLang, vLLM, llama.cpp, LMDeploy, TGI, and
of course OpenAI itself. No SDK dependency; httpx only.

Thinking models: Qwen3-style chat templates only emit the empty
``<think>\n\n</think>\n\n`` block when ``enable_thinking=false`` is passed
through ``chat_template_kwargs``. The client sends that by default
(``disable_thinking=True``) and transparently falls back to omitting the
field on HTTP 400/422, so backends that do not understand it keep working.

Token budget: the request asks for a single completion token (only the first
sampled token is scored). ``max_tokens`` is sent by default; backends that
renamed the field (OpenAI's GPT-5.x family) are handled by naming
``max_completion_tokens`` in ``extra_body``, or by latching after the first
rejection that asks for it.

Scoring parameters: ``temperature`` and ``logit_bias`` are first-class arguments
because they are applied before the backend computes the returned logprobs, so
they move the reported probabilities. Measured: ``temperature=0.5`` reproduces the
p**2 renormalisation of the label distribution to three decimals.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .schemas import JevBridgeError


def check_temperature(value: Any) -> float:
    """Validate the scoring temperature (a non-negative finite number)."""
    try:
        t = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"temperature must be a number, got {value!r}")
    if t != t or t in (float("inf"), float("-inf")) or t < 0:
        raise ValueError(f"temperature must be a finite number >= 0, got {t}")
    return t


def check_logit_bias(value: Any) -> Dict[str, float]:
    """Validate a logit bias map: token -> additive logit bias."""
    if not value:
        return {}
    if not isinstance(value, dict):
        raise ValueError("logit_bias must be a JSON object of token -> bias")
    out: Dict[str, float] = {}
    for token, bias in value.items():
        if not isinstance(token, str) or not token:
            raise ValueError("logit_bias keys must be non-empty token strings")
        try:
            b = float(bias)
        except (TypeError, ValueError):
            raise ValueError(f"logit_bias[{token!r}] must be a number, got {bias!r}")
        if not -128.0 <= b <= 128.0:
            raise ValueError(f"logit_bias[{token!r}] must be within [-128, 128], got {b}")
        out[token] = b
    return out


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "dummy",
        model: str = "default",
        *,
        timeout: float = 60.0,
        max_concurrency: int = 16,
        extra_body: Optional[Dict[str, Any]] = None,
        disable_thinking: bool = True,
        temperature: float = 0.0,
        logit_bias: Optional[Dict[str, float]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.extra_body = dict(extra_body or {})
        # Both are applied before the backend computes the returned logprobs, so
        # unlike the rest of the sampling surface they change the reported score.
        self.temperature = check_temperature(temperature)
        self.logit_bias = check_logit_bias(logit_bias)
        self._disable_thinking = disable_thinking and "chat_template_kwargs" not in self.extra_body
        self._thinking_kwargs: Dict[str, Any] = {
            "enable_thinking": False,
            "preserve_thinking": False,
        }
        # Some backends (OpenAI's GPT-5.x family) reject `max_tokens` outright and
        # want `max_completion_tokens`; latched on the first such rejection.
        self._max_tokens_key = "max_tokens"
        limits = httpx.Limits(max_connections=max_concurrency)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(timeout),
            limits=limits,
        )
        self._sem = asyncio.Semaphore(max_concurrency)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _budget_key(self) -> str:
        """Which field carries the token budget on the wire.

        ``max_tokens`` is the de-facto default, but OpenAI's GPT-5.x family rejects
        it and accepts only ``max_completion_tokens``. A value named explicitly in
        :attr:`extra_body` decides the field up front (so the *first* request is
        already valid); otherwise the choice is latched after the first rejection
        that names the other field.
        """
        for key in ("max_completion_tokens", "max_tokens"):
            if key in self.extra_body:
                return key
        return self._max_tokens_key

    async def first_token_logprobs(
        self,
        messages: List[Dict[str, str]],
        top_logprobs: int,
        *,
        stop_tokens: Optional[List[str]] = None,
        max_completion_tokens: int = 1,
    ) -> Tuple[str, List[Tuple[str, float]], Dict[str, Any]]:
        """Score the first sampled token.

        Returns ``(text, [(token, logprob), ...], usage)``.

        Raises :class:`JevBridgeError` (502) on transport/HTTP errors so the
        API layer can map it to a Jev-ish error envelope.
        """
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            self._budget_key(): max_completion_tokens,
            "logprobs": True,
            "top_logprobs": max(1, top_logprobs),
        }
        if self.logit_bias:
            body["logit_bias"] = dict(self.logit_bias)
        if stop_tokens:
            body["stop"] = stop_tokens
        body.update(self.extra_body)
        if self._disable_thinking:
            body["chat_template_kwargs"] = dict(self._thinking_kwargs)
        try:
            async with self._sem:
                resp = await self._client.post("/chat/completions", json=body)
            if resp.status_code in (400, 422) and "max_tokens" in body \
                    and "max_completion_tokens" in resp.text:
                # backend renamed the field: send the token budget the other way
                self._max_tokens_key = "max_completion_tokens"
                retry = {k: v for k, v in body.items() if k != "max_tokens"}
                retry.setdefault(self._max_tokens_key, max_completion_tokens)
                async with self._sem:
                    resp = await self._client.post("/chat/completions", json=retry)
            if resp.status_code in (400, 422) and "chat_template_kwargs" in body:
                # backend does not accept chat_template_kwargs: retry without it
                self._disable_thinking = False
                retry = {k: v for k, v in body.items() if k != "chat_template_kwargs"}
                async with self._sem:
                    resp = await self._client.post("/chat/completions", json=retry)
        except httpx.HTTPError as e:
            raise JevBridgeError(f"backend request failed: {e}", status=502, err_type="backend_error") from e
        if resp.status_code != 200:
            raise JevBridgeError(
                f"backend returned HTTP {resp.status_code}: {resp.text[:500]}",
                status=502,
                err_type="backend_error",
            )
        try:
            data = resp.json()
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
            lp = choice["logprobs"]["content"][0]
            top = [(t["token"], float(t["logprob"])) for t in lp["top_logprobs"]]
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise JevBridgeError(
                f"backend response missing logprobs (is logprobs enabled?): {e}",
                status=502,
                err_type="backend_error",
            ) from e
        usage = data.get("usage") or {}
        return text, top, usage
