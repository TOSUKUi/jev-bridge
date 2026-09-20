"""Async OpenAI-compatible chat-completions client with logprob capture.

Only needs ``POST {base_url}/chat/completions`` with ``logprobs=True`` and
``top_logprobs`` — supported by SGLang, vLLM, llama.cpp, LMDeploy, TGI, and
of course OpenAI itself. No SDK dependency; httpx only.

Thinking models: Qwen3-style chat templates only emit the empty
``<think>\n\n</think>\n\n`` block when ``enable_thinking=false`` is passed
through ``chat_template_kwargs``. The client sends that by default
(``disable_thinking=True``) and transparently falls back to omitting the
field on HTTP 400/422, so backends that do not understand it keep working.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .schemas import JevBridgeError


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
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.extra_body = dict(extra_body or {})
        self._disable_thinking = disable_thinking and "chat_template_kwargs" not in self.extra_body
        self._thinking_kwargs: Dict[str, Any] = {
            "enable_thinking": False,
            "preserve_thinking": False,
        }
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
            "temperature": 0.0,
            "max_tokens": max_completion_tokens,
            "logprobs": True,
            "top_logprobs": max(1, top_logprobs),
        }
        if stop_tokens:
            body["stop"] = stop_tokens
        body.update(self.extra_body)
        if self._disable_thinking:
            body["chat_template_kwargs"] = dict(self._thinking_kwargs)
        try:
            async with self._sem:
                resp = await self._client.post("/chat/completions", json=body)
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
