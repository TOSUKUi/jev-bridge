"""Backend client tests: body construction, thinking-kwargs fallback, parsing."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import httpx
import pytest

from jev_bridge.backend import OpenAICompatClient
from jev_bridge.schemas import JevBridgeError

MESSAGES = [{"role": "user", "content": "pick one"}]


def _ok_response(with_usage: bool = True) -> Dict[str, Any]:
    body = {
        "choices": [
            {
                "message": {"content": "A"},
                "logprobs": {
                    "content": [
                        {
                            "token": "A",
                            "logprob": -0.01,
                            "top_logprobs": [
                                {"token": "A", "logprob": -0.01},
                                {"token": "B", "logprob": -4.0},
                            ],
                        }
                    ]
                },
            }
        ]
    }
    if with_usage:
        body["usage"] = {"prompt_tokens": 7, "completion_tokens": 1}
    return body


def _client_with_transport(handler) -> OpenAICompatClient:
    client = OpenAICompatClient("http://example.invalid/v1", model="m")
    client._client = httpx.AsyncClient(  # type: ignore[assignment]
        base_url="http://example.invalid/v1",
        transport=httpx.MockTransport(handler),
    )
    return client


def test_sends_logprobs_and_thinking_kwargs():
    seen: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_ok_response())

    client = _client_with_transport(handler)

    async def run():
        text, top, usage = await client.first_token_logprobs(MESSAGES, 20)
        await client.aclose()
        return text, top, usage

    text, top, usage = asyncio.run(run())
    assert text == "A"
    assert top[0] == ("A", -0.01)
    assert usage["prompt_tokens"] == 7
    body = seen[0]
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 20
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 1
    assert body["chat_template_kwargs"] == {"enable_thinking": False, "preserve_thinking": False}


def test_retries_without_chat_template_kwargs_on_400():
    seen: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        seen.append(body)
        if "chat_template_kwargs" in body:
            return httpx.Response(400, json={"error": "unknown field chat_template_kwargs"})
        return httpx.Response(200, json=_ok_response())

    client = _client_with_transport(handler)

    async def run():
        text, top, _ = await client.first_token_logprobs(MESSAGES, 5)
        # a second call must not re-send the rejected field
        await client.first_token_logprobs(MESSAGES, 5)
        await client.aclose()
        return text

    assert asyncio.run(run()) == "A"
    assert len(seen) == 3
    assert "chat_template_kwargs" not in seen[1]
    assert "chat_template_kwargs" not in seen[2]


def test_extra_body_kwargs_take_precedence():
    seen: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_ok_response())

    client = OpenAICompatClient(
        "http://example.invalid/v1",
        model="m",
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    client._client = httpx.AsyncClient(  # type: ignore[assignment]
        base_url="http://example.invalid/v1", transport=httpx.MockTransport(handler)
    )

    async def run():
        await client.first_token_logprobs(MESSAGES, 5)
        await client.aclose()

    asyncio.run(run())
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": True}


def test_missing_logprobs_raises_backend_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "A"}}]})

    client = _client_with_transport(handler)

    async def run():
        with pytest.raises(JevBridgeError) as ei:
            await client.first_token_logprobs(MESSAGES, 5)
        await client.aclose()
        return ei.value

    err = asyncio.run(run())
    assert err.status == 502
    assert err.err_type == "backend_error"


def test_http_error_raises_backend_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = _client_with_transport(handler)

    async def run():
        with pytest.raises(JevBridgeError) as ei:
            await client.first_token_logprobs(MESSAGES, 5)
        await client.aclose()
        return ei.value

    err = asyncio.run(run())
    assert err.status == 502
