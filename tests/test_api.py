"""End-to-end API tests with a fake OpenAI-compatible backend (no network)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from fastapi.testclient import TestClient

import jev_bridge.app as app_module
from jev_bridge.scorer import QuestionScorer

# A plausible first-token distribution covering letters, booleans and digits.
FAKE_TOP: List[Tuple[str, float]] = [
    ("A", -0.05),
    ("B", -4.0),
    ("C", -7.5),
    ("true", -0.02),
    ("false", -6.0),
    ("0", -8.0),
    ("1", -2.5),
    ("2", -0.03),
]


class FakeClient:
    def __init__(self) -> None:
        self.calls: List[List[Dict[str, str]]] = []

    async def first_token_logprobs(
        self, messages: List[Dict[str, str]], top_logprobs: int
    ) -> Tuple[str, List[Tuple[str, float]], Dict[str, Any]]:
        self.calls.append(messages)
        return "A", list(FAKE_TOP), {"prompt_tokens": 10, "completion_tokens": 1}

    async def aclose(self) -> None:
        return None


def _client_with_fake_backend() -> Tuple[TestClient, FakeClient]:
    fake = FakeClient()

    def fake_build_state() -> Dict[str, Any]:
        scorer = QuestionScorer(fake, prefill_assistant=True, top_k=20)
        return {"client": fake, "scorer": scorer}

    app_module.build_state = fake_build_state  # type: ignore[assignment]
    return TestClient(app_module.app), fake


BODY = {
    "model": "jev-latest",
    "state": {"message": "charged twice", "order": "A-104"},
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "payment issues", "shipping": "delivery", "returns": "refunds"},
        },
        "is_urgent": {"type": "noul", "instructions": "Urgent?"},
        "severity": {
            "type": "score",
            "instructions": "How bad?",
            "criteria": ["cosmetic", "workaround", "blocking"],
        },
    },
}


def test_systemone_end_to_end():
    client, fake = _client_with_fake_backend()
    with client:
        resp = client.post("/v1/systemone", json=BODY)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["model"] == "jev-latest"
    assert set(data["answers"]) == {"department", "is_urgent", "severity"}

    dept = data["answers"]["department"]
    assert dept["type"] == "choice"
    assert dept["choice"] == "billing"
    assert abs(sum(dept["probabilities"].values()) - 1.0) < 1e-3
    # Jev keys choice probabilities by option name, and the argmax option is
    # what the answer reports as "choice".
    assert set(dept["probabilities"]) == {"billing", "shipping", "returns"}
    assert dept["choice"] == max(dept["probabilities"], key=dept["probabilities"].get)

    assert data["answers"]["is_urgent"]["noul"] > 0.9

    sev = data["answers"]["severity"]
    assert sev["type"] == "score"
    assert 1.5 < sev["score"] <= 2.0
    assert sev["legend"]["2"] == "blocking"

    # shared prefix across per-question calls (radix-cache friendly)
    assert len(fake.calls) == 3
    prefixes = {call[0]["content"] for call in fake.calls}
    assert len(prefixes) == 1
    assert len({call[-2]["content"] for call in fake.calls}) == 3  # distinct question turns


def test_shares_prefix_state_and_prefills_think_block():
    client, fake = _client_with_fake_backend()
    with client:
        client.post("/v1/systemone", json=BODY)
    first = fake.calls[0]
    assert "A-104" in first[0]["content"]  # state in system message
    assert first[-1]["role"] == "assistant"
    assert first[-1]["content"] == "<think></think>"


def test_invalid_question_type_returns_400():
    client, _ = _client_with_fake_backend()
    with client:
        resp = client.post(
            "/v1/systemone",
            json={"state": "s", "questions": {"q": {"type": "banana"}}},
        )
    assert resp.status_code == 400
    assert "banana" in resp.json()["error"]["message"]


def test_missing_state_returns_400():
    client, _ = _client_with_fake_backend()
    with client:
        resp = client.post("/v1/systemone", json={"questions": {"q": {"type": "noul"}}})
    assert resp.status_code == 400


def test_backend_error_maps_to_502(monkeypatch):
    class ExplodingClient(FakeClient):
        async def first_token_logprobs(self, messages, top_logprobs):  # type: ignore[override]
            from jev_bridge.schemas import JevBridgeError

            raise JevBridgeError("backend exploded", status=502, err_type="backend_error")

    fake = ExplodingClient()

    def fake_build_state() -> Dict[str, Any]:
        return {"client": fake, "scorer": QuestionScorer(fake)}

    app_module.build_state = fake_build_state  # type: ignore[assignment]
    client = TestClient(app_module.app)
    with client:
        resp = client.post(
            "/v1/systemone",
            json={"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}},
        )
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "backend_error"


def test_health():
    client, _ = _client_with_fake_backend()
    with client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
