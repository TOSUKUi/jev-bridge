"""Unit tests for probability math, schemas, labels, and prompt building."""

from __future__ import annotations

import math

import pytest

from jev_bridge.labels import labels_for
from jev_bridge.probability import (
    confidence_from,
    probabilities_map,
    restricted_softmax,
    score_answer,
)
from jev_bridge.prompts import build_messages, question_user_message, serialize_state, user_message
from jev_bridge.schemas import JevBridgeError, Question, SystemOneRequest


# ---------- probability ----------

def test_restricted_softmax_sums_to_one():
    probs = restricted_softmax([0.0, -1.0, -2.0])
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-9)
    assert probs[0] > probs[1] > probs[2]


def test_confidence_linear_matches_jev_example():
    # TypeSafe docs example: probabilities {0:0.0, 1:0.7, 2:0.3} -> confidence 0.54
    probs = [0.0, 0.7, 0.3]
    c = confidence_from(probs, method="linear")
    assert abs(c - 0.54) < 0.02


def test_confidence_uniform_is_zero_linear():
    assert confidence_from([0.25, 0.25, 0.25, 0.25]) == 0.0


def test_confidence_one_hot():
    assert confidence_from([1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_confidence_entropy_method():
    # one-hot -> 1.0; uniform -> 0.0
    assert confidence_from([1.0, 0.0], method="entropy") == pytest.approx(1.0)
    assert confidence_from([0.5, 0.5], method="entropy") == pytest.approx(0.0)


def test_score_answer_between_rungs():
    # p(level1)=0.7, p(level2)=0.3 -> score 1.3 (docs example)
    assert score_answer([0.0, 0.7, 0.3]) == pytest.approx(1.3)


def test_probabilities_map_rounding():
    m = probabilities_map(["A", "B"], [0.98765, 0.01235])
    assert m == {"A": 0.9877, "B": 0.0123}
    # a three-way coin flip must not report 0.9999: the residual goes to the
    # largest entry so the contract (probabilities sum to 1) holds exactly
    three = probabilities_map(["A", "B", "C"], [1 / 3, 1 / 3, 1 / 3])
    assert three == {"A": 0.3334, "B": 0.3333, "C": 0.3333}
    assert abs(sum(three.values()) - 1.0) < 1e-9


# ---------- labels ----------

def test_labels_letters_then_digits():
    assert labels_for(3) == ["A", "B", "C"]
    assert labels_for(26)[-1] == "Z"
    assert labels_for(27)[:3] == ["0", "1", "2"]


# ---------- schemas ----------

def _choice_spec():
    return {"type": "choice", "instructions": "pick", "criteria": {"a": "x", "b": "y"}}


def test_parse_valid_request():
    req = SystemOneRequest.parse(
        {"model": "jev-latest", "state": "hello", "questions": {"q1": _choice_spec()}}
    )
    assert req.questions["q1"].levels() == ["a", "b"]


def test_parse_rejects_unknown_type():
    with pytest.raises(JevBridgeError):
        SystemOneRequest.parse(
            {"state": "s", "questions": {"q": {"type": "banana"}}}
        )


def test_parse_rejects_choice_without_criteria():
    with pytest.raises(JevBridgeError):
        SystemOneRequest.parse({"state": "s", "questions": {"q": {"type": "choice"}}})


def test_parse_rejects_score_bad_criteria():
    with pytest.raises(JevBridgeError):
        SystemOneRequest.parse(
            {"state": "s", "questions": {"q": {"type": "score", "criteria": ["only"]}}}
        )


def test_parse_rejects_too_many_options():
    crit = {f"o{i}": "d" for i in range(256)}
    with pytest.raises(JevBridgeError):
        SystemOneRequest.parse(
            {"state": "s", "questions": {"q": {"type": "choice", "criteria": crit}}}
        )


def test_state_serialization():
    assert serialize_state("plain") == "plain"
    d = {"a": 1}
    assert "a" in serialize_state(d)


# ---------- prompts ----------

def test_choice_prompt_contains_labels_and_state_hint():
    q = Question(key="q", type="choice", instructions="pick one", criteria={"a": "alpha", "b": "beta"})
    q.option_labels = ["A", "B"]
    msg = question_user_message(q)
    assert "A: a — alpha" in msg
    assert "B: b — beta" in msg
    assert "letter" in msg


def test_choice_prompt_uses_number_for_digit_labels():
    q = Question(key="q", type="choice", instructions="pick one", criteria={"a": "alpha"})
    q.option_labels = ["0"]  # >26-option fallback labels are digits
    msg = question_user_message(q)
    assert "number" in msg


def test_build_messages_prefill_assistant():
    q = Question(key="q", type="noul", instructions="is it?")
    msgs = build_messages("the-state", q, prefill_assistant=True)
    assert msgs[-1] == {"role": "assistant", "content": "<think></think>"}
    assert "the-state" in msgs[1]["content"]
    assert "the-state" not in msgs[0]["content"]  # system prompt is static (prefix-friendly)
    msgs2 = build_messages("the-state", q, prefill_assistant=False)
    assert msgs2[-1]["role"] == "user"


def test_user_message_with_images():
    msg = user_message("the state", "the question", ["data:image/png;base64,AAAA"])
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    assert msg["content"][0]["type"] == "image_url"
    assert msg["content"][-1]["type"] == "text"
    assert "the state" in msg["content"][-1]["text"]
    assert "the question" in msg["content"][-1]["text"]


def test_user_message_without_images_is_plain_string():
    msg = user_message("s", "q", None)
    assert isinstance(msg["content"], str)


def test_build_messages_attaches_images_to_user_turn():
    q = Question(key="q", type="noul", instructions="is it?")
    msgs = build_messages("STATE", q, prefill_assistant=False, image_urls=["http://x/i.png"])
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"][0]["image_url"]["url"] == "http://x/i.png"


def test_noul_prompt_mentions_true_false():
    q = Question(key="q", type="noul", instructions="urgent?")
    msg = question_user_message(q)
    assert '"true"' in msg and '"false"' in msg
