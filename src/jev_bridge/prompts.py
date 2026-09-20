"""Prompt construction.

Design: the STATE goes into a shared system message; each question becomes a
single user turn ("state-system, question-user"). All question calls share the
exact same system prefix, so backends with prefix caching (SGLang radix
cache, vLLM APC) prefill the state once and then only process the tiny
per-question suffix.

The final assistant message is an empty think-block prefill
(``<think></think>``) so thinking-mode models (e.g. Qwen3.x) answer with the
label token immediately instead of emitting ``<think>`` first. Disable with
``JEVB_PREFILL_ASSISTANT=0`` for models where that prefix is wrong.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .labels import LETTERS
from .schemas import JevBridgeError, Question

STATE_SYSTEM_TEMPLATE = """\
You are a decision engine. You never explain, never apologize, and never \
write sentences. You read the STATE and answer the user's question by \
outputting EXACTLY ONE answer label and nothing else.

STATE:
{state}"""


def serialize_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    try:
        return json.dumps(state, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError) as e:
        raise JevBridgeError(f"'state' is not JSON-serializable: {e}")


def question_user_message(q: Question) -> str:
    lines: List[str] = []
    if q.type == "choice":
        labels = q.option_labels or LETTERS[: len(q.criteria)]
        lines.append(f"Question: {q.instructions}" if q.instructions else "Question:")
        lines.append("Options:")
        for label, (key, desc) in zip(labels, q.criteria.items()):
            if desc:
                lines.append(f"{label}: {key} — {desc}")
            else:
                lines.append(f"{label}: {key}")
        word = "letter" if labels and labels[0].isalpha() else "number"
        lines.append(f"Respond with ONLY the {word} of the best option.")
    elif q.type == "score":
        labels = q.option_labels or [str(i) for i in range(len(q.criteria))]
        lines.append(f"Question: {q.instructions}" if q.instructions else "Question:")
        lines.append("Scale levels, lowest to highest:")
        for label, desc in zip(labels, q.criteria):
            lines.append(f"{label}: {desc}")
        lines.append("Respond with ONLY the number of the best-fitting level.")
    else:  # noul
        lines.append(f"Question: {q.instructions}" if q.instructions else "Question:")
        crit = q.criteria if isinstance(q.criteria, dict) else {}
        if crit:
            lines.append(f"true means: {crit.get('true', 'yes')}")
            lines.append(f"false means: {crit.get('false', 'no')}")
        lines.append('Respond with ONLY "true" or "false".')
    return "\n".join(lines)


def build_messages(state_text: str, q: Question, *, prefill_assistant: bool) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": STATE_SYSTEM_TEMPLATE.format(state=state_text)},
        {"role": "user", "content": question_user_message(q)},
    ]
    if prefill_assistant:
        messages.append({"role": "assistant", "content": "<think></think>"})
    return messages
