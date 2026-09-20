"""Prompt construction.

Message layout (one user turn, images first):

    system:    static instruction, no state
    user:      [image parts...] + "STATE:\n{state}\n\n{question block}"
    assistant: "<think></think>" prefill (optional)

Why one user message instead of ``system(state) + user(question)``:
OpenAI-compatible templates reject images in system messages, and some
templates (Llama-style) require strict user/assistant alternation. Putting
everything in a single user turn keeps both happy. Prefix caching is
unaffected: the differing part (the question) is at the *end* of the token
stream, so SGLang's radix cache / vLLM's APC still reuse the
[system + images + state] prefill across every question of a request.

The assistant prefill suppresses thinking-model preamble (Qwen3.x emits
``<think>`` otherwise). Disable with ``JEVB_PREFILL_ASSISTANT=0``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .labels import LETTERS
from .schemas import JevBridgeError, Question

SYSTEM_INSTRUCTION = """\
You are a decision engine. You never explain, never apologize, and never \
write sentences. You read the STATE and answer the user's question by \
outputting EXACTLY ONE answer label and nothing else."""


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


def user_message(
    state_text: str,
    question_text: str,
    image_urls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    body = f"STATE:\n{state_text}\n\n{question_text}"
    if not image_urls:
        return {"role": "user", "content": body}
    parts: List[Dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": url}} for url in image_urls
    ]
    parts.append({"type": "text", "text": body})
    return {"role": "user", "content": parts}


def build_messages(
    state_text: str,
    q: Question,
    *,
    prefill_assistant: bool,
    image_urls: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        user_message(state_text, question_user_message(q), image_urls),
    ]
    if prefill_assistant:
        messages.append({"role": "assistant", "content": "<think></think>"})
    return messages
