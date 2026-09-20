"""Token-budgeted option labels: map each answer option to a distinct single
answer token without hardcoding tokenizer specifics.

For small option counts (<= 26) we use letters A..Z — robust on every
instruct tokenizer and self-describing in the prompt. For larger sets we
fall back to digit tokens (0..9 then multi-char digit strings), which stay
collision-free up to the 255-option Jev limit as long as the backend can
return >= top_logprobs(=len(options)) entries.
"""

from __future__ import annotations

from typing import List

LETTERS = [chr(ord("A") + i) for i in range(26)]


def labels_for(n: int) -> List[str]:
    if n <= 0:
        return []
    if n <= len(LETTERS):
        return LETTERS[:n]
    # digit tokens: "0".."9", "10", "11", ... up to 255
    return [str(i) for i in range(n)]


def verify_single_token(labels: List[str], token_ids_per_label: List[object]) -> List[str]:
    """Warn (by returning fallback labels) if any label tokenizes to >1 token.

    ``token_ids_per_label`` entries are lists of ids produced by the caller's
    tokenizer probe; a label with more than one id is replaced by its index
    string, which the prompt text also shows. The bridge itself never assumes
    this succeeded — scoring re-checks via the backend.
    """
    out: List[str] = []
    for i, (label, ids) in enumerate(zip(labels, token_ids_per_label)):
        out.append(label if ids is not None and len(ids) == 1 else str(i))
    return out
