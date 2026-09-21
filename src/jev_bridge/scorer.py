"""Question scorer: one logprob call per question, fan-out with a semaphore.

Given the first-token top-logprob list, each option's score is its logprob if
the option's token appears in the top-K list, otherwise a floor value derived
from the worst observed logprob (options outside the top-K are vanishingly
unlikely relative to what the model did want). Probabilities are then a
restricted softmax over the option labels, so they always sum to 1 — the same
contract as Jev's Choice/Score answers.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from . import prompts
from .backend import OpenAICompatClient
from .images import normalize_image
from .labels import LETTERS, labels_for
from .probability import confidence_from, probabilities_map, restricted_softmax, score_answer
from .schemas import Question, SystemOneRequest

DEFAULT_TOP_K = int(os.environ.get("JEVB_TOP_K", "20"))


class QuestionScorer:
    def __init__(
        self,
        client: OpenAICompatClient,
        *,
        prefill_assistant: bool = True,
        top_k: int = DEFAULT_TOP_K,
        confidence_method: str = "linear",
    ):
        self.client = client
        self.prefill_assistant = prefill_assistant
        self.top_k = top_k
        self.confidence_method = confidence_method

    def _labels(self, q: Question) -> List[str]:
        if q.type == "noul":
            return ["true", "false"]
        n = len(q.levels())
        if q.type == "score":
            # digits 0..9 are single tokens and match Jev's probabilities/legend
            # keys ("0", "1", ...); score criteria are capped at 10 levels.
            return [str(i) for i in range(n)]
        return LETTERS[:n] if n <= len(LETTERS) else labels_for(n)

    def _request_top_k(self, n_labels: int) -> int:
        # Ask for at least enough slots to cover every label, capped by config.
        return max(self.top_k, min(n_labels, self.top_k * 2))

    @staticmethod
    def _collect(labels: List[str], top: List[Tuple[str, float]]) -> List[float]:
        """Return one logprob per label, from the observed top-token list."""
        top_map: Dict[str, float] = {}
        for tok, lp in top:
            top_map.setdefault(tok, lp)
            top_map.setdefault(tok.strip(), lp)
        floor = (min(lp for _, lp in top) - 8.0) if top else -20.0
        lps: List[float] = []
        for label in labels:
            if label in top_map:
                lps.append(top_map[label])
            elif label.lower() in top_map:
                lps.append(top_map[label.lower()])
            else:
                lps.append(floor)
        return lps

    async def score(
        self, state_text: str, q: Question, image_urls: Optional[List[str]] = None
    ) -> Tuple[Dict, Dict[str, Any]]:
        labels = self._labels(q)
        q.option_labels = labels
        messages = prompts.build_messages(
            state_text,
            q,
            prefill_assistant=self.prefill_assistant,
            image_urls=image_urls or None,
        )
        _text, top, usage = await self.client.first_token_logprobs(
            messages, self._request_top_k(len(labels))
        )
        probs = restricted_softmax(self._collect(labels, top))

        answer: Dict = {"type": q.type}
        if q.type == "choice":
            best = max(range(len(labels)), key=lambda i: probs[i])
            answer["choice"] = q.levels()[best]
            answer["confidence"] = round(
                confidence_from(probs, method=self.confidence_method), 4
            )
            # Jev keys choice probabilities by the criteria option names
            # (e.g. {"billing": 0.94, "shipping": 0.06}), not by prompt labels.
            answer["probabilities"] = probabilities_map(q.levels(), probs)
        elif q.type == "noul":
            # labels order is [true, false]; the answer is p(true)
            answer["noul"] = round(probs[0], 4)
        else:  # score
            answer["score"] = round(score_answer(probs), 4)
            answer["confidence"] = round(
                confidence_from(probs, method=self.confidence_method), 4
            )
            answer["probabilities"] = probabilities_map(labels, probs)
            answer["legend"] = {label: str(desc) for label, desc in zip(labels, q.criteria)}
        return answer, usage


class PrimedStates:
    """Remembers which state prefixes were warmed recently.

    Priming costs one sequential round trip, so it is only worth paying once per
    state per cache lifetime — server-side prefix caches live on the order of
    minutes (OpenAI documents eviction after 5–10 min of inactivity; vLLM and
    SGLang evict under memory pressure). Bounded LRU so a long-running server
    does not accumulate state hashes.
    """

    def __init__(self, ttl: float, max_entries: int = 64) -> None:
        self._ttl = ttl
        self._max = max_entries
        self._seen: "OrderedDict[str, float]" = OrderedDict()
        self._lock = asyncio.Lock()

    async def claim(self, state_text: str) -> bool:
        """Return True (and record the claim) when this state should be primed."""
        key = hashlib.sha256(state_text.encode("utf-8")).hexdigest()[:16]
        async with self._lock:
            now = time.monotonic()
            seen_at = self._seen.get(key)
            if seen_at is not None and now - seen_at < self._ttl:
                self._seen.move_to_end(key)
                return False
            self._seen[key] = now
            self._seen.move_to_end(key)
            while len(self._seen) > self._max:
                self._seen.popitem(last=False)
            return True


async def score_all(
    scorer: QuestionScorer,
    request: SystemOneRequest,
    *,
    max_concurrency: int = 8,
    allow_local_images: bool = False,
    prime_prefix: bool = False,
    prime_min_state_chars: int = 1200,
    primed: Optional[PrimedStates] = None,
) -> Tuple[Dict[str, Dict], Dict[str, int]]:
    """Score all questions concurrently.

    Returns ``(answers, usage)`` where usage aggregates backend-reported
    prompt/completion tokens across the per-question calls.

    Prefix caching: every question of a request shares the same
    ``[system + images + STATE]`` bytes, and the question block is the only
    difference. On a *cold* prefix cache that shared prefill is computed once per
    concurrent call, so firing N questions at once pays the prefill N times.
    With ``prime_prefix`` on, the first question goes out alone (for states at
    least ``prime_min_state_chars`` long, and at most once per state per
    ``ttl``), which warms the prefix so the remaining questions hit it.
    """
    state_text = prompts.serialize_state(request.state)
    sem = asyncio.Semaphore(max_concurrency)
    image_urls = [
        normalize_image(ref, allow_local=allow_local_images)
        for ref in request.images
    ]

    async def run(key: str, q: Question) -> Tuple[str, Dict, Dict[str, Any]]:
        async with sem:
            answer, usage = await scorer.score(state_text, q, image_urls)
        return key, answer, usage

    pending = list(request.questions.items())
    results: List[Tuple[str, Dict, Dict[str, Any]]] = []
    did_prime = False
    if (
        prime_prefix
        and primed is not None
        and len(pending) > 1
        and len(state_text) >= prime_min_state_chars
        and await primed.claim(state_text)
    ):
        key, q = pending.pop(0)
        results.append(await run(key, q))
        did_prime = True
    results.extend(await asyncio.gather(*(run(k, q) for k, q in pending)))

    answers: Dict[str, Dict] = {}
    prompt_tokens = completion_tokens = 0
    for key, answer, usage in results:
        answers[key] = answer
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
    usage_out = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "questions": len(answers),
    }
    if did_prime:
        # operator-visible: this request paid one sequential round trip to warm the cache
        usage_out["prefix_primed"] = 1
    return answers, usage_out
