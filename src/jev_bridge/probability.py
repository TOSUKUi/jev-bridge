"""Probability math for one-token answer distributions.

All options are scored from the same first-token distribution, so the
computation is a restricted softmax over the labels' logprobs followed by
small derived statistics:

- probabilities: softmax over the observed label logprobs
- choice:       argmax label, confidence
- noul:         p(true)  (the ``noul`` field)
- score:        expected level  sum(i * p_i)  — can land between rungs, like Jev
- confidence:   by default the linear rescaling (K*p_max - 1)/(K - 1), which
                best matches confidence values observable in TypeSafe's
                published Choice/Score examples ({0:.7,1:.7,2:.3} -> 0.54);
                ``max_prob`` and ``entropy`` (1 - H/ln K) are available.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence


def restricted_softmax(logprobs: Sequence[float]) -> List[float]:
    m = max(logprobs)
    exps = [math.exp(lp - m) for lp in logprobs]
    total = sum(exps)
    return [e / total for e in exps]


def confidence_from(
    probs: Sequence[float],
    method: str = "linear",
    p_max: Optional[float] = None,
) -> float:
    """Derive a Jev-style confidence (0..1) from the answer distribution."""
    if not probs:
        return 0.0
    if p_max is None:
        p_max = max(probs)
    k = len(probs)
    if method == "max_prob":
        c = p_max
    elif method == "entropy":
        if k <= 1:
            return 1.0
        h = -sum(p * math.log(p) for p in probs if p > 0.0)
        c = 1.0 - h / math.log(k)
    else:  # linear (default): (K*p_max - 1)/(K - 1), uniform -> 0, peaky -> 1
        if k <= 1:
            return 1.0
        c = (k * p_max - 1.0) / (k - 1.0)
    return min(1.0, max(0.0, c))


def score_answer(probs: Sequence[float]) -> float:
    """Probability-weighted position on the ordered scale: sum(i * p_i)."""
    return sum(i * p for i, p in enumerate(probs))


def probabilities_map(labels: Sequence[str], probs: Sequence[float]) -> Dict[str, float]:
    return {label: round(p, 4) for label, p in zip(labels, probs)}


def legend_map(labels: Sequence[str], descriptions: Sequence[str]) -> Dict[str, str]:
    return {label: desc for label, desc in zip(labels, descriptions)}
