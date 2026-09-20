"""Pydantic-free request/response schemas (plain dataclasses + dicts).

Kept dependency-light on purpose: the Jev wire format is JSON in / JSON out,
and clients send arbitrarily keyed question maps, so dicts are normalized
into typed views instead of the other way around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

QUESTION_TYPES = ("choice", "noul", "score")
CHOICE_MAX_OPTIONS = 255
SCORE_MAX_LEVELS = 10


class JevBridgeError(Exception):
    """Base error with an HTTP status and a Jev-ish error payload."""

    def __init__(self, message: str, *, status: int = 400, err_type: str = "invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.err_type = err_type

    def to_payload(self) -> Dict[str, Any]:
        return {"error": {"message": str(self), "type": self.err_type}}


@dataclass
class Question:
    """One normalized question."""

    key: str
    type: str
    instructions: str = ""
    # choice: label -> description; noul: {"true": ..., "false": ...} or empty;
    # score: ordered level descriptions (lowest first).
    criteria: Any = None
    # runtime-filled scoring tokens (text of the single answer token per option)
    option_tokens: List[str] = field(default_factory=list)
    option_labels: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.type not in QUESTION_TYPES:
            raise JevBridgeError(
                f"question '{self.key}' has unknown type '{self.type}' (expected one of {QUESTION_TYPES})"
            )
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not self.criteria:
                raise JevBridgeError(
                    f"question '{self.key}': choice requires non-empty 'criteria' mapping label -> description"
                )
            if len(self.criteria) > CHOICE_MAX_OPTIONS:
                raise JevBridgeError(
                    f"question '{self.key}': choice supports at most {CHOICE_MAX_OPTIONS} options "
                    f"(got {len(self.criteria)}); split into multiple questions instead"
                )
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not (2 <= len(self.criteria) <= SCORE_MAX_LEVELS):
                raise JevBridgeError(
                    f"question '{self.key}': score requires 'criteria' as an ordered list of 2..{SCORE_MAX_LEVELS} "
                    "level descriptions (lowest first)"
                )
        # noul: criteria optional ({"true": ..., "false": ...} descriptions allowed)

    def levels(self) -> List[str]:
        """Ordered answer labels."""
        if self.type == "choice":
            return [str(k) for k in self.criteria.keys()]
        if self.type == "score":
            return [str(i) for i in range(len(self.criteria))]
        return ["true", "false"]


@dataclass
class SystemOneRequest:
    model: Optional[str]
    state: Any
    questions: Dict[str, Question]
    images: List[str]
    raw: Dict[str, Any]

    @classmethod
    def parse(cls, body: Any) -> "SystemOneRequest":
        if not isinstance(body, dict):
            raise JevBridgeError("request body must be a JSON object")
        questions_raw = body.get("questions")
        if not isinstance(questions_raw, dict) or not questions_raw:
            raise JevBridgeError("'questions' must be a non-empty JSON object")
        questions: Dict[str, Question] = {}
        for key, spec in questions_raw.items():
            if not isinstance(spec, dict):
                raise JevBridgeError(f"question '{key}' must be a JSON object")
            q = Question(
                key=str(key),
                type=str(spec.get("type", "")).lower(),
                instructions=str(spec.get("instructions", "")),
                criteria=spec.get("criteria"),
            )
            q.validate()
            questions[str(key)] = q
        state = body.get("state")
        images_raw = body.get("images")
        if state is None and not images_raw:
            raise JevBridgeError("'state' is required (or supply 'images')")
        images: List[str] = []
        if images_raw is not None:
            if isinstance(images_raw, str):
                images_raw = [images_raw]
            if not isinstance(images_raw, list):
                raise JevBridgeError("'images' must be a string or an array of image references")
            images = [str(entry) for entry in images_raw]
        return cls(
            model=body.get("model"),
            state=state if state is not None else "",
            questions=questions,
            images=images,
            raw=body,
        )
