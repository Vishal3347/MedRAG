"""Cheap follow-up handling: no LLM call, so no quota is spent on query rewriting."""
from __future__ import annotations

import re
from typing import Sequence

_FOLLOW_UP = re.compile(
    r"^(and|also|what about|how about|why|how|what else|can you|could you|explain|tell me more|"
    r"is it|are they|does it|do they)\b|\b(it|its|they|them|their|this|that|these|those|he|she)\b",
    re.I,
)


def build_retrieval_query(history: Sequence[tuple[str, str]], question: str, max_chars: int = 400) -> str:
    """history: (role, content) pairs, oldest first. Short or pronoun-heavy follow-ups are
    prefixed with the previous user question so retrieval has the topic."""
    question = question.strip()
    previous = next((c for r, c in reversed(history) if r == "user" and c.strip()), None)
    if previous and (len(question.split()) <= 6 or _FOLLOW_UP.search(question)):
        return f"{previous.strip()} {question}"[:max_chars]
    return question[:max_chars]
