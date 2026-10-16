"""Token counting. The pipeline uses the embedding model's tokenizer when available; this
heuristic (roughly WordPiece-like) keeps chunking usable and testable without model downloads."""
from __future__ import annotations

import re
from typing import Callable

TokenCounter = Callable[[str], int]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def approx_token_count(text: str) -> int:
    return sum(1 + (len(tok) - 1) // 6 for tok in _TOKEN_RE.findall(text))
