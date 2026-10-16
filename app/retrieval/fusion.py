from __future__ import annotations

import re
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def rrf_fuse(
    rankings: Sequence[Sequence[str]], *, k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d)), ranks starting at 1."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for w, ranking in zip(weights, rankings):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
            first_seen.setdefault(doc_id, len(first_seen))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    if len(words) <= n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def dedupe(
    items: Sequence[T],
    *,
    text_of: Callable[[T], str],
    hash_of: Callable[[T], str | None],
    threshold: float = 0.85,
) -> list[T]:
    """Drop exact duplicates (same content hash) and near-duplicates (shingle Jaccard >= threshold).
    Input order is preserved, so higher-ranked items win."""
    kept: list[T] = []
    seen_hashes: set[str] = set()
    kept_shingles: list[set[tuple[str, ...]]] = []
    for item in items:
        h = hash_of(item)
        if h and h in seen_hashes:
            continue
        sh = _shingles(text_of(item))
        if sh and any(len(sh & o) / len(sh | o) >= threshold for o in kept_shingles if o):
            continue
        if h:
            seen_hashes.add(h)
        kept.append(item)
        kept_shingles.append(sh)
    return kept
