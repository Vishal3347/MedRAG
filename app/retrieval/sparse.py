"""BM25-style sparse vectors for Qdrant.

Documents are encoded client-side with the BM25 term-frequency component; Qdrant's `IDF` modifier on
the sparse vector field supplies the inverse document frequency at query time, which keeps the index
correct as documents are added. Terms are hashed into a 32-bit index space (no vocabulary file).
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter

K1 = 1.2
B = 0.75
AVG_DOC_LEN = 300.0  # typical chunk length after stop-word removal

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'/.][a-z0-9]+)*")
_SPLIT_RE = re.compile(r"[-'/.]")

# Deliberately tiny: articles, auxiliaries and prepositions only. Negations (no, not, without),
# units and numbers are kept because they carry meaning in medical text.
STOPWORDS = frozenset(
    "a an the and or of in on at to for with by from as is are was were be been being it its this that these those "
    "which who whom whose than then so such into over under about between within during also can may might "
    "could would should will shall has have had do does did".split()
)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        parts = _SPLIT_RE.split(raw)
        if len(parts) == 1:
            if raw not in STOPWORDS and (len(raw) > 1 or raw.isdigit()):
                tokens.append(raw)
            continue
        tokens.append(raw)  # keep the compound ("beta-blocker", "mg/dl", "5.5")
        if all(p.isalpha() for p in parts):
            tokens.extend(p for p in parts if len(p) > 1 and p not in STOPWORDS)
    return tokens


def term_index(term: str) -> int:
    return int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest(), "big")


def encode_document(text: str) -> tuple[list[int], list[float]]:
    tokens = tokenize(text)
    if not tokens:
        return [term_index("__empty__")], [1.0]
    length = len(tokens)
    weights: dict[int, float] = {}
    for term, tf in Counter(tokens).items():
        w = tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / AVG_DOC_LEN))
        idx = term_index(term)
        weights[idx] = weights.get(idx, 0.0) + w
    indices = sorted(weights)
    return indices, [round(weights[i], 5) for i in indices]


def encode_query(text: str) -> tuple[list[int], list[float]]:
    terms = sorted({term_index(t) for t in tokenize(text)})
    return terms, [1.0] * len(terms)
