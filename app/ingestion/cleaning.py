"""Conservative text cleaning.

Deliberately NOT done: NFKC normalisation (it would turn mm² into mm2 and µ into μ), case folding,
stop-word removal, number or unit rewriting, anything that could touch negations ("no", "not",
"without"), drug names, doses or scientific notation.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from app.ingestion.models import RawPage

_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"})
_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"), None)
_INLINE_WS = re.compile(r"[ \t\u00a0\u2009\u202f]+")
_HYPHEN_BREAK = re.compile(r"[A-Za-z]{3,}-$")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_LIGATURES).translate(_INVISIBLE)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INLINE_WS.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def join_lines(lines: list[str]) -> str:
    """Join wrapped lines into one paragraph, repairing end-of-line hyphenation."""
    out = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not out:
            out = line
        elif out.endswith("-") and line[0].islower():
            if _HYPHEN_BREAK.search(out):
                out = out[:-1] + line  # "hypo-" + "glycemia" -> "hypoglycemia"
            else:
                out = out + line  # "T-" + "cell" -> "T-cell"
        else:
            out = out + " " + line
    return out


def _norm_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def in_margin(y: float, height: float, margin: float = 0.085) -> bool:
    return y >= 0 and height > 0 and (y < height * margin or y > height * (1 - margin))


def find_running_keys(pages: list[RawPage], body_size: float = 0.0, min_pages: int = 5) -> set[str]:
    """Margin lines (running headers/footers, page numbers) that repeat across many pages."""
    counts: Counter[str] = Counter()
    for page in pages:
        seen: set[str] = set()
        for line in page.lines:
            if not in_margin(line.y, page.height):
                continue
            if body_size and line.size > body_size * 1.3:
                continue
            key = _norm_key(line.text)
            if key and key not in seen:
                seen.add(key)
                counts[key] += 1
    return {k for k, c in counts.items() if c >= min_pages}


def is_running_line(text: str, y: float, height: float, size: float, body_size: float, keys: set[str]) -> bool:
    if not keys or not in_margin(y, height):
        return False
    if body_size and size > body_size * 1.3:
        return False
    return _norm_key(text) in keys
