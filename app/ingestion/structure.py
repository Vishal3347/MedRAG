"""Turn raw lines into paragraph / heading blocks (structure-aware chunking input)."""
from __future__ import annotations

import re
from collections import Counter

from app.ingestion.cleaning import clean_text, is_running_line, join_lines
from app.ingestion.models import Block, Line, RawPage

CHAPTER_RE = re.compile(r"^(chapter|part|unit)\s+(?:[ivxlc]+|\d+)\b[.:]?", re.I)
CAPTION_RE = re.compile(r"^(fig(?:ure)?s?\.?|table|box|plate|panel)\s*[\dA-Z]", re.I)
SENTENCE_END = re.compile(r"[.!?:;)\]\"'”’]$")


def compute_body_size(pages: list[RawPage]) -> float:
    counts: Counter[float] = Counter()
    for page in pages:
        for line in page.lines:
            if line.size > 0:
                counts[round(line.size, 1)] += len(line.text.strip())
    return counts.most_common(1)[0][0] if counts else 0.0


def classify_heading(text: str, size: float, bold: bool, n_lines: int, body: float) -> tuple[bool, int]:
    n = len(text)
    if n < 3 or n > 160 or CAPTION_RE.match(text) or re.fullmatch(r"[\d\W]+", text):
        return False, 0
    ratio = size / body if size > 0 and body > 0 else 1.0
    if n <= 120 and ((CHAPTER_RE.match(text) and (ratio >= 1.1 or bold)) or ratio >= 1.5):
        return True, 1
    if ratio >= 1.12 and n_lines <= 3 and not text.endswith((".", ",")):
        return True, 2
    if bold and n <= 90 and n_lines <= 2 and ratio >= 0.95 and not text.endswith((".", ",", ";")):
        return True, 2
    return False, 0


def _group_lines(lines: list[Line]) -> list[list[Line]]:
    groups: list[list[Line]] = []
    for line in lines:
        if groups and groups[-1][0].block == line.block:
            groups[-1].append(line)
        else:
            groups.append([line])
    return groups


def build_blocks(pages: list[RawPage], body_size: float = 0.0, running_keys: set[str] | None = None) -> list[Block]:
    running_keys = running_keys or set()
    raw: list[Block] = []
    for page in pages:
        lines = [
            l
            for l in page.lines
            if l.text.strip() and not is_running_line(l.text, l.y, page.height, l.size, body_size, running_keys)
        ]
        for group in _group_lines(lines):
            text = join_lines([clean_text(l.text) for l in group])
            if not text:
                continue
            total = sum(len(l.text) for l in group) or 1
            size = sum(l.size * len(l.text) for l in group) / total
            bold = all(l.bold for l in group)
            is_heading, level = classify_heading(text, size, bold, len(group), body_size)
            raw.append(Block(text, page.number, page.number, is_heading, level))

    merged: list[Block] = []
    for block in raw:
        prev = merged[-1] if merged else None
        if prev is not None:
            # A paragraph interrupted by a layout/page break continues in the next block.
            if (
                not prev.is_heading
                and not block.is_heading
                and not SENTENCE_END.search(prev.text)
                and block.text[:1].islower()
            ):
                prev.text = join_lines([prev.text, block.text])
                prev.page_end = block.page_end
                continue
            # "Chapter 12" followed by its title on the next line/page: one chapter heading.
            if (
                prev.is_heading
                and block.is_heading
                and prev.level == 1
                and block.level == 1
                and block.page_start - prev.page_end <= 1
            ):
                prev.text = f"{prev.text.rstrip(':.')}: {block.text}"
                prev.page_end = block.page_end
                continue
        merged.append(block)
    return merged
