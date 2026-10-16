"""Token-aware, structure-aware chunker.

* Chunks never cross a chapter boundary and (once a section has enough text) never cross a
  section boundary.
* Section headings are kept inline ("## Heading") and the heading path is part of the embedded text,
  so a chunk always carries its context.
* Chunks are packed up to `target_tokens`; consecutive chunks inside one section overlap by about
  `overlap_tokens` (whole sentences/paragraphs where possible).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.models import Block, Chunk, hash_text
from app.ingestion.tokens import TokenCounter, approx_token_count

_ABBREVIATIONS = {
    "e.g.", "i.e.", "vs.", "dr.", "fig.", "figs.", "approx.", "no.", "eq.", "al.", "etc.", "ca.", "cf.",
    "ref.", "refs.", "mr.", "mrs.", "st.", "inc.", "vol.", "pp.", "p.", "sp.", "spp.",
}
_SENT_BOUNDARY = re.compile(r"([.!?])\s+(?=[A-Z0-9(\[“\"])")


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for match in _SENT_BOUNDARY.finditer(text):
        end = match.end(1)
        words = text[start:end].split()
        last = words[-1].lower() if words else ""
        if last in _ABBREVIATIONS or (len(last) == 2 and last[0].isalpha() and last[1] == "."):
            continue
        sentences.append(text[start:end].strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


@dataclass(slots=True)
class _Unit:
    text: str
    page_start: int
    page_end: int
    tokens: int
    is_heading: bool = False
    para_start: bool = True


class Chunker:
    def __init__(
        self,
        *,
        document_id: str,
        document_title: str,
        target_tokens: int = 600,
        overlap_tokens: int = 80,
        section_break_tokens: int = 150,
        count_tokens: TokenCounter = approx_token_count,
    ):
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        self.document_id = document_id
        self.document_title = document_title
        self.target = target_tokens
        self.overlap = overlap_tokens
        self.section_break = section_break_tokens
        self.count = count_tokens

    # ------------------------------------------------------------------ public
    def chunk(self, blocks: list[Block]) -> list[Chunk]:
        self._out: list[Chunk] = []
        self._buf: list[_Unit] = []
        self._buf_tokens = 0
        self._new_tokens = 0
        self._chapter = ""
        self._section = ""
        self._buf_chapter = ""
        self._buf_section = ""
        for block in blocks:
            if block.is_heading:
                self._on_heading(block)
            else:
                for unit in self._paragraph_units(block):
                    self._add(unit)
        self._flush()
        return self._out

    # ----------------------------------------------------------------- headings
    def _on_heading(self, block: Block) -> None:
        if block.level <= 1:
            self._flush()
            self._chapter = block.text
            self._section = ""
            return
        if self._new_tokens >= self.section_break:
            self._flush()
        elif self._new_tokens == 0:
            self._buf, self._buf_tokens = [], 0  # drop overlap carried from the previous section
        self._section = block.text
        self._add(_Unit(f"## {block.text}", block.page_start, block.page_end, self.count(block.text) + 2, True, True))

    # ------------------------------------------------------------------ packing
    def _add(self, unit: _Unit) -> None:
        if self._buf and self._buf_tokens + unit.tokens > self.target and self._new_tokens > 0:
            self._flush(carry=True)
            if self._buf_tokens + unit.tokens > self.target:
                self._buf, self._buf_tokens = [], 0
        if not self._buf:
            self._buf_chapter, self._buf_section = self._chapter, self._section
        self._buf.append(unit)
        self._buf_tokens += unit.tokens
        if not unit.is_heading:
            self._new_tokens += unit.tokens

    def _flush(self, carry: bool = False) -> None:
        trailing: list[_Unit] = []
        if carry:
            while self._buf and self._buf[-1].is_heading:
                trailing.insert(0, self._buf.pop())  # keep a heading together with the text it introduces
        emitted = self._buf
        self._emit(emitted)
        tail = self._overlap_tail(emitted) if carry else []
        self._buf = tail + trailing
        self._buf_tokens = sum(u.tokens for u in self._buf)
        self._new_tokens = 0
        self._buf_chapter, self._buf_section = self._chapter, self._section

    def _emit(self, units: list[_Unit]) -> None:
        if not units or self._new_tokens == 0:
            return
        parts: list[str] = []
        for i, u in enumerate(units):
            if i == 0:
                parts.append(u.text)
            else:
                parts.append(("\n\n" if (u.para_start or u.is_heading) else " ") + u.text)
        text = "".join(parts).strip()
        if not text:
            return
        self._out.append(
            Chunk(
                document_id=self.document_id,
                document_title=self.document_title,
                chunk_index=len(self._out),
                page_start=min(u.page_start for u in units),
                page_end=max(u.page_end for u in units),
                chapter=self._buf_chapter,
                section_heading=self._buf_section,
                text=text,
                token_count=sum(u.tokens for u in units),
                content_hash=hash_text(text),
            )
        )

    def _overlap_tail(self, units: list[_Unit]) -> list[_Unit]:
        if self.overlap <= 0 or not units:
            return []
        tail: list[_Unit] = []
        total = 0
        for u in reversed(units):
            if total + u.tokens > self.overlap:
                break
            tail.insert(0, u)
            total += u.tokens
        if not tail and not units[-1].is_heading:
            last = units[-1]
            picked: list[str] = []
            for word in reversed(last.text.split()):
                t = self.count(word)
                if total + t > self.overlap:
                    break
                picked.insert(0, word)
                total += t
            if picked:
                tail = [_Unit(" ".join(picked), last.page_start, last.page_end, total, False, False)]
        return tail

    # --------------------------------------------------------------- paragraphs
    def _paragraph_units(self, block: Block) -> list[_Unit]:
        tokens = self.count(block.text)
        if tokens <= self.target:
            return [_Unit(block.text, block.page_start, block.page_end, tokens)]
        units: list[_Unit] = []
        first = True
        for sentence in split_sentences(block.text):
            t = self.count(sentence)
            pieces = [sentence] if t <= self.target else self._split_long(sentence)
            for piece in pieces:
                units.append(_Unit(piece, block.page_start, block.page_end, self.count(piece), False, first))
                first = False
        return units

    def _split_long(self, sentence: str) -> list[str]:
        pieces: list[str] = []
        current: list[str] = []
        total = 0
        for word in sentence.split():
            t = self.count(word) or 1
            if current and total + t > self.target:
                pieces.append(" ".join(current))
                current, total = [], 0
            current.append(word)
            total += t
        if current:
            pieces.append(" ".join(current))
        return pieces
