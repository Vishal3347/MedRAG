from __future__ import annotations

import hashlib
from dataclasses import dataclass


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


@dataclass(slots=True)
class Line:
    text: str
    size: float = 0.0  # 0.0 means "unknown" (OCR / fallback extraction)
    bold: bool = False
    block: int = 0
    y: float = -1.0  # -1.0 means "unknown"


@dataclass(slots=True)
class RawPage:
    number: int  # 1-based PDF page index
    height: float
    lines: list[Line]
    source: str = "text"  # text | plumber | ocr | empty


@dataclass(slots=True)
class Block:
    text: str
    page_start: int
    page_end: int
    is_heading: bool = False
    level: int = 0  # 1 = chapter, 2 = section


@dataclass(slots=True)
class Chunk:
    document_id: str
    document_title: str
    chunk_index: int
    page_start: int
    page_end: int
    chapter: str
    section_heading: str
    text: str
    token_count: int
    content_hash: str

    @property
    def embed_text(self) -> str:
        """Heading path + text: what actually gets embedded and BM25-indexed."""
        path = " > ".join(p for p in (self.chapter, self.section_heading) if p)
        return f"{path}\n{self.text}" if path else self.text

    def payload(self) -> dict:
        return {
            "schema_version": 1,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chapter": self.chapter,
            "section_heading": self.section_heading,
            "chunk_index": self.chunk_index,
            "content_hash": self.content_hash,
            "token_count": self.token_count,
            "text": self.text,
        }
