from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=40)
    document_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_\-]{1,64}$")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class SourceOut(BaseModel):
    ref: str
    chunk_id: str
    document_id: str
    document_title: str
    chapter: str
    section_heading: str
    page_start: int
    page_end: int
    snippet: str
    score: float | None = None
