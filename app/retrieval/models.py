from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Hit:
    id: str
    score: float
    payload: dict


@dataclass(slots=True)
class PointRecord:
    id: str
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]
    payload: dict


@dataclass(slots=True)
class Candidate:
    id: str
    payload: dict
    rrf_score: float
    dense_score: float | None = None
    sparse_score: float | None = None

    @property
    def text(self) -> str:
        return str(self.payload.get("text", ""))


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[Candidate]
    confident: bool
    best_dense: float | None
    reason: str
    stats: dict = field(default_factory=dict)
