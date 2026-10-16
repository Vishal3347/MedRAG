"""Embedders (lazy-loaded, thread-safe): sentence-transformers (default, used for ingestion) and a light ONNX
backend (fastembed) for small hosts. Both use the same BAAI/bge-small-en-v1.5 weights and the same query prefix."""
from __future__ import annotations

import logging
import threading
from typing import Callable

log = logging.getLogger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str, *, dim: int | None = None, batch_size: int = 32, max_seq_length: int = 512):
        self.model_name = model_name
        self.expected_dim = dim
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self._model = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    log.info("loading embedding model", extra={"model": self.model_name})
                    model = SentenceTransformer(self.model_name, device="cpu")
                    model.max_seq_length = min(self.max_seq_length, model.max_seq_length or self.max_seq_length)
                    dim = model.get_sentence_embedding_dimension()
                    if self.expected_dim and dim != self.expected_dim:
                        raise RuntimeError(
                            f"{self.model_name} produces {dim}-dim vectors but EMBEDDING_DIM={self.expected_dim}"
                        )
                    self._model = model
        return self._model

    def warmup(self) -> None:
        self._load()
        self.embed_query("warmup")

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    @property
    def max_tokens(self) -> int:
        return int(self._load().max_seq_length)

    def embed_documents(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        model = self._load()
        with self._lock:
            vectors = model.encode(
                texts,
                batch_size=batch_size or self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        prefix = BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return self.embed_documents([prefix + text], batch_size=1)[0]

    def token_counter(self) -> Callable[[str], int]:
        """Exact token counts using the embedding model's own tokenizer."""
        tokenizer = self._load().tokenizer
        return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


class FastEmbedEmbedder:
    """ONNX runtime embedder (no torch, a few hundred MB of RAM). Query-time only: ingestion needs the exact
    tokenizer-based `token_counter`, so ingest with the sentence-transformers backend."""

    def __init__(self, model_name: str, *, dim: int | None = None, batch_size: int = 32,
                 max_seq_length: int = 512, cache_dir: str | None = None):
        self.model_name = model_name
        self.expected_dim = dim
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.cache_dir = cache_dir or None
        self._model = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    log.info("loading embedding model (fastembed)", extra={"model": self.model_name})
                    model = TextEmbedding(self.model_name, cache_dir=self.cache_dir)
                    dim = len(next(iter(model.embed(["dimension probe"]))))
                    if self.expected_dim and dim != self.expected_dim:
                        raise RuntimeError(
                            f"{self.model_name} produces {dim}-dim vectors but EMBEDDING_DIM={self.expected_dim}"
                        )
                    self._dim = dim
                    self._model = model
        return self._model

    def warmup(self) -> None:
        self._load()
        self.embed_query("warmup")

    @property
    def dimension(self) -> int:
        self._load()
        return self._dim

    @property
    def max_tokens(self) -> int:
        return self.max_seq_length

    def embed_documents(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        import numpy as np

        model = self._load()
        with self._lock:
            vectors = list(model.embed(texts, batch_size=batch_size or self.batch_size))
        out = []
        for v in vectors:  # unit length, like sentence-transformers' normalize_embeddings=True
            v = np.asarray(v, dtype="float32")
            norm = float(np.linalg.norm(v)) or 1.0
            out.append((v / norm).tolist())
        return out

    def embed_query(self, text: str) -> list[float]:
        prefix = BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return self.embed_documents([prefix + text], batch_size=1)[0]

    def token_counter(self) -> Callable[[str], int]:
        raise NotImplementedError("Ingest with EMBEDDING_BACKEND=sentence-transformers (needs the model's tokenizer).")


def make_embedder(settings):
    backend = getattr(settings, "embedding_backend", "sentence-transformers")
    if backend == "fastembed":
        return FastEmbedEmbedder(
            settings.embedding_model, dim=settings.embedding_dim, batch_size=settings.embedding_batch_size,
            cache_dir=getattr(settings, "fastembed_cache_dir", "") or None,
        )
    if backend == "sentence-transformers":
        return Embedder(settings.embedding_model, dim=settings.embedding_dim, batch_size=settings.embedding_batch_size)
    raise ValueError(f"Unknown EMBEDDING_BACKEND '{backend}' (use sentence-transformers or fastembed)")
