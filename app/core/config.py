"""Application settings, loaded from environment variables and an optional `.env` file."""
from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Qdrant
    qdrant_url: str = ""
    qdrant_api_key: SecretStr = SecretStr("")
    qdrant_collection: str = "medical_textbook"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 32
    embedding_backend: str = "sentence-transformers"  # or "fastembed" (ONNX, small hosts; query-time only)
    fastembed_cache_dir: str = ""

    # Ingestion
    chunk_tokens: int = 600
    chunk_overlap_tokens: int = 80

    # Retrieval
    candidates_per_retriever: int = 20
    context_chunks: int = 5
    rrf_k: int = 60
    min_dense_score: float = 0.45

    # LLM router
    llm_chain: str = "groq:openai/gpt-oss-120b,groq:openai/gpt-oss-20b,gemini:gemini-3.8-flash,ollama:llama3.1:8b"
    groq_api_key: SecretStr = SecretStr("")
    gemini_api_key: SecretStr = SecretStr("")
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.2
    llm_max_output_tokens: int = 900
    llm_max_input_tokens: int = 4500  # prompt budget (estimate); see build_messages
    llm_first_token_timeout_s: float = 20.0
    llm_stall_timeout_s: float = 30.0
    llm_connect_timeout_s: float = 10.0
    stream_commit_chars: int = 120

    # API
    admin_api_key: SecretStr = SecretStr("")
    cors_origins: str = "http://localhost:3000"
    rate_limit_per_minute: int = 20
    trust_proxy: bool = False
    max_history_messages: int = 4
    log_level: str = "INFO"
    log_json: bool = True

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
