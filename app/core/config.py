"""Typed settings loaded from .env."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    All values are loaded from environment variables (.env in local dev).
    Nothing hardcoded — this is the single source of truth for tunables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- runtime ---
    app_env: Literal["local", "staging", "prod"] = "local"
    log_level: str = "INFO"
    log_json: bool = True

    # --- llm provider routing ---
    llm_provider: Literal["ollama", "openai", "nim"] = "ollama"

    # Ollama (default, local, $0)
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "qwen3.5:9b"
    ollama_embed_model: str = "nomic-embed-text"

    # OpenAI (optional, used for cost-comparison in demo)
    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    # NIM (optional)
    nim_api_key: str = ""
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"

    # --- cost table (USD per 1K tokens) ---
    # Used by cost middleware to estimate $ per request.
    # Ollama is local so all its multipliers are 0 — but we still count tokens
    # to show "would have cost $X on OpenAI".
    openai_gpt4o_mini_input_per_1k: float = 0.00015
    openai_gpt4o_mini_output_per_1k: float = 0.0006
    openai_embed_small_per_1k: float = 0.00002

    # --- qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "properties"
    qdrant_ef: int = Field(default=64, ge=4, le=512)
    qdrant_m: int = Field(default=16, ge=4, le=64)

    # --- redis ---
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86400  # 24h

    # --- search pipeline ---
    retrieval_top_k: int = 50
    rerank_top_k: int = 10
    rerank_batch_size: int = 10

    @property
    def is_local(self) -> bool:
        return self.app_env == "local"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton. Use this everywhere instead of `Settings()`."""
    return Settings()
