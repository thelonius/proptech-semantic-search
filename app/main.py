"""FastAPI entry point.

Wiring:
  1. Config (pydantic-settings)
  2. Logging (structlog, JSON)
  3. Middleware: request context + cost tracking + CORS
  4. Routers: /healthz /readyz /metrics /search /ingest
  5. Lifespan: warm up clients, close them on shutdown
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import health, ingest, search
from app.core.config import get_settings
from app.core.cost import RequestContextMiddleware
from app.core.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(level=settings.log_level, json_output=settings.log_json)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info(
        "app_starting",
        version=__version__,
        env=settings.app_env,
        llm_provider=settings.llm_provider,
        ollama_llm=settings.ollama_llm_model,
        ollama_embed=settings.ollama_embed_model,
    )
    yield
    logger.info("app_stopping")


app = FastAPI(
    title="PropTech Semantic Search",
    version=__version__,
    description=(
        "Lifestyle-based real estate search. "
        "Intent parsing (LLM) → multi-vector retrieval (Qdrant) → LLM rerank + explain."
    ),
    lifespan=lifespan,
)

# Middleware (order matters: outermost first)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Request-ID",
        "X-Duration-Ms",
        "X-Cost-USD",
        "X-Cost-Shadow-OpenAI-USD",
        "X-LLM-Calls",
        "X-LLM-Tokens-In",
        "X-LLM-Tokens-Out",
    ],
)

# Routers
app.include_router(health.router)
app.include_router(search.router)
app.include_router(ingest.router)


@app.get("/", tags=["meta"])
async def root() -> dict[str, object]:
    return {
        "name": "proptech-semantic-search",
        "version": __version__,
        "docs": "/docs",
        "metrics": "/metrics",
        "providers": {
            "llm": settings.llm_provider,
            "embed": settings.embed_provider,
            "llm_model": (
                settings.nim_llm_model if settings.llm_provider == "nim"
                else settings.openai_llm_model if settings.llm_provider == "openai"
                else settings.ollama_llm_model
            ),
            "embed_model": (
                settings.nim_embed_model if settings.embed_provider == "nim"
                else settings.openai_embed_model if settings.embed_provider == "openai"
                else settings.ollama_embed_model
            ),
        },
    }
