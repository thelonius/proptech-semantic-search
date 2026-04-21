"""DTO models for /search and /ingest."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Intent(BaseModel):
    """Structured intent parsed from natural-language query."""

    household: Literal[
        "single",
        "couple",
        "family_with_child",
        "family_with_children",
        "elderly",
        "student",
        "investor",
        "unknown",
    ] = "unknown"
    pets: list[str] = Field(default_factory=list)
    intent_kind: Literal["buy", "rent", "invest", "unknown"] = "unknown"

    # Hard filters — mapped directly to Qdrant payload filters
    min_rooms: int | None = None
    max_rooms: int | None = None
    max_price_usd: float | None = None
    preferred_locations: list[str] = Field(default_factory=list)

    # Soft preferences — used by reranker prompt only
    implicit_needs: list[str] = Field(default_factory=list)  # ["park_nearby","quiet"]
    style_preferences: list[str] = Field(default_factory=list)  # ["modern","bright"]

    # LLM explanation of how it understood the query — good for debugging
    rationale: str = ""

    # Some LLMs (llama-3.1-8b) emit `null` for empty list fields instead of `[]`.
    # Coerce None -> [] at parse time so we don't need a schema rewrite.
    @field_validator("pets", "preferred_locations", "implicit_needs", "style_preferences", mode="before")
    @classmethod
    def _none_to_empty(cls, v: object) -> object:
        return [] if v is None else v


class PropertyHit(BaseModel):
    """One search result."""

    property_id: str
    title: str
    location: str
    price_usd: float | None = None
    rooms: int | None = None
    score: float
    rerank_score: float | None = None
    reasons: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    with_explain: bool = True


class SearchResponse(BaseModel):
    request_id: str
    query: str
    intent: Intent
    hits: list[PropertyHit]
    # Mirror of X-Cost-USD headers for convenience
    cost_usd: float
    shadow_openai_usd: float
    latency_ms: float
    stages_ms: dict[str, float] = Field(default_factory=dict)
