"""Search endpoint.

Pipeline:
  1. Intent parsing  — LLM (qwen3.5:9b) → structured Intent
  2. Retrieval       — embed query → Qdrant hybrid search → top-K
  3. Rerank + explain (TODO)
"""

import time

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import get_settings
from app.core.cost import current_usage
from app.core.logging import get_logger, request_id_ctx
from app.models.search import Intent, PropertyHit, SearchRequest, SearchResponse
from app.services.llm import LLMClient, get_embed, get_llm, LLMError
from app.services.qdrant_client import QdrantService, get_qdrant

logger = get_logger(__name__)
router = APIRouter(prefix="/search", tags=["search"])
settings = get_settings()


INTENT_SYSTEM_PROMPT = """You parse a real-estate search query in natural language into strict JSON.

The user describes their LIFE SCENARIO, not filters. Your job:
1. Identify the household type, pets, purchase intent.
2. Derive hard filters (rooms, price) only if explicit.
3. Extract implicit_needs — lifestyle hints the user didn't spell out:
   - family_with_child → ["park_nearby", "quiet", "school_nearby"]
   - elderly → ["elevator", "clinic_nearby", "no_stairs"]
   - student → ["transit_nearby", "affordable", "shared_ok"]
   - remote_worker → ["quiet", "good_internet_area"]
4. Extract style_preferences (bright, modern, spacious, etc.) if mentioned.
5. Write a short rationale (1 sentence) explaining how you understood the query.

Return ONLY valid JSON with these fields:
  household: one of [single, couple, family_with_child, family_with_children, elderly, student, investor, unknown]
  pets: list of strings
  intent_kind: one of [buy, rent, invest, unknown]
  min_rooms, max_rooms: int or null
  max_price_usd: float or null
  preferred_locations: list of strings
  implicit_needs: list of strings
  style_preferences: list of strings
  rationale: short string
"""


def _compose_retrieval_text(query: str, intent: Intent) -> str:
    """Compose the text we embed for retrieval.

    We don't just embed the raw query — we enrich it with derived needs so the
    vector search captures implicit lifestyle cues.
    """
    parts = [query]
    if intent.implicit_needs:
        parts.append("Lifestyle needs: " + ", ".join(intent.implicit_needs))
    if intent.style_preferences:
        parts.append("Style: " + ", ".join(intent.style_preferences))
    if intent.household != "unknown":
        parts.append(f"For {intent.household.replace('_', ' ')}")
    return ". ".join(parts)


@router.post("", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    request: Request,
    llm: LLMClient = Depends(get_llm),
    embed: LLMClient = Depends(get_embed),
    qdrant: QdrantService = Depends(get_qdrant),
) -> SearchResponse:
    stages: dict[str, float] = {}

    # --- Stage 1: intent parsing ---
    t0 = time.perf_counter()
    try:
        raw = await llm.parse_json(
            system=INTENT_SYSTEM_PROMPT,
            user=req.query,
            temperature=0.0,
        )
        intent = Intent.model_validate(raw)
    except LLMError as e:
        logger.error("intent_parse_failed", error=str(e))
        raise HTTPException(status_code=502, detail=f"Intent parse failed: {e}") from e
    except Exception as e:  # pydantic ValidationError etc.
        logger.warning("intent_validation_failed", error=str(e))
        intent = Intent(rationale=f"fallback: {e.__class__.__name__}")
    stages["intent_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # --- Stage 2: retrieval (embed query + Qdrant hybrid search) ---
    t0 = time.perf_counter()
    retrieval_text = _compose_retrieval_text(req.query, intent)
    try:
        vectors = await embed.embed(inputs=retrieval_text)
        qvec = vectors[0]
    except LLMError as e:
        logger.error("embed_failed", error=str(e))
        raise HTTPException(status_code=502, detail=f"Embedding failed: {e}") from e

    hits_raw = await qdrant.search(
        vector=qvec,
        top_k=req.top_k,
        max_price_usd=intent.max_price_usd,
        min_rooms=intent.min_rooms,
        max_rooms=intent.max_rooms,
        preferred_locations=intent.preferred_locations,
    )
    stages["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    hits: list[PropertyHit] = []
    for h in hits_raw:
        p = h["payload"]
        hits.append(
            PropertyHit(
                property_id=str(h["id"]),
                title=str(p.get("title", "")),
                location=str(p.get("location", "")),
                price_usd=p.get("price_usd"),
                rooms=p.get("rooms"),
                score=h["score"],
            )
        )

    # TODO Stage 3: LLM rerank with explanations

    usage = current_usage()
    cost_usd = usage.total_real_usd if usage else 0.0
    shadow_usd = usage.total_shadow_openai_usd if usage else 0.0

    return SearchResponse(
        request_id=request_id_ctx.get(),
        query=req.query,
        intent=intent,
        hits=hits,
        cost_usd=cost_usd,
        shadow_openai_usd=shadow_usd,
        latency_ms=sum(stages.values()),
        stages_ms=stages,
    )
