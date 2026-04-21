"""Qdrant wrapper — covers collection lifecycle, upsert, and hybrid-ish search.

We start with a single text-vector per point (768-dim, nomic-embed-text).
Image-vector + YOLOv8 tags land in phase 2 once text path is proven end-to-end.
"""

from __future__ import annotations

import time
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from app.core import metrics
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

TEXT_VECTOR_SIZE = 768  # nomic-embed-text


class QdrantService:
    def __init__(self) -> None:
        self._client = AsyncQdrantClient(url=settings.qdrant_url, prefer_grpc=False)
        self._collection = settings.qdrant_collection

    @property
    def client(self) -> AsyncQdrantClient:
        return self._client

    @property
    def collection(self) -> str:
        return self._collection

    async def ensure_collection(self, recreate: bool = False) -> None:
        exists = await self._client.collection_exists(self._collection)
        if exists and recreate:
            await self._client.delete_collection(self._collection)
            exists = False
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=TEXT_VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
                hnsw_config=HnswConfigDiff(
                    m=settings.qdrant_m,
                    ef_construct=128,
                    full_scan_threshold=10_000,
                ),
            )
            logger.info("qdrant_collection_created", name=self._collection, size=TEXT_VECTOR_SIZE)

    async def upsert_batch(self, points: list[PointStruct]) -> None:
        if not points:
            return
        await self._client.upsert(collection_name=self._collection, points=points, wait=True)

    async def search(
        self,
        *,
        vector: list[float],
        top_k: int = 50,
        max_price_usd: float | None = None,
        min_rooms: int | None = None,
        max_rooms: int | None = None,
        preferred_locations: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid: hard filter (payload) + dense retrieval (cosine).

        Hard filters ONLY for structured numeric payload (price, rooms). Free-text
        fields like `preferred_locations` are left to the embedding — LLMs often
        emit noisy tokens there ("quiet", "house with backyard") that don't exist
        as real location values and would wipe the result set under a Qdrant
        `should` filter. Location hints are absorbed into the vector upstream by
        `_compose_retrieval_text`.

        Returns list of {id, score, payload}.
        """
        must: list[FieldCondition] = []
        if max_price_usd is not None:
            must.append(FieldCondition(key="price_usd", range=Range(lte=max_price_usd)))
        if min_rooms is not None:
            must.append(FieldCondition(key="rooms", range=Range(gte=min_rooms)))
        if max_rooms is not None:
            must.append(FieldCondition(key="rooms", range=Range(lte=max_rooms)))

        # NOTE: `preferred_locations` intentionally not used as a hard filter
        # here — see docstring above.
        _ = preferred_locations  # consumed upstream in _compose_retrieval_text

        flt: Filter | None = Filter(must=must) if must else None

        start = time.perf_counter()
        try:
            res = await self._client.query_points(
                collection_name=self._collection,
                query=vector,
                query_filter=flt,
                limit=top_k,
                with_payload=True,
            )
            hits = res.points
        finally:
            metrics.qdrant_search_duration_seconds.observe(time.perf_counter() - start)

        return [
            {"id": h.id, "score": float(h.score), "payload": h.payload or {}} for h in hits
        ]

    async def close(self) -> None:
        await self._client.close()


_svc: QdrantService | None = None


def get_qdrant() -> QdrantService:
    global _svc
    if _svc is None:
        _svc = QdrantService()
    return _svc


async def close_qdrant() -> None:
    global _svc
    if _svc is not None:
        await _svc.close()
        _svc = None
