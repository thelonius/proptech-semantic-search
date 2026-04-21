"""Ingest endpoint — stub for now.

Actual bulk ingestion is done via `python -m scripts.ingest_hf`. This HTTP
endpoint is reserved for streaming single-property ingestion once the
retrieval stage is wired up.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("")
async def ingest_stub() -> dict[str, str]:
    return {
        "status": "not_implemented",
        "hint": "Use `python -m scripts.ingest_hf --limit 100` for bulk ingest",
    }
