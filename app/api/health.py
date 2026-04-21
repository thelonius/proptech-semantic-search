"""Health checks + Prometheus metrics endpoint."""

from fastapi import APIRouter, Response

from app.core import metrics as m
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness — process is alive. Always 200 if the app responds at all."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz() -> dict[str, object]:
    """Readiness — can we actually serve traffic?

    TODO: ping Qdrant and Redis; for the demo we return a static shape
    so the endpoint is in place for k8s probes.
    """
    return {
        "status": "ok",
        "checks": {
            "qdrant": "unknown",
            "redis": "unknown",
            "ollama": "unknown",
        },
    }


@router.get("/metrics", response_class=Response)
async def metrics_endpoint() -> Response:
    """Prometheus scrape endpoint."""
    payload = generate_latest(m.registry)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
