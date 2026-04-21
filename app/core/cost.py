"""Cost tracking — the centerpiece of the demo.

Every LLM/embedding call goes through `LLMUsage.record(...)`. The cost
middleware (below) picks up the per-request total and adds an `X-Cost-USD`
response header. Prometheus counters track cumulative spend.

For local Ollama the real cost is $0, BUT we still compute the "shadow"
OpenAI cost for the same token volume — that's what gets shown to founders:

    Real cost: $0.000000 (Ollama, local)
    Equivalent on OpenAI: $0.004287

This is exactly what a cost-obsessed startup wants to see.
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core import metrics
from app.core.config import get_settings
from app.core.logging import get_logger, request_id_ctx

logger = get_logger(__name__)
settings = get_settings()

Direction = Literal["input", "output"]
Kind = Literal["completion", "embedding"]

# Per-request usage accumulator. Populated by LLM/embedding wrappers,
# read by the cost middleware at response time.
_usage_ctx: ContextVar["RequestUsage | None"] = ContextVar("llm_usage", default=None)


@dataclass
class CallRecord:
    """One LLM or embedding call."""

    provider: str
    model: str
    kind: Kind
    input_tokens: int
    output_tokens: int
    latency_s: float
    real_cost_usd: float
    shadow_openai_usd: float
    shadow_nim_usd: float
    cache_hit: bool = False


@dataclass
class RequestUsage:
    """Aggregated usage for a single HTTP request."""

    request_id: str
    calls: list[CallRecord] = field(default_factory=list)

    def add(self, rec: CallRecord) -> None:
        self.calls.append(rec)

    @property
    def total_real_usd(self) -> float:
        return sum(c.real_cost_usd for c in self.calls)

    @property
    def total_shadow_openai_usd(self) -> float:
        return sum(c.shadow_openai_usd for c in self.calls)

    @property
    def total_shadow_nim_usd(self) -> float:
        return sum(c.shadow_nim_usd for c in self.calls)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)


def current_usage() -> RequestUsage | None:
    """Get the usage accumulator bound to the current request."""
    return _usage_ctx.get()


def _openai_shadow_cost(kind: Kind, input_tokens: int, output_tokens: int) -> float:
    """Cost if the workload had run on OpenAI (gpt-4o-mini + embedding-3-small)."""
    s = settings
    if kind == "completion":
        return (
            input_tokens / 1000.0 * s.openai_gpt4o_mini_input_per_1k
            + output_tokens / 1000.0 * s.openai_gpt4o_mini_output_per_1k
        )
    return input_tokens / 1000.0 * s.openai_embed_small_per_1k


def _nim_shadow_cost(kind: Kind, input_tokens: int, output_tokens: int) -> float:
    """Cost if the workload had run on NVIDIA NIM (llama-3.3-70b + nv-embedqa-e5)."""
    s = settings
    if kind == "completion":
        return (
            input_tokens / 1000.0 * s.nim_llama70b_input_per_1k
            + output_tokens / 1000.0 * s.nim_llama70b_output_per_1k
        )
    return input_tokens / 1000.0 * s.nim_embed_per_1k


def _real_cost(provider: str, kind: Kind, input_tokens: int, output_tokens: int) -> float:
    """What we actually paid for this call."""
    if provider == "ollama":
        return 0.0
    if provider == "openai":
        return _openai_shadow_cost(kind, input_tokens, output_tokens)
    if provider == "nim":
        return _nim_shadow_cost(kind, input_tokens, output_tokens)
    return 0.0


def record_call(
    *,
    provider: str,
    model: str,
    kind: Kind,
    input_tokens: int,
    output_tokens: int,
    latency_s: float,
    status: str = "ok",
    cache_hit: bool = False,
) -> CallRecord:
    """Record a single LLM/embedding call.

    Updates Prometheus counters AND (if inside a request) the per-request
    accumulator so the cost middleware can add X-Cost-USD header.
    """
    real = _real_cost(provider, kind, input_tokens, output_tokens)
    shadow_oai = _openai_shadow_cost(kind, input_tokens, output_tokens)
    shadow_nim = _nim_shadow_cost(kind, input_tokens, output_tokens)
    rec = CallRecord(
        provider=provider,
        model=model,
        kind=kind,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=latency_s,
        real_cost_usd=real,
        shadow_openai_usd=shadow_oai,
        shadow_nim_usd=shadow_nim,
        cache_hit=cache_hit,
    )

    # Prometheus (always, even outside request context — e.g. ingestion)
    metrics.llm_calls_total.labels(
        provider=provider, model=model, kind=kind, status=status
    ).inc()
    metrics.llm_tokens_total.labels(
        provider=provider, model=model, direction="input"
    ).inc(input_tokens)
    metrics.llm_tokens_total.labels(
        provider=provider, model=model, direction="output"
    ).inc(output_tokens)
    metrics.llm_latency_seconds.labels(
        provider=provider, model=model, kind=kind
    ).observe(latency_s)
    metrics.llm_cost_usd_total.labels(
        provider=provider, model=model, kind=kind
    ).inc(real)
    metrics.llm_cost_usd_shadow_openai_total.labels(kind=kind).inc(shadow_oai)
    metrics.llm_cost_usd_shadow_nim_total.labels(kind=kind).inc(shadow_nim)

    # Per-request accumulator
    usage = _usage_ctx.get()
    if usage is not None:
        usage.add(rec)

    return rec


# ---------- middleware ----------


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Bind request_id + usage accumulator + timing + cost headers.

    Responsibilities:
      1. Generate request_id (or reuse X-Request-ID header)
      2. Set up ContextVars for logging and cost
      3. Record request duration
      4. Attach cost headers to every response
      5. Emit a single structured access-log line with cost summary
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        req_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        rid_token = request_id_ctx.set(req_id)
        usage = RequestUsage(request_id=req_id)
        usage_token = _usage_ctx.set(usage)

        start = time.perf_counter()
        response: Response | None = None
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return self._finalize(response, request, usage, req_id, start, status_code)
        except Exception:
            duration_s = time.perf_counter() - start
            path = request.url.path
            method = request.method
            metrics.http_requests_total.labels(
                method=method, path=path, status="500"
            ).inc()
            metrics.http_request_duration_seconds.labels(
                method=method, path=path
            ).observe(duration_s)
            logger.exception(
                "request_failed",
                method=method,
                path=path,
                duration_ms=round(duration_s * 1000, 1),
            )
            raise
        finally:
            _usage_ctx.reset(usage_token)
            request_id_ctx.reset(rid_token)

    def _finalize(
        self,
        response: Response,
        request: Request,
        usage: "RequestUsage",
        req_id: str,
        start: float,
        status_code: int,
    ) -> Response:
        duration_s = time.perf_counter() - start
        route = request.scope.get("route")
        path = route.path if route else request.url.path  # type: ignore[union-attr]
        method = request.method

        metrics.http_requests_total.labels(
            method=method, path=path, status=str(status_code)
        ).inc()
        metrics.http_request_duration_seconds.labels(
            method=method, path=path
        ).observe(duration_s)

        response.headers["X-Request-ID"] = req_id
        response.headers["X-Duration-Ms"] = f"{duration_s * 1000:.1f}"
        response.headers["X-Cost-USD"] = f"{usage.total_real_usd:.6f}"
        response.headers["X-Cost-Shadow-OpenAI-USD"] = (
            f"{usage.total_shadow_openai_usd:.6f}"
        )
        response.headers["X-Cost-Shadow-NIM-USD"] = (
            f"{usage.total_shadow_nim_usd:.6f}"
        )
        response.headers["X-LLM-Calls"] = str(len(usage.calls))
        response.headers["X-LLM-Tokens-In"] = str(usage.total_input_tokens)
        response.headers["X-LLM-Tokens-Out"] = str(usage.total_output_tokens)

        logger.info(
            "request_completed",
            method=method,
            path=path,
            status=status_code,
            duration_ms=round(duration_s * 1000, 1),
            llm_calls=len(usage.calls),
            tokens_in=usage.total_input_tokens,
            tokens_out=usage.total_output_tokens,
            cost_usd=round(usage.total_real_usd, 6),
            shadow_openai_usd=round(usage.total_shadow_openai_usd, 6),
            shadow_nim_usd=round(usage.total_shadow_nim_usd, 6),
        )
        return response
