"""Prometheus metrics registry.

All app-level counters/histograms live here.
Exposed via /metrics endpoint (see app.api.health).
"""

from prometheus_client import CollectorRegistry, Counter, Histogram

# Dedicated registry so we don't pollute the default one in tests
registry = CollectorRegistry()

# --- request-level ---

http_requests_total = Counter(
    "psem_http_requests_total",
    "Total HTTP requests by method, path, status",
    ["method", "path", "status"],
    registry=registry,
)

http_request_duration_seconds = Histogram(
    "psem_http_request_duration_seconds",
    "HTTP request duration (end-to-end)",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=registry,
)

# --- LLM / embedding ---

llm_calls_total = Counter(
    "psem_llm_calls_total",
    "LLM/embedding calls by provider, model, kind, status",
    ["provider", "model", "kind", "status"],  # kind: completion|embedding
    registry=registry,
)

llm_tokens_total = Counter(
    "psem_llm_tokens_total",
    "Total tokens consumed by provider, model, direction (input/output)",
    ["provider", "model", "direction"],
    registry=registry,
)

llm_latency_seconds = Histogram(
    "psem_llm_latency_seconds",
    "LLM/embedding call latency",
    ["provider", "model", "kind"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=registry,
)

# --- cost (this is THE metric founders care about) ---

llm_cost_usd_total = Counter(
    "psem_llm_cost_usd_total",
    "Cumulative USD cost of LLM calls (local models report 0, but we still tag)",
    ["provider", "model", "kind"],
    registry=registry,
)

# "Shadow" costs — what the same workload would have cost on each provider.
# Shown on Grafana alongside real cost — lets us compare Ollama / NIM / OpenAI
# for the exact same query load.
llm_cost_usd_shadow_openai_total = Counter(
    "psem_llm_cost_usd_shadow_openai_total",
    "Cumulative USD cost IF we had used OpenAI (for provider comparison)",
    ["kind"],
    registry=registry,
)

llm_cost_usd_shadow_nim_total = Counter(
    "psem_llm_cost_usd_shadow_nim_total",
    "Cumulative USD cost IF we had used NVIDIA NIM (70B) (for provider comparison)",
    ["kind"],
    registry=registry,
)

# --- cache ---

cache_ops_total = Counter(
    "psem_cache_ops_total",
    "Cache operations by kind and result",
    ["kind", "result"],  # kind: intent|embedding|rerank; result: hit|miss|error
    registry=registry,
)

# --- retrieval ---

qdrant_search_duration_seconds = Histogram(
    "psem_qdrant_search_duration_seconds",
    "Qdrant search latency",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=registry,
)
