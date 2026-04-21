# PropTech Semantic Search — Demo

Lifestyle-based real estate matching: natural language query → multi-vector retrieval (text + image) → LLM rerank with explanations.

Built as a **portfolio demo** for the Spacenplace AI/ML Engineer role. Focus: production-grade plumbing (cost tracking, observability, graceful degradation, evaluation harness) — not feature count.

## Status

🚧 **In progress.** This is a live development branch.

| Stage | Status |
|---|---|
| Scaffold (FastAPI, Docker infra, Makefile) | ✅ done |
| Cost middleware + Prometheus metrics | ✅ done |
| Ollama LLM client (qwen3.5:9b + nomic-embed-text) | ✅ done |
| Intent parsing (stage 1 of pipeline) | ✅ done |
| HF dataset ingestion | ⏳ next |
| Qdrant retrieval (stage 2) | ⏳ next |
| LLM reranker (stage 3) | ⏳ |
| Eval harness (precision@5, recall@10, MRR) | ⏳ next |
| k6 load test | ⏳ |
| ADR docs | ⏳ |

## Design highlights

- **100% local LLM** (Ollama on Apple Silicon) → $0 real cost
- **Cost observability**: every response carries `X-Cost-USD` + `X-Cost-Shadow-OpenAI-USD` headers
  (real cost vs what it would have cost on OpenAI — proves the value of self-hosting)
- **Structured JSON logs** with request-id correlation
- **Prometheus metrics** for every layer: HTTP, LLM (tokens/cost/latency), Qdrant, cache

## Quickstart

Requires: Python 3.11+, Docker Desktop, Ollama with `qwen3.5:9b` + `nomic-embed-text`.

```bash
make install              # venv + deps
make env                  # copy .env.example → .env
make up                   # start infra (Qdrant, Redis, Prometheus, Grafana)
make dev                  # run FastAPI with reload
```

Open http://localhost:8000/docs — try `POST /search` with:

```json
{ "query": "family with kids and a dog looking for a house with a yard" }
```

Response headers show per-request cost:

```
X-Cost-USD: 0.000000
X-Cost-Shadow-OpenAI-USD: 0.004287
X-LLM-Tokens-In: 412
X-LLM-Tokens-Out: 138
```

## Stack

- **Backend**: FastAPI, asyncio, httpx, structlog, tenacity
- **LLM**: Ollama (qwen3.5:9b for completion, nomic-embed-text for embeddings)
- **Vector DB**: Qdrant (multi-vector, hybrid search)
- **Cache**: Redis
- **Observability**: Prometheus + Grafana
- **ML** (optional extras): PyTorch, transformers, Ultralytics (YOLOv8), Pillow

## License

MIT
