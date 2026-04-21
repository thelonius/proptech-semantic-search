# PropTech Semantic Search — Demo

Lifestyle-based real estate matching: natural language query → multi-vector retrieval (text + image) → LLM rerank with explanations.

Built as a **portfolio demo** for the Spacenplace AI/ML Engineer role. Focus: production-grade plumbing (cost tracking, observability, graceful degradation, evaluation harness) — not feature count.

## Status

🟢 **Working end-to-end.** Search pipeline is live; further polish in progress.

| Stage | Status |
|---|---|
| Scaffold (FastAPI, Docker infra, Makefile) | ✅ |
| Cost middleware + Prometheus metrics | ✅ |
| Ollama LLM client (qwen3.5:9b + nomic-embed-text) | ✅ |
| Intent parsing (stage 1) | ✅ |
| HF dataset ingestion (100 properties, Nigerian real estate) | ✅ |
| Qdrant retrieval (stage 2, cosine + hard filters) | ✅ |
| Eval harness (precision@K, recall@K, MRR, cost summary) | ✅ |
| LLM-assisted gold labeler (`scripts/label_queries.py`) | ✅ |
| LLM reranker (stage 3) | ⏳ |
| Grafana cost dashboard | ⏳ |
| k6 load test | ⏳ |
| Chaos/failure-mode tests | ⏳ |
| Scaling + cost-breakdown docs | ⏳ |

## Design highlights

- **100% local LLM** (Ollama on Apple Silicon) → $0 real cost
- **Cost observability**: every response carries `X-Cost-USD` + `X-Cost-Shadow-OpenAI-USD` headers
  (real cost vs what it would have cost on OpenAI — proves the value of self-hosting)
- **Structured JSON logs** with request-id correlation
- **Prometheus metrics** for every layer: HTTP, LLM (tokens/cost/latency), Qdrant, cache

## Quickstart

Requires: Python 3.11+, Docker Desktop, Ollama with `qwen3.5:9b` + `nomic-embed-text`.

```bash
make install              # venv + base deps
make install-ml           # torch, transformers, datasets (only for ingestion)
make env                  # copy .env.example → .env
make up                   # start infra (Qdrant, Redis, Prometheus, Grafana)
make ingest               # load 100 Nigerian real-estate listings into Qdrant
make dev                  # run FastAPI with reload
```

Open http://localhost:8000/docs — try `POST /search` with:

```json
{
  "query": "Family with two kids and a dog looking for a quiet house with a backyard",
  "top_k": 5
}
```

Response headers show per-request cost and plumbing:

```
X-Request-ID: a84de388401b40a8
X-Cost-USD: 0.000000          # we ran it locally
X-Cost-Shadow-OpenAI-USD: 0.000161   # what OpenAI would have charged
X-LLM-Calls: 2
X-LLM-Tokens-In: 367
X-LLM-Tokens-Out: 185
X-Duration-Ms: 31148.2
```

Response body includes the parsed intent:

```json
{
  "intent": {
    "household": "family_with_children",
    "pets": ["dog"],
    "implicit_needs": ["park_nearby", "quiet", "school_nearby"],
    "style_preferences": ["quiet"],
    "rationale": "User mentions family with kids and a dog..."
  },
  "hits": [
    { "property_id": "85", "title": "5 bedroom detached duplex for sale",
      "location": "Opebi, Ikeja, Lagos", "score": 0.611 },
    ...
  ],
  "stages_ms": { "intent_ms": 30200, "retrieval_ms": 938 }
}
```

## Eval harness

```bash
make eval                 # runs evals/queries.yaml, writes evals/results/<ts>.md
```

Output metrics per query: `precision@K`, `recall@K`, `MRR`, `latency`, `cost`.
Reports are markdown so they diff nicely in git. Real and shadow-OpenAI cost
are rolled up side-by-side — the "savings from local LLM" line is the headline.

To bootstrap gold labels (which properties are relevant for each query):

```bash
.venv/bin/python -m scripts.label_queries --limit-properties 100
# review evals/queries.labeled.yaml, then move it onto queries.yaml
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
