# PropTech Semantic Search — Demo

Lifestyle-based real-estate matching: natural-language query → LLM intent parsing → multi-vector retrieval (Qdrant) → (upcoming) LLM rerank with explanations.

Built as a **portfolio demo** for an AI/ML Engineer role at Spacenplace (PropTech). Focus: production-grade plumbing (cost tracking, observability, multi-provider LLM, evaluation harness) — not feature count.

## Status

🟢 **Working end-to-end on 3 LLM providers.**

| Stage | Status |
|---|---|
| Scaffold (FastAPI, Docker infra, Makefile) | ✅ |
| Cost middleware (real + shadow OpenAI + shadow NIM) | ✅ |
| Prometheus metrics for every layer | ✅ |
| Multi-provider LLM: Ollama / NIM / OpenAI (split LLM vs embed) | ✅ |
| Intent parsing (stage 1) | ✅ |
| HF dataset ingestion (100 properties, Nigerian real estate) | ✅ |
| Qdrant retrieval (stage 2, cosine + hard filters) | ✅ |
| Eval harness (precision@K, recall@K, MRR, cost, side-by-side providers) | ✅ |
| Provider-benchmark eval reports (Ollama vs NIM) | ✅ |
| LLM-assisted gold labeler (`scripts/label_queries.py`) with checkpointing | ✅ |
| LLM reranker (stage 3) | ⏳ |
| Grafana cost dashboard | ⏳ |
| k6 load test + chaos tests | ⏳ |

## Highlights

- **Multi-provider LLM** in one codebase: `LLM_PROVIDER=ollama|nim|openai`
- **Split chat / embed routing** (`EMBED_PROVIDER` separate, sticky to Qdrant vector dim)
- **Cost tracking every request**:
  `X-Cost-USD` (real), `X-Cost-Shadow-OpenAI-USD`, `X-Cost-Shadow-NIM-USD`
  — founders see what they pay now vs each alternative on the same workload
- **Structured JSON logs** with `request_id` correlation
- **Prometheus metrics** for every layer: HTTP, LLM (tokens/cost/latency), Qdrant, cache
- **Eval harness** writes diff-friendly Markdown reports (`evals/results/*.md`)
- **Architecture Decision Records** in `docs/adr/`

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

## Provider benchmark (same 15 queries, same dataset)

Verified end-to-end with both providers. Both reports are checked in under
`evals/results/` — Markdown format, diff-friendly.

| Metric | **Ollama** (qwen3.5:9b, local M1) | **NIM** (llama-3.1-8b, cloud GPU) | Ratio |
|---|---|---|---|
| Mean latency | 42 254 ms | 1 908 ms | **22× faster** |
| Median latency | 39 722 ms | 1 509 ms | **26× faster** |
| Success rate | 12/15 (80%) | 15/15 (100%) | — |
| Real cost (15 queries) | **$0.000000** | $0.000589 | — |
| Shadow OpenAI (same load) | $0.002137 | $0.001892 | NIM is **3.2× cheaper than OpenAI** |

Pick Ollama for dev / privacy / zero-cost experiments.
Pick NIM for low-latency prod at a fraction of OpenAI's bill.
OpenAI stays available as a compatibility target.

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
