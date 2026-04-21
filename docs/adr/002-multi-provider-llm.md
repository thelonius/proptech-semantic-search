# ADR-002: Multi-provider LLM with split chat/embed routing

**Status:** accepted
**Date:** 2026-04-21
**Supersedes:** partial refinement of ADR-001 (Ollama-first stays, but no longer "only")

## Context

ADR-001 defaulted everything to local Ollama (qwen3.5:9b + nomic-embed-text)
to keep cost at $0 and prove we can self-host models. That story remains
important for the target role. But running real traffic exposed two issues:

1. **Ollama intent latency on M1** is 30–50s per call (9B reasoning model
   even with `think=false`). Unacceptable UX for a "semantic search" demo.
2. **Qwen3.5 JSON mode is unreliable** — ~20% of intent calls returned
   invalid JSON with control chars despite the lenient parser (`_parse_json_loose`).

A hosted option (NVIDIA NIM) delivers the same workload in ~1–3s with fewer
JSON failures. But swapping entirely to NIM would:

- Break the "$0 cost" story (even if pennies per query)
- Undercut the "self-hosted models" signal that the target vacancy flags
- Create a new failure mode: external network / API key / free-tier quota

A third concern: the **embedding vector dim is tied to the Qdrant collection**.
Ollama nomic-embed-text = 768, NIM nv-embedqa-e5-v5 = 1024, OpenAI
text-embedding-3-small = 1536. Changing the embed provider requires a full
re-index, which is an operational event, not a toggle.

## Decision

Split provider routing into two independent knobs:

```
LLM_PROVIDER   = ollama | nim | openai   # chat/completion
EMBED_PROVIDER = ollama | nim | openai   # embeddings
```

- `LLM_PROVIDER` is **free to switch** per-deploy or even per-request (not
  implemented yet, but possible via header): affects quality, latency, cost.
- `EMBED_PROVIDER` is **sticky** to the collection's vector dimension:
  changing it requires re-ingesting all properties.

Default deployment: `LLM_PROVIDER=ollama`, `EMBED_PROVIDER=ollama`
(zero cost, fully local, matches existing 768-dim collection).
For demo / founder review we flip `LLM_PROVIDER=nim` to show:
- ~20× lower latency
- ~0 JSON parse failures
- Shadow cost comparison across all 3 providers simultaneously

## Consequences

**+** One HTTP client abstraction (`OpenAICompatibleClient`) covers both NIM
     and real OpenAI — both speak OpenAI schema on `/v1/chat/completions`.

**+** Cost middleware now tracks **three** cost figures on every response:
     `X-Cost-USD` (real), `X-Cost-Shadow-OpenAI-USD`, `X-Cost-Shadow-NIM-USD`.
     Founders can see in one dashboard "what we pay now" vs "what we'd pay
     on either commercial provider" for the exact same workload.

**+** `get_llm()` and `get_embed()` are separate factory dependencies in
     FastAPI — clean to override in tests.

**−** Three providers × two roles = six code paths. Currently only
     `(ollama,ollama)`, `(nim,ollama)`, `(openai,*)` are tested end-to-end;
     the rest work by construction but aren't in CI yet.

**−** Re-indexing on `EMBED_PROVIDER` change is a manual operation (drop +
     re-ingest). `make reindex` target is a TODO.

## NIM quirks discovered

During implementation, real NIM calls exposed provider-specific behavior
that the abstraction must absorb:

1. **Cold-path variance** — free-tier NIM endpoints show 1–33s jitter on
   the same prompt, depending on worker assignment. Mitigation: 180s httpx
   timeout + tenacity retry with exponential backoff.
2. **Embeddings require `input_type`** — NIM's `nv-embedqa-e5-v5` expects
   `input_type: "query" | "passage"` in the request body (not in OpenAI spec).
   `OpenAICompatibleClient.embed` adds it when `provider == "nim"`.
3. **`response_format: {type: json_object}`** is respected by NIM for most
   Llama variants. Testing showed llama-3.1-8b is stable; llama-3.3-70b
   is higher quality but noticeably slower on free tier.

## Choice of default NIM model

`meta/llama-3.1-8b-instruct` — sub-2s latency, clean JSON output, sufficient
for structured intent extraction. The 70B variant is kept available for
later use in the reranking stage where quality matters more than latency.

## Revisit when

- Want prompt caching / request deduplication (Redis layer)
- Want per-request provider routing (`X-Prefer-Provider` header)
- Consider hybrid: cheap model for intent, expensive for rerank
