# ADR-001: Local Ollama as default LLM provider

**Status:** accepted
**Date:** 2026-04-21

## Context

The pipeline makes 2–3 LLM calls per `/search` request (intent parsing,
optionally reranking) plus embeddings during ingestion. Three realistic
options:

1. **OpenAI** (`gpt-4o-mini`, `text-embedding-3-small`)
2. **NVIDIA NIM** (hosted inference)
3. **Ollama local** (`qwen3.5:9b`, `nomic-embed-text`)

The target company is a very early PropTech startup that explicitly flags
*"оптимизация стоимости AI-сервисов"* and *"развёртывание и поддержки моделей
на серверах"* as core requirements.

## Decision

**Default to Ollama. Keep OpenAI code paths for cost comparison only.**

Rationale:
- **$0 real cost** at any volume — the value proposition we want to show.
- Qwen3.5:9B on Apple Silicon (MPS) is fast enough (~2–5s warm)
  for the demo and gives high-quality structured JSON via `format: json`.
- Demonstrates **self-hosted model deployment**, which is a hard requirement
  in the target role.
- Code stays provider-agnostic via a single `OllamaClient` abstraction —
  switching to OpenAI / NIM later is a config flag.

## Consequences

**+** Zero marginal cost → no API budget needed for eval runs.
**+** Proves production capability for local model serving.
**+** Realistic latency (single-machine) surfaces the need for batching
     and keep-alive; these patterns become visible in metrics.

**−** Cold start of 9B model on M1 is ~30s — mitigated by keeping Ollama's
     default 5-min model keep-alive and a 180s httpx timeout.
**−** Qwen3 is a reasoning model — need `think=false` to avoid burning tokens
     on internal chain-of-thought.

## Alternatives considered

- **OpenAI only**: simplest, but $ scales with eval iteration and gives the
  wrong signal to a cost-sensitive startup.
- **NIM**: attractive (NVIDIA prod stack), but requires NGC key; endpoint
  returned 403 during exploration — skipping for now, NIM branch can be added
  later via the same `LLMClient` abstraction.

## Revisit when

- Query volume exceeds ~50 RPS (local may become bottleneck)
- Need for multi-node model serving arises
- Qwen releases a non-thinking variant of similar quality at smaller size
