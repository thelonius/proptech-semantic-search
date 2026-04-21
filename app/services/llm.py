"""LLM + embedding client with built-in cost tracking.

Providers: ollama (default, local, $0), openai, nim.
All calls record tokens/latency/cost into Prometheus and per-request accumulator
via `app.core.cost.record_call(...)`.

For Ollama we rely on the API response's `prompt_eval_count` and `eval_count`
for accurate token accounting (llama.cpp tokenizer, not tiktoken).

For OpenAI we use `tiktoken` as fallback if the API doesn't return usage.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.cost import record_call
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ---------- token counting fallback ----------

_TIKTOKEN_ENCODERS: dict[str, Any] = {}


def _count_tokens_openai(text: str, model: str = "gpt-4o-mini") -> int:
    enc = _TIKTOKEN_ENCODERS.get(model)
    if enc is None:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        _TIKTOKEN_ENCODERS[model] = enc
    return len(enc.encode(text))


# ---------- base client ----------


class LLMError(Exception):
    pass


# Invalid JSON control chars — anything < 0x20 except tab/newline/CR
_INVALID_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# Accidental markdown fences around JSON
_MD_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


def _parse_json_loose(text: str) -> dict[str, Any]:
    """Parse JSON that LLMs might mangle.

    Strategy:
      1. Try as-is.
      2. Strip markdown fences.
      3. Strip stray control chars inside strings.
      4. Extract outermost {...} block and retry.
    """
    if not text:
        raise LLMError("LLM returned empty content")

    candidates: list[str] = [text]
    fence_stripped = _MD_FENCE.sub("", text).strip()
    if fence_stripped != text:
        candidates.append(fence_stripped)
    candidates.append(_INVALID_CTRL.sub(" ", fence_stripped))

    # Last ditch: extract first {...} balanced block
    brace_start = fence_stripped.find("{")
    brace_end = fence_stripped.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        candidates.append(
            _INVALID_CTRL.sub(" ", fence_stripped[brace_start : brace_end + 1])
        )

    last_err: Exception | None = None
    for candidate in candidates:
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError as e:
            last_err = e
    logger.warning("llm_json_parse_failed", preview=text[:400], error=str(last_err))
    raise LLMError(f"LLM returned invalid JSON: {last_err}")


class OllamaClient:
    """Minimal async Ollama client.

    Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
    Endpoints used: /api/chat (completion), /api/embed (embeddings).
    """

    def __init__(self, base_url: str, llm_model: str, embed_model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.llm_model = llm_model
        self.embed_model = embed_model
        # Keep-alive helps on bursty workloads
        # Generous timeout — first-load of a 9B model on CPU/MPS can take 20–30s.
        # Keep-alive helps on bursty workloads.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(180.0, connect=5.0),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )

    async def close(self) -> None:
        await self._client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        reraise=True,
    )
    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.2,
        model: str | None = None,
        think: bool = False,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Run a chat completion. Returns {"content": str, "raw": dict}.

        For Qwen3 / reasoning models: pass `think=False` to skip the reasoning
        prefix (otherwise the model eats its output budget on internal thinking).
        """
        mdl = model or self.llm_model
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: dict[str, Any] = {
            "model": mdl,
            "messages": messages,
            "stream": False,
            "think": think,  # disable reasoning prefix (Ollama >=0.20)
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"

        start = time.perf_counter()
        status = "ok"
        in_tokens = 0
        out_tokens = 0
        try:
            r = await self._client.post("/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            content = data.get("message", {}).get("content", "") or ""
            # If thinking slipped through (older Ollama / unsupported model),
            # fall back to the thinking field so caller still gets text.
            if not content and data.get("message", {}).get("thinking"):
                content = data["message"]["thinking"]
            # Ollama provides exact tokenizer counts
            in_tokens = int(data.get("prompt_eval_count") or 0)
            out_tokens = int(data.get("eval_count") or 0)
            if in_tokens == 0:
                in_tokens = _count_tokens_openai(
                    "\n".join(m.get("content", "") for m in messages)
                )
            if out_tokens == 0:
                out_tokens = _count_tokens_openai(content)
            return {"content": content, "raw": data}
        except httpx.HTTPError as e:
            status = "error"
            # httpx.ReadTimeout has empty str() — include type name for visibility
            err_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            logger.error("ollama_chat_failed", error=err_msg, model=mdl)
            raise LLMError(f"Ollama chat failed: {err_msg}") from e
        finally:
            latency = time.perf_counter() - start
            record_call(
                provider="ollama",
                model=mdl,
                kind="completion",
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                latency_s=latency,
                status=status,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        reraise=True,
    )
    async def embed(
        self,
        *,
        inputs: str | list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """Embed text(s). Always returns a list of vectors (even for single input)."""
        mdl = model or self.embed_model
        texts = [inputs] if isinstance(inputs, str) else list(inputs)

        start = time.perf_counter()
        status = "ok"
        in_tokens = 0
        try:
            r = await self._client.post(
                "/api/embed",
                json={"model": mdl, "input": texts},
            )
            r.raise_for_status()
            data = r.json()
            vectors = data.get("embeddings", [])
            in_tokens = int(data.get("prompt_eval_count") or 0) or sum(
                _count_tokens_openai(t) for t in texts
            )
            return vectors
        except httpx.HTTPError as e:
            status = "error"
            logger.error("ollama_embed_failed", error=str(e), model=mdl)
            raise LLMError(f"Ollama embed failed: {e}") from e
        finally:
            latency = time.perf_counter() - start
            record_call(
                provider="ollama",
                model=mdl,
                kind="embedding",
                input_tokens=in_tokens,
                output_tokens=0,
                latency_s=latency,
                status=status,
            )

    async def parse_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Convenience: chat in JSON mode and parse result.

        Robust to common LLM JSON failure modes:
        - Stray markdown fences (```json ... ```)
        - Raw control characters inside strings (e.g. \\x0b from Qwen)
        - Trailing commas
        """
        out = await self.chat(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            json_mode=True,
            temperature=temperature,
        )
        content = out["content"]
        return _parse_json_loose(content)


class OpenAICompatibleClient:
    """OpenAI-compatible client — works for NVIDIA NIM and real OpenAI.

    NIM exposes /v1/chat/completions and /v1/embeddings with OpenAI's schema,
    so the same code handles both. The `provider` label differentiates them
    in cost/metrics tracking.
    """

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        llm_model: str,
        embed_model: str,
    ) -> None:
        if not api_key:
            raise LLMError(f"{provider.upper()}_API_KEY is not set")
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.llm_model = llm_model
        self.embed_model = embed_model
        # NIM free tier has noticeable jitter (some calls ~1s, some ~30s for 70B).
        # Give ourselves headroom; tenacity retries handle transient timeouts.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(180.0, connect=15.0),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        reraise=True,
    )
    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.2,
        model: str | None = None,
        think: bool = False,  # unused; accepted for interface parity with Ollama
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        mdl = model or self.llm_model
        payload: dict[str, Any] = {
            "model": mdl,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            # OpenAI spec — NIM accepts the same
            payload["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        status = "ok"
        in_tokens = 0
        out_tokens = 0
        try:
            r = await self._client.post("/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
            content = (
                data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            )
            usage = data.get("usage") or {}
            in_tokens = int(usage.get("prompt_tokens") or 0)
            out_tokens = int(usage.get("completion_tokens") or 0)
            if in_tokens == 0:
                in_tokens = _count_tokens_openai(
                    "\n".join(m.get("content", "") for m in messages)
                )
            if out_tokens == 0:
                out_tokens = _count_tokens_openai(content)
            return {"content": content, "raw": data}
        except httpx.HTTPError as e:
            status = "error"
            err_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            logger.error(f"{self.provider}_chat_failed", error=err_msg, model=mdl)
            raise LLMError(f"{self.provider} chat failed: {err_msg}") from e
        finally:
            latency = time.perf_counter() - start
            record_call(
                provider=self.provider,
                model=mdl,
                kind="completion",
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                latency_s=latency,
                status=status,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        reraise=True,
    )
    async def embed(
        self,
        *,
        inputs: str | list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        mdl = model or self.embed_model
        texts = [inputs] if isinstance(inputs, str) else list(inputs)

        start = time.perf_counter()
        status = "ok"
        in_tokens = 0
        try:
            payload: dict[str, Any] = {"model": mdl, "input": texts}
            # NIM-specific: some NV models require input_type
            if self.provider == "nim":
                payload["input_type"] = "query"
                payload["encoding_format"] = "float"
            r = await self._client.post("/embeddings", json=payload)
            r.raise_for_status()
            data = r.json()
            vectors = [item["embedding"] for item in data.get("data", [])]
            usage = data.get("usage") or {}
            in_tokens = int(usage.get("prompt_tokens") or 0) or sum(
                _count_tokens_openai(t) for t in texts
            )
            return vectors
        except httpx.HTTPError as e:
            status = "error"
            logger.error(f"{self.provider}_embed_failed", error=str(e), model=mdl)
            raise LLMError(f"{self.provider} embed failed: {e}") from e
        finally:
            latency = time.perf_counter() - start
            record_call(
                provider=self.provider,
                model=mdl,
                kind="embedding",
                input_tokens=in_tokens,
                output_tokens=0,
                latency_s=latency,
                status=status,
            )

    async def parse_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        out = await self.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            json_mode=True,
            temperature=temperature,
        )
        return _parse_json_loose(out["content"])


# ---------- union type + factory ----------

# The two clients expose the same call surface. We use them interchangeably
# via duck-typing (no formal Protocol needed for the size of this codebase).
LLMClient = OllamaClient | OpenAICompatibleClient


def _make_client(provider: str) -> LLMClient:
    if provider == "ollama":
        return OllamaClient(
            base_url=settings.ollama_base_url,
            llm_model=settings.ollama_llm_model,
            embed_model=settings.ollama_embed_model,
        )
    if provider == "nim":
        return OpenAICompatibleClient(
            provider="nim",
            base_url=settings.nim_base_url,
            api_key=settings.nim_api_key,
            llm_model=settings.nim_llm_model,
            embed_model=settings.nim_embed_model,
        )
    if provider == "openai":
        return OpenAICompatibleClient(
            provider="openai",
            base_url="https://api.openai.com/v1",
            api_key=settings.openai_api_key,
            llm_model=settings.openai_llm_model,
            embed_model=settings.openai_embed_model,
        )
    raise LLMError(f"Unknown provider: {provider!r}")


# Two separate singletons — LLM (chat) and embed may be different providers.
# Key reason: Qdrant collection has a fixed vector dim tied to embed provider;
# switching LLM is free, switching embed requires re-indexing.
_llm_client: LLMClient | None = None
_embed_client: LLMClient | None = None


def get_llm() -> LLMClient:
    """Return the chat/completion client by LLM_PROVIDER."""
    global _llm_client
    if _llm_client is None:
        _llm_client = _make_client(settings.llm_provider)
    return _llm_client


def get_embed() -> LLMClient:
    """Return the embeddings client by EMBED_PROVIDER.

    Often the same provider as LLM, but may differ to keep Qdrant collection
    dim compatible. Reuses the LLM client when providers match to save
    connection pool setup.
    """
    global _embed_client
    if _embed_client is None:
        if settings.embed_provider == settings.llm_provider:
            _embed_client = get_llm()
        else:
            _embed_client = _make_client(settings.embed_provider)
    return _embed_client


async def close_llm() -> None:
    global _llm_client, _embed_client
    if _embed_client is not None and _embed_client is not _llm_client:
        await _embed_client.close()
    _embed_client = None
    if _llm_client is not None:
        await _llm_client.close()
    _llm_client = None
