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
        """Convenience: chat in JSON mode and parse result."""
        out = await self.chat(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            json_mode=True,
            temperature=temperature,
        )
        try:
            return json.loads(out["content"])
        except json.JSONDecodeError as e:
            logger.warning("llm_json_parse_failed", content=out["content"][:400])
            raise LLMError(f"LLM returned invalid JSON: {e}") from e


# ---------- singleton ----------

_client: OllamaClient | None = None


def get_llm() -> OllamaClient:
    """Return the active LLM client.

    For the demo we default to Ollama. If LLM_PROVIDER=openai we'd add an
    OpenAI-compatible client here; deliberately not adding yet to keep scope tight.
    """
    global _client
    if _client is None:
        if settings.llm_provider != "ollama":
            raise NotImplementedError(
                f"Provider {settings.llm_provider!r} not wired yet — use ollama."
            )
        _client = OllamaClient(
            base_url=settings.ollama_base_url,
            llm_model=settings.ollama_llm_model,
            embed_model=settings.ollama_embed_model,
        )
    return _client


async def close_llm() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None
