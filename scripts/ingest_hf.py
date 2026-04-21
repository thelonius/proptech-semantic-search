"""Ingest HuggingFace real-estate dataset into Qdrant.

Usage:
  python -m scripts.ingest_hf --limit 100 --recreate

Source: Binaryy/multimodal-real-estate-search
  - 1041 Nigerian property listings (Lagos, Abuja, ...)
  - Columns: Title, Location, Details, image
  - ~70 MB parquet

We only use TEXT here (title + location + details → single embedding).
Image embeddings are phase 2 — proved text path first.
"""

from __future__ import annotations

import asyncio
import re
import sys
from typing import Any

import typer
from qdrant_client.models import PointStruct
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.llm import close_llm, get_llm
from app.services.qdrant_client import close_qdrant, get_qdrant

settings = get_settings()
configure_logging(level="INFO", json_output=False)
logger = get_logger(__name__)
console = Console()


# --- light NLP extraction for payload (pre-LLM; cheap, good enough for demo) ---

_PRICE_RE = re.compile(
    r"(?:₦|N|NGN|\$|USD)\s?([\d,]+(?:\.\d+)?)(?:\s?(?:million|m|k|,000))?", re.IGNORECASE
)
_ROOMS_RE = re.compile(r"(\d+)\s*(?:bedroom|bed|br)\b", re.IGNORECASE)


def _extract_rooms(text: str) -> int | None:
    m = _ROOMS_RE.search(text or "")
    return int(m.group(1)) if m else None


def _extract_price_usd(text: str) -> float | None:
    """Very rough: look for N/₦/$ values; convert Naira → USD via fixed rate.

    Honest caveat: prices in this dataset are often embedded in marketing copy,
    with inconsistent formatting. This is good enough for demo filtering.
    NGN→USD fixed at 1500 (typical 2024-26 rate).
    """
    t = text or ""
    m = _PRICE_RE.search(t)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        val = float(raw)
    except ValueError:
        return None
    low = m.group(0).lower()
    if "million" in low or low.strip().endswith("m"):
        val *= 1_000_000
    elif low.strip().endswith("k"):
        val *= 1_000
    # Assume Naira if either currency symbol absent or ₦/N used
    if "$" in low or "usd" in low:
        return round(val, 2)
    return round(val / 1500, 2)


def _norm_location(loc: str) -> list[str]:
    """Split 'Lekki Phase 1, Lekki, Lagos' into normalized tokens."""
    parts = [p.strip().lower() for p in (loc or "").split(",") if p.strip()]
    return parts


async def _load_dataset(limit: int) -> list[dict[str, Any]]:
    """Lazy import — keeps base install lean when ingestion isn't needed."""
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError:
        console.print(
            "[red]datasets not installed.[/red] Run: [bold]make install-ml[/bold]"
        )
        sys.exit(2)

    console.print(
        f"[cyan]Loading [bold]Binaryy/multimodal-real-estate-search[/bold] "
        f"(limit={limit})[/cyan]"
    )
    ds = load_dataset("Binaryy/multimodal-real-estate-search", split="train")
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        rows.append(
            {
                "id": i,
                "title": row.get("Title", "") or "",
                "location": row.get("Location", "") or "",
                "details": row.get("Details", "") or "",
            }
        )
    return rows


def _build_embed_text(row: dict[str, Any]) -> str:
    """Compose the text we embed. Concise, keeps signal, skips boilerplate."""
    parts = [row["title"], row["location"], row["details"]]
    return "\n".join(p for p in parts if p)[:2000]  # cap to stay under model ctx


async def _ingest(limit: int, recreate: bool, batch_size: int) -> None:
    qdrant = get_qdrant()
    await qdrant.ensure_collection(recreate=recreate)

    llm = get_llm()

    rows = await _load_dataset(limit)
    console.print(f"[green]Loaded {len(rows)} rows[/green]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding + upserting", total=len(rows))

        # Batch embed for efficiency
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            texts = [_build_embed_text(r) for r in chunk]
            vectors = await llm.embed(inputs=texts)

            points = []
            for row, vec in zip(chunk, vectors, strict=False):
                full_text = _build_embed_text(row)
                payload = {
                    "title": row["title"][:200],
                    "location": row["location"][:200],
                    "location_norm": _norm_location(row["location"]),
                    "details": row["details"][:2000],
                    "rooms": _extract_rooms(full_text),
                    "price_usd": _extract_price_usd(full_text),
                }
                points.append(PointStruct(id=int(row["id"]), vector=vec, payload=payload))

            await qdrant.upsert_batch(points)
            progress.update(task, advance=len(chunk))

    console.print(f"[green]✓ Ingested {len(rows)} properties into "
                  f"[bold]{settings.qdrant_collection}[/bold][/green]")

    # Quick stats
    info = await qdrant.client.get_collection(settings.qdrant_collection)
    console.print(f"  Collection size: [bold]{info.points_count}[/bold] points")
    console.print(f"  Vector dim:      [bold]{info.config.params.vectors.size}[/bold]")


def main(
    limit: int = typer.Option(100, help="Number of properties to ingest"),
    recreate: bool = typer.Option(False, help="Drop collection first"),
    batch_size: int = typer.Option(8, help="Batch size for embeddings"),
) -> None:
    """Ingest properties from HuggingFace dataset into Qdrant."""
    try:
        asyncio.run(_ingest(limit=limit, recreate=recreate, batch_size=batch_size))
    finally:
        asyncio.run(close_llm())
        asyncio.run(close_qdrant())


if __name__ == "__main__":
    typer.run(main)
