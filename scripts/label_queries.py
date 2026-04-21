"""LLM-assisted labeling for eval queries.

For each query, we ask the LLM to classify each property in the collection
as 'relevant', 'partial', or 'not_relevant'. Output: updated queries.yaml
with bootstrapped gold labels — meant for HUMAN REVIEW before committing.

Why this exists:
- Hand-labeling 15 queries × 100 properties = 1500 judgments. Not feasible.
- LLM judgments are noisy but great for "bootstrap then review".
- The script writes to a SEPARATE file (queries.labeled.yaml) — the operator
  diffs and promotes to queries.yaml manually. We do NOT auto-overwrite gold.

Usage:
    python -m scripts.label_queries --limit-properties 100
    # review evals/queries.labeled.yaml
    # mv evals/queries.labeled.yaml evals/queries.yaml
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
import sys

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.llm import close_llm, get_llm
from app.services.qdrant_client import close_qdrant, get_qdrant

settings = get_settings()
configure_logging(level="WARNING", json_output=False)
logger = get_logger(__name__)
console = Console()


JUDGE_SYSTEM = """You are a real-estate search relevance judge.

Given:
  - a USER QUERY describing their life scenario / preferences
  - a list of PROPERTIES (id, title, location, details excerpt)

Classify EACH property against the query into one of:
  - "relevant"     — strong match on the user's key needs
  - "partial"      — matches some aspects, not ideal but defensible
  - "not_relevant" — doesn't match meaningful aspects

Rules:
- Be strict on "relevant". Only if property clearly fits the scenario.
- Use "partial" generously when there's *some* signal (right area, some features).
- Consider IMPLICIT needs: "family with kids" → prefer houses over studios,
  "elderly" → prefer ground floor / bungalow / elevator, etc.
- Do NOT make up facts — judge ONLY on what the property text actually says.

Return strict JSON:
{ "judgments": [ {"id": "<int>", "label": "relevant|partial|not_relevant", "reason": "<one short phrase>"}, ... ] }

Judge ALL provided properties. No ordering. No extra commentary.
"""


def _short(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s[:n]


async def _judge_batch(
    llm: Any, query: str, properties: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Judge a batch of properties in one LLM call."""
    prop_lines = []
    for p in properties:
        prop_lines.append(
            f"[{p['id']}] {_short(p['title'], 80)} | {_short(p['location'], 50)} "
            f"| {_short(p['details'], 280)}"
        )
    user = "QUERY: " + query + "\n\nPROPERTIES:\n" + "\n".join(prop_lines)

    try:
        raw = await llm.parse_json(system=JUDGE_SYSTEM, user=user, temperature=0.0)
    except Exception as e:
        logger.warning("judge_batch_failed", error=str(e), query=query[:60])
        return [{"id": p["id"], "label": "not_relevant", "reason": "llm_error"} for p in properties]
    judgments = raw.get("judgments", []) or []
    return judgments


async def _fetch_properties(limit: int) -> list[dict[str, Any]]:
    """Pull properties from Qdrant (scroll API)."""
    qdrant = get_qdrant()
    points: list[dict[str, Any]] = []
    offset = None
    while len(points) < limit:
        batch, offset = await qdrant.client.scroll(
            collection_name=settings.qdrant_collection,
            limit=min(64, limit - len(points)),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in batch:
            points.append(
                {
                    "id": p.id,
                    "title": (p.payload or {}).get("title", ""),
                    "location": (p.payload or {}).get("location", ""),
                    "details": (p.payload or {}).get("details", ""),
                }
            )
        if offset is None:
            break
    return points[:limit]


async def _label(
    queries_path: Path,
    out_path: Path,
    batch_size: int,
    limit_properties: int,
) -> None:
    data = yaml.safe_load(queries_path.read_text(encoding="utf-8")) or {}
    queries = data.get("queries", [])
    if not queries:
        console.print(f"[red]No queries in {queries_path}[/red]")
        return

    props = await _fetch_properties(limit_properties)
    console.print(f"[cyan]Loaded {len(props)} properties from Qdrant[/cyan]")

    llm = get_llm()

    total_batches = len(queries) * ((len(props) + batch_size - 1) // batch_size)
    updated_queries: list[dict[str, Any]] = []

    def log(msg: str) -> None:
        # Plain print with immediate flush — friendly to log files (unlike rich.Progress)
        print(msg, flush=True)
        sys.stdout.flush()

    log(f"[labeling] queries={len(queries)}  properties={len(props)}  "
        f"batch_size={batch_size}  total_batches={total_batches}")
    log(f"[labeling] incremental write -> {out_path}")

    def _write_incremental() -> None:
        """Write whatever we've labeled so far. Called after each query so a
        crash / timeout doesn't lose the earlier batches' work."""
        out_path.write_text(
            yaml.safe_dump(
                {"queries": updated_queries}, sort_keys=False, allow_unicode=True,
            ),
            encoding="utf-8",
        )

    batch_n = 0
    try:
        for q_idx, q in enumerate(queries, 1):
            q_id = str(q["id"])
            query_text = str(q["query"])
            relevant: list[str] = []
            partial: list[str] = []

            for start in range(0, len(props), batch_size):
                batch_n += 1
                chunk = props[start : start + batch_size]
                t0 = time.monotonic()
                judgments = await _judge_batch(llm, query_text, chunk)
                dt_s = time.monotonic() - t0
                # Normalize ids (LLM may return str or int)
                by_id = {str(j.get("id")): j for j in judgments if "id" in j}
                batch_rel = 0
                batch_par = 0
                for p in chunk:
                    j = by_id.get(str(p["id"]))
                    if not j:
                        continue
                    label = j.get("label", "not_relevant")
                    if label == "relevant":
                        relevant.append(str(p["id"]))
                        batch_rel += 1
                    elif label == "partial":
                        partial.append(str(p["id"]))
                        batch_par += 1
                log(
                    f"  [{batch_n}/{total_batches}] q={q_id} "
                    f"props={start}-{start+len(chunk)} "
                    f"rel={batch_rel} par={batch_par} took={dt_s:.1f}s"
                )

            log(
                f"[query {q_idx}/{len(queries)}] {q_id}: "
                f"relevant={len(relevant)} partial={len(partial)}"
            )
            updated_queries.append(
                {
                    "id": q_id,
                    "query": query_text,
                    "relevant": sorted(set(relevant), key=int),
                    "partial": sorted(set(partial), key=int),
                    "notes": str(q.get("notes", "") or "auto-labeled, REVIEW BEFORE USE"),
                }
            )
            # Checkpoint after each query — don't lose work on crash/timeout
            _write_incremental()
    finally:
        # Final write in case of crash before completion
        _write_incremental()
        await close_llm()
        await close_qdrant()

    out = {"queries": updated_queries}
    out_path.write_text(
        yaml.safe_dump(out, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    # Stats
    console.print()
    console.rule("[bold]Labeling summary[/bold]")
    for q in updated_queries:
        console.print(
            f"  [cyan]{q['id']:<28}[/cyan] "
            f"relevant={len(q['relevant']):>2}  partial={len(q['partial']):>2}"
        )
    console.print()
    console.print(f"[green]→ Wrote {out_path}[/green]")
    console.print("[yellow]  REVIEW before replacing queries.yaml.[/yellow]")


def main(
    queries: Path = typer.Option(
        Path("evals/queries.yaml"), exists=True, readable=True
    ),
    out: Path = typer.Option(Path("evals/queries.labeled.yaml")),
    batch_size: int = typer.Option(10, help="Properties per LLM judging call"),
    limit_properties: int = typer.Option(100, help="How many properties to judge"),
) -> None:
    asyncio.run(
        _label(
            queries_path=queries,
            out_path=out,
            batch_size=batch_size,
            limit_properties=limit_properties,
        )
    )


if __name__ == "__main__":
    typer.run(main)
