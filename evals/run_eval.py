"""Evaluation harness for the lifestyle-based search pipeline.

Runs a suite of life-scenario queries against /search, compares results to
gold labels, computes information-retrieval metrics (precision@K, recall@K, MRR),
and writes a markdown report.

    python -m evals.run_eval --queries evals/queries.yaml
    python -m evals.run_eval --endpoint http://localhost:8000 --k 5 10

Design notes:
- We treat Qdrant IDs as strings (property_id from /search response).
- Gold is an OR-set of "relevant" IDs plus optional "partial" (half credit).
- MRR is computed against the full relevant set.
- Cost is tallied from response headers — the demo's own cost middleware in action.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import statistics as stats
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import typer
import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

console = Console()


@dataclass
class EvalQuery:
    id: str
    query: str
    relevant: list[str] = field(default_factory=list)          # full credit
    partial: list[str] = field(default_factory=list)           # half credit
    notes: str = ""


@dataclass
class EvalResult:
    q: EvalQuery
    hits: list[dict[str, Any]]
    intent: dict[str, Any]
    cost_usd: float
    shadow_openai_usd: float
    latency_ms: float
    stages_ms: dict[str, float]
    error: str | None = None

    def hit_ids(self) -> list[str]:
        return [str(h["property_id"]) for h in self.hits]

    def precision_at(self, k: int) -> float:
        top = self.hit_ids()[:k]
        if not top:
            return 0.0
        hits = sum(
            1.0 if pid in self.q.relevant else 0.5 if pid in self.q.partial else 0.0
            for pid in top
        )
        return hits / k

    def recall_at(self, k: int) -> float:
        if not self.q.relevant:
            return 0.0
        top = self.hit_ids()[:k]
        hits = sum(1 for pid in top if pid in self.q.relevant)
        return hits / len(self.q.relevant)

    def reciprocal_rank(self) -> float:
        for i, pid in enumerate(self.hit_ids(), start=1):
            if pid in self.q.relevant:
                return 1.0 / i
        return 0.0


def _load_queries(path: Path) -> list[EvalQuery]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    queries = [
        EvalQuery(
            id=str(q["id"]),
            query=str(q["query"]),
            relevant=[str(x) for x in q.get("relevant", [])],
            partial=[str(x) for x in q.get("partial", [])],
            notes=str(q.get("notes", "")),
        )
        for q in data.get("queries", [])
    ]
    if not queries:
        console.print(f"[red]No queries found in {path}[/red]")
        sys.exit(2)
    return queries


async def _run_one(
    client: httpx.AsyncClient, endpoint: str, q: EvalQuery, top_k: int
) -> EvalResult:
    """Run one query. NEVER raises — always returns an EvalResult (maybe with .error)."""
    try:
        r = await client.post(
            f"{endpoint}/search",
            json={"query": q.query, "top_k": top_k, "with_explain": False},
            timeout=600.0,  # cold start on 9B can exceed default 300s
        )
    except (httpx.TimeoutException, httpx.TransportError) as e:
        return EvalResult(
            q=q, hits=[], intent={}, cost_usd=0.0, shadow_openai_usd=0.0,
            latency_ms=0.0, stages_ms={}, error=f"{type(e).__name__}: {e}",
        )

    if r.status_code >= 400:
        # Still pick up what metadata we can from headers
        return EvalResult(
            q=q,
            hits=[],
            intent={},
            cost_usd=float(r.headers.get("X-Cost-USD", 0.0) or 0.0),
            shadow_openai_usd=float(r.headers.get("X-Cost-Shadow-OpenAI-USD", 0.0) or 0.0),
            latency_ms=float(r.headers.get("X-Duration-Ms", 0.0) or 0.0),
            stages_ms={},
            error=f"HTTP {r.status_code}: {r.text[:200]}",
        )

    body = r.json()
    return EvalResult(
        q=q,
        hits=body.get("hits", []),
        intent=body.get("intent", {}),
        cost_usd=float(r.headers.get("X-Cost-USD", body.get("cost_usd", 0.0))),
        shadow_openai_usd=float(
            r.headers.get("X-Cost-Shadow-OpenAI-USD", body.get("shadow_openai_usd", 0.0))
        ),
        latency_ms=float(r.headers.get("X-Duration-Ms", body.get("latency_ms", 0.0))),
        stages_ms=body.get("stages_ms", {}),
    )


def _aggregate(results: list[EvalResult], ks: list[int]) -> dict[str, Any]:
    def m(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(stats.mean(vals), 4),
            "median": round(stats.median(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    ok = [r for r in results if r.error is None]
    agg: dict[str, Any] = {
        "n_queries": len(results),
        "n_ok": len(ok),
        "n_failed": len(results) - len(ok),
        "mrr": m([r.reciprocal_rank() for r in ok]),
        "cost_real_usd": round(sum(r.cost_usd for r in results), 6),
        "cost_shadow_openai_usd": round(sum(r.shadow_openai_usd for r in results), 6),
        "latency_ms": m([r.latency_ms for r in ok]),
    }
    for k in ks:
        agg[f"precision@{k}"] = m([r.precision_at(k) for r in ok])
        agg[f"recall@{k}"] = m([r.recall_at(k) for r in ok])
    return agg


def _write_markdown(
    out_path: Path,
    queries_path: Path,
    endpoint: str,
    ks: list[int],
    agg: dict[str, Any],
    results: list[EvalResult],
) -> None:
    lines: list[str] = []
    lines.append(f"# Eval report — {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"- Endpoint: `{endpoint}`")
    lines.append(f"- Queries: `{queries_path}` ({agg['n_queries']})")
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Metric | Mean | Median | Min | Max |")
    lines.append("|---|---|---|---|---|")
    for k in ks:
        p = agg[f"precision@{k}"]
        lines.append(
            f"| precision@{k} | {p['mean']:.3f} | {p['median']:.3f} | {p['min']:.3f} | {p['max']:.3f} |"
        )
        r = agg[f"recall@{k}"]
        lines.append(
            f"| recall@{k} | {r['mean']:.3f} | {r['median']:.3f} | {r['min']:.3f} | {r['max']:.3f} |"
        )
    mrr = agg["mrr"]
    lines.append(
        f"| MRR | {mrr['mean']:.3f} | {mrr['median']:.3f} | {mrr['min']:.3f} | {mrr['max']:.3f} |"
    )
    lat = agg["latency_ms"]
    lines.append(
        f"| latency_ms | {lat['mean']:.1f} | {lat['median']:.1f} | {lat['min']:.1f} | {lat['max']:.1f} |"
    )
    lines.append("")
    lines.append("## Cost")
    lines.append("")
    lines.append(f"- Real (Ollama, local): **${agg['cost_real_usd']:.6f}**")
    lines.append(f"- Shadow on OpenAI:      **${agg['cost_shadow_openai_usd']:.6f}**")
    if agg["cost_shadow_openai_usd"] > 0:
        savings = agg["cost_shadow_openai_usd"] - agg["cost_real_usd"]
        pct = (savings / agg["cost_shadow_openai_usd"]) * 100
        lines.append(f"- **Savings vs OpenAI: ${savings:.6f} ({pct:.1f}%)**")
    lines.append("")
    lines.append(f"- Successful: {agg['n_ok']} / {agg['n_queries']}")
    if agg["n_failed"]:
        lines.append(f"- **Failed: {agg['n_failed']}**")
    lines.append("")
    lines.append("## Per-query")
    lines.append("")
    lines.append("| id | p@1 | p@5 | r@10 | MRR | latency_ms | cost$ (shadow) | status |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        status = "✓" if r.error is None else f"✗ {r.error[:40]}"
        lines.append(
            f"| {r.q.id} | {r.precision_at(1):.2f} | {r.precision_at(5):.2f} | "
            f"{r.recall_at(10):.2f} | {r.reciprocal_rank():.2f} | "
            f"{r.latency_ms:.0f} | {r.shadow_openai_usd:.6f} | {status} |"
        )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(
    queries: Path = typer.Option(
        Path("evals/queries.yaml"),
        help="YAML with queries + gold labels",
        exists=True,
        readable=True,
    ),
    endpoint: str = typer.Option("http://localhost:8000", help="Base URL of the API"),
    k: list[int] = typer.Option([1, 5, 10], help="K values for precision/recall"),
    top_k: int = typer.Option(10, help="top_k to request from /search"),
    out_dir: Path = typer.Option(Path("evals/results"), help="Directory for reports"),
    json_out: bool = typer.Option(False, help="Also write JSON result file"),
    concurrency: int = typer.Option(1, help="Parallel queries (be gentle with LLM)"),
) -> None:
    """Run eval suite; write markdown report to evals/results/."""
    asyncio.run(_run_all(queries, endpoint, k, top_k, out_dir, json_out, concurrency))


async def _run_all(
    queries_path: Path,
    endpoint: str,
    ks: list[int],
    top_k: int,
    out_dir: Path,
    json_out: bool,
    concurrency: int,
) -> None:
    qs = _load_queries(queries_path)
    console.print(
        f"[cyan]Running {len(qs)} queries against {endpoint} (k={ks}, concurrency={concurrency})[/cyan]"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[EvalResult] = []
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=600.0) as client:
        # Pre-warm: one throwaway request so the LLM is in memory
        # before we start the timed eval.
        console.print("[yellow]Warming up LLM (may take 30-60s on cold start)...[/yellow]")
        try:
            await _run_one(client, endpoint, qs[0], top_k)
            console.print("[green]✓ Warmup complete[/green]")
        except Exception as e:
            console.print(f"[yellow]Warmup failed ({e}), proceeding anyway[/yellow]")


        async def _one(q: EvalQuery) -> EvalResult:
            async with sem:
                return await _run_one(client, endpoint, q, top_k)

        with Progress(
            TextColumn("[bold cyan]{task.description}[/bold cyan]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running eval", total=len(qs))
            pending = [asyncio.create_task(_one(q)) for q in qs]
            for fut in asyncio.as_completed(pending):
                results.append(await fut)
                progress.update(task, advance=1)

    # Sort results by query id order (asyncio.as_completed is order-independent)
    order = {q.id: i for i, q in enumerate(qs)}
    results.sort(key=lambda r: order[r.q.id])

    agg = _aggregate(results, ks)

    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d_%H-%M-%S")
    md_path = out_dir / f"{stamp}.md"
    _write_markdown(md_path, queries_path, endpoint, ks, agg, results)
    console.print(f"[green]✓ Report:[/green] {md_path}")

    if json_out:
        json_path = out_dir / f"{stamp}.json"
        json_path.write_text(
            json.dumps(
                {
                    "meta": {
                        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
                        "endpoint": endpoint,
                        "queries_path": str(queries_path),
                        "ks": ks,
                    },
                    "aggregate": agg,
                    "results": [
                        {
                            "id": r.q.id,
                            "query": r.q.query,
                            "hits": r.hit_ids(),
                            "precision@5": r.precision_at(5),
                            "recall@10": r.recall_at(10),
                            "reciprocal_rank": r.reciprocal_rank(),
                            "latency_ms": r.latency_ms,
                            "cost_usd": r.cost_usd,
                            "shadow_openai_usd": r.shadow_openai_usd,
                            "intent": r.intent,
                        }
                        for r in results
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        console.print(f"[green]✓ JSON:[/green] {json_path}")

    # Pretty console summary
    console.print()
    console.rule("[bold]Summary[/bold]")
    for k in ks:
        p = agg[f"precision@{k}"]["mean"]
        r = agg[f"recall@{k}"]["mean"]
        console.print(f"  [cyan]precision@{k}[/cyan]: {p:.3f}   "
                      f"[cyan]recall@{k}[/cyan]: {r:.3f}")
    console.print(f"  [cyan]MRR[/cyan]:          {agg['mrr']['mean']:.3f}")
    console.print(f"  [cyan]mean latency[/cyan]: {agg['latency_ms']['mean']:.0f} ms")
    console.print()
    console.print(
        f"  [bold]cost:[/bold] real=${agg['cost_real_usd']:.6f}   "
        f"shadow_openai=${agg['cost_shadow_openai_usd']:.6f}"
    )


if __name__ == "__main__":
    typer.run(main)
