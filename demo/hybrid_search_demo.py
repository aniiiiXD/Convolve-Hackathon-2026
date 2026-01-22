#!/usr/bin/env python3
"""
MediSync Hybrid Search Demo
Demonstrates Qdrant's multi-vector search with RRF fusion.
Run: python3 demo/hybrid_search_demo.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich import box

console = Console()


def print_header(title: str):
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold white]  {title}[/bold white]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")


def demo_hybrid_search():
    """Demonstrate hybrid search capabilities"""
    print_header("HYBRID SEARCH DEMO")

    console.print(Panel.fit(
        "[bold]Qdrant Hybrid Search[/bold]\n\n"
        "Combines multiple retrieval strategies:\n"
        "[blue]• Sparse vectors (BM42/SPLADE)[/blue] - Keyword precision\n"
        "[green]• Dense vectors (Gemini 768d)[/green] - Semantic understanding\n"
        "[magenta]• RRF Fusion[/magenta] - Optimal ranking",
        border_style="cyan"
    ))

    # 1. Search Architecture
    console.print("\n[bold yellow]1. Search Architecture[/bold yellow]\n")

    tree = Tree("[bold]Hybrid Search Pipeline[/bold]")

    query_node = tree.add("[white]Query: 'chest pain elevated troponin cardiac'[/white]")

    prefetch = query_node.add("[cyan]Prefetch Stage[/cyan]")
    sparse = prefetch.add("[blue]Sparse Prefetch (BM42)[/blue]")
    sparse.add("Tokenize → IDF weights → Top 100 candidates")
    sparse.add("Good for: exact medical terms, drug names")

    dense = prefetch.add("[green]Dense Prefetch (Gemini)[/green]")
    dense.add("Embed → 768-dim vector → Top 100 candidates")
    dense.add("Good for: semantic similarity, concepts")

    fusion = query_node.add("[magenta]RRF Fusion[/magenta]")
    fusion.add("Combine rankings from both prefetches")
    fusion.add("Score = sum(1 / (k + rank_i)) for each method")
    fusion.add("Returns optimally ranked results")

    console.print(tree)

    # 2. Live Search Demo
    console.print("\n[bold yellow]2. Live Hybrid Search[/bold yellow]\n")

    query = "chest pain elevated troponin diabetic patient"
    console.print(f"[dim]Query: '{query}'[/dim]\n")

    try:
        from medisync.service_agents.advanced_retrieval_agent import AdvancedRetrievalPipeline

        pipeline = AdvancedRetrievalPipeline(clinic_id="Clinic-A")

        start = time.time()
        results, metrics = pipeline.search(query, limit=5)
        elapsed = (time.time() - start) * 1000

        console.print(f"[green]Search completed in {elapsed:.0f}ms[/green]\n")

        # Show metrics
        console.print("[bold]Pipeline Metrics:[/bold]")
        console.print(f"  Total candidates evaluated: {metrics.total_candidates}")
        console.print(f"  Stage timings:")
        for stage, timing in metrics.stage_timings.items():
            console.print(f"    • {stage}: {timing*1000:.0f}ms")

        # Show results
        if results:
            console.print(f"\n[bold]Top {len(results)} Results:[/bold]\n")

            table = Table(box=box.SIMPLE, show_header=True)
            table.add_column("#", style="dim", width=3)
            table.add_column("Score", style="cyan", width=8)
            table.add_column("Patient", style="green", width=10)
            table.add_column("Content Preview", width=45)

            for i, r in enumerate(results, 1):
                score = r.score
                patient = r.payload.get('patient_id', 'N/A')
                content = r.payload.get('text_content', '')[:45] + '...'

                score_color = "green" if score > 0.7 else "yellow" if score > 0.5 else "white"
                table.add_row(str(i), f"[{score_color}]{score:.3f}[/{score_color}]", patient, content)

            console.print(table)
        else:
            console.print("[dim]No results found[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

    # 3. Why Hybrid?
    console.print("\n[bold yellow]3. Why Hybrid Search?[/bold yellow]\n")

    comparison = Table(box=box.ROUNDED, show_header=True)
    comparison.add_column("Query Type", style="white")
    comparison.add_column("Sparse Only", style="blue")
    comparison.add_column("Dense Only", style="green")
    comparison.add_column("Hybrid", style="magenta")

    comparison.add_row(
        "Exact drug name\n'Metformin 500mg'",
        "[green]Excellent[/green]",
        "[yellow]Good[/yellow]",
        "[green]Excellent[/green]"
    )
    comparison.add_row(
        "Conceptual query\n'heart attack symptoms'",
        "[yellow]Fair[/yellow]",
        "[green]Excellent[/green]",
        "[green]Excellent[/green]"
    )
    comparison.add_row(
        "Mixed query\n'chest pain troponin diabetic'",
        "[yellow]Good[/yellow]",
        "[yellow]Good[/yellow]",
        "[green]Excellent[/green]"
    )
    comparison.add_row(
        "Typo/variant\n'cardiack arrest'",
        "[red]Poor[/red]",
        "[green]Good[/green]",
        "[green]Good[/green]"
    )

    console.print(comparison)

    # 4. Named Vectors
    console.print("\n[bold yellow]4. Named Vectors in clinical_records[/bold yellow]\n")

    vectors = Table(box=box.SIMPLE, show_header=True)
    vectors.add_column("Vector Name", style="cyan")
    vectors.add_column("Type", style="green")
    vectors.add_column("Dimensions", style="yellow")
    vectors.add_column("Use Case", style="white")

    vectors.add_row("dense_text", "Dense", "768", "Semantic search")
    vectors.add_row("sparse_code", "Sparse", "Variable", "Keyword/BM42 search")
    vectors.add_row("image_clip", "Dense", "512", "Multimodal (X-rays)")

    console.print(vectors)

    # 5. Code Example
    console.print("\n[bold yellow]5. Code Example[/bold yellow]\n")

    code = '''
from qdrant_client import models

# Hybrid search with prefetch chains
results = client.query_points(
    collection_name="clinical_records",
    prefetch=[
        # Stage 1: Sparse retrieval (BM42)
        models.Prefetch(
            query=sparse_vector,
            using="sparse_code",
            limit=100,
            filter=clinic_filter
        ),
        # Stage 2: Dense retrieval (Gemini)
        models.Prefetch(
            query=dense_vector,
            using="dense_text",
            limit=100,
            filter=clinic_filter
        )
    ],
    # Stage 3: RRF Fusion
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10,
    with_payload=True
)
'''

    console.print(Panel(code, title="Qdrant Hybrid Search Code", border_style="green"))

    print_header("DEMO COMPLETE")


if __name__ == "__main__":
    demo_hybrid_search()
