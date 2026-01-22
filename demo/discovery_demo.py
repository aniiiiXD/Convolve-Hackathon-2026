#!/usr/bin/env python3
"""
MediSync Discovery API Demo
Demonstrates Qdrant's Discovery API for context-aware medical search.
Run: python3 demo/discovery_demo.py
"""

import sys
import os

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


def demo_discovery_api():
    """Demonstrate Discovery API capabilities"""
    print_header("DISCOVERY API DEMO")

    console.print(Panel.fit(
        "[bold]Qdrant Discovery API[/bold]\n\n"
        "Find cases similar to 'target' but biased by context.\n"
        "[green]+ Positive context[/green]: Find more like these\n"
        "[red]- Negative context[/red]: Find less like these",
        border_style="cyan"
    ))

    # 1. Basic Discovery Search
    console.print("\n[bold yellow]1. Basic Discovery Search[/bold yellow]\n")

    from medisync.service_agents.discovery_agent import DiscoveryService

    console.print("[dim]Query: Find 'cardiac emergency' cases[/dim]")
    console.print("[dim]  + Positive: chest pain, elevated troponin[/dim]")
    console.print("[dim]  - Negative: trauma, accident[/dim]\n")

    results = DiscoveryService.discover_contextual(
        target_text="cardiac emergency",
        positive_texts=["chest pain", "elevated troponin", "shortness of breath"],
        negative_texts=["trauma", "accident", "injury"],
        limit=5,
        clinic_id="Clinic-A"
    )

    console.print(f"[green]Found {len(results)} results[/green]\n")

    if results:
        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Score", style="cyan", width=8)
        table.add_column("Patient", width=10)
        table.add_column("Content", width=50)

        for r in results[:5]:
            content = r.payload.get('text_content', '')[:50] + '...'
            patient = r.payload.get('patient_id', 'N/A')
            table.add_row(f"{r.score:.3f}", patient, content)

        console.print(table)

    # 2. How Discovery Works
    console.print("\n[bold yellow]2. How Discovery API Works[/bold yellow]\n")

    tree = Tree("[bold]Discovery Query Process[/bold]")
    step1 = tree.add("[cyan]1. Embed target text[/cyan]")
    step1.add("'cardiac emergency' → 768-dim vector")

    step2 = tree.add("[green]2. Embed positive context[/green]")
    step2.add("'chest pain' → vector")
    step2.add("'elevated troponin' → vector")

    step3 = tree.add("[red]3. Embed negative context[/red]")
    step3.add("'trauma' → vector")
    step3.add("'accident' → vector")

    step4 = tree.add("[magenta]4. Create context pairs[/magenta]")
    step4.add("(positive[0], negative[0]) → pair 1")
    step4.add("(positive[1], negative[1]) → pair 2")

    step5 = tree.add("[yellow]5. Execute DiscoverQuery[/yellow]")
    step5.add("Qdrant biases results toward positive")
    step5.add("Qdrant biases results away from negative")

    console.print(tree)

    # 3. Clinical Use Case: Differential Diagnosis
    console.print("\n[bold yellow]3. Clinical Use Case: Differential Diagnosis[/bold yellow]\n")

    console.print(Panel(
        "[bold]Scenario:[/bold] Patient with chest pain\n\n"
        "[green]Confirmed findings (+):[/green]\n"
        "  • Troponin elevated\n"
        "  • ST changes on ECG\n"
        "  • History of CAD\n\n"
        "[red]Ruled out (-):[/red]\n"
        "  • Aortic dissection (CT negative)\n"
        "  • Pulmonary embolism (D-dimer normal)",
        title="Clinical Context",
        border_style="blue"
    ))

    from medisync.service_agents.differential_diagnosis_agent import DifferentialDiagnosisAgent

    try:
        agent = DifferentialDiagnosisAgent("Clinic-A")
        console.print("\n[dim]Running differential diagnosis with Discovery API...[/dim]\n")

        # This would normally call agent.generate_differential()
        # For demo, we show the concept

        console.print("[bold green]Discovery API enables:[/bold green]")
        console.print("  • Find cases similar to confirmed cardiac findings")
        console.print("  • Exclude cases similar to ruled-out conditions")
        console.print("  • Rank diagnoses by contextual similarity")

    except Exception as e:
        console.print(f"[dim]Demo mode: {e}[/dim]")

    # 4. Code Example
    console.print("\n[bold yellow]4. Code Example[/bold yellow]\n")

    code = '''
from medisync.service_agents.discovery_agent import DiscoveryService

# Find cardiac cases biased by clinical context
results = DiscoveryService.discover_contextual(
    target_text="acute myocardial infarction",
    positive_texts=[
        "elevated troponin",
        "ST elevation",
        "chest pain radiating to arm"
    ],
    negative_texts=[
        "normal troponin",
        "pulmonary embolism",
        "musculoskeletal pain"
    ],
    limit=10,
    clinic_id="Clinic-A"
)
'''

    console.print(Panel(code, title="Python Code", border_style="green"))

    print_header("DEMO COMPLETE")


if __name__ == "__main__":
    demo_discovery_api()
