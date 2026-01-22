#!/usr/bin/env python3
"""
MediSync Global Insights Demo
Demonstrates cross-clinic anonymized insights with K-anonymity.
Run: python3 demo/global_insights_demo.py
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


def demo_global_insights():
    """Demonstrate cross-clinic insights with privacy"""
    print_header("GLOBAL INSIGHTS DEMO")

    console.print(Panel.fit(
        "[bold]Cross-Clinic Medical Insights[/bold]\n\n"
        "MediSync enables clinics to share anonymized insights\n"
        "while preserving patient privacy through K-anonymity.\n\n"
        "[yellow]Key Parameters:[/yellow]\n"
        "  K = 20 (minimum records per group)\n"
        "  min_clinics = 5 (minimum contributing clinics)",
        border_style="cyan"
    ))

    # 1. K-Anonymity Explained
    console.print("\n[bold yellow]1. K-Anonymity with 5-Clinic Threshold[/bold yellow]\n")

    tree = Tree("[bold]Privacy Requirements[/bold]")

    k_anon = tree.add("[cyan]K-Anonymity (K=20)[/cyan]")
    k_anon.add("At least 20 records per condition/treatment group")
    k_anon.add("Prevents individual identification")
    k_anon.add("Groups with <20 records are suppressed")

    clinic_div = tree.add("[green]Clinic Diversity (min_clinics=5)[/green]")
    clinic_div.add("At least 5 different clinics must contribute")
    clinic_div.add("Prevents clinic re-identification")
    clinic_div.add("Ensures statistical validity across practices")

    console.print(tree)

    # 2. What Gets Shared
    console.print("\n[bold yellow]2. What Gets Shared (Example Insight)[/bold yellow]\n")

    console.print(Panel(
        "[bold]Insight: Diabetes + Metformin Effectiveness[/bold]\n\n"
        "[dim]Type:[/dim] treatment_outcome\n"
        "[dim]Condition:[/dim] Type 2 Diabetes (generalized)\n"
        "[dim]Treatment:[/dim] Metformin monotherapy\n\n"
        "[green]Aggregated Statistics:[/green]\n"
        "  • Sample size: 1,250 patients\n"
        "  • Contributing clinics: 45\n"
        "  • Success rate: 72%\n"
        "  • Median time to HbA1c control: 3.2 months\n\n"
        "[yellow]Age Distribution:[/yellow]\n"
        "  • 30-40: 15%\n"
        "  • 40-50: 35%\n"
        "  • 50-60: 32%\n"
        "  • 60-70: 18%\n\n"
        "[red]NOT Shared:[/red]\n"
        "  • Individual patient IDs\n"
        "  • Specific clinic names\n"
        "  • Exact ages (only brackets)\n"
        "  • Raw clinical notes",
        title="Anonymized Insight",
        border_style="green"
    ))

    # 3. Data Flow
    console.print("\n[bold yellow]3. How Data Flows Between Clinics[/bold yellow]\n")

    flow_tree = Tree("[bold]Insight Generation Pipeline[/bold]")

    step1 = flow_tree.add("[cyan]Step 1: Local Analysis[/cyan]")
    step1.add("Doctor queries local clinical_records")
    step1.add("InsightsGeneratorAgent analyzes patterns")

    step2 = flow_tree.add("[green]Step 2: Aggregation[/green]")
    step2.add("Group by condition + treatment")
    step2.add("Count records per group")
    step2.add("Count contributing clinics")

    step3 = flow_tree.add("[yellow]Step 3: K-Anonymity Check[/yellow]")
    step3.add("Require: sample_size >= 20")
    step3.add("Require: clinic_count >= 5")
    step3.add("Suppress outliers (top/bottom 5%)")

    step4 = flow_tree.add("[magenta]Step 4: PII Removal[/magenta]")
    step4.add("Strip SSN, phone, email patterns")
    step4.add("Generalize ages to brackets")
    step4.add("Hash patient/clinic IDs")

    step5 = flow_tree.add("[red]Step 5: Global Storage[/red]")
    step5.add("Insert into global_medical_insights collection")
    step5.add("Available to all authorized doctors")

    console.print(flow_tree)

    # 4. Qdrant Collections
    console.print("\n[bold yellow]4. Qdrant Collection Architecture[/bold yellow]\n")

    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("Collection", style="cyan")
    table.add_column("Scope", style="green")
    table.add_column("Privacy Level", style="yellow")
    table.add_column("Access", style="white")

    table.add_row(
        "clinical_records",
        "Per-Clinic",
        "Full PHI",
        "Clinic doctors only"
    )
    table.add_row(
        "feedback_analytics",
        "Per-Clinic",
        "Hashed queries",
        "System + Analytics"
    )
    table.add_row(
        "global_medical_insights",
        "Cross-Clinic",
        "K-Anonymized",
        "All doctors (read)"
    )

    console.print(table)

    # 5. Query Global Insights
    console.print("\n[bold yellow]5. Querying Global Insights[/bold yellow]\n")

    try:
        from medisync.service_agents.insights_agent import GlobalInsightsService

        console.print("[dim]Querying global insights for 'diabetes treatment'...[/dim]\n")

        results = GlobalInsightsService.query_global_insights(
            query="diabetes treatment effectiveness",
            limit=3
        )

        if results:
            console.print(f"[green]Found {len(results)} global insights[/green]\n")
            for i, r in enumerate(results, 1):
                insight_type = r.payload.get('insight_type', 'unknown')
                condition = r.payload.get('condition', 'N/A')
                sample = r.payload.get('sample_size', 0)
                clinics = r.payload.get('clinic_count', 0)
                console.print(f"  {i}. [{insight_type}] {condition}")
                console.print(f"     [dim]Sample: {sample}, Clinics: {clinics}[/dim]")
        else:
            console.print("[dim]No global insights found (collection may be empty)[/dim]")

    except Exception as e:
        console.print(f"[dim]Demo mode - query skipped: {e}[/dim]")

    # 6. Code Example
    console.print("\n[bold yellow]6. Code Example[/bold yellow]\n")

    code = '''
from medisync.core_agents.privacy_agent import PrivacyFilter

# K-Anonymity check before sharing
records = get_clinic_records(condition="diabetes", treatment="metformin")

anonymized = PrivacyFilter.apply_k_anonymity(
    records=records,
    k=20,              # Minimum 20 records
    min_clinics=5,     # Minimum 5 clinics
    grouping_keys=['condition', 'treatment']
)

# Only groups meeting both thresholds are returned
for group in anonymized:
    GlobalInsightsService.publish_insight(
        insight_type="treatment_outcome",
        condition=group['condition'],  # Generalized
        treatment=group['treatment'],  # Generalized
        statistics=group['aggregated_stats'],
        sample_size=group['count'],
        clinic_count=group['clinic_count']
    )
'''

    console.print(Panel(code, title="Python Code", border_style="green"))

    # Summary
    console.print("\n[bold yellow]Key Takeaways[/bold yellow]\n")
    console.print("  [green]✓[/green] Patient privacy preserved through K=20 anonymity")
    console.print("  [green]✓[/green] Clinic identity protected by 5-clinic minimum")
    console.print("  [green]✓[/green] Medical insights flow across institutions")
    console.print("  [green]✓[/green] Better population health analytics")
    console.print("  [green]✓[/green] All stored in Qdrant with hybrid search")

    print_header("DEMO COMPLETE")


if __name__ == "__main__":
    demo_global_insights()
