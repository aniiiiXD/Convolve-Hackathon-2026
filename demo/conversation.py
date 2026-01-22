#!/usr/bin/env python3
"""
MediSync Interactive Demo - Clinical Conversation Simulation
Run: python3 demo/conversation.py

This simulates a realistic clinical workflow showing MediSync's capabilities.
Perfect for hackathon presentations and live demos.
"""

import asyncio
import time
import sys
import os
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.live import Live

# Suppress noisy HTTP logs during demo
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

console = Console()

# Typing effect for realistic conversation
def type_text(text: str, delay: float = 0.02, style: str = ""):
    """Simulate typing effect"""
    for char in text:
        if style:
            console.print(char, end="", style=style)
        else:
            console.print(char, end="")
        sys.stdout.flush()
        time.sleep(delay)
    print()

def slow_print(text: str, delay: float = 0.5):
    """Print with delay"""
    console.print(text)
    time.sleep(delay)

def wait_for_enter(prompt: str = ""):
    """Wait for user to press Enter to continue"""
    console.print(f"\n[dim]{prompt}Press Enter to continue...[/dim]")
    input()

def clear_screen():
    """Clear terminal"""
    console.clear()

# ============================================================================
# DEMO SCENES
# ============================================================================

async def scene_intro():
    """Introduction scene"""
    clear_screen()
    console.print(Panel.fit(
        "[bold cyan]MediSync[/bold cyan] [white]Clinical AI Demo[/white]\n\n"
        "[dim]Qdrant Convolve 4.0 Pan-IIT Hackathon[/dim]",
        border_style="cyan"
    ))

    console.print("\n[bold yellow]SCENARIO:[/bold yellow]")
    console.print("""
    Dr. Strange is starting his morning shift at the cardiac care unit.
    A new patient has just arrived in the ER with concerning symptoms.

    Let's see how MediSync helps with rapid diagnosis and decision support.
    """)
    wait_for_enter()

async def scene_login():
    """Doctor login scene"""
    clear_screen()
    console.print(Panel("[bold]Scene 1: Authentication[/bold]", border_style="blue"))

    console.print("\n[dim]Terminal:[/dim]")
    console.print("[green]$[/green] medisync login\n")
    time.sleep(0.5)

    type_text("Username: ", delay=0.03)
    time.sleep(0.3)
    type_text("Dr_Strange", delay=0.05, style="cyan")

    type_text("Password: ", delay=0.03)
    time.sleep(0.3)
    type_text("********", delay=0.05, style="dim")

    time.sleep(0.5)

    # Actual login
    from medisync.service_agents.gatekeeper_agent import AuthService
    user = AuthService.login("Dr_Strange")

    if user:
        console.print("\n[bold green]✓ Authentication Successful[/bold green]")
        console.print(f"  [dim]User:[/dim] {user.username}")
        console.print(f"  [dim]Role:[/dim] [cyan]{user.role}[/cyan]")
        console.print(f"  [dim]Clinic:[/dim] {user.clinic_id}")
        console.print(f"  [dim]Session:[/dim] {user.id[:16]}...")

    wait_for_enter()
    return user

async def scene_alert():
    """Vigilance alert scene"""
    clear_screen()
    console.print(Panel("[bold]Scene 2: Morning Alerts[/bold]", border_style="red"))

    console.print("\n[bold yellow]MEDISYNC VIGILANCE SYSTEM[/bold yellow]")
    console.print("[dim]Checking for critical alerts...[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        progress.add_task("Scanning patient records...", total=None)
        time.sleep(1.5)

    # Simulated alerts
    alerts = [
        ("CRITICAL", "New ER admission - 65yo male, chest pain, elevated troponin", "red"),
        ("HIGH", "Patient P-103: BP trending upward over 3 days (145/95 → 168/102)", "yellow"),
        ("WARNING", "Patient P-107: Missed follow-up appointment (48 hours)", "yellow"),
    ]

    console.print("[bold]Active Alerts:[/bold]\n")

    for severity, message, color in alerts:
        time.sleep(0.3)
        if severity == "CRITICAL":
            console.print(Panel(
                f"[bold]{message}[/bold]\n\n[dim]Arrived: 2 minutes ago | Triage: Immediate[/dim]",
                title=f"[bold red]● {severity}[/bold red]",
                border_style="red"
            ))
        else:
            console.print(f"  [{color}]●[/{color}] [bold]{severity}[/bold]: {message}")

    console.print("\n[bold cyan]→ Dr. Strange focuses on the CRITICAL alert[/bold cyan]")
    wait_for_enter()

async def scene_patient_intake():
    """New patient intake scene"""
    clear_screen()
    console.print(Panel("[bold]Scene 3: Patient Intake[/bold]", border_style="green"))

    console.print("\n[bold yellow]NEW PATIENT - ER ADMISSION[/bold yellow]\n")

    patient_info = """
┌─────────────────────────────────────────────────────────────┐
│  PATIENT: John Doe (P-ER-2024)          AGE: 65  SEX: Male  │
├─────────────────────────────────────────────────────────────┤
│  CHIEF COMPLAINT:                                           │
│  "Crushing chest pain for the past 30 minutes"              │
│                                                             │
│  VITAL SIGNS:                                               │
│  • BP: 88/56 mmHg (Hypotensive)                            │
│  • HR: 112 bpm (Tachycardic)                               │
│  • RR: 24/min (Tachypneic)                                 │
│  • SpO2: 94% on room air                                   │
│  • Temp: 37.1°C                                            │
│                                                             │
│  PRESENTING SYMPTOMS:                                       │
│  • Crushing substernal chest pain                          │
│  • Radiation to left arm and jaw                           │
│  • Diaphoresis (profuse sweating)                          │
│  • Shortness of breath                                     │
│  • Nausea                                                  │
│                                                             │
│  MEDICAL HISTORY:                                           │
│  • Hypertension (10 years)                                 │
│  • Type 2 Diabetes (8 years)                               │
│  • Hyperlipidemia                                          │
│  • Previous PCI to LAD (2 years ago)                       │
│                                                             │
│  INITIAL LABS:                                              │
│  • Troponin I: 2.4 ng/mL [bold red](CRITICAL - Normal <0.04)[/bold red]     │
└─────────────────────────────────────────────────────────────┘
"""
    console.print(patient_info)

    console.print("\n[bold cyan]→ Dr. Strange: \"Let me search for similar cases...\"[/bold cyan]")
    wait_for_enter()

async def scene_hybrid_search(session):
    """Hybrid search demonstration"""
    clear_screen()
    console.print(Panel("[bold]Scene 4: Hybrid Search - Finding Similar Cases[/bold]", border_style="cyan"))

    console.print("\n[dim]Dr. Strange queries MediSync:[/dim]")
    console.print("[green]>[/green] [cyan]search chest pain elevated troponin diabetic patient STEMI[/cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Running hybrid search...", total=None)

        # Actual search using AdvancedRetrievalPipeline
        from medisync.service_agents.advanced_retrieval_agent import AdvancedRetrievalPipeline
        pipeline = AdvancedRetrievalPipeline(clinic_id="Clinic-A")
        retrieval_results, metrics = pipeline.search(
            "chest pain elevated troponin diabetic patient STEMI",
            limit=5
        )

        time.sleep(0.5)

    console.print("[bold green]✓ Hybrid Search Complete[/bold green]")
    console.print(f"[dim]  Method: Sparse (BM42) + Dense (Gemini 768d) → RRF Fusion[/dim]")
    total_time = sum(metrics.stage_timings.values()) if metrics.stage_timings else 0
    console.print(f"[dim]  Time: {total_time:.0f}ms | Candidates: {metrics.total_candidates}[/dim]\n")

    # Display search architecture
    console.print("[bold yellow]Search Pipeline:[/bold yellow]")
    pipeline_tree = Tree("Query")
    sparse = pipeline_tree.add("[blue]Sparse Prefetch (BM42)[/blue] → 100 candidates")
    dense = pipeline_tree.add("[green]Dense Prefetch (Gemini)[/green] → 100 candidates")
    fusion = pipeline_tree.add("[magenta]RRF Fusion[/magenta] → Optimal ranking")
    fusion.add("[cyan]Top 5 results[/cyan]")
    console.print(pipeline_tree)

    console.print("\n[bold yellow]Similar Cases Found:[/bold yellow]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=8)
    table.add_column("Patient", width=10)
    table.add_column("Summary", width=50)

    for i, result in enumerate(retrieval_results[:5], 1):
        score = result.score
        patient = result.payload.get('patient_id', 'Unknown')
        text_content = result.payload.get('text_content', '')
        content = text_content[:60] + '...' if text_content else 'N/A'

        score_color = "green" if score > 0.7 else "yellow" if score > 0.5 else "white"
        table.add_row(str(i), f"[{score_color}]{score:.3f}[/{score_color}]", patient, content)

    console.print(table)

    console.print("\n[bold cyan]→ Dr. Strange: \"These cases confirm my suspicion. Let me run differential diagnosis...\"[/bold cyan]")
    wait_for_enter()
    return retrieval_results

async def scene_discovery_api():
    """Discovery API demonstration"""
    clear_screen()
    console.print(Panel("[bold]Scene 5: Discovery API - Context-Aware Diagnosis[/bold]", border_style="magenta"))

    console.print("\n[dim]Dr. Strange uses Discovery API with clinical context:[/dim]")
    console.print("""
[green]>[/green] [cyan]discover[/cyan]
   [dim]target:[/dim]    "acute cardiac event"
   [dim]positive:[/dim] ["chest pain", "elevated troponin", "ST changes"]
   [dim]negative:[/dim] ["trauma", "pulmonary embolism"]
""")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        progress.add_task("Running context-aware discovery...", total=None)

        # Actual discovery search
        from medisync.service_agents.discovery_agent import DiscoveryService

        results = DiscoveryService.discover_contextual(
            target_text="acute cardiac event",
            positive_texts=["chest pain", "elevated troponin", "ST changes"],
            negative_texts=["trauma", "pulmonary embolism"],
            limit=5,
            clinic_id="Clinic-A"
        )
        time.sleep(0.5)

    console.print("[bold green]✓ Discovery Search Complete[/bold green]")
    console.print("[dim]  Method: Qdrant DiscoverQuery with context vectors[/dim]\n")

    console.print("[bold yellow]How Discovery API Works:[/bold yellow]")
    discovery_tree = Tree("[bold]Discovery Query[/bold]")
    target = discovery_tree.add("[cyan]Target Vector[/cyan]: 'acute cardiac event'")
    context = discovery_tree.add("[magenta]Context Pairs[/magenta]")
    context.add("[green]+ Positive[/green]: chest pain, elevated troponin, ST changes")
    context.add("[red]- Negative[/red]: trauma, pulmonary embolism")
    discovery_tree.add("[yellow]Result[/yellow]: Cases biased toward cardiac, away from trauma")
    console.print(discovery_tree)

    console.print(f"\n[bold]Found {len(results)} contextually relevant cases[/bold]")

    console.print("\n[bold cyan]→ Dr. Strange: \"Now let me generate the differential diagnosis...\"[/bold cyan]")
    wait_for_enter()

async def scene_differential_diagnosis():
    """Differential diagnosis scene"""
    clear_screen()
    console.print(Panel("[bold]Scene 6: Differential Diagnosis Generation[/bold]", border_style="yellow"))

    console.print("\n[dim]MediSync generates differential diagnosis based on evidence...[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        progress.add_task("Analyzing symptoms and evidence...", total=None)
        time.sleep(1)
        progress.add_task("Correlating with similar cases...", total=None)
        time.sleep(0.5)
        progress.add_task("Calculating confidence scores...", total=None)
        time.sleep(0.5)

    console.print("[bold green]✓ Differential Diagnosis Generated[/bold green]\n")

    diagnoses = [
        ("ST-Elevation Myocardial Infarction (STEMI)", 94, "red", [
            "Troponin I: 2.4 ng/mL (60x normal)",
            "ST elevation in V1-V4",
            "Crushing chest pain with radiation",
            "History of previous PCI"
        ]),
        ("Unstable Angina / NSTEMI", 60, "yellow", [
            "Similar presentation pattern",
            "Cardiac risk factors present",
            "Could not rule out without full ECG"
        ]),
        ("Aortic Dissection", 20, "dim", [
            "Sudden onset severe pain",
            "Hypotension present",
            "Less likely: no tearing quality, no BP differential"
        ]),
        ("Pulmonary Embolism", 15, "dim", [
            "Dyspnea present",
            "Less likely: no pleuritic pain, no DVT history"
        ]),
    ]

    console.print("[bold yellow]DIFFERENTIAL DIAGNOSIS[/bold yellow]\n")

    for name, confidence, color, evidence in diagnoses:
        # Confidence bar
        filled = int(confidence / 10)
        bar = "█" * filled + "░" * (10 - filled)

        if confidence >= 90:
            console.print(Panel(
                f"[bold]Confidence: [{color}]{confidence}%[/{color}][/bold]  [{color}]{bar}[/{color}]\n\n" +
                "[bold]Supporting Evidence:[/bold]\n" +
                "\n".join(f"  • {e}" for e in evidence),
                title=f"[bold {color}]#1 {name}[/bold {color}]",
                border_style=color
            ))
        else:
            console.print(f"\n[{color}]#{diagnoses.index((name, confidence, color, evidence))+1} {name}[/{color}]")
            console.print(f"   Confidence: [{color}]{confidence}%[/{color}]  [{color}]{bar}[/{color}]")
            console.print(f"   [dim]Evidence: {evidence[0]}[/dim]")

    console.print("\n[bold cyan]→ Dr. Strange: \"94% confidence for STEMI. Let me see the evidence graph...\"[/bold cyan]")
    wait_for_enter()

async def scene_evidence_graph():
    """Evidence graph visualization"""
    clear_screen()
    console.print(Panel("[bold]Scene 7: Evidence Graph - Explainable AI[/bold]", border_style="green"))

    console.print("\n[dim]MediSync generates explainable reasoning chain...[/dim]\n")

    from medisync.clinical_agents.explanation.evidence_graph_agent import EvidenceGraphAgent

    agent = EvidenceGraphAgent()
    graph = agent.generate_diagnostic_graph(
        patient_context="65 year old male with history of hypertension, diabetes, and hyperlipidemia. Previous PCI to LAD.",
        symptoms=["crushing substernal chest pain", "radiation to left arm", "shortness of breath", "diaphoresis"],
        evidence_records=[
            {"id": "lab-001", "score": 0.95, "text_content": "Troponin I: 2.4 ng/mL (elevated)"},
            {"id": "ecg-001", "score": 0.92, "text_content": "ECG: ST elevation V1-V4"},
            {"id": "vitals-001", "score": 0.88, "text_content": "BP 88/56, HR 112"},
        ],
        diagnosis_candidates=[
            {"diagnosis": "STEMI", "confidence": 0.94},
            {"diagnosis": "Unstable Angina", "confidence": 0.60},
        ],
        recommendations=["ACTIVATE CATH LAB", "Aspirin 325mg", "Heparin bolus"]
    )

    console.print("[bold yellow]EVIDENCE GRAPH[/bold yellow]")
    console.print("[dim]This shows how MediSync reached its diagnosis[/dim]\n")

    # Print ASCII graph
    console.print(graph.to_ascii())

    console.print("\n[bold green]✓ Full reasoning chain documented[/bold green]")
    console.print("[dim]  Export formats: ASCII, GraphViz DOT, JSON[/dim]")

    console.print("\n[bold cyan]→ Dr. Strange: \"Clear reasoning. Time to act!\"[/bold cyan]")
    wait_for_enter()

async def scene_recommendations():
    """Final recommendations scene"""
    clear_screen()
    console.print(Panel("[bold]Scene 8: Clinical Recommendations[/bold]", border_style="red"))

    console.print("\n[bold red]⚡ IMMEDIATE ACTIONS REQUIRED[/bold red]\n")

    actions = [
        ("CRITICAL", "ACTIVATE CARDIAC CATH LAB - STEMI ALERT", "Immediate revascularization within 90 minutes"),
        ("URGENT", "Aspirin 325mg chewed", "Antiplatelet therapy"),
        ("URGENT", "Heparin 60 units/kg IV bolus", "Anticoagulation"),
        ("URGENT", "Clopidogrel 600mg loading dose", "Dual antiplatelet therapy"),
        ("HIGH", "Morphine 2-4mg IV for pain", "Pain management + preload reduction"),
        ("HIGH", "Supplemental O2 to maintain SpO2 >94%", "Oxygenation support"),
        ("HIGH", "Cardiology consult STAT", "Specialist involvement"),
    ]

    for priority, action, rationale in actions:
        if priority == "CRITICAL":
            console.print(Panel(
                f"[bold]{action}[/bold]\n\n[dim]{rationale}[/dim]",
                title="[bold red]● CRITICAL[/bold red]",
                border_style="red"
            ))
        elif priority == "URGENT":
            console.print(f"  [yellow]●[/yellow] [bold]URGENT[/bold]: {action}")
            console.print(f"    [dim]{rationale}[/dim]")
        else:
            console.print(f"  [blue]●[/blue] [bold]HIGH[/bold]: {action}")
            console.print(f"    [dim]{rationale}[/dim]")

    console.print("\n" + "="*60)
    console.print("[bold green]DOOR-TO-BALLOON TIME TRACKING[/bold green]")
    console.print("="*60)
    console.print(f"  Patient Arrival:     {datetime.now().strftime('%H:%M:%S')}")
    console.print(f"  Diagnosis Made:      {datetime.now().strftime('%H:%M:%S')} (T+3 min)")
    console.print(f"  Cath Lab Activated:  [yellow]PENDING[/yellow]")
    console.print(f"  Target PCI:          < 90 minutes")
    console.print("="*60)

    wait_for_enter()

async def scene_summary():
    """Demo summary"""
    clear_screen()
    console.print(Panel("[bold]Demo Complete - Feature Summary[/bold]", border_style="cyan"))

    console.print("\n[bold yellow]MEDISYNC CAPABILITIES DEMONSTRATED[/bold yellow]\n")

    features = [
        ("Qdrant Features", [
            "Hybrid Search (Sparse + Dense + RRF Fusion)",
            "Discovery API (Context-aware search)",
            "Prefetch Chains (Multi-stage retrieval)",
            "Named Vectors (Separate embedding spaces)",
            "Payload Filters (Privacy isolation)"
        ]),
        ("Clinical AI", [
            "Differential Diagnosis Generation",
            "Evidence Graphs (Explainable AI)",
            "Vigilance Monitoring (Proactive alerts)",
            "Similar Case Retrieval"
        ]),
        ("Privacy & Security", [
            "Role-based access control",
            "Clinic isolation",
            "Patient record isolation",
            "K-anonymity (K>=20)"
        ])
    ]

    for category, items in features:
        console.print(f"\n[bold cyan]{category}[/bold cyan]")
        for item in items:
            console.print(f"  [green]✓[/green] {item}")

    console.print("\n" + "="*60)
    console.print("[bold]Key Differentiator:[/bold] All features use [cyan]Qdrant native APIs[/cyan]")
    console.print("No external re-rankers or ColBERT - just Qdrant!")
    console.print("="*60)

    console.print("\n[bold green]Thank you for watching the MediSync demo![/bold green]")
    console.print("[dim]Qdrant Convolve 4.0 Pan-IIT Hackathon[/dim]\n")

# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Run the full demo"""
    try:
        await scene_intro()
        session = await scene_login()
        await scene_alert()
        await scene_patient_intake()
        results = await scene_hybrid_search(session)
        await scene_discovery_api()
        await scene_differential_diagnosis()
        await scene_evidence_graph()
        await scene_recommendations()
        await scene_summary()

    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
