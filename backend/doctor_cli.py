from app.agent.core import ClinicalAgent
from app.services.qdrant_ops import initialize_collections
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich import print as rprint
import sys

console = Console()

def setup_app():
    console.rule("[bold blue]MediSync System Initialization")
    with console.status("[bold green]Connecting to Neural Memory..."):
        initialize_collections()
        # Mock Login
        agent = ClinicalAgent(clinic_id="Mayo_Clinic_01", doctor_id="Dr_Strange")
    rprint("[bold green]✓ System Online")
    return agent

def display_menu():
    console.print("\n")
    console.rule("[bold blue]Main Menu")
    console.print("[1] 📝 [bold]New Patient Note[/bold] (Ingest)")
    console.print("[2] 🔍 [bold]Query Patient History[/bold] (Recall)")
    console.print("[3] 🧠 [bold]Ask Assistant[/bold] (Reason)")
    console.print("[4] ❌ [bold]Exit[/bold]")

def handle_ingest(agent):
    rprint(Panel("Enter the patient's clinical note below.", title="Ingestion Mode", border_style="green"))
    patient_id = Prompt.ask("Patient ID", default="P-99999")
    text = Prompt.ask("Clinical Note")
    
    with console.status("Encoding and Memorizing..."):
        pid = agent.ingest_note(patient_id, text)
    rprint(f"[bold green]✓ Saved to Memory ID: {pid}[/bold green]")

def handle_recall(agent):
    query = Prompt.ask("Search Query")
    with console.status("Performing Hybrid Retrieval..."):
        results = agent.recall(query)
    
    table = Table(title=f"Results for '{query}'")
    table.add_column("Score", justify="right", style="cyan")
    table.add_column("Patient", style="magenta")
    table.add_column("Snippet", style="white")

    for p in results:
        table.add_row(
            f"{p.score:.4f}",
            p.payload['patient_id'],
            p.payload['text_content'][:60] + "..."
        )
    console.print(table)

def handle_reason(agent):
    query = Prompt.ask("Clinical Question")
    with console.status("Thinking..."):
        # 1. Recall
        context = agent.recall(query, limit=3)
        # 2. Reason
        answer = agent.reason(query, context)
    
    rprint(Panel(answer, title="AI Assessment", border_style="yellow"))

def main():
    agent = setup_app()
    
    while True:
        display_menu()
        choice = Prompt.ask("Select Option", choices=["1", "2", "3", "4"])
        
        if choice == "1":
            handle_ingest(agent)
        elif choice == "2":
            handle_recall(agent)
        elif choice == "3":
            handle_reason(agent)
        elif choice == "4":
            rprint("[bold red]Shutting down...[/bold red]")
            sys.exit(0)

if __name__ == "__main__":
    main()
