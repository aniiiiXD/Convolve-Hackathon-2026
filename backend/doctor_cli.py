from app.agent.core import ClinicalAgent
from app.services.qdrant_ops import initialize_collections
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
import time
import sys

console = Console()

def main():
    console.clear()
    console.rule("[bold blue]MediSync AI: Clinical Intelligence Agent")
    console.print("[italic]Connected to specific clinic tenancy: Mayo_Clinic_01[/italic]\n")

    # Init
    with console.status("Waking up agent..."):
        initialize_collections()
        agent = ClinicalAgent(clinic_id="Mayo_Clinic_01", doctor_id="Dr_Strange")
    
    console.print("[bold green]Agent Ready.[/bold green] Talk to me naturally. (e.g., 'Add a note for P-101...', 'Search for fracture...')\n")

    while True:
        try:
            user_input = Prompt.ask("[bold yellow]Dr. Strange[/bold yellow]")
            if user_input.lower() in ["exit", "quit"]:
                console.print("[red]Disconnecting...[/red]")
                break
            
            console.print("") # Spacing

            # Process the stream
            # We use a visual loop to render thoughts distinct form answers
            for step_type, message in agent.process_request(user_input):
                if step_type == "THOUGHT":
                    console.print(f"[dim]💭 {message}[/dim]")
                    time.sleep(0.3) # Readability
                elif step_type == "ACTION":
                    console.print(f"[bold cyan]⚡ {message}[/bold cyan]")
                    time.sleep(0.5)
                elif step_type == "SYSTEM":
                    console.print(f"[green]🖥️ {message}[/green]")
                elif step_type == "ANSWER":
                    console.print(Panel(Markdown(message), title="MediSync", border_style="blue"))
            
            console.print("") # Spacing

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
