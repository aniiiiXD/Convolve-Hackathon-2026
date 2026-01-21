from medisync.service_agents.gatekeeper_agent import AuthService
from medisync.clinical_agents.reasoning.doctor_agent import DoctorAgent
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from datetime import datetime
import time

console = Console()

def main():
    console.clear()
    console.rule("[bold blue]MediSync AI: Doctor Portal")
    console.print(Panel.fit("Welcome to the [bold cyan]Clinical Workspace[/bold cyan].\nSecure access to patient records and discovery tools.", border_style="blue"))
    
    # 1. Login
    user = None
    while not user:
        username = Prompt.ask("Login as")
        user = AuthService.login(username)
        if user and user.role != "DOCTOR":
            console.print("[red]Access Denied. Doctors only.[/red]")
            user = None
    
    # 2. Init Agent
    with console.status("Initializing Clinical Agent..."):
        agent = DoctorAgent(user)
        
    console.print(f"\n[bold green]Welcome, {user.user_id}.[/bold green] (Clinic: {user.clinic_id})")
    console.print("Commands: 'add note...', 'search...', 'discover <target> context: <context>'")

    # 3. Loop
    while True:
        try:
            user_input = Prompt.ask(f"[bold blue]{user.user_id}[/bold blue]")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # Agent ReAct Loop
            for step_type, message in agent.process_request(user_input):
                 if step_type == "THOUGHT":
                    console.print(f"[dim]💭 {message}[/dim]")
                 elif step_type == "ACTION":
                    console.print(f"[bold cyan]⚡ {message}[/bold cyan]")
                 elif step_type == "SYSTEM":
                    console.print(f"[green]🖥️ {message}[/green]")
                 elif step_type == "ANSWER":
                    console.print(Panel(Markdown(message), title="MediSync", border_style="blue"))
                 elif step_type == "RESULTS":
                    table = Table(title="Search Results", show_header=True, header_style="bold magenta")
                    table.add_column("Patient", style="cyan", width=12)
                    table.add_column("Date", style="dim")
                    table.add_column("Clinical Note", style="white")
                    table.add_column("Score", justify="right")

                    for point in message:
                        payload = point.payload
                        score = f"{point.score:.2f}" if hasattr(point, "score") and point.score else "N/A"
                        ts = float(payload.get("timestamp", 0))
                        date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if ts > 0 else "Unknown"
                        
                        # Truncate content
                        content = payload.get("text_content", "")
                        preview = (content[:75] + '...') if len(content) > 75 else content
                        
                        table.add_row(
                            payload.get("patient_id", "Unknown"),
                            date_str,
                            preview,
                            score
                        )
                    console.print(table)
                    
        except KeyboardInterrupt:
            break
    
    console.print("[red]Logged out.[/red]")

if __name__ == "__main__":
    main()
