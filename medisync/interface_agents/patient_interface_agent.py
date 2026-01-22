from medisync.service_agents.gatekeeper_agent import AuthService, UserRole
from medisync.clinical_agents.reasoning.patient_agent import PatientAgent
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
    console.print(Panel.fit(
        "[bold magenta]MyHealth Companion[/bold magenta]\n"
        "[dim]AI-Powered Personal Health Journal[/dim]",
        border_style="magenta"
    ))

    # 1. Login
    user = None
    while not user:
        username = Prompt.ask("[bold magenta]Patient ID[/bold magenta]", default="P-101")
        with console.status("Unlocking secure journal...", spinner="hearts"):
            time.sleep(0.5)
            user = AuthService.login(username)
            if user and user.role != UserRole.PATIENT:
                 console.print("[red]⛔ Access restricted to Patients only.[/red]")
                 user = None
            elif not user:
                console.print("[red]❌ ID not found.[/red]")

    # 2. Init Agent
    with console.status("[magenta]Loading health history...[/magenta]", spinner="dots"):
        agent = PatientAgent(user)
        time.sleep(0.5)
        
    console.print(f"\n[bold green]Welcome back, {user.user_id}.[/bold green]")
    
    menu = """
    [bold]How can I help you today?[/bold]
    • [magenta]Log "I have a headache"[/magenta]
    • [magenta]Show my history[/magenta]
    • [magenta]Any insights?[/magenta]
    • [magenta]Am I at risk?[/magenta]
    """
    console.print(Panel(menu, border_style="grey50"))

    # 3. Loop
    while True:
        try:
            print()
            user_input = Prompt.ask(f"[bold magenta]You[/bold magenta]")
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                console.print("[magenta]Take care! 👋[/magenta]")
                break
            
            with console.status("[magenta]Thinking...[/magenta]", spinner="dots"):
                # Agent ReAct Loop
                for step_type, message in agent.process_request(user_input):
                     if step_type == "THOUGHT":
                        console.log(f"[dim]🧠 {message}[/dim]")
                     elif step_type == "ACTION":
                        console.log(f"[bold cyan]⚡ {message}[/bold cyan]")
                     elif step_type == "ANSWER":
                        console.print(Panel(Markdown(message), title="Response", border_style="magenta"))
                     elif step_type == "RESULTS":
                        if not message:
                            console.print("[dim]No records found.[/dim]")
                            continue
                            
                        table = Table(title="Journal Entries", show_header=True, header_style="bold magenta", border_style="magenta")
                        table.add_column("Date", style="dim", width=12)
                        table.add_column("Entry / Record", style="white")
                        table.add_column("Type", style="cyan", justify="right")

                        for point in message:
                            payload = point.payload
                            ts = float(payload.get("timestamp", 0))
                            date_str = datetime.fromtimestamp(ts).strftime('%b %d') if ts > 0 else "-"
                            content = payload.get("text_content", "")
                            type_ = payload.get("type", "entry").upper()
                            
                            table.add_row(date_str, content, type_)
                        console.print(table)
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            
    console.print("[dim]Journal closed.[/dim]")

if __name__ == "__main__":
    main()
