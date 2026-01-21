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
    console.print(Panel.fit(
        "[bold white]MediSync AI[/bold white] [bold cyan]Clinical Workbench[/bold cyan]\n"
        "[dim]Secure Search & Discovery System[/dim]", 
        border_style="cyan", subtitle="v2.0 Native Rerank"
    ))
    
    # 1. Login
    user = None
    while not user:
        username = Prompt.ask("[bold cyan]Login ID[/bold cyan]", default="Dr_Strange")
        with console.status("[cyan]Authenticating...[/cyan]", spinner="dots"):
            user = AuthService.login(username)
            if user and user.role != "DOCTOR":
                console.print("[red]⛔ Access Denied: User is not authorized as a Doctor.[/red]")
                user = None
            elif not user:
                 console.print("[red]❌ User not found.[/red]")
    
    # 2. Init Agent
    with console.status("[bold cyan]Initializing Clinical Engine & Reranker...[/bold cyan]", spinner="simpleDotsScrolling"):
        agent = DoctorAgent(user)
        # Warmup / Check reranker
        if agent.use_reranker and agent.reranker:
            console.log(f"Reranker: [green]Active[/green] ({agent.reranker.reranker_model})")
        else:
            console.log("Reranker: [dim]Disabled[/dim]")
        time.sleep(1) # Visual pause
        
    console.print(f"\n[bold green]✓ Session Active[/bold green] | User: [white]{user.user_id}[/white] | Clinic: [white]{user.clinic_id}[/white]")
    
    help_text = """
    [bold]Available Commands:[/bold]
    • [cyan]search <query>[/cyan]       - Search patient records
    • [cyan]note <text>[/cyan]           - Record a clinical observation
    • [cyan]discover <topic> context: <ctx>[/cyan] - Deep contextual discovery
    • [cyan]recommend <symptoms>[/cyan]  - Find similar cases/treatments
    • [cyan]global <query>[/cyan]        - Search global medical insights
    • [dim]exit[/dim]                   - Logout
    """
    console.print(Panel(help_text, title="Command Menu", border_style="grey50"))

    # 3. Loop
    while True:
        try:
            print() # Spacer
            user_input = Prompt.ask(f"[bold cyan]medisync[/bold cyan] >")
            
            if user_input.lower() in ["exit", "quit", "logout"]:
                console.print("[yellow]Saving session... Logged out.[/yellow]")
                break
            
            # Agent ReAct Loop
            with console.status("[cyan]Processing...[/cyan]", spinner="earth"):
                response_gen = agent.process_request(user_input)
                
                # Consume generator
                for step_type, message in response_gen:
                     if step_type == "THOUGHT":
                        console.log(f"[dim]🤖 {message}[/dim]")
                     
                     elif step_type == "ACTION":
                        console.log(f"[bold yellow]⚡ {message}[/bold yellow]")
                     
                     elif step_type == "SYSTEM":
                        console.print(f"[green]✓ {message}[/green]")
                     
                     elif step_type == "ANSWER":
                        console.print(Panel(Markdown(message), title="MediSync AI", border_style="cyan"))
                     
                     elif step_type == "RESULTS":
                        if not message:
                            console.print("[yellow]No relevant results found.[/yellow]")
                            continue

                        table = Table(title=f"Retrieval Results ({len(message)})", show_header=True, header_style="bold magenta", border_style="cyan")
                        table.add_column("Patient", style="cyan", width=12)
                        table.add_column("Date", style="dim")
                        table.add_column("Clinical Note", style="white")
                        table.add_column("Relevance", justify="right")

                        for point in message:
                            payload = point.payload
                            score = point.score if hasattr(point, "score") else 0.0
                            score_fmt = f"{score:.3f}" if score else "N/A"
                            
                            ts = float(payload.get("timestamp", 0))
                            date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts > 0 else "-"
                            
                            # Truncate content nicely
                            content = payload.get("text_content", "")
                            preview = (content[:80] + '...') if len(content) > 80 else content
                            
                            table.add_row(
                                payload.get("patient_id", "Unknown"),
                                date_str,
                                preview,
                                f"[bold]{score_fmt}[/bold]"
                            )
                        console.print(table)
                    
                     elif step_type == "GLOBAL_INSIGHTS":
                        # Add handling for global insights table
                        table = Table(title="Global Medical Insights", show_header=True, border_style="green")
                        table.add_column("Condition", style="bold white")
                        table.add_column("Evidence", style="dim")
                        for point in message:
                            p = point.payload
                            table.add_row(p.get('condition','-'), p.get('insight_text','-')[:100])
                        console.print(table)

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]System Error:[/bold red] {e}")
            # console.print_exception() # For dev mode

    console.print("[dim]Session terminated.[/dim]")

if __name__ == "__main__":
    main()
