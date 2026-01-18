from medisync.services.auth import AuthService
from medisync.agents.reasoning.patient import PatientAgent
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
import time

console = Console()

def main():
    console.clear()
    console.rule("[bold magenta]MediSync AI: Patient Companion")
    
    # 1. Login
    user = None
    while not user:
        username = Prompt.ask("Login as")
        user = AuthService.login(username)
        if user and user.role != "PATIENT":
             console.print("[red]Access Denied. Patients only.[/red]")
             user = None

    # 2. Init Agent
    with console.status("Loading Personal Health Data..."):
        agent = PatientAgent(user)
        
    console.print(f"\n[bold green]Hello, {user.user_id}.[/bold green]")
    console.print("I am your private health companion.")
    console.print("Try: 'Log a symptom', 'Show my history', 'Any insights?'")

    # 3. Loop
    while True:
        try:
            user_input = Prompt.ask(f"[bold magenta]{user.user_id}[/bold magenta]")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            for step_type, message in agent.process_request(user_input):
                 if step_type == "THOUGHT":
                    console.print(f"[dim]💭 {message}[/dim]")
                 elif step_type == "ACTION":
                    console.print(f"[bold cyan]⚡ {message}[/bold cyan]")
                 elif step_type == "ANSWER":
                    console.print(Panel(Markdown(message), title="Your Health", border_style="magenta"))
                    
        except KeyboardInterrupt:
            break
            
    console.print("[red]Goodbye.[/red]")

if __name__ == "__main__":
    main()
