from medisync.services.auth import AuthService
from medisync.agents.reasoning.doctor import DoctorAgent
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
import time

console = Console()

def main():
    console.clear()
    console.rule("[bold blue]MediSync AI: Doctor Portal")
    
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
                    
        except KeyboardInterrupt:
            break
    
    console.print("[red]Logged out.[/red]")

if __name__ == "__main__":
    main()
