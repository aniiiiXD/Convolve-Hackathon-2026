"""
Intensive Test Script for Doctor-Patient Conversation
Tests MediSync system with ~20 dialogues from each side
"""
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from medisync.services.auth import AuthService
from medisync.agents.reasoning.doctor import DoctorAgent
from medisync.agents.reasoning.patient import PatientAgent
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

class ConversationTester:
    def __init__(self):
        self.test_clinic_id = "TEST-CLINIC-001"
        self.test_doctor_username = "dr_test_intensive"
        self.test_patient_username = "patient_test_intensive"
        self.doctor_agent = None
        self.patient_agent = None
        self.test_results = {
            "doctor_actions": 0,
            "patient_actions": 0,
            "successful_ingests": 0,
            "successful_searches": 0,
            "successful_diary_logs": 0,
            "errors": []
        }

    def setup(self):
        """Setup test users and agents"""
        console.print("\n[bold cyan]═══ Setting Up Test Environment ═══[/bold cyan]\n")

        # Register test users
        doctor_user = AuthService.register_user(
            username=self.test_doctor_username,
            role="DOCTOR",
            clinic_id=self.test_clinic_id
        )

        patient_user = AuthService.register_user(
            username=self.test_patient_username,
            role="PATIENT",
            clinic_id=self.test_clinic_id
        )

        # Initialize agents
        self.doctor_agent = DoctorAgent(doctor_user)
        self.patient_agent = PatientAgent(patient_user)

        console.print(f"✓ Doctor Agent: {self.test_doctor_username}", style="green")
        console.print(f"✓ Patient Agent: {self.test_patient_username}", style="green")
        console.print(f"✓ Clinic ID: {self.test_clinic_id}", style="green")

    def process_doctor_input(self, input_text, dialogue_num):
        """Process doctor input and display results"""
        console.print(f"\n[bold blue]👨‍⚕️  Doctor (Dialogue #{dialogue_num}):[/bold blue]")
        console.print(f"   [italic]{input_text}[/italic]")

        self.test_results["doctor_actions"] += 1

        try:
            for step_type, content in self.doctor_agent.process_request(input_text):
                if step_type == "THOUGHT":
                    console.print(f"   💭 {content}", style="dim cyan")
                elif step_type == "ACTION":
                    console.print(f"   ⚡ {content}", style="yellow")
                elif step_type == "SYSTEM":
                    console.print(f"   🖥️  {content}", style="green")
                    self.test_results["successful_ingests"] += 1
                elif step_type == "ANSWER":
                    console.print(Panel(content, title="Response", border_style="blue"))
                elif step_type == "RESULTS":
                    self._display_search_results(content, "Doctor Search")
                    self.test_results["successful_searches"] += 1

        except Exception as e:
            console.print(f"   ❌ ERROR: {str(e)}", style="bold red")
            self.test_results["errors"].append(f"Doctor Dialogue {dialogue_num}: {str(e)}")

    def process_patient_input(self, input_text, dialogue_num):
        """Process patient input and display results"""
        console.print(f"\n[bold magenta]🧑 Patient (Dialogue #{dialogue_num}):[/bold magenta]")
        console.print(f"   [italic]{input_text}[/italic]")

        self.test_results["patient_actions"] += 1

        try:
            for step_type, content in self.patient_agent.process_request(input_text):
                if step_type == "THOUGHT":
                    console.print(f"   💭 {content}", style="dim magenta")
                elif step_type == "ACTION":
                    console.print(f"   ⚡ {content}", style="yellow")
                elif step_type == "ANSWER":
                    console.print(Panel(content, title="Response", border_style="magenta"))
                    if "diary" in content.lower():
                        self.test_results["successful_diary_logs"] += 1
                elif step_type == "RESULTS":
                    self._display_search_results(content, "Patient History")

        except Exception as e:
            console.print(f"   ❌ ERROR: {str(e)}", style="bold red")
            self.test_results["errors"].append(f"Patient Dialogue {dialogue_num}: {str(e)}")

    def _display_search_results(self, results, title):
        """Display search results in a formatted table"""
        if not results:
            console.print(f"   📭 No results found", style="yellow")
            return

        table = Table(title=title, show_header=True, header_style="bold")
        table.add_column("ID", style="dim", width=12)
        table.add_column("Type", width=8)
        table.add_column("Content", width=60)
        table.add_column("Score", justify="right", width=8)

        for point in results[:5]:  # Show top 5 results
            point_id = str(point.id)[:8] + "..."
            point_type = point.payload.get("type", "N/A")
            content = point.payload.get("text_content", "")[:60] + "..."
            score = f"{point.score:.3f}" if hasattr(point, 'score') and point.score else "N/A"

            table.add_row(point_id, point_type, content, score)

        console.print(table)

    def run_intensive_conversation(self):
        """Run intensive test conversation with ~20 dialogues from each side"""
        console.print("\n[bold yellow]═══ Starting Intensive Conversation Test ═══[/bold yellow]\n")

        # PHASE 1: Initial Patient Visit (Day 1)
        console.print("\n[bold cyan]═══ PHASE 1: Initial Visit (Day 1) ═══[/bold cyan]")

        # Patient expresses initial symptoms
        self.process_patient_input(
            "I've been feeling pain in my right index finger for 3 days. It hurts when I bend it.",
            1
        )

        time.sleep(1)

        # Doctor takes note after examination
        self.process_doctor_input(
            "add note P-001 Patient presents with right index finger pain, duration 3 days. Swelling noted at PIP joint. Tenderness on palpation. No visible deformity. Possible tendon strain.",
            1
        )

        time.sleep(1)

        # Patient asks about cause
        self.process_patient_input(
            "log diary: Doctor said it might be a tendon strain. I think I hurt it playing basketball last week.",
            2
        )

        time.sleep(1)

        # Doctor orders X-ray
        self.process_doctor_input(
            "add note P-001 Ordered X-ray to rule out fracture. Patient reports basketball injury last week. Prescribed ibuprofen 400mg TID for pain and inflammation.",
            2
        )

        time.sleep(1)

        # PHASE 2: Follow-up Visit (Day 5)
        console.print("\n[bold cyan]═══ PHASE 2: Follow-up Visit (Day 5) ═══[/bold cyan]")

        # Patient reports worsening symptoms
        self.process_patient_input(
            "log symptom: The pain has gotten worse. My finger is now swollen and bruised. I can barely move it.",
            3
        )

        time.sleep(1)

        # Doctor reviews X-ray results
        self.process_doctor_input(
            "add note P-001 X-ray results show hairline fracture of proximal phalanx on right index finger. Diagnosis confirmed. Finger fracture requires immobilization.",
            3
        )

        time.sleep(1)

        # Doctor adds special treatment note
        self.process_doctor_input(
            "add note P-001 Patients with finger fractures need special treatment: buddy taping to middle finger, aluminum splint for 4-6 weeks, follow-up in 2 weeks for progress check.",
            4
        )

        time.sleep(1)

        # Patient acknowledges treatment plan
        self.process_patient_input(
            "log diary: Doctor confirmed it's a fracture. I need to wear a splint for 4-6 weeks and tape my fingers together. No basketball for a while.",
            4
        )

        time.sleep(1)

        # PHASE 3: Patient Education
        console.print("\n[bold cyan]═══ PHASE 3: Patient Education ═══[/bold cyan]")

        # Doctor adds educational note
        self.process_doctor_input(
            "add note P-001 Educated patient on RICE protocol: Rest, Ice 20min every 2-3 hours, Compression with buddy tape, Elevation above heart level. Avoid weight-bearing activities.",
            5
        )

        time.sleep(1)

        # Patient logs understanding
        self.process_patient_input(
            "log diary: Need to remember RICE - Rest, Ice, Compression, Elevation. Ice for 20 minutes every few hours.",
            5
        )

        time.sleep(1)

        # PHASE 4: Week 1 Progress
        console.print("\n[bold cyan]═══ PHASE 4: Week 1 Progress Check ═══[/bold cyan]")

        # Patient reports side effects
        self.process_patient_input(
            "log symptom: The ibuprofen is helping with pain but I'm getting stomach upset. Should I be worried?",
            6
        )

        time.sleep(1)

        # Doctor adjusts medication
        self.process_doctor_input(
            "add note P-001 Patient reports gastric upset from ibuprofen. Switched to acetaminophen 500mg QID. Advised to take with food. Healing progressing normally.",
            6
        )

        time.sleep(1)

        # Patient checks history
        self.process_patient_input(
            "show me my history",
            7
        )

        time.sleep(1)

        # PHASE 5: Week 2 Follow-up
        console.print("\n[bold cyan]═══ PHASE 5: Week 2 Follow-up ═══[/bold cyan]")

        # Patient reports improvement
        self.process_patient_input(
            "log diary: Swelling has gone down a lot. Pain is much better. I can move my other fingers normally now.",
            8
        )

        time.sleep(1)

        # Doctor documents progress
        self.process_doctor_input(
            "add note P-001 Two-week follow-up: Swelling reduced by 70%, bruising fading. ROM improved in adjacent fingers. Patient compliant with splint. Continue current treatment plan.",
            7
        )

        time.sleep(1)

        # Doctor searches for similar cases
        self.process_doctor_input(
            "search for finger fracture cases in the clinic",
            8
        )

        time.sleep(1)

        # PHASE 6: Complications
        console.print("\n[bold cyan]═══ PHASE 6: Minor Complication ═══[/bold cyan]")

        # Patient reports new issue
        self.process_patient_input(
            "log symptom: The tape is causing some skin irritation and redness where it touches my skin.",
            9
        )

        time.sleep(1)

        # Doctor addresses complication
        self.process_doctor_input(
            "add note P-001 Patient developed contact dermatitis from adhesive tape. Applied hypoallergenic tape with gauze padding. Prescribed hydrocortisone 1% cream BID for irritation.",
            9
        )

        time.sleep(1)

        # PHASE 7: Week 4 Check
        console.print("\n[bold cyan]═══ PHASE 7: Week 4 Progress ═══[/bold cyan]")

        # Patient reports status
        self.process_patient_input(
            "log diary: Halfway through treatment. Finger feels much better. No more pain unless I accidentally bump it. Skin irritation cleared up with the cream.",
            10
        )

        time.sleep(1)

        # Doctor orders follow-up X-ray
        self.process_doctor_input(
            "add note P-001 Week 4 check: Clinical healing progressing well. Ordered follow-up X-ray to assess bone healing. If favorable, may reduce splinting to nighttime only.",
            10
        )

        time.sleep(1)

        # Patient asks about recovery timeline
        self.process_patient_input(
            "log diary: Doctor says if the X-ray looks good, I might only need the splint at night soon. Hoping to get back to basketball in a few weeks.",
            11
        )

        time.sleep(1)

        # PHASE 8: Good News
        console.print("\n[bold cyan]═══ PHASE 8: X-ray Results ═══[/bold cyan]")

        # Doctor reviews X-ray
        self.process_doctor_input(
            "add note P-001 Follow-up X-ray shows excellent callus formation. Fracture line barely visible. Cleared for nighttime-only splinting. Begin gentle ROM exercises 3x daily.",
            11
        )

        time.sleep(1)

        # Patient celebrates progress
        self.process_patient_input(
            "log diary: Great news! The fracture is healing well. I can stop wearing the splint during the day. Just need to do some gentle exercises and wear it at night.",
            12
        )

        time.sleep(1)

        # PHASE 9: Physical Therapy
        console.print("\n[bold cyan]═══ PHASE 9: Rehabilitation Phase ═══[/bold cyan]")

        # Doctor prescribes exercises
        self.process_doctor_input(
            "add note P-001 PT protocol: Gentle fist clenching 10 reps 3x/day, finger spread exercises, putty resistance training. Refer to occupational therapy if stiffness persists beyond 6 weeks.",
            12
        )

        time.sleep(1)

        # Patient starts exercises
        self.process_patient_input(
            "log diary: Started doing the exercises the doctor showed me. Finger is stiff but getting better each day. The putty exercises are helping.",
            13
        )

        time.sleep(1)

        # PHASE 10: Week 6 Final Check
        console.print("\n[bold cyan]═══ PHASE 10: Week 6 Final Assessment ═══[/bold cyan]")

        # Patient reports full recovery
        self.process_patient_input(
            "log diary: Week 6 - finger feels almost normal! Still a bit stiff in the morning but loosens up quickly. Can do most normal activities without pain.",
            14
        )

        time.sleep(1)

        # Doctor final assessment
        self.process_doctor_input(
            "add note P-001 Week 6 final assessment: Fracture fully healed. ROM restored to 95% of normal. Minimal residual stiffness. Cleared to discontinue splint. Gradual return to sports over 2 weeks.",
            13
        )

        time.sleep(1)

        # Patient asks about sports
        self.process_patient_input(
            "log diary: Doctor cleared me to return to basketball gradually! Need to tape it for support and start with non-contact drills first.",
            15
        )

        time.sleep(1)

        # PHASE 11: Additional Cases (Build Knowledge Base)
        console.print("\n[bold cyan]═══ PHASE 11: Additional Case Documentation ═══[/bold cyan]")

        # Doctor adds more diverse cases
        self.process_doctor_input(
            "add note P-002 Patient with middle finger fracture from door injury. Similar treatment protocol. Healing time expected 5-6 weeks with proper immobilization.",
            14
        )

        time.sleep(1)

        self.process_doctor_input(
            "add note P-003 Thumb fracture case - different treatment needed due to thumb's importance in grip. Requires thumb spica splint instead of buddy taping.",
            15
        )

        time.sleep(1)

        self.process_doctor_input(
            "add note P-004 Ring finger fracture with rotational deformity. Required closed reduction before splinting. Emphasizes importance of checking alignment.",
            16
        )

        time.sleep(1)

        # PHASE 12: Patient Seeks Health Insights
        console.print("\n[bold cyan]═══ PHASE 12: Patient Health Insights ═══[/bold cyan]")

        # Patient asks for insights
        self.process_patient_input(
            "give me health insights based on my history",
            16
        )

        time.sleep(1)

        # Patient reflects on journey
        self.process_patient_input(
            "log diary: Looking back at my treatment journey. The key lessons: catch injuries early, follow doctor's orders, be patient with healing, and don't rush recovery.",
            17
        )

        time.sleep(1)

        # PHASE 13: Doctor's Clinical Analysis
        console.print("\n[bold cyan]═══ PHASE 13: Clinical Knowledge Search ═══[/bold cyan]")

        # Doctor searches for fracture patterns
        self.process_doctor_input(
            "search for all finger fracture treatment protocols",
            17
        )

        time.sleep(1)

        self.process_doctor_input(
            "search for complications in finger fracture healing",
            18
        )

        time.sleep(1)

        # PHASE 14: Preventive Care Discussion
        console.print("\n[bold cyan]═══ PHASE 14: Prevention Education ═══[/bold cyan]")

        # Doctor adds prevention note
        self.process_doctor_input(
            "add note P-001 Prevention counseling provided: proper warm-up before sports, protective taping for high-risk activities, importance of early evaluation for injuries.",
            19
        )

        time.sleep(1)

        # Patient logs prevention tips
        self.process_patient_input(
            "log diary: Doctor gave me tips to prevent future injuries: always warm up before basketball, tape fingers if needed, and don't ignore pain signals.",
            18
        )

        time.sleep(1)

        # PHASE 15: Long-term Follow-up Plan
        console.print("\n[bold cyan]═══ PHASE 15: Long-term Care Plan ═══[/bold cyan]")

        # Doctor sets follow-up schedule
        self.process_doctor_input(
            "add note P-001 Long-term plan: No further follow-up needed unless symptoms recur. Patient educated on red flags: persistent pain, deformity, numbness. Return immediately if any concerns.",
            20
        )

        time.sleep(1)

        # Patient final log
        self.process_patient_input(
            "log diary: Treatment complete! Finger is fully healed. Know what to watch for in the future. Grateful for the great care and detailed guidance throughout recovery.",
            19
        )

        time.sleep(1)

        # Patient checks complete history
        self.process_patient_input(
            "show me my complete history",
            20
        )

        time.sleep(1)

        # Doctor final search for knowledge base
        self.process_doctor_input(
            "search for patient P-001 complete treatment history",
            21
        )

    def display_test_summary(self):
        """Display comprehensive test summary"""
        console.print("\n\n[bold green]═══════════════════════════════════════════════════[/bold green]")
        console.print("[bold green]          TEST SUMMARY REPORT                        [/bold green]")
        console.print("[bold green]═══════════════════════════════════════════════════[/bold green]\n")

        summary_table = Table(title="Test Statistics", show_header=True, header_style="bold cyan")
        summary_table.add_column("Metric", style="yellow", width=40)
        summary_table.add_column("Count", justify="right", style="green", width=10)

        summary_table.add_row("Total Doctor Actions", str(self.test_results["doctor_actions"]))
        summary_table.add_row("Total Patient Actions", str(self.test_results["patient_actions"]))
        summary_table.add_row("Successful Doctor Ingests", str(self.test_results["successful_ingests"]))
        summary_table.add_row("Successful Searches", str(self.test_results["successful_searches"]))
        summary_table.add_row("Successful Patient Diary Logs", str(self.test_results["successful_diary_logs"]))
        summary_table.add_row("Total Errors", str(len(self.test_results["errors"])),
                            style="red" if self.test_results["errors"] else "green")

        console.print(summary_table)

        if self.test_results["errors"]:
            console.print("\n[bold red]Errors Encountered:[/bold red]")
            for error in self.test_results["errors"]:
                console.print(f"  ❌ {error}", style="red")
        else:
            console.print("\n[bold green]✓ All tests completed successfully with no errors![/bold green]")

        # Calculate success rate
        total_actions = self.test_results["doctor_actions"] + self.test_results["patient_actions"]
        success_rate = ((total_actions - len(self.test_results["errors"])) / total_actions * 100) if total_actions > 0 else 0

        console.print(f"\n[bold cyan]Success Rate: {success_rate:.2f}%[/bold cyan]")
        console.print("\n[bold green]═══════════════════════════════════════════════════[/bold green]\n")

    def run(self):
        """Main test execution"""
        try:
            self.setup()
            self.run_intensive_conversation()
            self.display_test_summary()

        except KeyboardInterrupt:
            console.print("\n\n[bold yellow]Test interrupted by user[/bold yellow]")
            self.display_test_summary()
        except Exception as e:
            console.print(f"\n\n[bold red]Fatal error: {str(e)}[/bold red]")
            import traceback
            console.print(traceback.format_exc())
            self.display_test_summary()


if __name__ == "__main__":
    console.print("[bold magenta]╔══════════════════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║   MediSync Intensive Conversation Test Suite    ║[/bold magenta]")
    console.print("[bold magenta]║   Testing Doctor-Patient Interaction System     ║[/bold magenta]")
    console.print("[bold magenta]╚══════════════════════════════════════════════════╝[/bold magenta]")

    tester = ConversationTester()
    tester.run()
