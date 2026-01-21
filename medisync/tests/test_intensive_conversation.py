"""
Intensive Test Script for Doctor-Patient Conversation with Learning System
Tests MediSync system with ~20 dialogues from each side
INCLUDES: Feedback collection, re-ranking, global insights, analytics, privacy
"""
import time
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from medisync.services.auth import AuthService
from medisync.agents.reasoning.doctor import DoctorAgent
from medisync.agents.reasoning.patient import PatientAgent
from medisync.services.feedback_service import FeedbackService
from medisync.services.analytics_service import AnalyticsService
from medisync.services.global_insights import GlobalInsightsService
from medisync.core.privacy import PrivacyFilter, PrivacyValidator
from medisync.models.model_registry import get_registry, ModelType
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Enable feedback and re-ranking for testing
os.environ["FEEDBACK_ENABLED"] = "true"
os.environ["USE_RERANKER"] = "true"

class ConversationTester:
    def __init__(self):
        self.test_clinic_id = "TEST-CLINIC-001"
        self.test_doctor_username = "dr_test_intensive"
        self.test_patient_username = "patient_test_intensive"
        self.doctor_agent = None
        self.patient_agent = None
        self.doctor_user = None
        self.patient_user = None
        self.test_results = {
            "doctor_actions": 0,
            "patient_actions": 0,
            "successful_ingests": 0,
            "successful_searches": 0,
            "successful_diary_logs": 0,
            "feedback_queries_logged": 0,
            "feedback_interactions_logged": 0,
            "feedback_outcomes_logged": 0,
            "global_insights_queries": 0,
            "privacy_checks_passed": 0,
            "reranking_attempts": 0,
            "errors": []
        }

    def setup(self):
        """Setup test users and agents"""
        console.print("\n[bold cyan]═══ Setting Up Test Environment ═══[/bold cyan]\n")

        # Register test users
        self.doctor_user = AuthService.register_user(
            username=self.test_doctor_username,
            role="DOCTOR",
            clinic_id=self.test_clinic_id
        )

        self.patient_user = AuthService.register_user(
            username=self.test_patient_username,
            role="PATIENT",
            clinic_id=self.test_clinic_id
        )

        # Initialize agents
        self.doctor_agent = DoctorAgent(self.doctor_user)
        self.patient_agent = PatientAgent(self.patient_user)

        console.print(f"✓ Doctor Agent: {self.test_doctor_username}", style="green")
        console.print(f"✓ Patient Agent: {self.test_patient_username}", style="green")
        console.print(f"✓ Clinic ID: {self.test_clinic_id}", style="green")
        console.print(f"✓ Feedback Collection: {'ENABLED' if self.doctor_agent.feedback_middleware.enabled else 'DISABLED'}", style="green")
        console.print(f"✓ Re-ranker: {'ENABLED' if self.doctor_agent.use_reranker else 'DISABLED'}", style="cyan")

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
                    # Test re-ranking if enabled
                    if self.doctor_agent.use_reranker:
                        self.test_results["reranking_attempts"] += 1
                elif step_type == "GLOBAL_INSIGHTS":
                    self._display_global_insights(content)
                    self.test_results["global_insights_queries"] += 1

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

    def _display_global_insights(self, insights):
        """Display global insights in a formatted table"""
        if not insights:
            console.print(f"   📭 No global insights found", style="yellow")
            return

        table = Table(title="Global Medical Insights", show_header=True, header_style="bold cyan")
        table.add_column("Condition", width=20)
        table.add_column("Treatment", width=20)
        table.add_column("Sample Size", justify="right", width=12)
        table.add_column("Clinics", justify="right", width=10)

        for insight in insights[:5]:
            table.add_row(
                insight.get('condition', 'N/A'),
                insight.get('treatment', 'N/A'),
                str(insight.get('sample_size', 0)),
                str(insight.get('clinic_count', 0))
            )

        console.print(table)

    def test_feedback_collection(self):
        """Test feedback collection system"""
        console.print("\n[bold yellow]═══ Testing Feedback Collection ═══[/bold yellow]")

        try:
            # Get feedback statistics
            stats = FeedbackService.get_query_statistics(days=1)

            console.print(f"✓ Total Queries Logged: {stats['total_queries']}", style="green")
            console.print(f"✓ Queries with Clicks: {stats['queries_with_clicks']}", style="green")
            console.print(f"✓ Click-Through Rate: {stats['click_through_rate']}%", style="green")

            self.test_results["feedback_queries_logged"] = stats['total_queries']
            self.test_results["feedback_interactions_logged"] = stats['queries_with_clicks']

        except Exception as e:
            console.print(f"❌ Feedback collection test failed: {e}", style="red")
            self.test_results["errors"].append(f"Feedback collection: {str(e)}")

    def test_clinical_outcomes(self):
        """Test clinical outcome feedback"""
        console.print("\n[bold yellow]═══ Testing Clinical Outcome Feedback ═══[/bold yellow]")

        try:
            # Record a test outcome
            if self.doctor_agent.feedback_middleware.current_query_id:
                self.doctor_agent.record_clinical_outcome(
                    patient_id="P-001",
                    outcome_type="led_to_diagnosis",
                    confidence_level=5
                )

                self.test_results["feedback_outcomes_logged"] += 1
                console.print("✓ Clinical outcome recorded successfully", style="green")
            else:
                console.print("⚠ No active query for outcome tracking", style="yellow")

        except Exception as e:
            console.print(f"❌ Outcome feedback test failed: {e}", style="red")
            self.test_results["errors"].append(f"Clinical outcomes: {str(e)}")

    def test_global_insights(self):
        """Test global insights querying"""
        console.print("\n[bold yellow]═══ Testing Global Insights System ═══[/bold yellow]")

        try:
            # Query global insights
            insights = self.doctor_agent.query_global_insights(
                query="finger fracture treatment",
                limit=3
            )

            if insights:
                console.print(f"✓ Found {len(insights)} global insights", style="green")
                self._display_global_insights(insights)
            else:
                console.print("⚠ No global insights available yet (normal for new system)", style="yellow")

        except Exception as e:
            console.print(f"❌ Global insights test failed: {e}", style="red")
            self.test_results["errors"].append(f"Global insights: {str(e)}")

    def test_analytics(self):
        """Test analytics service"""
        console.print("\n[bold yellow]═══ Testing Analytics Dashboard ═══[/bold yellow]")

        try:
            # Get comprehensive analytics
            dashboard = AnalyticsService.get_comprehensive_dashboard(
                days=1,
                clinic_id=self.test_clinic_id
            )

            console.print("✓ Analytics Dashboard Generated", style="green")

            # Display key metrics
            search_metrics = dashboard.get('search_metrics', {})
            ranking_metrics = dashboard.get('ranking_metrics', {})

            metrics_table = Table(title="Key Metrics", show_header=True)
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="yellow")

            metrics_table.add_row("Total Queries", str(search_metrics.get('total_queries', 0)))
            metrics_table.add_row("CTR", f"{search_metrics.get('click_through_rate', 0)}%")
            metrics_table.add_row("MRR", f"{ranking_metrics.get('mean_reciprocal_rank', 0):.3f}")
            metrics_table.add_row("Avg Click Position", f"{ranking_metrics.get('average_click_position', 0):.1f}")

            console.print(metrics_table)

        except Exception as e:
            console.print(f"❌ Analytics test failed: {e}", style="red")
            self.test_results["errors"].append(f"Analytics: {str(e)}")

    def test_privacy_compliance(self):
        """Test privacy compliance features"""
        console.print("\n[bold yellow]═══ Testing Privacy Compliance ═══[/bold yellow]")

        try:
            # Test PII removal
            test_text = "Patient SSN: 123-45-6789, Phone: 555-123-4567"
            sanitized = PrivacyFilter.remove_pii(test_text)

            if "123-45-6789" not in sanitized and "555-123-4567" not in sanitized:
                console.print("✓ PII removal working correctly", style="green")
                self.test_results["privacy_checks_passed"] += 1
            else:
                console.print("❌ PII not properly removed", style="red")
                self.test_results["errors"].append("Privacy: PII removal failed")

            # Test K-anonymity
            records = [
                {"condition": "fracture", "treatment": "cast", "clinic_id": f"c{i}"}
                for i in range(25)
            ]

            filtered = PrivacyFilter.apply_k_anonymity(
                records=records,
                k=20,
                min_clinics=1,
                grouping_keys=['condition', 'treatment']
            )

            if len(filtered) == 25:
                console.print("✓ K-anonymity enforcement working (K=20)", style="green")
                self.test_results["privacy_checks_passed"] += 1
            else:
                console.print(f"❌ K-anonymity failed: {len(filtered)}/25", style="red")

            # Test generalization
            age_bracket = PrivacyFilter.generalize_age(45)
            if age_bracket == "40-50":
                console.print("✓ Age generalization working correctly", style="green")
                self.test_results["privacy_checks_passed"] += 1

        except Exception as e:
            console.print(f"❌ Privacy compliance test failed: {e}", style="red")
            self.test_results["errors"].append(f"Privacy: {str(e)}")

    def test_model_registry(self):
        """Test model registry"""
        console.print("\n[bold yellow]═══ Testing Model Registry ═══[/bold yellow]")

        try:
            registry = get_registry()

            # Check embedder models
            embedders = registry.list_models(ModelType.EMBEDDER)
            console.print(f"✓ Found {len(embedders)} embedder model(s)", style="green")

            # Check re-ranker models
            rerankers = registry.list_models(ModelType.RERANKER)
            console.print(f"✓ Found {len(rerankers)} re-ranker model(s)", style="green")

            if embedders or rerankers:
                console.print("✓ Model registry operational", style="green")
            else:
                console.print("⚠ No models registered yet (normal for new system)", style="yellow")

        except Exception as e:
            console.print(f"❌ Model registry test failed: {e}", style="red")
            self.test_results["errors"].append(f"Model registry: {str(e)}")

    def run_learning_system_tests(self):
        """Run all learning system tests"""
        console.print("\n\n[bold cyan]╔══════════════════════════════════════════════════╗[/bold cyan]")
        console.print("[bold cyan]║     Learning System Integration Tests           ║[/bold cyan]")
        console.print("[bold cyan]╚══════════════════════════════════════════════════╝[/bold cyan]")

        self.test_feedback_collection()
        time.sleep(0.5)

        self.test_clinical_outcomes()
        time.sleep(0.5)

        self.test_global_insights()
        time.sleep(0.5)

        self.test_analytics()
        time.sleep(0.5)

        self.test_privacy_compliance()
        time.sleep(0.5)

        self.test_model_registry()

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
        summary_table.add_row("─" * 40, "─" * 10)
        summary_table.add_row("[bold cyan]Learning System Metrics[/bold cyan]", "")
        summary_table.add_row("Feedback Queries Logged", str(self.test_results["feedback_queries_logged"]))
        summary_table.add_row("Feedback Interactions Logged", str(self.test_results["feedback_interactions_logged"]))
        summary_table.add_row("Clinical Outcomes Logged", str(self.test_results["feedback_outcomes_logged"]))
        summary_table.add_row("Global Insights Queries", str(self.test_results["global_insights_queries"]))
        summary_table.add_row("Privacy Checks Passed", str(self.test_results["privacy_checks_passed"]))
        summary_table.add_row("Re-ranking Attempts", str(self.test_results["reranking_attempts"]))
        summary_table.add_row("─" * 40, "─" * 10)
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
            self.run_learning_system_tests()
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
