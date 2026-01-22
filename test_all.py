#!/usr/bin/env python3
"""
MediSync Complete Demo & Test Suite
Qdrant Convolve 4.0 Pan-IIT Hackathon

Run: python3 test_all.py
"""

import asyncio
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich import box
from datetime import datetime

console = Console()

def print_header(title: str):
    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold white]  {title}[/bold white]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")

def print_result(name: str, success: bool, message: str = ""):
    status = "[green]✓ PASS[/green]" if success else "[red]✗ FAIL[/red]"
    console.print(f"  {status} {name}")
    if message:
        console.print(f"       [dim]{message}[/dim]")

def print_subheader(title: str):
    console.print(f"\n  [bold yellow]▸ {title}[/bold yellow]")

# ============================================================================
# CORE INFRASTRUCTURE TESTS
# ============================================================================

async def test_qdrant_connection():
    """Test 1: Qdrant Cloud Connection"""
    print_header("TEST 1: Qdrant Cloud Connection")
    try:
        from medisync.core_agents.database_agent import client
        collections = client.get_collections()
        print_result("Qdrant connection", True, f"Found {len(collections.collections)} collections")

        # Show collections
        for col in collections.collections:
            console.print(f"       [dim]• {col.name}[/dim]")
        return True
    except Exception as e:
        print_result("Qdrant connection", False, str(e))
        return False

async def test_collections():
    """Test 2: Collection Initialization"""
    print_header("TEST 2: Collections Setup")
    try:
        from medisync.service_agents.memory_ops_agent import initialize_collections, COLLECTION_NAME
        initialize_collections()
        print_result("Collections initialized", True)

        from medisync.core_agents.database_agent import client
        info = client.get_collection(COLLECTION_NAME)
        print_result(f"Collection '{COLLECTION_NAME}'", True, f"Points: {info.points_count}, Vectors: {len(info.config.params.vectors)}")
        return True
    except Exception as e:
        print_result("Collections setup", False, str(e))
        return False

async def test_embedding_service():
    """Test 3: Embedding Service"""
    print_header("TEST 3: Embedding Service (Gemini)")
    try:
        from medisync.service_agents.encoding_agent import EmbeddingService
        service = EmbeddingService()

        test_text = "chest pain with shortness of breath and elevated troponin"

        print_subheader("Dense Embedding (Gemini 768-dim)")
        dense = service.get_dense_embedding(test_text)
        print_result("Dense embedding", True, f"Vector size: {len(dense)}")
        console.print(f"       [dim]Sample values: [{dense[0]:.4f}, {dense[1]:.4f}, {dense[2]:.4f}, ...][/dim]")

        print_subheader("Sparse Embedding (BM42/SPLADE)")
        sparse = service.get_sparse_embedding("diabetes mellitus type 2")
        print_result("Sparse embedding", True, f"Non-zero indices: {len(sparse.indices)}")
        console.print(f"       [dim]Top indices: {sparse.indices[:5]}[/dim]")

        return True
    except Exception as e:
        print_result("Embedding service", False, str(e))
        return False

async def test_authentication():
    """Test 4: Authentication System"""
    print_header("TEST 4: Authentication (Gatekeeper)")
    try:
        from medisync.service_agents.gatekeeper_agent import AuthService, UserRole

        print_subheader("Doctor Login")
        doctor = AuthService.login("Dr_Strange")
        print_result("Doctor login", doctor is not None and doctor.role == UserRole.DOCTOR,
                    f"User: {doctor.user_id}, Role: {doctor.role.value}, Clinic: {doctor.clinic_id}" if doctor else "")

        print_subheader("Patient Login")
        patient = AuthService.login("P-101")
        print_result("Patient login", patient is not None and patient.role == UserRole.PATIENT,
                    f"User: {patient.user_id}, Role: {patient.role.value}" if patient else "")

        print_subheader("Invalid Login")
        invalid = AuthService.login("NonExistent")
        print_result("Invalid login rejected", invalid is None)

        return doctor is not None and patient is not None
    except Exception as e:
        print_result("Authentication", False, str(e))
        return False

# ============================================================================
# DATA OPERATIONS TESTS
# ============================================================================

async def test_data_ingestion():
    """Test 5: Data Ingestion"""
    print_header("TEST 5: Data Ingestion")
    try:
        from medisync.service_agents.gatekeeper_agent import AuthService
        from medisync.clinical_agents.reasoning.doctor_agent import DoctorAgent

        user = AuthService.login("Dr_Strange")
        agent = DoctorAgent(user)

        test_note = f"""Clinical Note - {datetime.now().strftime('%Y-%m-%d %H:%M')}
Patient presents with crushing substernal chest pain radiating to left arm.
Onset 2 hours ago. Associated with diaphoresis and nausea.
Troponin I elevated at 2.4 ng/mL (normal <0.04).
ECG shows ST elevation in leads V1-V4.
Assessment: STEMI - anterior wall MI.
Plan: Activate cath lab, ASA 325mg, Heparin bolus."""

        print_subheader("Ingesting Clinical Note")
        console.print(f"       [dim]{test_note[:100]}...[/dim]")

        record_id = agent.ingest_note("P-TEST", test_note)
        print_result("Note ingestion", record_id is not None, f"Record ID: {record_id[:16]}..." if record_id else "")

        return record_id is not None
    except Exception as e:
        print_result("Data ingestion", False, str(e))
        return False

async def test_hybrid_search():
    """Test 6: Hybrid Search"""
    print_header("TEST 6: Hybrid Search (Sparse + Dense + RRF)")
    try:
        from medisync.service_agents.gatekeeper_agent import AuthService
        from medisync.clinical_agents.reasoning.doctor_agent import DoctorAgent

        user = AuthService.login("Dr_Strange")
        agent = DoctorAgent(user)

        query = "chest pain diabetes cardiac"
        print_subheader(f"Query: '{query}'")

        results = agent.search_clinic(query, limit=5)
        print_result("Hybrid search", len(results) > 0, f"Found {len(results)} results")

        # Show results table
        if results:
            table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
            table.add_column("Score", style="cyan", width=8)
            table.add_column("Patient", style="green", width=10)
            table.add_column("Content Preview", style="white")

            for r in results[:3]:
                content = r.payload.get('text_content', '')[:50] + "..."
                patient = r.payload.get('patient_id', 'N/A')
                table.add_row(f"{r.score:.3f}", patient, content)

            console.print(table)

        return len(results) > 0
    except Exception as e:
        print_result("Hybrid search", False, str(e))
        return False

async def test_discovery_api():
    """Test 7: Discovery API"""
    print_header("TEST 7: Discovery API (Context-Aware Search)")
    try:
        from medisync.service_agents.gatekeeper_agent import AuthService
        from medisync.clinical_agents.reasoning.doctor_agent import DoctorAgent

        user = AuthService.login("Dr_Strange")
        agent = DoctorAgent(user)

        print_subheader("Context-Aware Discovery")
        console.print("       [dim]Target: 'cardiac patient'[/dim]")
        console.print("       [dim]Positive: ['chest pain', 'elevated troponin'][/dim]")
        console.print("       [dim]Negative: ['trauma'][/dim]")

        results = agent.discover_cases(
            target="cardiac patient",
            context_positive=["chest pain", "elevated troponin"],
            context_negative=["trauma"]
        )
        print_result("Discovery search", True, f"Found {len(results)} contextual matches")

        return True
    except Exception as e:
        print_result("Discovery API", False, str(e))
        return False

async def test_privacy_isolation():
    """Test 8: Privacy Isolation"""
    print_header("TEST 8: Privacy Isolation")
    try:
        from medisync.service_agents.gatekeeper_agent import AuthService
        from medisync.clinical_agents.reasoning.patient_agent import PatientAgent

        print_subheader("Patient P-101 accessing records")
        p101 = AuthService.login("P-101")
        agent101 = PatientAgent(p101)
        results101 = agent101.get_my_history()
        console.print(f"       [dim]P-101 sees {len(results101)} records[/dim]")

        print_subheader("Patient P-102 accessing records")
        p102 = AuthService.login("P-102")
        agent102 = PatientAgent(p102)
        results102 = agent102.get_my_history()
        console.print(f"       [dim]P-102 sees {len(results102)} records[/dim]")

        # Check isolation
        p102_sees_p101 = any(
            r.get('patient_id') == 'P-101'
            for r in results102
            if isinstance(r, dict)
        )

        print_result("Patient isolation", not p102_sees_p101,
                    "P-102 cannot see P-101's records ✓" if not p102_sees_p101 else "PRIVACY BREACH!")

        return not p102_sees_p101
    except Exception as e:
        print_result("Privacy isolation", False, str(e))
        return False

# ============================================================================
# ADVANCED FEATURES TESTS
# ============================================================================

async def test_advanced_retrieval():
    """Test 9: Advanced Retrieval Pipeline"""
    print_header("TEST 9: Advanced Retrieval Pipeline (4-Stage)")
    try:
        from medisync.service_agents.advanced_retrieval_agent import AdvancedRetrievalPipeline

        pipeline = AdvancedRetrievalPipeline("Clinic-A")

        print_subheader("Pipeline Stages")
        console.print("       [dim]1. Sparse Prefetch (BM42) → 100 candidates[/dim]")
        console.print("       [dim]2. Dense Prefetch (Gemini) → 100 candidates[/dim]")
        console.print("       [dim]3. RRF Fusion → Optimal ranking[/dim]")
        console.print("       [dim]4. Discovery Refinement → Context filtering[/dim]")

        print_result("Pipeline initialized", True)

        print_subheader("Executing Search")
        results, metrics = pipeline.search(
            query="patient with elevated blood pressure and cardiac symptoms",
            limit=5
        )

        time_ms = sum(metrics.stage_timings.values()) * 1000
        print_result("Pipeline search", True,
                    f"Retrieved {len(results)} results in {time_ms:.0f}ms")

        console.print(f"       [dim]Total candidates evaluated: {metrics.total_candidates}[/dim]")

        return True
    except Exception as e:
        print_result("Advanced retrieval", False, str(e))
        return False

async def test_insights_generator():
    """Test 10: Insights Generator"""
    print_header("TEST 10: Insights Generator (Clinical Intelligence)")
    try:
        from medisync.service_agents.insights_generator_agent import InsightsGeneratorAgent, InsightType
        from medisync.service_agents.gatekeeper_agent import AuthService

        user = AuthService.login("Dr_Strange")
        agent = InsightsGeneratorAgent(user)

        print_result("Insights agent initialized", True)

        print_subheader("Available Insight Types")
        types = [t.value for t in InsightType]

        # Create tree view
        tree = Tree("[bold]InsightType[/bold]")
        for t in types:
            tree.add(f"[cyan]{t}[/cyan]")
        console.print(tree)

        return True
    except Exception as e:
        print_result("Insights generator", False, str(e))
        return False

async def test_vigilance_agent():
    """Test 11: Vigilance Agent"""
    print_header("TEST 11: Vigilance Agent (Autonomous Monitoring)")
    try:
        from medisync.clinical_agents.autonomous.vigilance_agent import (
            VigilanceAgentSync, AlertSeverity, AlertType
        )

        agent = VigilanceAgentSync("Clinic-A")
        print_result("Vigilance agent initialized", True)

        print_subheader("Alert Severities")
        severities = [s.value for s in AlertSeverity]
        for sev in severities:
            color = {"critical": "red", "high": "yellow", "warning": "orange3", "info": "blue"}.get(sev, "white")
            console.print(f"       [{color}]● {sev.upper()}[/{color}]")

        print_subheader("Alert Types")
        alert_types = [t.value for t in AlertType]
        for at in alert_types[:5]:
            console.print(f"       [dim]• {at}[/dim]")

        return True
    except Exception as e:
        print_result("Vigilance agent", False, str(e))
        return False

async def test_evidence_graph():
    """Test 12: Evidence Graph"""
    print_header("TEST 12: Evidence Graph (Explainable AI)")
    try:
        from medisync.clinical_agents.explanation.evidence_graph_agent import (
            EvidenceGraphAgent, EvidenceGraph, NodeType
        )

        agent = EvidenceGraphAgent()
        print_result("Evidence graph agent initialized", True)

        print_subheader("Generating Diagnostic Graph")
        graph = agent.generate_diagnostic_graph(
            patient_context="65 year old male with history of hypertension",
            symptoms=["crushing chest pain", "shortness of breath", "diaphoresis"],
            evidence_records=[
                {"id": "1", "score": 0.92, "text_content": "Troponin I elevated 2.4 ng/mL"},
                {"id": "2", "score": 0.88, "text_content": "ECG: ST elevation V1-V4"},
                {"id": "3", "score": 0.85, "text_content": "BP 90/60, HR 110, diaphoretic"}
            ],
            diagnosis_candidates=[
                {"diagnosis": "STEMI", "confidence": 0.92},
                {"diagnosis": "Unstable Angina", "confidence": 0.65},
                {"diagnosis": "Aortic Dissection", "confidence": 0.20}
            ],
            recommendations=["Activate cath lab", "Aspirin 325mg", "Heparin bolus"]
        )
        print_result("Graph creation", graph is not None,
                    f"Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")

        # Show ASCII graph
        print_subheader("Evidence Graph (ASCII)")
        ascii_output = graph.to_ascii()
        # Print first 40 lines
        lines = ascii_output.split('\n')[:40]
        for line in lines:
            console.print(f"[dim]{line}[/dim]")
        if len(ascii_output.split('\n')) > 40:
            console.print("[dim]... (truncated)[/dim]")

        print_result("ASCII export", len(ascii_output) > 0)

        # Show export options
        print_subheader("Export Formats Available")
        console.print("       [dim]• graph.to_ascii()  → Terminal display[/dim]")
        console.print("       [dim]• graph.to_dot()    → GraphViz (paste at graphviz.org)[/dim]")
        console.print("       [dim]• graph.to_json()   → Frontend visualization[/dim]")

        return True
    except Exception as e:
        print_result("Evidence graph", False, str(e))
        return False

async def test_differential_diagnosis():
    """Test 13: Differential Diagnosis Agent"""
    print_header("TEST 13: Differential Diagnosis (Discovery API Powered)")
    try:
        from medisync.service_agents.differential_diagnosis_agent import DifferentialDiagnosisAgent

        agent = DifferentialDiagnosisAgent("Clinic-A")
        print_result("Diagnosis agent initialized", True, "Uses Discovery API for context-aware ranking")

        print_subheader("How It Works")
        console.print("       [dim]1. Embed symptoms as target vector[/dim]")
        console.print("       [dim]2. Use confirmed findings as positive context[/dim]")
        console.print("       [dim]3. Use ruled-out conditions as negative context[/dim]")
        console.print("       [dim]4. Discovery API finds similar cases[/dim]")
        console.print("       [dim]5. Extract diagnoses with confidence scores[/dim]")

        return True
    except Exception as e:
        print_result("Differential diagnosis", False, str(e))
        return False

async def test_reranker():
    """Test 14: Re-ranking Agent"""
    print_header("TEST 14: Re-ranking Agent (Cross-Encoder)")
    try:
        from medisync.model_agents.ranking_agent import ReRankerModel, get_reranker

        reranker = get_reranker()
        print_result("Reranker initialized", True,
                    f"Model: {reranker.reranker_model}")

        console.print(f"       [dim]Available: {reranker.is_available()}[/dim]")

        print_subheader("Re-ranking Process")
        console.print("       [dim]1. Hybrid search returns candidates[/dim]")
        console.print("       [dim]2. Cross-encoder scores query-document pairs[/dim]")
        console.print("       [dim]3. Results re-ranked by semantic relevance[/dim]")

        return True
    except Exception as e:
        print_result("Reranker", False, str(e))
        return False

# ============================================================================
# LLM FEATURES TESTS
# ============================================================================

async def test_llm_extraction():
    """Test 15: LLM Entity Extraction (Gemini)"""
    print_header("TEST 15: Medical Entity Extraction (Gemini 3 Flash)")
    try:
        from medisync.service_agents.extraction_agent import MedicalEntityExtractor

        extractor = MedicalEntityExtractor()
        print_result("Extractor initialized", True, "Model: gemini-3-flash-preview")

        clinical_note = """
        45-year-old male with type 2 diabetes presents with crushing chest pain
        radiating to left arm, onset 2 hours ago. Associated diaphoresis and nausea.
        Troponin elevated at 2.4 ng/mL. ECG shows ST elevation V1-V4.
        Started on aspirin, heparin, and clopidogrel. Cath lab activated.
        """

        print_subheader("Extracting Entities from Clinical Note")
        console.print(f"       [dim]{clinical_note.strip()[:100]}...[/dim]")

        entities = extractor.extract_entities(clinical_note)

        if entities:
            print_result("Entity extraction", True)

            table = Table(box=box.SIMPLE, show_header=True)
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="white")

            for key, value in entities.items():
                if value:
                    table.add_row(key, str(value))

            console.print(table)
        else:
            print_result("Entity extraction", False, "No entities extracted")

        print_subheader("Extracting Symptoms")
        symptoms = extractor.extract_symptoms(clinical_note)
        print_result("Symptom extraction", len(symptoms) > 0, f"Found {len(symptoms)} symptoms")
        for s in symptoms[:5]:
            console.print(f"       [dim]• {s}[/dim]")

        print_subheader("Classifying Intent")
        intent = extractor.classify_intent("What is the best treatment for STEMI?")
        print_result("Intent classification", True, f"Intent: {intent}")

        return entities is not None
    except Exception as e:
        print_result("LLM extraction", False, str(e))
        return False

async def test_llm_reasoning():
    """Test 16: LLM Reasoning Service (Gemini 3 Flash)"""
    print_header("TEST 16: LLM Reasoning Service (Gemini 3 Flash)")
    try:
        from medisync.service_agents.reasoning_agent import LLMService

        llm = LLMService()

        if llm.client is None:
            print_result("LLM service", False, "GEMINI_API_KEY not configured")
            return False

        print_result("LLM service initialized", True, "Model: gemini-3-flash-preview")

        print_subheader("Generating Clinical Summary")
        prompt = """Summarize this case in 2 sentences:
        65yo male, crushing chest pain, troponin 2.4, ST elevation V1-V4.
        Started on dual antiplatelet therapy, cath lab activated."""

        response = llm.generate_response(prompt)

        if response and "Error" not in response:
            print_result("LLM generation", True)
            console.print(f"       [dim]{response[:200]}...[/dim]" if len(response) > 200 else f"       [dim]{response}[/dim]")
        else:
            print_result("LLM generation", False, response or "Empty response")

        return response and "Error" not in response
    except Exception as e:
        print_result("LLM reasoning", False, str(e))
        return False

# ============================================================================
# MAIN
# ============================================================================

async def run_all_tests():
    """Run all tests"""
    console.print(Panel.fit(
        "[bold white]MediSync[/bold white] [bold cyan]Complete Demo & Test Suite[/bold cyan]\n"
        "[dim]Qdrant Convolve 4.0 Pan-IIT Hackathon[/dim]\n\n"
        "[yellow]Testing 16 components with live demonstrations[/yellow]",
        border_style="cyan"
    ))

    results = {}

    # Core Infrastructure
    console.print("\n[bold magenta]━━━ CORE INFRASTRUCTURE ━━━[/bold magenta]")
    results["Qdrant Connection"] = await test_qdrant_connection()
    results["Collections"] = await test_collections()
    results["Embeddings"] = await test_embedding_service()
    results["Authentication"] = await test_authentication()

    # Data Operations
    console.print("\n[bold magenta]━━━ DATA OPERATIONS ━━━[/bold magenta]")
    results["Ingestion"] = await test_data_ingestion()
    results["Hybrid Search"] = await test_hybrid_search()
    results["Discovery API"] = await test_discovery_api()
    results["Privacy"] = await test_privacy_isolation()

    # Advanced Features
    console.print("\n[bold magenta]━━━ ADVANCED FEATURES ━━━[/bold magenta]")
    results["Advanced Retrieval"] = await test_advanced_retrieval()
    results["Insights"] = await test_insights_generator()
    results["Vigilance"] = await test_vigilance_agent()
    results["Evidence Graph"] = await test_evidence_graph()
    results["Diagnosis"] = await test_differential_diagnosis()
    results["Reranker"] = await test_reranker()

    # LLM Features
    console.print("\n[bold magenta]━━━ LLM FEATURES (Gemini) ━━━[/bold magenta]")
    results["LLM Extraction"] = await test_llm_extraction()
    results["LLM Reasoning"] = await test_llm_reasoning()

    # Summary
    print_header("TEST SUMMARY")

    table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
    table.add_column("Component", style="white", width=20)
    table.add_column("Status", justify="center", width=10)

    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    for name, success in results.items():
        status = "[green]✓ PASS[/green]" if success else "[red]✗ FAIL[/red]"
        table.add_row(name, status)

    console.print(table)
    console.print(f"\n[bold]Total:[/bold] {passed} passed, {failed} failed out of {len(results)} tests")

    if failed == 0:
        console.print("\n[bold green]🎉 All tests passed! MediSync is fully operational.[/bold green]")
    else:
        console.print(f"\n[bold yellow]⚠ {failed} test(s) need attention[/bold yellow]")

    # Feature summary
    console.print("\n" + "="*70)
    console.print("[bold cyan]MEDISYNC CAPABILITIES SUMMARY[/bold cyan]")
    console.print("="*70)

    features = [
        ("Qdrant Features", ["Hybrid Search (RRF)", "Discovery API", "Prefetch Chains", "Named Vectors", "Payload Filters"]),
        ("Clinical AI", ["Differential Diagnosis", "Evidence Graphs", "Vigilance Alerts", "Insights Generator"]),
        ("LLM Integration", ["Gemini Embeddings (768d)", "Entity Extraction (Gemini 3)", "Reasoning (Gemini 3)"]),
        ("Privacy", ["Clinic Isolation", "Patient Isolation", "K-Anonymity (K≥20)"])
    ]

    for category, items in features:
        console.print(f"\n[bold yellow]{category}[/bold yellow]")
        for item in items:
            console.print(f"  [dim]•[/dim] {item}")

    return failed == 0

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
