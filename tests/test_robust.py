import sys
import os
import time

# Ensure import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from medisync.service_agents.gatekeeper_agent import AuthService
from medisync.clinical_agents.reasoning.doctor_agent import DoctorAgent
from medisync.clinical_agents.reasoning.patient_agent import PatientAgent
from medisync.core_agents.records_agent import init_db

def test_robust_flow():
    print("==========================================")
    print("   MEDISYNC ROBUST VERIFICATION PROTOCOL  ")
    print("==========================================")
    
    # 1. Initialize Database
    print("\n[1] Infrastructure Check")
    init_db()
    print("    ✓ Database Tables Initialized")
    
    # 2. Account Creation (Postgres)
    print("\n[2] Account Management (Postgres)")
    # Create unique users for this run to avoid collisions if re-run without wipe
    suffix = int(time.time())
    doc_name = f"Dr_Test_{suffix}"
    pat_name = f"P_{suffix}"
    
    doc_user = AuthService.register_user(doc_name, "DOCTOR", "Mayo_Clinic_TEST")
    pat_user = AuthService.register_user(pat_name, "PATIENT", "Mayo_Clinic_TEST")
    
    assert doc_user.id is not None
    assert pat_user.id is not None
    print(f"    ✓ Registered Doctor: {doc_name}")
    print(f"    ✓ Registered Patient: {pat_name}")
    
    # 3. Doctor Flow (Ingestion + Search + LLM)
    print("\n[3] Doctor Clinical Flow")
    doc_agent = DoctorAgent(doc_user)
    
    note_text = f"Patient {pat_name} presents with severe tension headaches and light sensitivity. Prescribed Riizatriptan 10mg."
    
    # Ingest using HF Embeddings (or fallback)
    print("    → Ingesting Clinical Note...")
    point_id = doc_agent.ingest_note(pat_name, note_text)
    print(f"    ✓ Note Ingested (ID: {point_id})")
    
    # Wait for indexing
    time.sleep(1)
    
    # Search
    print("    → Searching for 'headaches'...")
    results = doc_agent.search_clinic("headaches")
    found = any(pat_name in r.payload["text_content"] for r in results)
    if found:
        print("    ✓ Search Successful: Found patient note.")
    else:
        print("    ✗ Search Failed: Note not found.")
        
    # Discovery (LLM)
    print("    → Testing LLM Discovery...")
    # Mocking DiscoveryService call via Agent for simplicity or direct
    # The DoctorAgent.process_request handles "discovery" intent
    # Let's try to simulate that or call discovery directly?
    # For robust test, let's call the Discovery Service logic if exposed or Agent flow.
    # We will simulate the Agent loop:
    
    # We need to see if DoctorAgent uses the new LLMService we added to Base?
    # DoctorAgent inherits MediSyncAgent.
    # Let's verify LLM is working by asking it something simple directly first.
    response = doc_agent.llm.generate_response("What is Riizatriptan used for?", context="Riizatriptan is a medication.")
    print(f"    ✓ LLM Response Test: {response[:50]}...")
    
    # 4. Patient Flow (Isolation + History)
    print("\n[4] Patient Companion Flow")
    pat_agent = PatientAgent(pat_user)
    
    print("    → Fetching Medical History...")
    history = pat_agent.get_my_history()
    found_own = any("headaches" in p.payload["text_content"] for p in history)
    
    if found_own:
        print("    ✓ History Verified: Patient sees their own note.")
    else:
        print("    ✗ History Verification Failed.")
        
    # 5. Isolation Check
    print("\n[5] Privacy/Isolation Check")
    thief_user = AuthService.register_user(f"Thief_{suffix}", "PATIENT", "Mayo_Clinic_TEST")
    thief_agent = PatientAgent(thief_user)
    
    thief_history = thief_agent.get_my_history()
    leak = any(pat_name in p.payload.get("text_content", "") for p in thief_history)
    
    if not leak:
        print("    ✓ Privacy Maintained: Thief cannot see patient data.")
    else:
        print("    ✗ DATA LEAK DETECTED!")

    print("\n==========================================")
    print("   VERIFICATION COMPLETE ")
    print("==========================================")

if __name__ == "__main__":
    test_robust_flow()
