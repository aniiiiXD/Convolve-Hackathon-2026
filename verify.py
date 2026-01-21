import sys
import os

# Ensure we can import from local directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from medisync.service_agents.gatekeeper_agent import AuthService
from medisync.clinical_agents.reasoning.doctor_agent import DoctorAgent
from medisync.clinical_agents.reasoning.patient_agent import PatientAgent
import time

def test_fusion():
    print("=== Starting Medisync Verification ===")
    
    # 1. Doctor Flow
    print("\n[1] Doctor Flow: Ingesting Note")
    doc_user = AuthService.login("Dr_Strange")
    assert doc_user.role == "DOCTOR"
    
    doc_agent = DoctorAgent(doc_user)
    note_text = "Patient P-101 exhibits signs of a distal radius fracture. Recommended X-ray."
    
    # Ingest
    point_id = doc_agent.ingest_note("P-101", note_text)
    print(f"    - Ingested Note ID: {point_id}")
    
    # Search
    print("    - Searching clinic records...")
    time.sleep(1) # Allow for indexing
    results = doc_agent.search_clinic("fracture")
    found = any("distal radius" in r.payload["text_content"] for r in results)
    if found:
        print("    - SUCCESS: Doctor found the note via search.")
    else:
        print("    - FAILED: Doctor could not find the note.")

    # 2. Patient Flow
    print("\n[2] Patient Flow: Accessing History")
    pat_user = AuthService.login("P-101")
    assert pat_user.role == "PATIENT"
    
    pat_agent = PatientAgent(pat_user)
    
    # My History
    time.sleep(1)
    history = pat_agent.get_my_history()
    found_own = any("distal radius" in p.payload["text_content"] for p in history)
    
    if found_own:
        print("    - SUCCESS: Patient P-101 found their own record.")
    else:
        print("    - FAILED: Patient P-101 could not find their record.")
        
    # 3. Isolation Test
    print("\n[3] Isolation Test (P-102 should NOT see P-101 data)")
    other_pat_user = AuthService.login("P-102")
    other_agent = PatientAgent(other_pat_user)
    
    other_history = other_agent.get_my_history()
    found_leak = any("distal radius" in p.payload["text_content"] for p in other_history)
    
    if not found_leak:
        print("    - SUCCESS: P-102 cannot see P-101's data.")
    else:
        print("    - FAILED: DATA LEAK! P-102 saw P-101's data.")

    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    test_fusion()
