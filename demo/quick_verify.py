#!/usr/bin/env python3
"""
MediSync Quick Verification - Qdrant Only
Tests core functionality in under 30 seconds.
Run: python3 demo/quick_verify.py
"""

import sys
import os
import time

# Ensure we can import from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from medisync.service_agents.gatekeeper_agent import AuthService, UserRole
from medisync.clinical_agents.reasoning.doctor_agent import DoctorAgent
from medisync.clinical_agents.reasoning.patient_agent import PatientAgent


def test_quick_verify():
    print("=" * 50)
    print("   MEDISYNC QUICK VERIFICATION (Qdrant Only)")
    print("=" * 50)

    passed = 0
    failed = 0

    # 1. Doctor Flow
    print("\n[1] Doctor Flow: Login + Ingest + Search")
    try:
        doc_user = AuthService.login("Dr_Strange")
        assert doc_user is not None
        assert doc_user.role == UserRole.DOCTOR
        print(f"    - Logged in as {doc_user.username} ({doc_user.role.value})")

        doc_agent = DoctorAgent(doc_user)
        note_text = f"Patient P-VERIFY exhibits signs of a distal radius fracture. Test at {time.time()}"

        # Ingest
        point_id = doc_agent.ingest_note("P-VERIFY", note_text)
        print(f"    - Ingested Note ID: {point_id[:16]}...")

        # Search
        time.sleep(1)  # Allow for indexing
        results = doc_agent.search_clinic("distal radius fracture")
        found = any("distal radius" in r.payload.get("text_content", "") for r in results)

        if found:
            print("    [PASS] Doctor found the note via search")
            passed += 1
        else:
            print("    [FAIL] Doctor could not find the note")
            failed += 1
    except Exception as e:
        print(f"    [FAIL] Error: {e}")
        failed += 1

    # 2. Patient Flow
    print("\n[2] Patient Flow: Login + Access History")
    try:
        pat_user = AuthService.login("P-101")
        assert pat_user is not None
        assert pat_user.role == UserRole.PATIENT
        print(f"    - Logged in as {pat_user.username} ({pat_user.role.value})")

        pat_agent = PatientAgent(pat_user)
        history = pat_agent.get_my_history()
        print(f"    - Found {len(history)} records in history")
        print("    [PASS] Patient can access their history")
        passed += 1
    except Exception as e:
        print(f"    [FAIL] Error: {e}")
        failed += 1

    # 3. Privacy/Isolation Check
    print("\n[3] Privacy/Isolation Check")
    try:
        p102_user = AuthService.login("P-102")
        p102_agent = PatientAgent(p102_user)
        p102_history = p102_agent.get_my_history()

        # Check P-102 cannot see P-101's data
        leak = any("P-101" in str(p.payload) for p in p102_history if hasattr(p, 'payload'))

        if not leak:
            print("    [PASS] P-102 cannot see P-101's data")
            passed += 1
        else:
            print("    [FAIL] DATA LEAK! P-102 saw P-101's data")
            failed += 1
    except Exception as e:
        print(f"    [FAIL] Error: {e}")
        failed += 1

    # Summary
    print("\n" + "=" * 50)
    print(f"   RESULTS: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = test_quick_verify()
    sys.exit(0 if success else 1)
