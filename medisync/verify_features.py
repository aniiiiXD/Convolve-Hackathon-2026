
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from medisync.agents.reasoning.doctor import DoctorAgent
from medisync.agents.reasoning.patient import PatientAgent
from medisync.services.auth import User
from unittest.mock import MagicMock, patch

def test_doctor_history():
    print("Testing Doctor History...")
    user = User(id="doc1", username="Dr. Smith", role="DOCTOR", clinic_id="C1")
    
    with patch('medisync.agents.reasoning.doctor.client') as mock_client:
        # Mock scroll return
        mock_client.scroll.return_value = ([], "next_page")
        
        doctor = DoctorAgent(user)
        # Mocking embedder to avoid real calls
        doctor.embedder = MagicMock()
        
        # Test get_patient_history
        doctor.get_patient_history("P-123")
        mock_client.scroll.assert_called_once()
        print("✓ get_patient_history called Qdrant scroll correctly")
        
        # Test process_request
        gen = doctor.process_request("show history of P-123")
        results = list(gen)
        assert any("Retrieving history for P-123" in str(r) for r in results), "Did not trigger history retrieval"
        print("✓ process_request triggered history intent")

def test_patient_state():
    print("\nTesting Patient State...")
    user = User(id="pat1", username="Jane Doe", role="PATIENT", clinic_id="C1")
    
    with patch('medisync.agents.reasoning.patient.client') as mock_client:
        patient = PatientAgent(user)
        patient.embedder = MagicMock()
        patient.log_diary = MagicMock()
        
        # 1. Test Generic Trigger
        print("1. Sending 'log a symptom'...")
        gen = patient.process_request("log a symptom")
        outputs = list(gen)
        assert patient.state == "AWAITING_LOG_CONTENT", f"State should be AWAITING_LOG_CONTENT, got {patient.state}"
        assert any("What symptom" in str(o) for o in outputs), "Did not ask for details"
        print("✓ Entered awaiting state")
        
        # 2. Test Follow-up
        print("2. Sending 'My head hurts'...")
        gen = patient.process_request("My head hurts")
        outputs = list(gen)
        assert patient.state == "IDLE", "State should reset to IDLE"
        patient.log_diary.assert_called_with("My head hurts")
        print("✓ Logged content and reset state")

def test_doctor_recommendations():
    print("\nTesting Doctor Recommendations...")
    user = User(id="doc1", username="Dr. Smith", role="DOCTOR", clinic_id="C1")
    
    with patch('medisync.agents.reasoning.doctor.client') as mock_client:
        doctor = DoctorAgent(user)
        doctor.embedder = MagicMock()
        
        # Test intent
        print("1. Sending 'recommend treatment for flu'...")
        gen = doctor.process_request("recommend treatment for flu")
        outputs = list(gen)
        assert any("Analyzing similar cases" in str(o) for o in outputs), "Did not trigger recommendation intent"
        print("✓ Triggered recommendation intent")
        
        # Verify it called search (since we wrapper search)
        mock_client.query_points.assert_called()
        print("✓ Called underlying search/query")

if __name__ == "__main__":
    try:
        test_doctor_history()
        test_patient_state()
        test_doctor_recommendations()
        print("\nAll verification tests passed!")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
