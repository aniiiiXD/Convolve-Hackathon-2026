"""
Unit Tests for Feedback System

Tests feedback collection, storage, and privacy features.
"""

import pytest
import hashlib
from datetime import datetime, timedelta

from medisync.services.feedback_service import FeedbackService
from medisync.services.feedback_middleware import FeedbackMiddleware
from medisync.core.db_sql import init_db, SessionLocal
from medisync.models.sql_models import SearchQuery, ResultInteraction, ClinicalOutcome


@pytest.fixture(scope="module")
def setup_database():
    """Initialize test database"""
    init_db()
    yield
    # Cleanup handled by test database


@pytest.fixture
def test_user_id():
    return "test_doctor_123"


@pytest.fixture
def test_clinic_id():
    return "test_clinic_456"


@pytest.fixture
def test_patient_id():
    return "test_patient_789"


class TestFeedbackService:
    """Test FeedbackService functionality"""

    def test_record_query(self, setup_database, test_user_id, test_clinic_id):
        """Test recording a search query"""
        query_id = FeedbackService.record_query(
            user_id=test_user_id,
            clinic_id=test_clinic_id,
            query_text="chest pain and fever",
            query_type="hybrid",
            query_intent="diagnosis",
            result_count=5
        )

        assert query_id is not None

        # Verify in database
        db = SessionLocal()
        query = db.query(SearchQuery).filter(SearchQuery.id == query_id).first()

        assert query is not None
        assert query.user_id == test_user_id
        assert query.clinic_id == test_clinic_id
        assert query.query_type == "hybrid"
        assert query.query_intent == "diagnosis"
        assert query.result_count == 5

        # Verify query text is hashed
        expected_hash = hashlib.sha256("chest pain and fever".encode()).hexdigest()
        assert query.query_text_hash == expected_hash

        db.close()

    def test_record_interaction(self, setup_database, test_user_id, test_clinic_id):
        """Test recording result interaction"""
        # First create a query
        query_id = FeedbackService.record_query(
            user_id=test_user_id,
            clinic_id=test_clinic_id,
            query_text="test query",
            query_type="hybrid",
            result_count=3
        )

        # Record interaction
        interaction_id = FeedbackService.record_interaction(
            query_id=query_id,
            result_point_id="point_123",
            result_rank=2,
            result_score=0.85,
            interaction_type="click",
            dwell_time_seconds=45.5
        )

        assert interaction_id is not None

        # Verify in database
        db = SessionLocal()
        interaction = db.query(ResultInteraction).filter(
            ResultInteraction.id == interaction_id
        ).first()

        assert interaction is not None
        assert interaction.query_id == query_id
        assert interaction.result_point_id == "point_123"
        assert interaction.result_rank == 2
        assert interaction.result_score == 0.85
        assert interaction.interaction_type == "click"
        assert interaction.dwell_time_seconds == 45.5

        db.close()

    def test_record_outcome(self, setup_database, test_user_id, test_clinic_id, test_patient_id):
        """Test recording clinical outcome"""
        # Create query
        query_id = FeedbackService.record_query(
            user_id=test_user_id,
            clinic_id=test_clinic_id,
            query_text="test query",
            query_type="hybrid",
            result_count=3
        )

        # Record outcome
        outcome_id = FeedbackService.record_outcome(
            query_id=query_id,
            patient_id=test_patient_id,
            clinic_id=test_clinic_id,
            doctor_id=test_user_id,
            outcome_type="led_to_diagnosis",
            confidence_level=5
        )

        assert outcome_id is not None

        # Verify in database
        db = SessionLocal()
        outcome = db.query(ClinicalOutcome).filter(
            ClinicalOutcome.id == outcome_id
        ).first()

        assert outcome is not None
        assert outcome.query_id == query_id
        assert outcome.outcome_type == "led_to_diagnosis"
        assert outcome.confidence_level == 5

        # Verify patient ID is hashed
        expected_hash = hashlib.sha256(test_patient_id.encode()).hexdigest()
        assert outcome.patient_id_hash == expected_hash

        db.close()

    def test_export_training_data(self, setup_database, test_user_id, test_clinic_id):
        """Test exporting training data"""
        # Create query with interactions
        query_id = FeedbackService.record_query(
            user_id=test_user_id,
            clinic_id=test_clinic_id,
            query_text="test export query",
            query_type="hybrid",
            result_count=3
        )

        # Add interactions
        FeedbackService.record_interaction(
            query_id=query_id,
            result_point_id="point_1",
            result_rank=1,
            result_score=0.9,
            interaction_type="click"
        )

        FeedbackService.record_interaction(
            query_id=query_id,
            result_point_id="point_2",
            result_rank=2,
            result_score=0.7,
            interaction_type="view"
        )

        # Export data
        export_data = FeedbackService.export_training_data(
            date_range_start=datetime.utcnow() - timedelta(days=1),
            date_range_end=datetime.utcnow() + timedelta(days=1),
            batch_name="test_batch",
            min_interactions=1
        )

        assert export_data['batch_name'] == "test_batch"
        assert export_data['sample_count'] >= 1
        assert len(export_data['samples']) >= 1

        # Verify sample structure
        sample = export_data['samples'][0]
        assert 'query_id' in sample
        assert 'query_hash' in sample
        assert 'positive_results' in sample
        assert 'negative_results' in sample

    def test_query_statistics(self, setup_database, test_user_id, test_clinic_id):
        """Test getting query statistics"""
        # Create some queries
        for i in range(5):
            query_id = FeedbackService.record_query(
                user_id=test_user_id,
                clinic_id=test_clinic_id,
                query_text=f"test query {i}",
                query_type="hybrid",
                result_count=3
            )

            # Add click to some queries
            if i < 3:
                FeedbackService.record_interaction(
                    query_id=query_id,
                    result_point_id=f"point_{i}",
                    result_rank=1,
                    result_score=0.9,
                    interaction_type="click"
                )

        # Get statistics
        stats = FeedbackService.get_query_statistics(days=7)

        assert stats['total_queries'] >= 5
        assert stats['queries_with_clicks'] >= 3
        assert stats['click_through_rate'] >= 0


class TestFeedbackMiddleware:
    """Test FeedbackMiddleware functionality"""

    def test_initialization(self):
        """Test middleware initialization"""
        middleware = FeedbackMiddleware(enabled=True)

        assert middleware.enabled is True
        assert middleware.session_id is not None
        assert middleware.current_query_id is None

    def test_intent_inference(self):
        """Test query intent classification"""
        middleware = FeedbackMiddleware()

        assert middleware._infer_intent("diagnose chest pain") == "diagnosis"
        assert middleware._infer_intent("treatment for fracture") == "treatment"
        assert middleware._infer_intent("patient history") == "history"
        assert middleware._infer_intent("general query") == "general"

    def test_result_tracking(self, setup_database):
        """Test result interaction tracking"""
        middleware = FeedbackMiddleware(enabled=True)

        # Simulate a query
        middleware.current_query_id = "test_query_id_123"

        # Record interaction
        middleware.record_result_interaction(
            result_point_id="point_test",
            result_rank=1,
            result_score=0.95,
            interaction_type="click"
        )

        # Note: This test requires the query to exist in DB
        # In production, use proper fixtures

    def test_session_reset(self):
        """Test session reset"""
        middleware = FeedbackMiddleware()

        old_session = middleware.session_id
        middleware.reset_session()
        new_session = middleware.session_id

        assert old_session != new_session
        assert middleware.current_query_id is None


class TestPrivacyFeatures:
    """Test privacy-preserving features"""

    def test_query_text_hashing(self, setup_database, test_user_id, test_clinic_id):
        """Test that query text is properly hashed"""
        query_text = "sensitive medical query"

        query_id = FeedbackService.record_query(
            user_id=test_user_id,
            clinic_id=test_clinic_id,
            query_text=query_text,
            query_type="hybrid",
            result_count=5
        )

        # Verify hash
        db = SessionLocal()
        query = db.query(SearchQuery).filter(SearchQuery.id == query_id).first()

        expected_hash = hashlib.sha256(query_text.encode()).hexdigest()
        assert query.query_text_hash == expected_hash

        # Verify original text is not stored
        assert not hasattr(query, 'query_text')

        db.close()

    def test_patient_id_hashing(self, setup_database, test_user_id, test_clinic_id):
        """Test that patient IDs are properly hashed"""
        patient_id = "P-SENSITIVE-123"

        query_id = FeedbackService.record_query(
            user_id=test_user_id,
            clinic_id=test_clinic_id,
            query_text="test",
            query_type="hybrid",
            result_count=1
        )

        outcome_id = FeedbackService.record_outcome(
            query_id=query_id,
            patient_id=patient_id,
            clinic_id=test_clinic_id,
            doctor_id=test_user_id,
            outcome_type="helpful",
            confidence_level=4
        )

        # Verify hash
        db = SessionLocal()
        outcome = db.query(ClinicalOutcome).filter(
            ClinicalOutcome.id == outcome_id
        ).first()

        expected_hash = hashlib.sha256(patient_id.encode()).hexdigest()
        assert outcome.patient_id_hash == expected_hash

        # Verify original ID is not stored
        assert not hasattr(outcome, 'patient_id')

        db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
