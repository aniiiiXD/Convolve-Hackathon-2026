"""
Comprehensive Tests for Advanced MediSync Features

Tests for:
- Advanced Retrieval Pipeline
- Differential Diagnosis Agent
- Insights Generator
- Vigilance Agent
- Change Detection
- Evidence Graph Generation
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from dataclasses import dataclass

# Test fixtures and mocks


@dataclass
class MockQdrantPoint:
    """Mock Qdrant point for testing"""
    id: str
    score: float
    payload: dict
    vector: dict = None


class MockQdrantResults:
    """Mock Qdrant query results"""
    def __init__(self, points):
        self.points = points


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing without actual Qdrant connection"""
    with patch('medisync.core_agents.database_agent.client') as mock_client:
        # Setup default return values
        mock_client.query_points.return_value = MockQdrantResults([
            MockQdrantPoint(
                id="point_1",
                score=0.85,
                payload={
                    "text_content": "Patient with chest pain and shortness of breath",
                    "clinic_id": "test_clinic",
                    "patient_id": "P-001",
                    "timestamp": datetime.now().timestamp()
                }
            ),
            MockQdrantPoint(
                id="point_2",
                score=0.72,
                payload={
                    "text_content": "Diagnosed with myocardial infarction, started treatment",
                    "clinic_id": "test_clinic",
                    "patient_id": "P-002",
                    "timestamp": datetime.now().timestamp()
                }
            ),
            MockQdrantPoint(
                id="point_3",
                score=0.65,
                payload={
                    "text_content": "Patient improved after medication adjustment",
                    "clinic_id": "test_clinic",
                    "patient_id": "P-001",
                    "timestamp": datetime.now().timestamp()
                }
            )
        ])

        mock_client.discover.return_value = [
            MockQdrantPoint(
                id="disc_1",
                score=0.88,
                payload={
                    "text_content": "Similar case with positive outcome",
                    "clinic_id": "test_clinic",
                    "patient_id": "P-003"
                }
            )
        ]

        mock_client.scroll.return_value = ([
            MockQdrantPoint(
                id="scroll_1",
                score=0.9,
                payload={
                    "text_content": "Patient history record",
                    "clinic_id": "test_clinic",
                    "patient_id": "P-001",
                    "timestamp": (datetime.now() - timedelta(days=5)).timestamp()
                },
                vector={"dense_text": [0.1] * 768}
            )
        ], None)

        mock_client.search.return_value = [
            MockQdrantPoint(
                id="search_1",
                score=0.82,
                payload={
                    "text_content": "Search result text",
                    "clinic_id": "test_clinic"
                }
            )
        ]

        mock_client.get_collection.return_value = Mock(points_count=100)

        yield mock_client


@pytest.fixture
def mock_embedder():
    """Mock embedding service"""
    with patch('medisync.service_agents.encoding_agent.EmbeddingService') as mock:
        instance = mock.return_value
        instance.get_dense_embedding.return_value = [0.1] * 768
        instance.get_sparse_embedding.return_value = Mock(
            indices=[1, 2, 3, 4, 5],
            values=[0.5, 0.3, 0.8, 0.2, 0.6]
        )
        yield instance


@pytest.fixture
def test_user():
    """Create test user"""
    from medisync.service_agents.gatekeeper_agent import User
    return User(
        id="test_doctor",
        username="Dr. Test",
        role="DOCTOR",
        clinic_id="test_clinic"
    )


# ==================== ADVANCED RETRIEVAL TESTS ====================

class TestAdvancedRetrievalPipeline:
    """Tests for AdvancedRetrievalPipeline"""

    def test_pipeline_initialization(self, mock_qdrant_client, mock_embedder):
        """Test pipeline initializes correctly"""
        from medisync.service_agents.advanced_retrieval_agent import AdvancedRetrievalPipeline

        pipeline = AdvancedRetrievalPipeline(clinic_id="test_clinic")

        assert pipeline.clinic_id == "test_clinic"
        assert pipeline.prefetch_limit == 100
        assert pipeline.final_limit == 10

    def test_basic_search(self, mock_qdrant_client, mock_embedder):
        """Test basic hybrid search"""
        from medisync.service_agents.advanced_retrieval_agent import AdvancedRetrievalPipeline

        pipeline = AdvancedRetrievalPipeline(clinic_id="test_clinic")
        pipeline.embedder = mock_embedder

        results, metrics = pipeline.search(
            query="chest pain symptoms",
            limit=5
        )

        assert metrics.total_candidates > 0
        assert 'embedding' in metrics.stage_timings

    def test_search_with_discovery_context(self, mock_qdrant_client, mock_embedder):
        """Test search with positive/negative context"""
        from medisync.service_agents.advanced_retrieval_agent import AdvancedRetrievalPipeline

        pipeline = AdvancedRetrievalPipeline(clinic_id="test_clinic")
        pipeline.embedder = mock_embedder

        results, metrics = pipeline.search(
            query="chest pain",
            limit=5,
            context_positive=["successful treatment"],
            context_negative=["adverse outcome"]
        )

        assert metrics.discovery_enabled is True

    def test_medical_precision_search(self, mock_qdrant_client, mock_embedder):
        """Test medical precision search with ruled-out conditions"""
        from medisync.service_agents.advanced_retrieval_agent import AdvancedRetrievalPipeline

        pipeline = AdvancedRetrievalPipeline(clinic_id="test_clinic")
        pipeline.embedder = mock_embedder

        results, metrics = pipeline.search_with_medical_precision(
            symptoms=["chest pain", "shortness of breath"],
            ruled_out=["pneumonia"],
            confirmed=["elevated troponin"],
            limit=5
        )

        assert metrics.discovery_enabled is True

    def test_retrieval_result_structure(self, mock_qdrant_client, mock_embedder):
        """Test that retrieval results have correct structure"""
        from medisync.service_agents.advanced_retrieval_agent import (
            AdvancedRetrievalPipeline, RetrievalResult
        )

        pipeline = AdvancedRetrievalPipeline(clinic_id="test_clinic")
        pipeline.embedder = mock_embedder

        results, _ = pipeline.search(query="test", limit=5)

        for result in results:
            assert hasattr(result, 'record_id')
            assert hasattr(result, 'score')
            assert hasattr(result, 'rank')
            assert hasattr(result, 'explanation')
            assert hasattr(result, 'payload')

    def test_explain_ranking(self, mock_qdrant_client, mock_embedder):
        """Test ranking explanation generation"""
        from medisync.service_agents.advanced_retrieval_agent import AdvancedRetrievalPipeline

        pipeline = AdvancedRetrievalPipeline(clinic_id="test_clinic")
        pipeline.embedder = mock_embedder

        results, _ = pipeline.search(query="test", limit=5)
        explanation = pipeline.explain_ranking(results)

        assert "Ranking Explanation" in explanation


# ==================== DIFFERENTIAL DIAGNOSIS TESTS ====================

class TestDifferentialDiagnosisAgent:
    """Tests for DifferentialDiagnosisAgent"""

    def test_agent_initialization(self, mock_qdrant_client, mock_embedder):
        """Test agent initializes correctly"""
        from medisync.service_agents.differential_diagnosis_agent import DifferentialDiagnosisAgent

        agent = DifferentialDiagnosisAgent(clinic_id="test_clinic")

        assert agent.clinic_id == "test_clinic"
        assert len(agent.symptom_diagnosis_map) > 0
        assert len(agent.diagnostic_tests) > 0

    def test_differential_diagnosis(self, mock_qdrant_client, mock_embedder):
        """Test differential diagnosis generation"""
        from medisync.service_agents.differential_diagnosis_agent import DifferentialDiagnosisAgent

        agent = DifferentialDiagnosisAgent(clinic_id="test_clinic")
        agent.embedder = mock_embedder

        result = agent.differential_diagnosis(
            presenting_symptoms=["chest pain", "shortness of breath"],
            ruled_out_diagnoses=["pneumonia"],
            patient_context="65 year old male"
        )

        assert result.primary_symptoms == ["chest pain", "shortness of breath"]
        assert "pneumonia" in result.ruled_out
        assert result.confidence_summary is not None

    def test_differential_with_confirmed_findings(self, mock_qdrant_client, mock_embedder):
        """Test differential with confirmed findings"""
        from medisync.service_agents.differential_diagnosis_agent import DifferentialDiagnosisAgent

        agent = DifferentialDiagnosisAgent(clinic_id="test_clinic")
        agent.embedder = mock_embedder

        result = agent.differential_diagnosis(
            presenting_symptoms=["fatigue", "weight loss"],
            confirmed_findings=["elevated blood glucose"]
        )

        assert result.recommended_next_steps is not None

    def test_explore_diagnostic_space(self, mock_qdrant_client, mock_embedder):
        """Test diagnostic space exploration"""
        from medisync.service_agents.differential_diagnosis_agent import DifferentialDiagnosisAgent

        agent = DifferentialDiagnosisAgent(clinic_id="test_clinic")
        agent.embedder = mock_embedder

        results = agent.explore_diagnostic_space(
            symptoms=["headache", "fever"]
        )

        # Should return list of exploration results
        assert isinstance(results, list)

    def test_suggest_tests(self, mock_qdrant_client, mock_embedder):
        """Test diagnostic test suggestions"""
        from medisync.service_agents.differential_diagnosis_agent import DifferentialDiagnosisAgent

        agent = DifferentialDiagnosisAgent(clinic_id="test_clinic")
        agent.embedder = mock_embedder

        suggestions = agent.suggest_tests(
            symptoms=["chest pain"],
            suspected_diagnoses=["myocardial infarction"],
            already_performed=[]
        )

        assert isinstance(suggestions, list)
        if suggestions:
            assert 'test_name' in suggestions[0]
            assert 'priority' in suggestions[0]


# ==================== INSIGHTS GENERATOR TESTS ====================

class TestInsightsGeneratorAgent:
    """Tests for InsightsGeneratorAgent"""

    def test_agent_initialization(self, mock_qdrant_client, mock_embedder, test_user):
        """Test agent initializes correctly"""
        from medisync.service_agents.insights_generator_agent import InsightsGeneratorAgent

        agent = InsightsGeneratorAgent(user=test_user)

        assert agent.clinic_id == test_user.clinic_id

    def test_temporal_trend_analysis(self, mock_qdrant_client, mock_embedder, test_user):
        """Test temporal trend analysis"""
        from medisync.service_agents.insights_generator_agent import InsightsGeneratorAgent

        agent = InsightsGeneratorAgent(user=test_user)
        agent.embedder = mock_embedder

        insights = agent.analyze_temporal_trends(
            condition="diabetes",
            time_window_days=30
        )

        assert isinstance(insights, list)

    def test_treatment_effectiveness(self, mock_qdrant_client, mock_embedder, test_user):
        """Test treatment effectiveness analysis"""
        from medisync.service_agents.insights_generator_agent import InsightsGeneratorAgent

        agent = InsightsGeneratorAgent(user=test_user)
        agent.embedder = mock_embedder

        insights = agent.analyze_treatment_effectiveness(
            treatment="insulin therapy",
            condition="diabetes"
        )

        assert isinstance(insights, list)

    def test_cohort_patterns(self, mock_qdrant_client, mock_embedder, test_user):
        """Test cohort pattern analysis"""
        from medisync.service_agents.insights_generator_agent import InsightsGeneratorAgent

        agent = InsightsGeneratorAgent(user=test_user)
        agent.embedder = mock_embedder

        insights = agent.analyze_cohort_patterns(
            cohort_criteria="elderly patients with diabetes"
        )

        assert isinstance(insights, list)

    def test_risk_pattern_detection(self, mock_qdrant_client, mock_embedder, test_user):
        """Test risk pattern detection"""
        from medisync.service_agents.insights_generator_agent import InsightsGeneratorAgent

        agent = InsightsGeneratorAgent(user=test_user)
        agent.embedder = mock_embedder

        insights = agent.detect_risk_patterns(
            risk_factor="uncontrolled hypertension"
        )

        assert isinstance(insights, list)

    def test_correlation_discovery(self, mock_qdrant_client, mock_embedder, test_user):
        """Test correlation discovery between factors"""
        from medisync.service_agents.insights_generator_agent import InsightsGeneratorAgent

        agent = InsightsGeneratorAgent(user=test_user)
        agent.embedder = mock_embedder

        insights = agent.discover_correlations(
            factor_a="diabetes",
            factor_b="hypertension"
        )

        assert isinstance(insights, list)

    def test_comprehensive_insights(self, mock_qdrant_client, mock_embedder, test_user):
        """Test comprehensive insights generation"""
        from medisync.service_agents.insights_generator_agent import InsightsGeneratorAgent

        agent = InsightsGeneratorAgent(user=test_user)
        agent.embedder = mock_embedder

        insights = agent.generate_comprehensive_insights(focus_area="cardiac")

        assert isinstance(insights, dict)
        assert 'temporal_trends' in insights
        assert 'risk_indicators' in insights

    def test_insight_structure(self, mock_qdrant_client, mock_embedder, test_user):
        """Test that generated insights have correct structure"""
        from medisync.service_agents.insights_generator_agent import (
            InsightsGeneratorAgent, GeneratedInsight
        )

        agent = InsightsGeneratorAgent(user=test_user)
        agent.embedder = mock_embedder

        insights = agent.detect_risk_patterns()

        for insight in insights:
            assert hasattr(insight, 'insight_type')
            assert hasattr(insight, 'title')
            assert hasattr(insight, 'description')
            assert hasattr(insight, 'confidence')
            assert hasattr(insight, 'recommendations')


# ==================== VIGILANCE AGENT TESTS ====================

class TestVigilanceAgent:
    """Tests for VigilanceAgent"""

    def test_sync_agent_initialization(self, mock_qdrant_client):
        """Test sync wrapper initializes correctly"""
        from medisync.clinical_agents.autonomous.vigilance_agent import VigilanceAgentSync

        agent = VigilanceAgentSync(clinic_id="test_clinic")

        assert agent.agent.clinic_id == "test_clinic"

    def test_alert_severity_levels(self):
        """Test alert severity enumeration"""
        from medisync.clinical_agents.autonomous.vigilance_agent import AlertSeverity

        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.HIGH.value == "high"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.INFO.value == "info"

    def test_alert_types(self):
        """Test alert type enumeration"""
        from medisync.clinical_agents.autonomous.vigilance_agent import AlertType

        assert AlertType.RISK_ELEVATION.value == "risk_elevation"
        assert AlertType.CONDITION_CHANGE.value == "condition_change"
        assert AlertType.MISSED_FOLLOWUP.value == "missed_followup"

    def test_alert_structure(self):
        """Test ClinicalAlert structure"""
        from medisync.clinical_agents.autonomous.vigilance_agent import (
            ClinicalAlert, AlertType, AlertSeverity
        )

        alert = ClinicalAlert(
            alert_id="ALR-TEST-001",
            alert_type=AlertType.RISK_ELEVATION,
            severity=AlertSeverity.HIGH,
            patient_id_hash="abc123",
            title="Test Alert",
            description="Test description",
            evidence=[{"type": "test", "value": 0.8}],
            recommended_actions=["Review patient records"]
        )

        alert_dict = alert.to_dict()

        assert alert_dict['alert_id'] == "ALR-TEST-001"
        assert alert_dict['severity'] == "high"
        assert alert_dict['alert_type'] == "risk_elevation"

    def test_get_alert_summary(self, mock_qdrant_client):
        """Test alert summary generation"""
        from medisync.clinical_agents.autonomous.vigilance_agent import VigilanceAgentSync

        agent = VigilanceAgentSync(clinic_id="test_clinic")
        summary = agent.get_summary()

        assert 'total_active' in summary
        assert 'critical' in summary
        assert 'high' in summary


# ==================== CHANGE DETECTION TESTS ====================

class TestChangeDetectionAgent:
    """Tests for ChangeDetectionAgent"""

    def test_agent_initialization(self, mock_qdrant_client):
        """Test agent initializes correctly"""
        from medisync.clinical_agents.autonomous.change_detection_agent import ChangeDetectionAgent

        agent = ChangeDetectionAgent(clinic_id="test_clinic")

        assert agent.clinic_id == "test_clinic"

    def test_state_change_structure(self):
        """Test StateChange structure"""
        from medisync.clinical_agents.autonomous.change_detection_agent import StateChange
        from datetime import datetime

        change = StateChange(
            patient_id="P-001",
            change_type="improvement",
            description="Patient showing signs of improvement",
            confidence=0.85,
            previous_embedding=[0.1] * 10,
            current_embedding=[0.2] * 10,
            semantic_shift=0.25,
            key_changes=["Positive indicators detected"],
            timestamp=datetime.now()
        )

        assert change.patient_id == "P-001"
        assert change.change_type == "improvement"
        assert change.confidence == 0.85

    def test_detect_changes(self, mock_qdrant_client, mock_embedder):
        """Test change detection for a patient"""
        from medisync.clinical_agents.autonomous.change_detection_agent import ChangeDetectionAgent

        agent = ChangeDetectionAgent(clinic_id="test_clinic")
        agent.embedder = mock_embedder

        changes = agent.detect_changes(
            patient_id="P-001",
            lookback_days=30
        )

        assert isinstance(changes, list)


# ==================== EVIDENCE GRAPH TESTS ====================

class TestEvidenceGraphAgent:
    """Tests for EvidenceGraphAgent"""

    def test_agent_initialization(self):
        """Test agent initializes correctly"""
        from medisync.clinical_agents.explanation.evidence_graph_agent import EvidenceGraphAgent

        agent = EvidenceGraphAgent()
        assert agent is not None

    def test_node_types(self):
        """Test node type enumeration"""
        from medisync.clinical_agents.explanation.evidence_graph_agent import NodeType

        assert NodeType.PATIENT.value == "patient"
        assert NodeType.SYMPTOM.value == "symptom"
        assert NodeType.DIAGNOSIS.value == "diagnosis"
        assert NodeType.RECOMMENDATION.value == "recommendation"

    def test_edge_types(self):
        """Test edge type enumeration"""
        from medisync.clinical_agents.explanation.evidence_graph_agent import EdgeType

        assert EdgeType.SUPPORTS.value == "supports"
        assert EdgeType.CONTRADICTS.value == "contradicts"
        assert EdgeType.LEADS_TO.value == "leads_to"

    def test_diagnostic_graph_generation(self):
        """Test diagnostic evidence graph generation"""
        from medisync.clinical_agents.explanation.evidence_graph_agent import EvidenceGraphAgent

        agent = EvidenceGraphAgent()

        graph = agent.generate_diagnostic_graph(
            patient_context="65 year old male",
            symptoms=["chest pain", "shortness of breath"],
            evidence_records=[
                {"id": "1", "score": 0.85, "text_content": "Similar MI case"},
                {"id": "2", "score": 0.72, "text_content": "Treatment protocol"}
            ],
            diagnosis_candidates=[
                {"diagnosis": "ACS", "confidence": 0.8, "explanation": "High likelihood"}
            ],
            recommendations=["Order troponin", "Get ECG"]
        )

        assert graph.graph_id is not None
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0

    def test_graph_to_dict(self):
        """Test graph serialization to dict"""
        from medisync.clinical_agents.explanation.evidence_graph_agent import EvidenceGraphAgent

        agent = EvidenceGraphAgent()

        graph = agent.generate_diagnostic_graph(
            patient_context="Test patient",
            symptoms=["fever"],
            evidence_records=[],
            diagnosis_candidates=[],
            recommendations=[]
        )

        graph_dict = graph.to_dict()

        assert 'graph_id' in graph_dict
        assert 'nodes' in graph_dict
        assert 'edges' in graph_dict
        assert 'summary' in graph_dict

    def test_graph_to_ascii(self):
        """Test ASCII representation generation"""
        from medisync.clinical_agents.explanation.evidence_graph_agent import EvidenceGraphAgent

        agent = EvidenceGraphAgent()

        graph = agent.generate_diagnostic_graph(
            patient_context="Test",
            symptoms=["headache"],
            evidence_records=[{"id": "1", "score": 0.8, "text_content": "Test"}],
            diagnosis_candidates=[{"diagnosis": "Migraine", "confidence": 0.7}],
            recommendations=["Rest"]
        )

        ascii_repr = graph.to_ascii()

        assert "EVIDENCE GRAPH" in ascii_repr
        assert "SYMPTOM" in ascii_repr or "headache" in ascii_repr

    def test_graph_to_dot(self):
        """Test DOT format export"""
        from medisync.clinical_agents.explanation.evidence_graph_agent import EvidenceGraphAgent

        agent = EvidenceGraphAgent()

        graph = agent.generate_diagnostic_graph(
            patient_context="Test",
            symptoms=["pain"],
            evidence_records=[],
            diagnosis_candidates=[],
            recommendations=[]
        )

        dot_repr = graph.to_dot()

        assert "digraph EvidenceGraph" in dot_repr
        assert "rankdir=LR" in dot_repr

    def test_treatment_graph_generation(self):
        """Test treatment evidence graph generation"""
        from medisync.clinical_agents.explanation.evidence_graph_agent import EvidenceGraphAgent

        agent = EvidenceGraphAgent()

        graph = agent.generate_treatment_graph(
            diagnosis="Diabetes",
            treatment_options=[
                {"name": "Metformin", "effectiveness": 0.8},
                {"name": "Insulin", "effectiveness": 0.9}
            ],
            similar_outcomes=[
                {"outcome": "positive", "description": "Good response"}
            ],
            recommendation="Start with Metformin"
        )

        assert graph.graph_id is not None
        assert "Diabetes" in graph.title

    def test_risk_assessment_graph(self):
        """Test risk assessment graph generation"""
        from medisync.clinical_agents.explanation.evidence_graph_agent import EvidenceGraphAgent

        agent = EvidenceGraphAgent()

        graph = agent.generate_risk_assessment_graph(
            patient_id_hash="abc123",
            risk_factors=[
                {"name": "Hypertension", "severity": 0.7},
                {"name": "Diabetes", "severity": 0.6}
            ],
            risk_score=0.75,
            similar_patients=[
                {"outcome": "negative", "similarity": 0.8}
            ],
            alerts=["High risk detected"]
        )

        assert graph.graph_id is not None
        assert graph.overall_confidence == 0.75


# ==================== INTEGRATION TESTS ====================

class TestEnhancedDoctorAgentIntegration:
    """Integration tests for EnhancedDoctorAgent"""

    def test_agent_initialization(self, mock_qdrant_client, mock_embedder, test_user):
        """Test enhanced agent initializes all components"""
        from medisync.clinical_agents.enhanced_doctor_agent import EnhancedDoctorAgent

        agent = EnhancedDoctorAgent(user=test_user)

        assert agent.clinic_id == test_user.clinic_id
        assert agent.retrieval_pipeline is not None
        assert agent.differential_agent is not None
        assert agent.insights_generator is not None
        assert agent.evidence_graph_agent is not None

    def test_enhanced_search(self, mock_qdrant_client, mock_embedder, test_user):
        """Test enhanced search returns all components"""
        from medisync.clinical_agents.enhanced_doctor_agent import EnhancedDoctorAgent

        agent = EnhancedDoctorAgent(user=test_user)
        agent.retrieval_pipeline.embedder = mock_embedder
        agent.generate_evidence_graphs = False  # Skip for faster test
        agent.auto_generate_insights = False

        result = agent.enhanced_search(
            query="chest pain",
            limit=5,
            include_evidence_graph=False,
            include_insights=False
        )

        assert result.results is not None
        assert result.pipeline_metrics is not None

    def test_run_differential_diagnosis(self, mock_qdrant_client, mock_embedder, test_user):
        """Test differential diagnosis through enhanced agent"""
        from medisync.clinical_agents.enhanced_doctor_agent import EnhancedDoctorAgent

        agent = EnhancedDoctorAgent(user=test_user)
        agent.differential_agent.embedder = mock_embedder
        agent.retrieval_pipeline.embedder = mock_embedder

        analysis = agent.run_differential_diagnosis(
            symptoms=["chest pain", "fatigue"],
            ruled_out=["pneumonia"],
            generate_graph=False
        )

        assert analysis.differential is not None
        assert analysis.recommendations is not None


# ==================== UTILITY TESTS ====================

class TestDataclassSerializations:
    """Test dataclass serialization methods"""

    def test_retrieval_result_to_dict(self):
        """Test RetrievalResult serialization"""
        from medisync.service_agents.advanced_retrieval_agent import (
            RetrievalResult, RetrievalStage
        )

        result = RetrievalResult(
            record_id="test_id",
            score=0.85,
            stage_scores={RetrievalStage.HYBRID: 0.85},
            payload={"text": "test"},
            rank=1,
            explanation="Test explanation"
        )

        result_dict = result.to_dict()

        assert result_dict['record_id'] == "test_id"
        assert result_dict['score'] == 0.85
        assert 'hybrid' in result_dict['stage_scores']

    def test_pipeline_metrics_to_dict(self):
        """Test PipelineMetrics serialization"""
        from medisync.service_agents.advanced_retrieval_agent import PipelineMetrics

        metrics = PipelineMetrics(
            total_candidates=100,
            stage_timings={'embedding': 0.1, 'search': 0.5},
            final_results=10,
            reranking_enabled=True,
            discovery_enabled=False
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict['total_candidates'] == 100
        assert metrics_dict['final_results'] == 10

    def test_generated_insight_to_dict(self):
        """Test GeneratedInsight serialization"""
        from medisync.service_agents.insights_generator_agent import (
            GeneratedInsight, InsightType
        )

        insight = GeneratedInsight(
            insight_type=InsightType.TEMPORAL_TREND,
            title="Test Trend",
            description="Description",
            confidence=0.8,
            evidence_ids=["1", "2"],
            metrics={"slope": 0.1},
            recommendations=["Do something"]
        )

        insight_dict = insight.to_dict()

        assert insight_dict['insight_type'] == "temporal_trend"
        assert insight_dict['confidence'] == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
