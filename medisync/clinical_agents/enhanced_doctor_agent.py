"""
Enhanced Doctor Agent

Integrates all next-level features into a unified clinical interface:
- Advanced 4-stage retrieval pipeline
- Differential diagnosis with Discovery API
- Evidence graph generation
- Proactive insights generation
- Autonomous vigilance integration
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from medisync.clinical_agents.base_clinical_agent import MediSyncAgent
from medisync.service_agents.gatekeeper_agent import User
from medisync.service_agents.encoding_agent import EmbeddingService
from medisync.service_agents.memory_ops_agent import COLLECTION_NAME, client
from medisync.service_agents.learning_middleware_agent import FeedbackMiddleware

# New advanced components
from medisync.service_agents.advanced_retrieval_agent import AdvancedRetrievalPipeline, RetrievalResult
from medisync.service_agents.differential_diagnosis_agent import DifferentialDiagnosisAgent, DifferentialResult
from medisync.service_agents.insights_generator_agent import InsightsGeneratorAgent, GeneratedInsight
from medisync.clinical_agents.explanation.evidence_graph_agent import EvidenceGraphAgent, EvidenceGraph
from medisync.clinical_agents.autonomous.vigilance_agent import VigilanceAgentSync, ClinicalAlert
from medisync.clinical_agents.autonomous.change_detection_agent import ChangeDetectionAgent, StateChange

logger = logging.getLogger(__name__)


@dataclass
class EnhancedSearchResult:
    """Enhanced search result with full pipeline details"""
    results: List[RetrievalResult]
    pipeline_metrics: Dict[str, Any]
    evidence_graph: Optional[EvidenceGraph]
    insights: List[GeneratedInsight]


@dataclass
class ClinicalAnalysis:
    """Complete clinical analysis result"""
    differential: Optional[DifferentialResult]
    evidence_graph: Optional[EvidenceGraph]
    insights: List[GeneratedInsight]
    alerts: List[ClinicalAlert]
    patient_changes: List[StateChange]
    recommendations: List[str]


class EnhancedDoctorAgent(MediSyncAgent):
    """
    Next-generation clinical agent with advanced capabilities.

    Features:
    - 4-stage retrieval (Sparse → Dense → ColBERT → Discovery)
    - Differential diagnosis with context-aware search
    - Automatic evidence graph generation
    - Proactive clinical insights
    - Patient state monitoring
    """

    def __init__(self, user: User):
        super().__init__(user)
        self.embedder = EmbeddingService()

        # Initialize advanced components
        self.retrieval_pipeline = AdvancedRetrievalPipeline(self.clinic_id)
        self.differential_agent = DifferentialDiagnosisAgent(self.clinic_id)
        self.insights_generator = InsightsGeneratorAgent(user)
        self.evidence_graph_agent = EvidenceGraphAgent()
        self.change_detector = ChangeDetectionAgent(self.clinic_id)

        # Vigilance agent (lazy loaded)
        self._vigilance_agent = None

        # Feedback tracking
        self.feedback_middleware = FeedbackMiddleware(
            enabled=os.getenv("FEEDBACK_ENABLED", "true").lower() == "true"
        )

        # Feature flags
        self.use_advanced_retrieval = os.getenv("USE_ADVANCED_RETRIEVAL", "true").lower() == "true"
        self.generate_evidence_graphs = os.getenv("GENERATE_EVIDENCE_GRAPHS", "true").lower() == "true"
        self.auto_generate_insights = os.getenv("AUTO_INSIGHTS", "true").lower() == "true"

        logger.info(f"EnhancedDoctorAgent initialized for clinic {self.clinic_id}")

    @property
    def vigilance_agent(self) -> VigilanceAgentSync:
        """Lazy load vigilance agent"""
        if self._vigilance_agent is None:
            self._vigilance_agent = VigilanceAgentSync(self.clinic_id)
        return self._vigilance_agent

    # ==================== ADVANCED SEARCH ====================

    def enhanced_search(
        self,
        query: str,
        limit: int = 10,
        context_positive: List[str] = None,
        context_negative: List[str] = None,
        include_evidence_graph: bool = True,
        include_insights: bool = True
    ) -> EnhancedSearchResult:
        """
        Perform advanced 4-stage search with evidence graph and insights.

        Args:
            query: Search query
            limit: Number of results
            context_positive: Contexts to include (for discovery)
            context_negative: Contexts to exclude (for discovery)
            include_evidence_graph: Generate visual evidence chain
            include_insights: Generate related insights

        Returns:
            EnhancedSearchResult with full analysis
        """
        # Execute 4-stage retrieval
        results, metrics = self.retrieval_pipeline.search(
            query=query,
            limit=limit,
            context_positive=context_positive,
            context_negative=context_negative
        )

        # Generate evidence graph
        evidence_graph = None
        if include_evidence_graph and self.generate_evidence_graphs and results:
            evidence_graph = self._generate_search_evidence_graph(query, results)

        # Generate insights
        insights = []
        if include_insights and self.auto_generate_insights:
            insights = self._generate_query_insights(query)

        return EnhancedSearchResult(
            results=results,
            pipeline_metrics={
                'total_candidates': metrics.total_candidates,
                'final_results': metrics.final_results,
                'timings': metrics.stage_timings,
                'colbert_enabled': metrics.colbert_enabled,
                'discovery_enabled': metrics.discovery_enabled
            },
            evidence_graph=evidence_graph,
            insights=insights
        )

    def medical_precision_search(
        self,
        symptoms: List[str],
        ruled_out: List[str] = None,
        confirmed: List[str] = None,
        limit: int = 10
    ) -> EnhancedSearchResult:
        """
        Specialized search for medical queries with high precision.

        Uses ColBERT for exact medical term matching.
        """
        results, metrics = self.retrieval_pipeline.search_with_medical_precision(
            symptoms=symptoms,
            ruled_out=ruled_out,
            confirmed=confirmed,
            limit=limit
        )

        # Generate evidence graph
        evidence_graph = None
        if self.generate_evidence_graphs and results:
            evidence_graph = self._generate_diagnostic_evidence_graph(
                symptoms, results, ruled_out
            )

        return EnhancedSearchResult(
            results=results,
            pipeline_metrics={
                'total_candidates': metrics.total_candidates,
                'final_results': metrics.final_results
            },
            evidence_graph=evidence_graph,
            insights=[]
        )

    # ==================== DIFFERENTIAL DIAGNOSIS ====================

    def run_differential_diagnosis(
        self,
        symptoms: List[str],
        ruled_out: List[str] = None,
        confirmed: List[str] = None,
        patient_context: str = None,
        generate_graph: bool = True
    ) -> ClinicalAnalysis:
        """
        Run comprehensive differential diagnosis.

        Uses Discovery API to find diagnoses that are:
        - Similar to confirmed symptoms
        - Dissimilar to ruled-out conditions
        """
        # Run differential diagnosis
        differential = self.differential_agent.differential_diagnosis(
            presenting_symptoms=symptoms,
            ruled_out_diagnoses=ruled_out,
            confirmed_findings=confirmed,
            patient_context=patient_context
        )

        # Generate evidence graph
        evidence_graph = None
        if generate_graph and self.generate_evidence_graphs:
            evidence_records = self._get_evidence_for_graph(symptoms)

            evidence_graph = self.evidence_graph_agent.generate_diagnostic_graph(
                patient_context=patient_context or "Patient",
                symptoms=symptoms,
                evidence_records=evidence_records,
                diagnosis_candidates=[
                    {
                        'diagnosis': c.diagnosis,
                        'confidence': c.confidence.value,
                        'explanation': c.explanation,
                        'suggested_tests': c.suggested_tests
                    }
                    for c in differential.candidates
                ],
                recommendations=differential.recommended_next_steps
            )

        # Generate related insights
        insights = []
        if self.auto_generate_insights and symptoms:
            insights = self.insights_generator.analyze_cohort_patterns(
                cohort_criteria=" ".join(symptoms)
            )

        return ClinicalAnalysis(
            differential=differential,
            evidence_graph=evidence_graph,
            insights=insights,
            alerts=[],
            patient_changes=[],
            recommendations=differential.recommended_next_steps
        )

    def explore_diagnostic_space(
        self,
        symptoms: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Explore diagnostic space without a specific target.

        Uses context-only Discovery to find unexpected possibilities.
        """
        return self.differential_agent.explore_diagnostic_space(symptoms)

    # ==================== INSIGHTS GENERATION ====================

    def generate_insights(
        self,
        focus_area: Optional[str] = None
    ) -> Dict[str, List[GeneratedInsight]]:
        """
        Generate comprehensive clinical insights.

        Returns insights categorized by type:
        - Temporal trends
        - Treatment effectiveness
        - Cohort patterns
        - Risk indicators
        - Anomalies
        """
        return self.insights_generator.generate_comprehensive_insights(focus_area)

    def get_insights_summary(self) -> Dict[str, Any]:
        """Get summary of available insights"""
        return self.insights_generator.get_insights_summary()

    def analyze_treatment_effectiveness(
        self,
        treatment: str,
        condition: Optional[str] = None
    ) -> List[GeneratedInsight]:
        """Analyze effectiveness of a specific treatment"""
        return self.insights_generator.analyze_treatment_effectiveness(
            treatment=treatment,
            condition=condition
        )

    def detect_risk_patterns(
        self,
        risk_factor: Optional[str] = None
    ) -> List[GeneratedInsight]:
        """Detect risk patterns in clinic data"""
        return self.insights_generator.detect_risk_patterns(risk_factor)

    def discover_correlations(
        self,
        factor_a: str,
        factor_b: str
    ) -> List[GeneratedInsight]:
        """Discover correlations between clinical factors"""
        return self.insights_generator.discover_correlations(factor_a, factor_b)

    # ==================== PATIENT MONITORING ====================

    def detect_patient_changes(
        self,
        patient_id: str,
        lookback_days: int = 30
    ) -> List[StateChange]:
        """
        Detect changes in a specific patient's state.
        """
        return self.change_detector.detect_changes(
            patient_id=patient_id,
            lookback_days=lookback_days
        )

    def detect_all_patient_changes(
        self,
        lookback_days: int = 30
    ) -> Dict[str, List[StateChange]]:
        """
        Detect changes across all patients.
        """
        return self.change_detector.detect_all_changes(lookback_days)

    def run_vigilance_cycle(self) -> Dict[str, Any]:
        """
        Run a single vigilance monitoring cycle.

        Returns alert summary and any new alerts.
        """
        self.vigilance_agent.run_single_cycle()
        alerts = self.vigilance_agent.get_alerts()

        return {
            'summary': self.vigilance_agent.get_summary(),
            'new_alerts': [a.to_dict() for a in alerts[:10]]
        }

    def get_active_alerts(self) -> List[ClinicalAlert]:
        """Get active clinical alerts"""
        return self.vigilance_agent.get_alerts()

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a clinical alert"""
        return self.vigilance_agent.agent.acknowledge_alert(
            alert_id, self.user.user_id
        )

    # ==================== PREDICTIVE ANALYSIS ====================

    def predict_patient_trajectory(
        self,
        patient_id: str
    ) -> List[GeneratedInsight]:
        """
        Generate predictive insights for a patient.

        Analyzes similar patient trajectories to predict outcomes.
        """
        return self.insights_generator.generate_predictive_insights(
            patient_id=patient_id,
            prediction_type="risk"
        )

    def compare_patient_trajectories(
        self,
        patient_id: str,
        reference_patient_id: str
    ) -> Dict[str, Any]:
        """
        Compare a patient's trajectory with a reference case.
        """
        return self.change_detector.compare_patient_trajectory(
            patient_id, reference_patient_id
        )

    # ==================== EVIDENCE GRAPHS ====================

    def generate_evidence_graph(
        self,
        analysis_type: str,
        **kwargs
    ) -> Optional[EvidenceGraph]:
        """
        Generate evidence graph for various analysis types.

        Args:
            analysis_type: 'diagnostic', 'treatment', or 'risk'
            **kwargs: Parameters for the specific graph type
        """
        if analysis_type == 'diagnostic':
            return self.evidence_graph_agent.generate_diagnostic_graph(**kwargs)
        elif analysis_type == 'treatment':
            return self.evidence_graph_agent.generate_treatment_graph(**kwargs)
        elif analysis_type == 'risk':
            return self.evidence_graph_agent.generate_risk_assessment_graph(**kwargs)
        else:
            logger.warning(f"Unknown graph type: {analysis_type}")
            return None

    # ==================== HELPER METHODS ====================

    def _generate_search_evidence_graph(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> EvidenceGraph:
        """Generate evidence graph from search results"""
        evidence_records = [
            {
                'id': r.record_id,
                'score': r.score,
                'text_content': r.payload.get('text_content', '')
            }
            for r in results
        ]

        # Extract symptoms from query (simplified)
        symptoms = query.split()[:5]

        return self.evidence_graph_agent.generate_diagnostic_graph(
            patient_context="Query-based search",
            symptoms=symptoms,
            evidence_records=evidence_records,
            diagnosis_candidates=[],
            recommendations=["Review search results for clinical relevance"]
        )

    def _generate_diagnostic_evidence_graph(
        self,
        symptoms: List[str],
        results: List[RetrievalResult],
        ruled_out: List[str] = None
    ) -> EvidenceGraph:
        """Generate evidence graph for diagnostic search"""
        evidence_records = [
            {
                'id': r.record_id,
                'score': r.score,
                'text_content': r.payload.get('text_content', '')
            }
            for r in results
        ]

        return self.evidence_graph_agent.generate_diagnostic_graph(
            patient_context=f"Ruled out: {', '.join(ruled_out or [])}",
            symptoms=symptoms,
            evidence_records=evidence_records,
            diagnosis_candidates=[],
            recommendations=["Continue diagnostic workup"]
        )

    def _get_evidence_for_graph(self, symptoms: List[str]) -> List[Dict]:
        """Get evidence records for graph generation"""
        query = " ".join(symptoms)
        results, _ = self.retrieval_pipeline.search(query, limit=5)

        return [
            {
                'id': r.record_id,
                'score': r.score,
                'text_content': r.payload.get('text_content', '')
            }
            for r in results
        ]

    def _generate_query_insights(self, query: str) -> List[GeneratedInsight]:
        """Generate insights related to a query"""
        # Quick temporal trend check
        return self.insights_generator.analyze_temporal_trends(
            condition=query,
            time_window_days=14
        )[:3]  # Limit to 3 insights

    # ==================== PROCESS REQUEST (CLI) ====================

    def process_request(self, user_input: str):
        """Enhanced ReAct Loop for Doctor CLI with all new features."""
        import time
        import re

        yield ("THOUGHT", f"Analyzing input with enhanced capabilities: '{user_input}'...")
        time.sleep(0.2)

        intent = self._classify_intent(user_input)
        lower_input = user_input.lower()

        if intent == "differential":
            # Extract symptoms
            symptoms = self._extract_symptoms(user_input)
            ruled_out = self._extract_ruled_out(user_input)

            yield ("ACTION", f"Running differential diagnosis for: {symptoms}")

            analysis = self.run_differential_diagnosis(
                symptoms=symptoms,
                ruled_out=ruled_out
            )

            if analysis.differential:
                yield ("DIFFERENTIAL", analysis.differential)
                if analysis.evidence_graph:
                    yield ("EVIDENCE_GRAPH", analysis.evidence_graph.to_ascii())

        elif intent == "insights":
            yield ("ACTION", "Generating clinical insights...")

            focus = self._extract_focus(user_input)
            insights = self.generate_insights(focus_area=focus)

            yield ("INSIGHTS", insights)

        elif intent == "alerts":
            yield ("ACTION", "Checking clinical alerts...")

            result = self.run_vigilance_cycle()
            yield ("ALERTS", result)

        elif intent == "changes":
            patient_id = self._extract_patient_id(user_input)
            if patient_id:
                yield ("ACTION", f"Detecting changes for {patient_id}...")
                changes = self.detect_patient_changes(patient_id)
                yield ("CHANGES", changes)
            else:
                yield ("ACTION", "Detecting changes across all patients...")
                changes = self.detect_all_patient_changes()
                yield ("CHANGES", changes)

        elif intent == "predict":
            patient_id = self._extract_patient_id(user_input)
            if patient_id:
                yield ("ACTION", f"Generating predictions for {patient_id}...")
                predictions = self.predict_patient_trajectory(patient_id)
                yield ("PREDICTIONS", predictions)
            else:
                yield ("ANSWER", "Please specify a patient ID for predictions.")

        elif intent == "search":
            yield ("ACTION", f"Running advanced 4-stage search...")

            result = self.enhanced_search(
                query=user_input,
                limit=5,
                include_evidence_graph=True,
                include_insights=True
            )

            yield ("RESULTS", result.results)
            if result.evidence_graph:
                yield ("EVIDENCE_GRAPH", result.evidence_graph.to_ascii())
            if result.insights:
                yield ("INSIGHTS", result.insights)

        elif intent == "correlations":
            factors = self._extract_factors(user_input)
            if len(factors) >= 2:
                yield ("ACTION", f"Discovering correlations between {factors[0]} and {factors[1]}...")
                insights = self.discover_correlations(factors[0], factors[1])
                yield ("INSIGHTS", insights)
            else:
                yield ("ANSWER", "Please specify two factors to correlate.")

        else:
            yield ("ANSWER",
                   "Enhanced commands: 'differential [symptoms]', 'insights [focus]', "
                   "'alerts', 'changes [patient]', 'predict [patient]', 'correlations [A] [B]', "
                   "or just search normally.")

    def _classify_intent(self, user_input: str) -> str:
        """Classify user intent"""
        lower = user_input.lower()

        if any(w in lower for w in ['differential', 'diagnose', 'ddx', 'ruled out']):
            return "differential"
        elif any(w in lower for w in ['insight', 'trend', 'pattern', 'analysis']):
            return "insights"
        elif any(w in lower for w in ['alert', 'warning', 'urgent', 'vigilance']):
            return "alerts"
        elif any(w in lower for w in ['change', 'progression', 'evolution']):
            return "changes"
        elif any(w in lower for w in ['predict', 'forecast', 'trajectory', 'prognosis']):
            return "predict"
        elif any(w in lower for w in ['correlate', 'relationship', 'between']):
            return "correlations"
        else:
            return "search"

    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text"""
        # Simple extraction - in production use medical NER
        common_symptoms = [
            'chest pain', 'shortness of breath', 'fatigue', 'headache',
            'fever', 'cough', 'nausea', 'dizziness', 'pain', 'swelling'
        ]
        found = [s for s in common_symptoms if s in text.lower()]
        return found if found else text.split()[:5]

    def _extract_ruled_out(self, text: str) -> List[str]:
        """Extract ruled-out conditions"""
        ruled_out = []
        lower = text.lower()

        if 'ruled out' in lower:
            parts = lower.split('ruled out')
            if len(parts) > 1:
                ruled_out = [p.strip() for p in parts[1].split(',')]

        return ruled_out

    def _extract_patient_id(self, text: str) -> Optional[str]:
        """Extract patient ID from text"""
        import re
        match = re.search(r'(P-\d+|patient[- ]?\w+)', text, re.IGNORECASE)
        return match.group(0) if match else None

    def _extract_focus(self, text: str) -> Optional[str]:
        """Extract focus area from text"""
        # Remove command words
        cleaned = text.lower()
        for word in ['insights', 'trends', 'patterns', 'show', 'get', 'generate']:
            cleaned = cleaned.replace(word, '')
        cleaned = cleaned.strip()
        return cleaned if cleaned else None

    def _extract_factors(self, text: str) -> List[str]:
        """Extract factors for correlation"""
        # Simple extraction between common words
        lower = text.lower()
        for connector in [' and ', ' with ', ' vs ', ' versus ']:
            if connector in lower:
                parts = lower.split(connector)
                return [p.strip() for p in parts[:2]]
        return []


def main():
    """CLI entry point"""
    from medisync.service_agents.gatekeeper_agent import User

    # Demo user
    user = User(
        id="demo_doctor",
        username="Dr. Demo",
        role="DOCTOR",
        clinic_id="demo_clinic"
    )

    agent = EnhancedDoctorAgent(user)

    print("Enhanced Doctor Agent initialized.")
    print("Commands: differential, insights, alerts, changes, predict, correlations, or search")

    # Demo differential diagnosis
    print("\n--- Demo: Differential Diagnosis ---")
    analysis = agent.run_differential_diagnosis(
        symptoms=["chest pain", "shortness of breath"],
        patient_context="65 year old male"
    )

    if analysis.differential:
        print(f"Top diagnosis: {analysis.differential.candidates[0].diagnosis if analysis.differential.candidates else 'None'}")
        print(f"Recommendations: {analysis.differential.recommended_next_steps[:2]}")

    if analysis.evidence_graph:
        print("\nEvidence Graph:")
        print(analysis.evidence_graph.to_ascii())


if __name__ == "__main__":
    main()
