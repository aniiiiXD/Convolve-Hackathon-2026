"""
Differential Diagnosis Agent

Uses Qdrant's Discovery API to perform sophisticated differential diagnosis
by escaping "similarity bubbles" and finding non-obvious diagnoses.

Key Capabilities:
- Context-aware diagnostic exploration
- Ruled-out condition filtering
- Symptom-to-diagnosis mapping
- Evidence-based diagnostic suggestions
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from qdrant_client import models

from medisync.core_agents.database_agent import client
from medisync.service_agents.memory_ops_agent import COLLECTION_NAME
from medisync.service_agents.encoding_agent import EmbeddingService

logger = logging.getLogger(__name__)


class DiagnosticConfidence(Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    SPECULATIVE = "speculative"


@dataclass
class DiagnosticCandidate:
    """Represents a potential diagnosis candidate"""
    diagnosis: str
    confidence: DiagnosticConfidence
    relevance_score: float
    supporting_evidence: List[Dict[str, Any]]
    differentiating_factors: List[str]
    suggested_tests: List[str]
    similar_cases: int
    explanation: str


@dataclass
class DifferentialResult:
    """Result of differential diagnosis analysis"""
    primary_symptoms: List[str]
    ruled_out: List[str]
    candidates: List[DiagnosticCandidate]
    recommended_next_steps: List[str]
    confidence_summary: str
    generated_at: datetime = field(default_factory=datetime.now)


class DifferentialDiagnosisAgent:
    """
    Advanced differential diagnosis using Qdrant's Discovery API.

    Uses context-aware search to find diagnoses that are:
    - Similar to confirmed symptoms/conditions
    - Dissimilar to ruled-out diagnoses
    - Based on evidence from similar cases
    """

    def __init__(self, clinic_id: str):
        self.clinic_id = clinic_id
        self.embedder = EmbeddingService()

        # Medical knowledge base (simplified - in production, use medical ontology)
        self.symptom_diagnosis_map = self._load_symptom_diagnosis_map()
        self.diagnostic_tests = self._load_diagnostic_tests()

    def differential_diagnosis(
        self,
        presenting_symptoms: List[str],
        ruled_out_diagnoses: List[str] = None,
        confirmed_findings: List[str] = None,
        patient_context: Optional[str] = None,
        limit: int = 5
    ) -> DifferentialResult:
        """
        Perform differential diagnosis using Discovery API.

        Args:
            presenting_symptoms: List of symptoms the patient presents with
            ruled_out_diagnoses: Diagnoses that have been ruled out
            confirmed_findings: Confirmed positive findings (lab results, etc.)
            patient_context: Additional patient context (age, history, etc.)
            limit: Maximum number of diagnostic candidates

        Returns:
            DifferentialResult with ranked diagnostic candidates
        """
        ruled_out_diagnoses = ruled_out_diagnoses or []
        confirmed_findings = confirmed_findings or []

        # Build target query from symptoms
        symptom_text = " ".join(presenting_symptoms)
        if patient_context:
            symptom_text = f"{patient_context}. Presenting with: {symptom_text}"

        target_embedding = self.embedder.get_dense_embedding(symptom_text)

        # Build context pairs for Discovery API
        context_pairs = self._build_context_pairs(
            confirmed_findings,
            ruled_out_diagnoses
        )

        # Execute Discovery search
        try:
            discovery_results = self._execute_discovery_search(
                target_embedding,
                context_pairs,
                limit * 3  # Get more candidates for filtering
            )
        except Exception as e:
            logger.error(f"Discovery search failed: {e}")
            # Fallback to standard search
            discovery_results = self._fallback_search(target_embedding, limit * 3)

        # Extract and rank diagnostic candidates
        candidates = self._extract_diagnostic_candidates(
            discovery_results,
            presenting_symptoms,
            ruled_out_diagnoses,
            limit
        )

        # Generate recommendations
        next_steps = self._generate_next_steps(candidates, presenting_symptoms)

        return DifferentialResult(
            primary_symptoms=presenting_symptoms,
            ruled_out=ruled_out_diagnoses,
            candidates=candidates,
            recommended_next_steps=next_steps,
            confidence_summary=self._generate_confidence_summary(candidates)
        )

    def explore_diagnostic_space(
        self,
        symptoms: List[str],
        exploration_radius: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Explore the diagnostic space around given symptoms.

        Uses context-only Discovery (no target) to find areas where
        positive symptom examples overlap, potentially revealing
        unexpected diagnostic possibilities.
        """
        results = []

        try:
            # Build positive context from symptoms
            positive_embeddings = [
                self.embedder.get_dense_embedding(symptom)
                for symptom in symptoms
            ]

            # Context-only discovery
            context_pairs = [
                models.ContextExamplePair(
                    positive=emb,
                    negative=None
                )
                for emb in positive_embeddings
            ]

            # Search without target (exploration mode)
            exploration_results = client.discover(
                collection_name=COLLECTION_NAME,
                context=context_pairs,
                limit=20,
                filter=models.Filter(must=[
                    models.FieldCondition(
                        key="clinic_id",
                        match=models.MatchValue(value=self.clinic_id)
                    )
                ]),
                using="dense_text"
            )

            # Analyze exploration results
            for result in exploration_results:
                text = result.payload.get('text_content', '')
                potential_diagnoses = self._extract_diagnoses_from_text(text)

                results.append({
                    "record_id": str(result.id),
                    "relevance_score": result.score,
                    "potential_diagnoses": potential_diagnoses,
                    "text_snippet": text[:200],
                    "exploration_insight": self._generate_exploration_insight(
                        symptoms, potential_diagnoses
                    )
                })

        except Exception as e:
            logger.error(f"Exploration failed: {e}", exc_info=True)

        return results

    def find_similar_presentations(
        self,
        symptoms: List[str],
        target_diagnosis: str,
        exclude_diagnosis: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find cases with similar presentations that led to a specific diagnosis.

        Useful for understanding how similar cases were diagnosed and treated.
        """
        similar_cases = []

        try:
            symptom_text = " ".join(symptoms)
            target_embedding = self.embedder.get_dense_embedding(symptom_text)

            # Build context
            positive_embedding = self.embedder.get_dense_embedding(
                f"{target_diagnosis} diagnosis confirmed successful treatment"
            )

            context_pairs = [
                models.ContextExamplePair(
                    positive=positive_embedding,
                    negative=self.embedder.get_dense_embedding(exclude_diagnosis) if exclude_diagnosis else None
                )
            ]

            results = client.discover(
                collection_name=COLLECTION_NAME,
                target=target_embedding,
                context=context_pairs,
                limit=limit,
                filter=models.Filter(must=[
                    models.FieldCondition(
                        key="clinic_id",
                        match=models.MatchValue(value=self.clinic_id)
                    )
                ]),
                using="dense_text"
            )

            for result in results:
                payload = result.payload
                similar_cases.append({
                    "case_id": str(result.id),
                    "similarity_score": result.score,
                    "patient_id_hash": payload.get('patient_id', '')[:8],
                    "text_summary": payload.get('text_content', '')[:300],
                    "timestamp": payload.get('timestamp'),
                    "outcome": self._infer_outcome(payload.get('text_content', ''))
                })

        except Exception as e:
            logger.error(f"Similar presentations search failed: {e}", exc_info=True)

        return similar_cases

    def suggest_tests(
        self,
        symptoms: List[str],
        suspected_diagnoses: List[str],
        already_performed: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest diagnostic tests based on symptoms and suspected diagnoses.
        """
        already_performed = already_performed or []
        suggestions = []

        # Map symptoms and diagnoses to tests
        relevant_tests = set()

        for diagnosis in suspected_diagnoses:
            if diagnosis.lower() in self.diagnostic_tests:
                for test in self.diagnostic_tests[diagnosis.lower()]:
                    if test not in already_performed:
                        relevant_tests.add(test)

        # Find what tests were performed in similar cases
        similar_cases = self.find_similar_presentations(
            symptoms,
            suspected_diagnoses[0] if suspected_diagnoses else "diagnosis",
            limit=20
        )

        # Extract tests from similar cases
        case_tests = self._extract_tests_from_cases(similar_cases)

        for test in relevant_tests:
            case_count = case_tests.get(test, 0)
            suggestions.append({
                "test_name": test,
                "reason": f"Recommended for suspected {', '.join(suspected_diagnoses[:2])}",
                "priority": "high" if case_count > 5 else "medium",
                "similar_cases_with_test": case_count
            })

        # Sort by priority and case count
        suggestions.sort(key=lambda x: (-['high', 'medium', 'low'].index(x['priority']), -x['similar_cases_with_test']))

        return suggestions[:10]

    def _build_context_pairs(
        self,
        confirmed_findings: List[str],
        ruled_out_diagnoses: List[str]
    ) -> List[models.ContextExamplePair]:
        """Build context pairs for Discovery API"""
        context_pairs = []

        # Positive context: confirmed findings should guide towards similar cases
        for finding in confirmed_findings[:3]:  # Limit context size
            positive_emb = self.embedder.get_dense_embedding(
                f"{finding} confirmed positive"
            )

            # Pair with first ruled-out if available
            negative_emb = None
            if ruled_out_diagnoses:
                negative_emb = self.embedder.get_dense_embedding(
                    f"{ruled_out_diagnoses[0]} diagnosis"
                )

            context_pairs.append(
                models.ContextExamplePair(
                    positive=positive_emb,
                    negative=negative_emb
                )
            )

        # Additional negative context for ruled-out diagnoses
        for i, diagnosis in enumerate(ruled_out_diagnoses[1:4]):  # Skip first (already used)
            # Use successful treatment as positive to contrast
            positive_emb = self.embedder.get_dense_embedding(
                "successful treatment positive outcome"
            )
            negative_emb = self.embedder.get_dense_embedding(
                f"{diagnosis} diagnosed confirmed"
            )

            context_pairs.append(
                models.ContextExamplePair(
                    positive=positive_emb,
                    negative=negative_emb
                )
            )

        return context_pairs

    def _execute_discovery_search(
        self,
        target_embedding: List[float],
        context_pairs: List[models.ContextExamplePair],
        limit: int
    ):
        """Execute Discovery API search"""
        return client.discover(
            collection_name=COLLECTION_NAME,
            target=target_embedding,
            context=context_pairs if context_pairs else None,
            limit=limit,
            filter=models.Filter(must=[
                models.FieldCondition(
                    key="clinic_id",
                    match=models.MatchValue(value=self.clinic_id)
                )
            ]),
            using="dense_text"
        )

    def _fallback_search(self, embedding: List[float], limit: int):
        """Fallback to standard search if Discovery fails"""
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=("dense_text", embedding),
            query_filter=models.Filter(must=[
                models.FieldCondition(
                    key="clinic_id",
                    match=models.MatchValue(value=self.clinic_id)
                )
            ]),
            limit=limit,
            with_payload=True
        )

    def _extract_diagnostic_candidates(
        self,
        results,
        symptoms: List[str],
        ruled_out: List[str],
        limit: int
    ) -> List[DiagnosticCandidate]:
        """Extract and rank diagnostic candidates from search results"""
        candidates = []
        seen_diagnoses = set()

        for result in results:
            text = result.payload.get('text_content', '').lower()

            # Extract potential diagnoses from text
            diagnoses = self._extract_diagnoses_from_text(text)

            for diagnosis in diagnoses:
                # Skip ruled-out diagnoses
                if any(r.lower() in diagnosis.lower() for r in ruled_out):
                    continue

                if diagnosis in seen_diagnoses:
                    continue
                seen_diagnoses.add(diagnosis)

                # Calculate confidence
                confidence = self._calculate_diagnostic_confidence(
                    diagnosis, symptoms, result.score
                )

                # Find supporting evidence
                evidence = [{
                    "source": "similar_case",
                    "record_id": str(result.id),
                    "relevance": result.score,
                    "snippet": text[:150]
                }]

                # Get differentiating factors
                diff_factors = self._get_differentiating_factors(diagnosis, ruled_out)

                # Get suggested tests
                tests = self.diagnostic_tests.get(diagnosis.lower(), [])[:3]

                candidate = DiagnosticCandidate(
                    diagnosis=diagnosis,
                    confidence=confidence,
                    relevance_score=result.score,
                    supporting_evidence=evidence,
                    differentiating_factors=diff_factors,
                    suggested_tests=tests,
                    similar_cases=1,
                    explanation=self._generate_candidate_explanation(
                        diagnosis, symptoms, confidence
                    )
                )
                candidates.append(candidate)

                if len(candidates) >= limit:
                    break

            if len(candidates) >= limit:
                break

        # Sort by relevance and confidence
        candidates.sort(key=lambda c: (
            -['speculative', 'low', 'moderate', 'high'].index(c.confidence.value),
            -c.relevance_score
        ))

        return candidates[:limit]

    def _extract_diagnoses_from_text(self, text: str) -> List[str]:
        """Extract potential diagnoses from text"""
        diagnoses = []
        text_lower = text.lower()

        # Common diagnosis patterns
        diagnosis_terms = [
            'diabetes', 'hypertension', 'pneumonia', 'bronchitis',
            'asthma', 'copd', 'heart failure', 'myocardial infarction',
            'stroke', 'tia', 'fracture', 'arthritis', 'osteoporosis',
            'depression', 'anxiety', 'infection', 'sepsis', 'anemia',
            'thyroid', 'hypothyroidism', 'hyperthyroidism', 'cancer',
            'migraine', 'epilepsy', 'parkinson', 'alzheimer',
            'ulcer', 'gastritis', 'hepatitis', 'cirrhosis',
            'kidney disease', 'uti', 'prostatitis'
        ]

        for term in diagnosis_terms:
            if term in text_lower:
                diagnoses.append(term.title())

        return diagnoses

    def _calculate_diagnostic_confidence(
        self,
        diagnosis: str,
        symptoms: List[str],
        relevance_score: float
    ) -> DiagnosticConfidence:
        """Calculate confidence level for a diagnostic candidate"""
        # Check symptom-diagnosis mapping
        mapped_diagnoses = set()
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            if symptom_lower in self.symptom_diagnosis_map:
                mapped_diagnoses.update(self.symptom_diagnosis_map[symptom_lower])

        diagnosis_lower = diagnosis.lower()

        if diagnosis_lower in mapped_diagnoses and relevance_score > 0.7:
            return DiagnosticConfidence.HIGH
        elif diagnosis_lower in mapped_diagnoses or relevance_score > 0.6:
            return DiagnosticConfidence.MODERATE
        elif relevance_score > 0.4:
            return DiagnosticConfidence.LOW
        else:
            return DiagnosticConfidence.SPECULATIVE

    def _get_differentiating_factors(
        self,
        diagnosis: str,
        ruled_out: List[str]
    ) -> List[str]:
        """Get factors that differentiate this diagnosis from ruled-out ones"""
        factors = []

        # Simplified - in production, use medical knowledge base
        diff_map = {
            'pneumonia': ['consolidation on imaging', 'productive cough', 'fever pattern'],
            'bronchitis': ['no consolidation', 'dry/wet cough transition', 'gradual onset'],
            'asthma': ['reversible obstruction', 'wheezing pattern', 'trigger history'],
            'heart failure': ['edema pattern', 'bnp levels', 'ejection fraction'],
            'myocardial infarction': ['troponin elevation', 'ecg changes', 'chest pain pattern']
        }

        diagnosis_lower = diagnosis.lower()
        if diagnosis_lower in diff_map:
            factors = diff_map[diagnosis_lower]

        return factors

    def _generate_candidate_explanation(
        self,
        diagnosis: str,
        symptoms: List[str],
        confidence: DiagnosticConfidence
    ) -> str:
        """Generate explanation for diagnostic candidate"""
        symptom_str = ", ".join(symptoms[:3])
        return (
            f"{diagnosis} is a {confidence.value} confidence candidate based on "
            f"presenting symptoms ({symptom_str}). "
            f"Discovery search found similar cases in clinical records."
        )

    def _generate_next_steps(
        self,
        candidates: List[DiagnosticCandidate],
        symptoms: List[str]
    ) -> List[str]:
        """Generate recommended next steps"""
        steps = []

        if not candidates:
            steps.append("Insufficient data for differential diagnosis")
            steps.append("Consider broader symptom evaluation")
            return steps

        # Top candidate recommendations
        top = candidates[0]
        steps.append(f"Primary consideration: {top.diagnosis}")

        if top.suggested_tests:
            steps.append(f"Recommended tests: {', '.join(top.suggested_tests)}")

        # Differentiating steps
        if len(candidates) > 1:
            steps.append(
                f"Differentiate between {candidates[0].diagnosis} and {candidates[1].diagnosis}"
            )

        # General recommendations
        if any(c.confidence == DiagnosticConfidence.SPECULATIVE for c in candidates):
            steps.append("Consider specialist consultation for uncertain diagnoses")

        return steps

    def _generate_confidence_summary(self, candidates: List[DiagnosticCandidate]) -> str:
        """Generate summary of diagnostic confidence"""
        if not candidates:
            return "No diagnostic candidates identified"

        high_conf = sum(1 for c in candidates if c.confidence == DiagnosticConfidence.HIGH)
        mod_conf = sum(1 for c in candidates if c.confidence == DiagnosticConfidence.MODERATE)

        if high_conf > 0:
            return f"{high_conf} high-confidence and {mod_conf} moderate-confidence candidates identified"
        elif mod_conf > 0:
            return f"{mod_conf} moderate-confidence candidates identified - consider additional testing"
        else:
            return "Low confidence candidates only - broader evaluation recommended"

    def _infer_outcome(self, text: str) -> str:
        """Infer outcome from text"""
        text_lower = text.lower()
        if any(w in text_lower for w in ['improved', 'resolved', 'recovered']):
            return "positive"
        elif any(w in text_lower for w in ['worsened', 'declined', 'deteriorated']):
            return "negative"
        return "neutral"

    def _extract_tests_from_cases(self, cases: List[Dict]) -> Dict[str, int]:
        """Extract test mentions from similar cases"""
        test_counts = {}
        test_terms = [
            'blood test', 'cbc', 'cmp', 'x-ray', 'ct scan', 'mri',
            'ultrasound', 'ecg', 'ekg', 'echo', 'biopsy', 'culture',
            'urinalysis', 'lipid panel', 'thyroid panel', 'a1c'
        ]

        for case in cases:
            text = case.get('text_summary', '').lower()
            for test in test_terms:
                if test in text:
                    test_counts[test] = test_counts.get(test, 0) + 1

        return test_counts

    def _generate_exploration_insight(
        self,
        symptoms: List[str],
        diagnoses: List[str]
    ) -> str:
        """Generate insight from exploration results"""
        if not diagnoses:
            return "No clear diagnostic pattern identified"

        return f"Symptoms may be associated with: {', '.join(diagnoses[:3])}"

    def _load_symptom_diagnosis_map(self) -> Dict[str, List[str]]:
        """Load symptom to diagnosis mapping"""
        # Simplified mapping - in production, use medical ontology
        return {
            'chest pain': ['myocardial infarction', 'angina', 'pneumonia', 'gerd'],
            'shortness of breath': ['heart failure', 'asthma', 'copd', 'pneumonia'],
            'fatigue': ['anemia', 'hypothyroidism', 'diabetes', 'depression'],
            'headache': ['migraine', 'tension headache', 'hypertension', 'sinusitis'],
            'fever': ['infection', 'pneumonia', 'uti', 'flu'],
            'cough': ['bronchitis', 'pneumonia', 'asthma', 'copd'],
            'joint pain': ['arthritis', 'gout', 'lupus', 'lyme disease'],
            'abdominal pain': ['gastritis', 'ulcer', 'appendicitis', 'gallstones'],
            'dizziness': ['vertigo', 'hypotension', 'anemia', 'dehydration'],
            'weight loss': ['diabetes', 'hyperthyroidism', 'cancer', 'depression']
        }

    def _load_diagnostic_tests(self) -> Dict[str, List[str]]:
        """Load diagnosis to test mapping"""
        return {
            'diabetes': ['hba1c', 'fasting glucose', 'glucose tolerance test'],
            'hypertension': ['blood pressure monitoring', 'ecg', 'renal function'],
            'pneumonia': ['chest x-ray', 'sputum culture', 'cbc'],
            'heart failure': ['bnp', 'echocardiogram', 'chest x-ray'],
            'myocardial infarction': ['troponin', 'ecg', 'coronary angiography'],
            'asthma': ['spirometry', 'peak flow', 'allergy testing'],
            'hypothyroidism': ['tsh', 'free t4', 'thyroid antibodies'],
            'anemia': ['cbc', 'iron studies', 'b12 and folate'],
            'infection': ['cbc', 'crp', 'blood culture'],
            'arthritis': ['x-ray', 'rheumatoid factor', 'anti-ccp']
        }


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Differential Diagnosis CLI")
    parser.add_argument("--clinic-id", required=True, help="Clinic ID")
    parser.add_argument("--symptoms", required=True, nargs="+", help="Presenting symptoms")
    parser.add_argument("--ruled-out", nargs="*", default=[], help="Ruled out diagnoses")
    parser.add_argument("--confirmed", nargs="*", default=[], help="Confirmed findings")

    args = parser.parse_args()

    agent = DifferentialDiagnosisAgent(args.clinic_id)

    result = agent.differential_diagnosis(
        presenting_symptoms=args.symptoms,
        ruled_out_diagnoses=args.ruled_out,
        confirmed_findings=args.confirmed
    )

    print(f"\n{'='*60}")
    print("DIFFERENTIAL DIAGNOSIS ANALYSIS")
    print(f"{'='*60}")
    print(f"\nSymptoms: {', '.join(result.primary_symptoms)}")
    print(f"Ruled Out: {', '.join(result.ruled_out) or 'None'}")
    print(f"\n{result.confidence_summary}")

    print(f"\n--- Diagnostic Candidates ---")
    for i, candidate in enumerate(result.candidates, 1):
        print(f"\n{i}. {candidate.diagnosis} [{candidate.confidence.value}]")
        print(f"   Relevance: {candidate.relevance_score:.2f}")
        print(f"   {candidate.explanation}")
        if candidate.suggested_tests:
            print(f"   Tests: {', '.join(candidate.suggested_tests)}")

    print(f"\n--- Next Steps ---")
    for step in result.recommended_next_steps:
        print(f"  • {step}")


if __name__ == "__main__":
    main()
