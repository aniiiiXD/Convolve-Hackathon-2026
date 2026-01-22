"""
Advanced Insights Generator Agent

Generates actionable clinical insights from current data using:
- Temporal trend analysis
- Treatment effectiveness patterns
- Cohort similarity analysis
- Risk pattern detection
- Anomaly detection
- Predictive indicators
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
from enum import Enum

from qdrant_client import models

from medisync.core_agents.database_agent import client
from medisync.service_agents.memory_ops_agent import COLLECTION_NAME, GLOBAL_INSIGHTS_COLLECTION
from medisync.service_agents.encoding_agent import EmbeddingService
from medisync.service_agents.gatekeeper_agent import User

logger = logging.getLogger(__name__)


class InsightType(Enum):
    TEMPORAL_TREND = "temporal_trend"
    TREATMENT_EFFECTIVENESS = "treatment_effectiveness"
    COHORT_PATTERN = "cohort_pattern"
    RISK_INDICATOR = "risk_indicator"
    ANOMALY = "anomaly"
    CORRELATION = "correlation"
    PREDICTION = "prediction"
    COMPARISON = "comparison"


@dataclass
class GeneratedInsight:
    """Represents a generated insight"""
    insight_type: InsightType
    title: str
    description: str
    confidence: float
    evidence_ids: List[str]
    metrics: Dict[str, Any]
    recommendations: List[str]
    urgency: str = "normal"  # low, normal, high, critical
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_type": self.insight_type.value,
            "title": self.title,
            "description": self.description,
            "confidence": self.confidence,
            "evidence_ids": self.evidence_ids,
            "metrics": self.metrics,
            "recommendations": self.recommendations,
            "urgency": self.urgency,
            "created_at": self.created_at.isoformat()
        }


class InsightsGeneratorAgent:
    """
    Generates actionable clinical insights from current data.

    This agent analyzes patterns across clinical records to surface:
    - Emerging trends
    - Treatment effectiveness
    - Risk indicators
    - Anomalies requiring attention
    """

    def __init__(self, user: User):
        self.user = user
        self.clinic_id = user.clinic_id
        self.embedder = EmbeddingService()

    # ==================== TEMPORAL TREND ANALYSIS ====================

    def analyze_temporal_trends(
        self,
        condition: Optional[str] = None,
        time_window_days: int = 30,
        min_occurrences: int = 3
    ) -> List[GeneratedInsight]:
        """
        Analyze temporal trends in clinical data.

        Detects:
        - Increasing/decreasing condition frequencies
        - Seasonal patterns
        - Emerging health concerns
        """
        insights = []

        try:
            # Get recent records within time window
            cutoff_time = datetime.now() - timedelta(days=time_window_days)
            cutoff_timestamp = cutoff_time.timestamp()

            # Query records with time filter
            filter_conditions = [
                models.FieldCondition(
                    key="clinic_id",
                    match=models.MatchValue(value=self.clinic_id)
                ),
                models.FieldCondition(
                    key="timestamp",
                    range=models.Range(gte=cutoff_timestamp)
                )
            ]

            if condition:
                # Add semantic search for condition
                condition_vec = self.embedder.get_dense_embedding(condition)
                results = client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=("dense_text", condition_vec),
                    query_filter=models.Filter(must=filter_conditions),
                    limit=500,
                    with_payload=True
                )
            else:
                # Get all recent records
                results, _ = client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=models.Filter(must=filter_conditions),
                    limit=500,
                    with_payload=True
                )

            if not results:
                return insights

            # Analyze temporal distribution
            time_buckets = self._bucket_by_time(results, bucket_days=7)

            # Detect trends
            if len(time_buckets) >= 2:
                trend = self._calculate_trend(time_buckets)

                if abs(trend['slope']) > 0.1:  # Significant trend
                    direction = "increasing" if trend['slope'] > 0 else "decreasing"

                    insight = GeneratedInsight(
                        insight_type=InsightType.TEMPORAL_TREND,
                        title=f"{direction.capitalize()} trend detected",
                        description=f"Clinical records show a {direction} trend over the past {time_window_days} days. "
                                   f"Weekly change rate: {abs(trend['slope']*100):.1f}%",
                        confidence=min(trend['r_squared'], 0.95),
                        evidence_ids=[str(r.id) for r in results[:10]],
                        metrics={
                            "trend_direction": direction,
                            "slope": trend['slope'],
                            "r_squared": trend['r_squared'],
                            "total_records": len(results),
                            "time_window_days": time_window_days
                        },
                        recommendations=self._generate_trend_recommendations(direction, trend),
                        urgency="high" if trend['slope'] > 0.3 else "normal"
                    )
                    insights.append(insight)

            # Detect spikes/anomalies in time series
            spike_insights = self._detect_temporal_spikes(time_buckets, results)
            insights.extend(spike_insights)

        except Exception as e:
            logger.error(f"Error analyzing temporal trends: {e}", exc_info=True)

        return insights

    def _bucket_by_time(self, records, bucket_days: int = 7) -> Dict[str, List]:
        """Group records into time buckets"""
        buckets = defaultdict(list)

        for record in records:
            timestamp = record.payload.get('timestamp', 0)
            if timestamp:
                dt = datetime.fromtimestamp(timestamp)
                bucket_key = dt.strftime(f"%Y-W%W")  # Week-based buckets
                buckets[bucket_key].append(record)

        return dict(sorted(buckets.items()))

    def _calculate_trend(self, time_buckets: Dict[str, List]) -> Dict[str, float]:
        """Calculate trend slope using simple linear regression"""
        counts = [len(records) for records in time_buckets.values()]

        if len(counts) < 2:
            return {'slope': 0, 'r_squared': 0}

        n = len(counts)
        x = list(range(n))

        # Simple linear regression
        x_mean = sum(x) / n
        y_mean = sum(counts) / n

        numerator = sum((x[i] - x_mean) * (counts[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # Normalize slope by mean
        normalized_slope = slope / y_mean if y_mean != 0 else 0

        # Calculate R-squared
        y_pred = [slope * x[i] + (y_mean - slope * x_mean) for i in range(n)]
        ss_res = sum((counts[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((counts[i] - y_mean) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {'slope': normalized_slope, 'r_squared': max(0, r_squared)}

    def _detect_temporal_spikes(self, time_buckets: Dict, records) -> List[GeneratedInsight]:
        """Detect unusual spikes in temporal data"""
        insights = []
        counts = [len(records) for records in time_buckets.values()]

        if len(counts) < 3:
            return insights

        mean_count = sum(counts) / len(counts)
        std_dev = (sum((c - mean_count) ** 2 for c in counts) / len(counts)) ** 0.5

        # Detect spikes (> 2 standard deviations)
        bucket_names = list(time_buckets.keys())
        for i, count in enumerate(counts):
            if std_dev > 0 and (count - mean_count) / std_dev > 2:
                insight = GeneratedInsight(
                    insight_type=InsightType.ANOMALY,
                    title=f"Unusual spike detected in {bucket_names[i]}",
                    description=f"Record count ({count}) is {((count-mean_count)/std_dev):.1f} standard deviations above normal ({mean_count:.0f})",
                    confidence=0.85,
                    evidence_ids=[str(r.id) for r in list(time_buckets.values())[i][:5]],
                    metrics={
                        "spike_count": count,
                        "mean_count": mean_count,
                        "std_deviations": (count - mean_count) / std_dev
                    },
                    recommendations=["Investigate the cause of this spike", "Review affected patient records"],
                    urgency="high"
                )
                insights.append(insight)

        return insights

    def _generate_trend_recommendations(self, direction: str, trend: Dict) -> List[str]:
        """Generate recommendations based on trend analysis"""
        recommendations = []

        if direction == "increasing":
            recommendations.append("Monitor this trend closely for potential outbreak or emerging issue")
            recommendations.append("Consider allocating additional resources if trend continues")
            if trend['slope'] > 0.3:
                recommendations.append("URGENT: Rapid increase detected - immediate review recommended")
        else:
            recommendations.append("Positive trend: Continue current practices")
            recommendations.append("Document successful interventions for future reference")

        return recommendations

    # ==================== TREATMENT EFFECTIVENESS ====================

    def analyze_treatment_effectiveness(
        self,
        treatment: str,
        condition: Optional[str] = None,
        min_cases: int = 5
    ) -> List[GeneratedInsight]:
        """
        Analyze treatment effectiveness based on outcome data.

        Compares:
        - Success rates across different treatments
        - Time to resolution
        - Patient satisfaction indicators
        """
        insights = []

        try:
            # Search for treatment-related records
            treatment_vec = self.embedder.get_dense_embedding(treatment)

            filter_conditions = [
                models.FieldCondition(
                    key="clinic_id",
                    match=models.MatchValue(value=self.clinic_id)
                )
            ]

            results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=("dense_text", treatment_vec),
                query_filter=models.Filter(must=filter_conditions),
                limit=200,
                score_threshold=0.6,
                with_payload=True
            )

            if len(results) < min_cases:
                return insights

            # Analyze outcomes from record text
            outcomes = self._extract_outcomes_from_records(results)

            if outcomes['total'] >= min_cases:
                success_rate = outcomes['positive'] / outcomes['total'] if outcomes['total'] > 0 else 0

                insight = GeneratedInsight(
                    insight_type=InsightType.TREATMENT_EFFECTIVENESS,
                    title=f"Treatment effectiveness: {treatment}",
                    description=f"Based on {outcomes['total']} cases, this treatment shows "
                               f"{success_rate*100:.1f}% positive outcomes.",
                    confidence=min(0.9, outcomes['total'] / 50),  # More cases = higher confidence
                    evidence_ids=[str(r.id) for r in results[:10]],
                    metrics={
                        "treatment": treatment,
                        "total_cases": outcomes['total'],
                        "positive_outcomes": outcomes['positive'],
                        "negative_outcomes": outcomes['negative'],
                        "neutral_outcomes": outcomes['neutral'],
                        "success_rate": success_rate
                    },
                    recommendations=self._generate_treatment_recommendations(success_rate, outcomes),
                    urgency="normal" if success_rate > 0.6 else "high"
                )
                insights.append(insight)

                # Compare with alternative treatments if data available
                comparison_insights = self._compare_treatments(treatment, condition)
                insights.extend(comparison_insights)

        except Exception as e:
            logger.error(f"Error analyzing treatment effectiveness: {e}", exc_info=True)

        return insights

    def _extract_outcomes_from_records(self, records) -> Dict[str, int]:
        """Extract outcome indicators from record text"""
        outcomes = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}

        positive_indicators = ['improved', 'resolved', 'better', 'successful', 'healed',
                              'recovered', 'stable', 'good response', 'effective']
        negative_indicators = ['worsened', 'failed', 'no improvement', 'declined',
                              'adverse', 'complication', 'unsuccessful', 'deteriorated']

        for record in records:
            text = record.payload.get('text_content', '').lower()
            outcomes['total'] += 1

            if any(ind in text for ind in positive_indicators):
                outcomes['positive'] += 1
            elif any(ind in text for ind in negative_indicators):
                outcomes['negative'] += 1
            else:
                outcomes['neutral'] += 1

        return outcomes

    def _generate_treatment_recommendations(self, success_rate: float, outcomes: Dict) -> List[str]:
        """Generate recommendations based on treatment effectiveness"""
        recommendations = []

        if success_rate >= 0.8:
            recommendations.append("High effectiveness - consider as primary treatment option")
            recommendations.append("Document protocol for standardization")
        elif success_rate >= 0.6:
            recommendations.append("Moderate effectiveness - suitable for many cases")
            recommendations.append("Consider patient-specific factors when prescribing")
        else:
            recommendations.append("Lower effectiveness - review alternative treatments")
            recommendations.append("Investigate factors contributing to poor outcomes")
            recommendations.append("Consider second-line treatments")

        if outcomes['negative'] > outcomes['positive']:
            recommendations.append("ALERT: Negative outcomes exceed positive - urgent review needed")

        return recommendations

    def _compare_treatments(self, treatment: str, condition: Optional[str]) -> List[GeneratedInsight]:
        """Compare treatment with alternatives"""
        # This would compare with other treatments for the same condition
        # Simplified implementation
        return []

    # ==================== COHORT ANALYSIS ====================

    def analyze_cohort_patterns(
        self,
        cohort_criteria: str,
        limit: int = 100
    ) -> List[GeneratedInsight]:
        """
        Analyze patterns within a patient cohort.

        Identifies:
        - Common characteristics
        - Shared risk factors
        - Treatment response patterns
        """
        insights = []

        try:
            # Find cohort using semantic search
            cohort_vec = self.embedder.get_dense_embedding(cohort_criteria)

            results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=("dense_text", cohort_vec),
                query_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="clinic_id",
                        match=models.MatchValue(value=self.clinic_id)
                    )
                ]),
                limit=limit,
                score_threshold=0.65,
                with_payload=True
            )

            if len(results) < 5:
                return insights

            # Analyze cohort characteristics
            patient_ids = set()
            common_terms = defaultdict(int)

            for record in results:
                patient_id = record.payload.get('patient_id')
                if patient_id:
                    patient_ids.add(patient_id)

                # Extract common terms
                text = record.payload.get('text_content', '')
                terms = self._extract_medical_terms(text)
                for term in terms:
                    common_terms[term] += 1

            # Get top common patterns
            top_patterns = sorted(common_terms.items(), key=lambda x: x[1], reverse=True)[:10]

            insight = GeneratedInsight(
                insight_type=InsightType.COHORT_PATTERN,
                title=f"Cohort analysis: {cohort_criteria[:50]}",
                description=f"Analyzed {len(patient_ids)} patients matching criteria. "
                           f"Common patterns identified across {len(results)} records.",
                confidence=0.75,
                evidence_ids=[str(r.id) for r in results[:10]],
                metrics={
                    "cohort_size": len(patient_ids),
                    "total_records": len(results),
                    "common_patterns": dict(top_patterns),
                    "criteria": cohort_criteria
                },
                recommendations=[
                    f"Most common finding: {top_patterns[0][0]} (found in {top_patterns[0][1]} records)" if top_patterns else "No clear patterns",
                    "Consider targeted interventions for this cohort",
                    "Monitor cohort for outcome tracking"
                ],
                urgency="normal"
            )
            insights.append(insight)

        except Exception as e:
            logger.error(f"Error analyzing cohort patterns: {e}", exc_info=True)

        return insights

    def _extract_medical_terms(self, text: str) -> List[str]:
        """Extract medical terms from text (simplified)"""
        # In production, use medical NER or terminology service
        common_medical_terms = [
            'diabetes', 'hypertension', 'fracture', 'infection', 'pain',
            'fever', 'cough', 'headache', 'nausea', 'fatigue',
            'anxiety', 'depression', 'arthritis', 'asthma', 'pneumonia'
        ]

        text_lower = text.lower()
        found_terms = [term for term in common_medical_terms if term in text_lower]
        return found_terms

    # ==================== RISK PATTERN DETECTION ====================

    def detect_risk_patterns(
        self,
        risk_factor: Optional[str] = None,
        lookback_days: int = 90
    ) -> List[GeneratedInsight]:
        """
        Detect risk patterns and early warning signs.

        Identifies:
        - Patients at elevated risk
        - Deterioration patterns
        - Risk factor correlations
        """
        insights = []

        try:
            cutoff_time = datetime.now() - timedelta(days=lookback_days)
            cutoff_timestamp = cutoff_time.timestamp()

            filter_conditions = [
                models.FieldCondition(
                    key="clinic_id",
                    match=models.MatchValue(value=self.clinic_id)
                ),
                models.FieldCondition(
                    key="timestamp",
                    range=models.Range(gte=cutoff_timestamp)
                )
            ]

            if risk_factor:
                risk_vec = self.embedder.get_dense_embedding(risk_factor)
                results = client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=("dense_text", risk_vec),
                    query_filter=models.Filter(must=filter_conditions),
                    limit=200,
                    score_threshold=0.6,
                    with_payload=True
                )
            else:
                # Search for general risk indicators
                risk_terms = "deteriorating worsening declining emergency urgent critical"
                risk_vec = self.embedder.get_dense_embedding(risk_terms)
                results = client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=("dense_text", risk_vec),
                    query_filter=models.Filter(must=filter_conditions),
                    limit=200,
                    score_threshold=0.5,
                    with_payload=True
                )

            if not results:
                return insights

            # Group by patient to identify patterns
            patient_records = defaultdict(list)
            for record in results:
                patient_id = record.payload.get('patient_id')
                if patient_id:
                    patient_records[patient_id].append(record)

            # Identify patients with multiple risk indicators
            high_risk_patients = []
            for patient_id, records in patient_records.items():
                if len(records) >= 2:  # Multiple concerning records
                    risk_score = self._calculate_patient_risk_score(records)
                    if risk_score > 0.6:
                        high_risk_patients.append({
                            'patient_id': patient_id,
                            'risk_score': risk_score,
                            'record_count': len(records)
                        })

            if high_risk_patients:
                high_risk_patients.sort(key=lambda x: x['risk_score'], reverse=True)

                insight = GeneratedInsight(
                    insight_type=InsightType.RISK_INDICATOR,
                    title=f"High-risk patients identified",
                    description=f"Found {len(high_risk_patients)} patients with elevated risk patterns "
                               f"in the past {lookback_days} days.",
                    confidence=0.8,
                    evidence_ids=[str(results[0].id)] if results else [],
                    metrics={
                        "high_risk_count": len(high_risk_patients),
                        "risk_factor": risk_factor or "general",
                        "lookback_days": lookback_days,
                        "top_risk_patients": [
                            {"patient_id": self._hash_patient_id(p['patient_id']),
                             "risk_score": p['risk_score']}
                            for p in high_risk_patients[:5]
                        ]
                    },
                    recommendations=[
                        f"Priority review needed for {len(high_risk_patients)} patients",
                        "Schedule follow-up appointments for high-risk cases",
                        "Consider preventive interventions"
                    ],
                    urgency="critical" if len(high_risk_patients) > 5 else "high"
                )
                insights.append(insight)

        except Exception as e:
            logger.error(f"Error detecting risk patterns: {e}", exc_info=True)

        return insights

    def _calculate_patient_risk_score(self, records) -> float:
        """Calculate risk score based on record patterns"""
        risk_keywords = ['critical', 'emergency', 'urgent', 'severe', 'worsening',
                        'deteriorating', 'uncontrolled', 'unstable']

        total_risk = 0
        for record in records:
            text = record.payload.get('text_content', '').lower()
            keyword_count = sum(1 for kw in risk_keywords if kw in text)
            record_risk = min(keyword_count * 0.2, 1.0)
            total_risk += record_risk

        # Normalize by number of records with diminishing returns
        return min(total_risk / (len(records) ** 0.5), 1.0)

    def _hash_patient_id(self, patient_id: str) -> str:
        """Hash patient ID for privacy"""
        return hashlib.sha256(patient_id.encode()).hexdigest()[:8]

    # ==================== CORRELATION DISCOVERY ====================

    def discover_correlations(
        self,
        factor_a: str,
        factor_b: str,
        min_co_occurrences: int = 5
    ) -> List[GeneratedInsight]:
        """
        Discover correlations between clinical factors.

        Analyzes:
        - Co-occurrence patterns
        - Conditional probabilities
        - Potential causal relationships
        """
        insights = []

        try:
            # Search for each factor
            vec_a = self.embedder.get_dense_embedding(factor_a)
            vec_b = self.embedder.get_dense_embedding(factor_b)

            filter_cond = models.Filter(must=[
                models.FieldCondition(
                    key="clinic_id",
                    match=models.MatchValue(value=self.clinic_id)
                )
            ])

            results_a = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=("dense_text", vec_a),
                query_filter=filter_cond,
                limit=200,
                score_threshold=0.65,
                with_payload=True
            )

            results_b = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=("dense_text", vec_b),
                query_filter=filter_cond,
                limit=200,
                score_threshold=0.65,
                with_payload=True
            )

            if not results_a or not results_b:
                return insights

            # Find patient overlap
            patients_a = {r.payload.get('patient_id') for r in results_a if r.payload.get('patient_id')}
            patients_b = {r.payload.get('patient_id') for r in results_b if r.payload.get('patient_id')}

            co_occurrence = patients_a & patients_b

            if len(co_occurrence) >= min_co_occurrences:
                # Calculate correlation metrics
                correlation_rate = len(co_occurrence) / min(len(patients_a), len(patients_b))

                insight = GeneratedInsight(
                    insight_type=InsightType.CORRELATION,
                    title=f"Correlation detected: {factor_a} & {factor_b}",
                    description=f"Found {len(co_occurrence)} patients with both factors. "
                               f"Correlation rate: {correlation_rate*100:.1f}%",
                    confidence=min(0.85, len(co_occurrence) / 20),
                    evidence_ids=[str(r.id) for r in results_a[:5]],
                    metrics={
                        "factor_a": factor_a,
                        "factor_b": factor_b,
                        "factor_a_patients": len(patients_a),
                        "factor_b_patients": len(patients_b),
                        "co_occurrence_count": len(co_occurrence),
                        "correlation_rate": correlation_rate
                    },
                    recommendations=[
                        f"Consider screening for {factor_b} in patients with {factor_a}",
                        "Investigate potential causal relationship",
                        "Update treatment protocols if correlation is clinically significant"
                    ],
                    urgency="normal" if correlation_rate < 0.5 else "high"
                )
                insights.append(insight)

        except Exception as e:
            logger.error(f"Error discovering correlations: {e}", exc_info=True)

        return insights

    # ==================== PREDICTIVE INSIGHTS ====================

    def generate_predictive_insights(
        self,
        patient_id: str,
        prediction_type: str = "risk"
    ) -> List[GeneratedInsight]:
        """
        Generate predictive insights for a patient.

        Uses:
        - Similar patient trajectories
        - Historical outcome patterns
        - Risk factor analysis
        """
        insights = []

        try:
            # Get patient's records
            patient_filter = models.Filter(must=[
                models.FieldCondition(
                    key="clinic_id",
                    match=models.MatchValue(value=self.clinic_id)
                ),
                models.FieldCondition(
                    key="patient_id",
                    match=models.MatchValue(value=patient_id)
                )
            ])

            patient_records, _ = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=patient_filter,
                limit=50,
                with_payload=True,
                with_vectors=True
            )

            if not patient_records:
                return insights

            # Get average embedding of patient's records
            patient_embedding = self._get_patient_centroid(patient_records)

            # Find similar patients (excluding current patient)
            similar_results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=("dense_text", patient_embedding),
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="clinic_id",
                            match=models.MatchValue(value=self.clinic_id)
                        )
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="patient_id",
                            match=models.MatchValue(value=patient_id)
                        )
                    ]
                ),
                limit=100,
                with_payload=True
            )

            # Analyze outcomes of similar patients
            similar_patients = defaultdict(list)
            for record in similar_results:
                sim_patient_id = record.payload.get('patient_id')
                if sim_patient_id:
                    similar_patients[sim_patient_id].append(record)

            # Extract outcome patterns
            outcomes = self._analyze_similar_patient_outcomes(similar_patients)

            if outcomes['total_similar'] >= 3:
                insight = GeneratedInsight(
                    insight_type=InsightType.PREDICTION,
                    title=f"Predictive analysis for patient",
                    description=f"Based on {outcomes['total_similar']} similar patients, "
                               f"predicted trajectory analysis completed.",
                    confidence=min(0.75, outcomes['total_similar'] / 20),
                    evidence_ids=[str(r.id) for r in similar_results[:5]],
                    metrics={
                        "patient_id_hash": self._hash_patient_id(patient_id),
                        "similar_patients_analyzed": outcomes['total_similar'],
                        "predicted_positive_rate": outcomes['positive_rate'],
                        "predicted_risk_score": outcomes['risk_score'],
                        "common_outcomes": outcomes['common_outcomes']
                    },
                    recommendations=self._generate_predictive_recommendations(outcomes),
                    urgency="high" if outcomes['risk_score'] > 0.6 else "normal"
                )
                insights.append(insight)

        except Exception as e:
            logger.error(f"Error generating predictive insights: {e}", exc_info=True)

        return insights

    def _get_patient_centroid(self, records) -> List[float]:
        """Calculate centroid embedding for patient records"""
        vectors = []
        for record in records:
            if hasattr(record, 'vector') and record.vector:
                if isinstance(record.vector, dict) and 'dense_text' in record.vector:
                    vectors.append(record.vector['dense_text'])

        if not vectors:
            # Fallback: embed concatenated text
            all_text = " ".join(r.payload.get('text_content', '') for r in records[:5])
            return self.embedder.get_dense_embedding(all_text)

        # Calculate centroid
        centroid = [sum(v[i] for v in vectors) / len(vectors) for i in range(len(vectors[0]))]
        return centroid

    def _analyze_similar_patient_outcomes(self, similar_patients: Dict) -> Dict:
        """Analyze outcomes of similar patients"""
        positive_indicators = ['improved', 'resolved', 'recovered', 'stable', 'better']
        negative_indicators = ['worsened', 'declined', 'deteriorated', 'critical']

        positive_count = 0
        negative_count = 0
        outcome_terms = defaultdict(int)

        for patient_id, records in similar_patients.items():
            patient_text = " ".join(r.payload.get('text_content', '').lower() for r in records)

            has_positive = any(ind in patient_text for ind in positive_indicators)
            has_negative = any(ind in patient_text for ind in negative_indicators)

            if has_positive and not has_negative:
                positive_count += 1
            elif has_negative:
                negative_count += 1

            # Track common terms
            for term in self._extract_medical_terms(patient_text):
                outcome_terms[term] += 1

        total = len(similar_patients)
        return {
            'total_similar': total,
            'positive_rate': positive_count / total if total > 0 else 0,
            'negative_rate': negative_count / total if total > 0 else 0,
            'risk_score': negative_count / total if total > 0 else 0,
            'common_outcomes': dict(sorted(outcome_terms.items(), key=lambda x: x[1], reverse=True)[:5])
        }

    def _generate_predictive_recommendations(self, outcomes: Dict) -> List[str]:
        """Generate recommendations based on predictive analysis"""
        recommendations = []

        if outcomes['risk_score'] > 0.5:
            recommendations.append("HIGH RISK: Consider proactive intervention")
            recommendations.append("Schedule closer monitoring")
        elif outcomes['positive_rate'] > 0.6:
            recommendations.append("Favorable prognosis based on similar cases")
            recommendations.append("Continue current treatment plan")
        else:
            recommendations.append("Mixed outcomes in similar patients - individualized approach recommended")

        if outcomes['common_outcomes']:
            top_outcome = list(outcomes['common_outcomes'].keys())[0]
            recommendations.append(f"Common finding in similar patients: {top_outcome}")

        return recommendations

    # ==================== COMPREHENSIVE ANALYSIS ====================

    def generate_comprehensive_insights(
        self,
        focus_area: Optional[str] = None
    ) -> Dict[str, List[GeneratedInsight]]:
        """
        Generate comprehensive insights across all analysis types.

        Returns categorized insights for dashboard display.
        """
        all_insights = {
            'temporal_trends': [],
            'treatment_effectiveness': [],
            'cohort_patterns': [],
            'risk_indicators': [],
            'correlations': [],
            'predictions': [],
            'anomalies': []
        }

        try:
            # Temporal trends
            all_insights['temporal_trends'] = self.analyze_temporal_trends(
                condition=focus_area,
                time_window_days=30
            )

            # Risk patterns
            all_insights['risk_indicators'] = self.detect_risk_patterns(
                risk_factor=focus_area,
                lookback_days=60
            )

            # Cohort patterns (if focus area provided)
            if focus_area:
                all_insights['cohort_patterns'] = self.analyze_cohort_patterns(
                    cohort_criteria=focus_area
                )

                all_insights['treatment_effectiveness'] = self.analyze_treatment_effectiveness(
                    treatment=focus_area
                )

            # Extract anomalies from all insights
            for category, insights in all_insights.items():
                anomalies = [i for i in insights if i.insight_type == InsightType.ANOMALY]
                all_insights['anomalies'].extend(anomalies)

            logger.info(
                f"Generated comprehensive insights: "
                f"{sum(len(v) for v in all_insights.values())} total insights"
            )

        except Exception as e:
            logger.error(f"Error generating comprehensive insights: {e}", exc_info=True)

        return all_insights

    def get_insights_summary(self) -> Dict[str, Any]:
        """Get a summary of available insights"""
        insights = self.generate_comprehensive_insights()

        total_insights = sum(len(v) for v in insights.values())
        critical_count = sum(
            1 for category in insights.values()
            for insight in category
            if insight.urgency in ['critical', 'high']
        )

        return {
            'total_insights': total_insights,
            'critical_alerts': critical_count,
            'categories': {k: len(v) for k, v in insights.items()},
            'generated_at': datetime.now().isoformat()
        }


# CLI Entry Point
def main():
    """CLI for testing insights generator"""
    import argparse
    from medisync.service_agents.gatekeeper_agent import User

    parser = argparse.ArgumentParser(description="Generate clinical insights")
    parser.add_argument("--clinic-id", required=True, help="Clinic ID")
    parser.add_argument("--focus", help="Focus area for insights")
    parser.add_argument("--type", choices=['trends', 'treatment', 'cohort', 'risk', 'comprehensive'],
                       default='comprehensive', help="Type of insights")

    args = parser.parse_args()

    # Create test user
    user = User(
        id="insights_user",
        username="insights_generator",
        role="DOCTOR",
        clinic_id=args.clinic_id
    )

    generator = InsightsGeneratorAgent(user)

    if args.type == 'comprehensive':
        insights = generator.generate_comprehensive_insights(focus_area=args.focus)
        for category, category_insights in insights.items():
            if category_insights:
                print(f"\n=== {category.upper()} ===")
                for insight in category_insights:
                    print(f"\n[{insight.urgency.upper()}] {insight.title}")
                    print(f"  {insight.description}")
                    print(f"  Confidence: {insight.confidence*100:.0f}%")
                    for rec in insight.recommendations[:2]:
                        print(f"  - {rec}")
    else:
        print(f"Running {args.type} analysis...")


if __name__ == "__main__":
    main()
