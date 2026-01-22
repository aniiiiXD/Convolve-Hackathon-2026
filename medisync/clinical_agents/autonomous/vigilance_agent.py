"""
Autonomous Vigilance Agent

Continuously monitors patient states and proactively alerts clinicians
about significant changes, risks, or anomalies requiring attention.

This agent operates autonomously, periodically scanning clinical data
and generating actionable alerts without explicit user queries.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import hashlib

from qdrant_client import models

from medisync.core_agents.database_agent import client
from medisync.service_agents.memory_ops_agent import COLLECTION_NAME
from medisync.service_agents.encoding_agent import EmbeddingService
from medisync.service_agents.discovery_agent import DiscoveryService

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    RISK_ELEVATION = "risk_elevation"
    CONDITION_CHANGE = "condition_change"
    MISSED_FOLLOWUP = "missed_followup"
    ANOMALY_DETECTED = "anomaly_detected"
    TREATMENT_CONCERN = "treatment_concern"
    SIMILAR_CASE_OUTCOME = "similar_case_outcome"


@dataclass
class ClinicalAlert:
    """Represents a proactive clinical alert"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    patient_id_hash: str  # Privacy: only hashed IDs
    title: str
    description: str
    evidence: List[Dict[str, Any]]
    recommended_actions: List[str]
    similar_cases: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    assigned_doctor_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "patient_id_hash": self.patient_id_hash,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "recommended_actions": self.recommended_actions,
            "similar_cases": self.similar_cases,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
            "assigned_doctor_id": self.assigned_doctor_id
        }


@dataclass
class PatientState:
    """Represents a patient's current clinical state"""
    patient_id: str
    patient_id_hash: str
    clinic_id: str
    risk_score: float
    last_visit: datetime
    record_count: int
    latest_embedding: List[float]
    key_conditions: List[str]
    key_treatments: List[str]
    trend: str  # "stable", "improving", "declining"


class VigilanceAgent:
    """
    Autonomous agent that continuously monitors patient states
    and generates proactive clinical alerts.

    Capabilities:
    - Patient state change detection
    - Risk elevation monitoring
    - Follow-up compliance tracking
    - Anomaly detection
    - Similar case outcome analysis
    """

    def __init__(
        self,
        clinic_id: str,
        check_interval_seconds: int = 300,
        alert_callback: Optional[Callable[[ClinicalAlert], None]] = None
    ):
        self.clinic_id = clinic_id
        self.check_interval = check_interval_seconds
        self.alert_callback = alert_callback
        self.embedder = EmbeddingService()
        self.running = False
        self.alerts: List[ClinicalAlert] = []
        self.patient_states: Dict[str, PatientState] = {}
        self._alert_counter = 0

        # Thresholds
        self.risk_threshold_warning = 0.5
        self.risk_threshold_high = 0.7
        self.risk_threshold_critical = 0.85
        self.followup_overdue_days = 14
        self.change_detection_threshold = 0.3

    # ==================== MONITORING LOOP ====================

    async def start_monitoring(self):
        """Start the autonomous monitoring loop"""
        self.running = True
        logger.info(f"VigilanceAgent started for clinic {self.clinic_id}")

        while self.running:
            try:
                await self._monitoring_cycle()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {e}", exc_info=True)
                await asyncio.sleep(60)  # Brief pause on error

    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.running = False
        logger.info(f"VigilanceAgent stopped for clinic {self.clinic_id}")

    async def _monitoring_cycle(self):
        """Execute one monitoring cycle"""
        logger.info(f"Running monitoring cycle for clinic {self.clinic_id}")

        # 1. Scan for patient state changes
        state_changes = await self._detect_state_changes()

        # 2. Evaluate risk for changed patients
        for patient_id, change_info in state_changes.items():
            await self._evaluate_patient_risk(patient_id, change_info)

        # 3. Check for overdue follow-ups
        await self._check_followup_compliance()

        # 4. Detect anomalies in recent data
        await self._detect_anomalies()

        # 5. Analyze outcomes of similar cases
        await self._analyze_similar_case_outcomes()

        logger.info(f"Monitoring cycle complete. Active alerts: {len(self.get_active_alerts())}")

    # ==================== STATE CHANGE DETECTION ====================

    async def _detect_state_changes(self) -> Dict[str, Dict]:
        """Detect significant changes in patient states"""
        changes = {}

        try:
            # Get recent records (last 24 hours)
            cutoff = datetime.now() - timedelta(hours=24)
            cutoff_timestamp = cutoff.timestamp()

            recent_records, _ = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="clinic_id",
                        match=models.MatchValue(value=self.clinic_id)
                    ),
                    models.FieldCondition(
                        key="timestamp",
                        range=models.Range(gte=cutoff_timestamp)
                    )
                ]),
                limit=500,
                with_payload=True,
                with_vectors=True
            )

            # Group by patient
            patient_records = defaultdict(list)
            for record in recent_records:
                patient_id = record.payload.get('patient_id')
                if patient_id:
                    patient_records[patient_id].append(record)

            # Compare with stored states
            for patient_id, records in patient_records.items():
                current_state = self._compute_patient_state(patient_id, records)

                if patient_id in self.patient_states:
                    previous_state = self.patient_states[patient_id]
                    change_magnitude = self._calculate_state_change(
                        previous_state, current_state
                    )

                    if change_magnitude > self.change_detection_threshold:
                        changes[patient_id] = {
                            'previous_state': previous_state,
                            'current_state': current_state,
                            'change_magnitude': change_magnitude,
                            'new_records': records
                        }

                # Update stored state
                self.patient_states[patient_id] = current_state

        except Exception as e:
            logger.error(f"Error detecting state changes: {e}", exc_info=True)

        return changes

    def _compute_patient_state(self, patient_id: str, records) -> PatientState:
        """Compute current state from patient records"""
        # Calculate risk score
        risk_score = self._calculate_risk_score(records)

        # Get latest embedding (average of recent records)
        embeddings = []
        for record in records[-5:]:  # Last 5 records
            if hasattr(record, 'vector') and record.vector:
                if isinstance(record.vector, dict) and 'dense_text' in record.vector:
                    embeddings.append(record.vector['dense_text'])

        if embeddings:
            latest_embedding = [
                sum(e[i] for e in embeddings) / len(embeddings)
                for i in range(len(embeddings[0]))
            ]
        else:
            # Fallback: embed concatenated text
            all_text = " ".join(r.payload.get('text_content', '')[:200] for r in records[-3:])
            latest_embedding = self.embedder.get_dense_embedding(all_text)

        # Extract key conditions and treatments
        key_conditions = self._extract_conditions(records)
        key_treatments = self._extract_treatments(records)

        # Determine trend
        trend = self._determine_trend(records)

        # Get latest visit
        timestamps = [r.payload.get('timestamp', 0) for r in records]
        latest_timestamp = max(timestamps) if timestamps else 0
        last_visit = datetime.fromtimestamp(latest_timestamp) if latest_timestamp else datetime.now()

        return PatientState(
            patient_id=patient_id,
            patient_id_hash=self._hash_id(patient_id),
            clinic_id=self.clinic_id,
            risk_score=risk_score,
            last_visit=last_visit,
            record_count=len(records),
            latest_embedding=latest_embedding,
            key_conditions=key_conditions,
            key_treatments=key_treatments,
            trend=trend
        )

    def _calculate_state_change(self, previous: PatientState, current: PatientState) -> float:
        """Calculate magnitude of state change"""
        changes = []

        # Risk score change
        risk_change = abs(current.risk_score - previous.risk_score)
        changes.append(risk_change)

        # Embedding similarity (cosine)
        if previous.latest_embedding and current.latest_embedding:
            dot_product = sum(a * b for a, b in zip(previous.latest_embedding, current.latest_embedding))
            norm_a = sum(a ** 2 for a in previous.latest_embedding) ** 0.5
            norm_b = sum(b ** 2 for b in current.latest_embedding) ** 0.5
            similarity = dot_product / (norm_a * norm_b) if norm_a and norm_b else 1.0
            embedding_change = 1 - similarity
            changes.append(embedding_change)

        # New conditions
        new_conditions = set(current.key_conditions) - set(previous.key_conditions)
        condition_change = len(new_conditions) * 0.2
        changes.append(min(condition_change, 1.0))

        # Trend change
        if previous.trend != current.trend:
            trend_change = 0.3 if current.trend == "declining" else 0.1
            changes.append(trend_change)

        return sum(changes) / len(changes) if changes else 0

    def _calculate_risk_score(self, records) -> float:
        """Calculate risk score from records"""
        risk_keywords = {
            'critical': 0.9, 'emergency': 0.85, 'urgent': 0.8,
            'severe': 0.7, 'worsening': 0.6, 'deteriorating': 0.65,
            'uncontrolled': 0.55, 'unstable': 0.5, 'concerning': 0.4
        }

        total_risk = 0
        weighted_count = 0

        for record in records:
            text = record.payload.get('text_content', '').lower()
            timestamp = record.payload.get('timestamp', 0)

            # More recent records weighted higher
            age_days = (datetime.now().timestamp() - timestamp) / 86400 if timestamp else 30
            recency_weight = max(0.1, 1 - (age_days / 30))

            for keyword, risk_value in risk_keywords.items():
                if keyword in text:
                    total_risk += risk_value * recency_weight
                    weighted_count += recency_weight

        return min(total_risk / max(weighted_count, 1), 1.0)

    def _extract_conditions(self, records) -> List[str]:
        """Extract medical conditions from records"""
        conditions = set()
        condition_terms = [
            'diabetes', 'hypertension', 'asthma', 'copd', 'heart failure',
            'cancer', 'infection', 'fracture', 'pneumonia', 'stroke'
        ]

        for record in records:
            text = record.payload.get('text_content', '').lower()
            for term in condition_terms:
                if term in text:
                    conditions.add(term)

        return list(conditions)

    def _extract_treatments(self, records) -> List[str]:
        """Extract treatments from records"""
        treatments = set()
        treatment_indicators = [
            'prescribed', 'started', 'administered', 'surgery',
            'therapy', 'treatment', 'medication'
        ]

        for record in records:
            text = record.payload.get('text_content', '').lower()
            for indicator in treatment_indicators:
                if indicator in text:
                    # Extract surrounding context (simplified)
                    treatments.add(indicator)

        return list(treatments)

    def _determine_trend(self, records) -> str:
        """Determine patient health trend"""
        if len(records) < 2:
            return "stable"

        positive_indicators = ['improved', 'better', 'resolved', 'stable']
        negative_indicators = ['worsened', 'declined', 'deteriorated']

        # Check recent records
        recent_text = " ".join(r.payload.get('text_content', '').lower() for r in records[-3:])

        has_positive = any(ind in recent_text for ind in positive_indicators)
        has_negative = any(ind in recent_text for ind in negative_indicators)

        if has_negative and not has_positive:
            return "declining"
        elif has_positive and not has_negative:
            return "improving"
        return "stable"

    # ==================== RISK EVALUATION ====================

    async def _evaluate_patient_risk(self, patient_id: str, change_info: Dict):
        """Evaluate risk for a patient with detected changes"""
        current_state = change_info['current_state']
        previous_state = change_info['previous_state']
        change_magnitude = change_info['change_magnitude']

        # Determine severity based on risk score
        if current_state.risk_score >= self.risk_threshold_critical:
            severity = AlertSeverity.CRITICAL
        elif current_state.risk_score >= self.risk_threshold_high:
            severity = AlertSeverity.HIGH
        elif current_state.risk_score >= self.risk_threshold_warning:
            severity = AlertSeverity.WARNING
        else:
            return  # No alert needed

        # Check if risk increased significantly
        risk_increase = current_state.risk_score - previous_state.risk_score
        if risk_increase < 0.1 and severity != AlertSeverity.CRITICAL:
            return  # Not significant enough

        # Find similar cases for context
        similar_cases = await self._find_similar_cases(current_state)

        # Generate alert
        alert = ClinicalAlert(
            alert_id=self._generate_alert_id(),
            alert_type=AlertType.RISK_ELEVATION,
            severity=severity,
            patient_id_hash=current_state.patient_id_hash,
            title=f"Risk elevation detected",
            description=(
                f"Patient risk score increased from {previous_state.risk_score:.2f} "
                f"to {current_state.risk_score:.2f}. "
                f"Trend: {current_state.trend}. "
                f"Change magnitude: {change_magnitude:.2f}."
            ),
            evidence=[
                {"type": "risk_score", "value": current_state.risk_score},
                {"type": "conditions", "value": current_state.key_conditions},
                {"type": "trend", "value": current_state.trend}
            ],
            recommended_actions=self._generate_risk_recommendations(current_state, similar_cases),
            similar_cases=similar_cases[:3]
        )

        self._register_alert(alert)

    async def _find_similar_cases(self, patient_state: PatientState, limit: int = 5) -> List[Dict]:
        """Find similar cases using Discovery API"""
        similar_cases = []

        try:
            # Use Discovery API to find similar cases with context
            results = DiscoveryService.discover_contextual(
                target_text=" ".join(patient_state.key_conditions),
                positive_texts=["successful treatment", "recovered", "improved"],
                negative_texts=["adverse outcome", "deteriorated"],
                limit=limit,
                clinic_id=self.clinic_id
            )

            for result in results:
                similar_cases.append({
                    "relevance_score": result.score if hasattr(result, 'score') else 0,
                    "outcome_type": self._infer_outcome(result.payload.get('text_content', '')),
                    "conditions": self._extract_conditions([result])
                })

        except Exception as e:
            logger.error(f"Error finding similar cases: {e}")

        return similar_cases

    def _infer_outcome(self, text: str) -> str:
        """Infer outcome from text"""
        text_lower = text.lower()
        if any(w in text_lower for w in ['improved', 'recovered', 'resolved']):
            return "positive"
        elif any(w in text_lower for w in ['worsened', 'deteriorated', 'failed']):
            return "negative"
        return "neutral"

    def _generate_risk_recommendations(
        self,
        state: PatientState,
        similar_cases: List[Dict]
    ) -> List[str]:
        """Generate recommendations based on risk analysis"""
        recommendations = []

        if state.risk_score >= self.risk_threshold_critical:
            recommendations.append("URGENT: Immediate clinical review required")
            recommendations.append("Consider emergency intervention")

        if state.trend == "declining":
            recommendations.append("Review current treatment plan")
            recommendations.append("Consider escalation of care")

        if state.key_conditions:
            recommendations.append(
                f"Focus areas: {', '.join(state.key_conditions[:3])}"
            )

        # Learn from similar cases
        positive_outcomes = [c for c in similar_cases if c.get('outcome_type') == 'positive']
        if positive_outcomes:
            recommendations.append(
                f"Similar cases with positive outcomes: {len(positive_outcomes)}/{len(similar_cases)}"
            )

        return recommendations

    # ==================== FOLLOW-UP COMPLIANCE ====================

    async def _check_followup_compliance(self):
        """Check for patients with overdue follow-ups"""
        try:
            cutoff = datetime.now() - timedelta(days=self.followup_overdue_days)

            for patient_id, state in self.patient_states.items():
                if state.last_visit < cutoff:
                    days_overdue = (datetime.now() - state.last_visit).days

                    # Severity based on risk and time overdue
                    if state.risk_score > self.risk_threshold_high or days_overdue > 30:
                        severity = AlertSeverity.HIGH
                    elif state.risk_score > self.risk_threshold_warning or days_overdue > 21:
                        severity = AlertSeverity.WARNING
                    else:
                        severity = AlertSeverity.INFO

                    alert = ClinicalAlert(
                        alert_id=self._generate_alert_id(),
                        alert_type=AlertType.MISSED_FOLLOWUP,
                        severity=severity,
                        patient_id_hash=state.patient_id_hash,
                        title=f"Follow-up overdue ({days_overdue} days)",
                        description=(
                            f"Patient has not been seen for {days_overdue} days. "
                            f"Current risk score: {state.risk_score:.2f}. "
                            f"Known conditions: {', '.join(state.key_conditions[:3]) or 'None recorded'}."
                        ),
                        evidence=[
                            {"type": "days_overdue", "value": days_overdue},
                            {"type": "last_visit", "value": state.last_visit.isoformat()},
                            {"type": "risk_score", "value": state.risk_score}
                        ],
                        recommended_actions=[
                            "Schedule follow-up appointment",
                            "Contact patient for wellness check",
                            "Review medication compliance"
                        ]
                    )

                    self._register_alert(alert)

        except Exception as e:
            logger.error(f"Error checking follow-up compliance: {e}", exc_info=True)

    # ==================== ANOMALY DETECTION ====================

    async def _detect_anomalies(self):
        """Detect anomalies in recent clinical data"""
        try:
            # Get recent records
            cutoff = datetime.now() - timedelta(hours=24)
            cutoff_timestamp = cutoff.timestamp()

            recent_records, _ = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="clinic_id",
                        match=models.MatchValue(value=self.clinic_id)
                    ),
                    models.FieldCondition(
                        key="timestamp",
                        range=models.Range(gte=cutoff_timestamp)
                    )
                ]),
                limit=100,
                with_payload=True,
                with_vectors=True
            )

            if len(recent_records) < 5:
                return

            # Calculate centroid of normal records
            embeddings = []
            for record in recent_records:
                if hasattr(record, 'vector') and isinstance(record.vector, dict):
                    if 'dense_text' in record.vector:
                        embeddings.append(record.vector['dense_text'])

            if len(embeddings) < 5:
                return

            centroid = [sum(e[i] for e in embeddings) / len(embeddings) for i in range(len(embeddings[0]))]

            # Find outliers (distance from centroid)
            distances = []
            for i, embedding in enumerate(embeddings):
                dist = sum((a - b) ** 2 for a, b in zip(embedding, centroid)) ** 0.5
                distances.append((dist, recent_records[i]))

            # Statistical outlier detection
            mean_dist = sum(d[0] for d in distances) / len(distances)
            std_dist = (sum((d[0] - mean_dist) ** 2 for d in distances) / len(distances)) ** 0.5

            for dist, record in distances:
                if std_dist > 0 and (dist - mean_dist) / std_dist > 2.5:
                    # This is an anomaly
                    patient_id = record.payload.get('patient_id', 'unknown')

                    alert = ClinicalAlert(
                        alert_id=self._generate_alert_id(),
                        alert_type=AlertType.ANOMALY_DETECTED,
                        severity=AlertSeverity.WARNING,
                        patient_id_hash=self._hash_id(patient_id),
                        title="Unusual clinical record detected",
                        description=(
                            f"This record is statistically unusual compared to recent clinic data. "
                            f"Distance from normal: {((dist - mean_dist) / std_dist):.1f} standard deviations."
                        ),
                        evidence=[
                            {"type": "anomaly_score", "value": (dist - mean_dist) / std_dist},
                            {"type": "record_snippet", "value": record.payload.get('text_content', '')[:200]}
                        ],
                        recommended_actions=[
                            "Review this record for accuracy",
                            "Verify patient data entry",
                            "Assess if this represents a genuine clinical concern"
                        ]
                    )

                    self._register_alert(alert)

        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}", exc_info=True)

    # ==================== SIMILAR CASE OUTCOMES ====================

    async def _analyze_similar_case_outcomes(self):
        """Analyze outcomes of similar cases to provide proactive insights"""
        try:
            # For high-risk patients, find similar cases and analyze outcomes
            high_risk_patients = [
                (pid, state) for pid, state in self.patient_states.items()
                if state.risk_score >= self.risk_threshold_warning
            ]

            for patient_id, state in high_risk_patients[:10]:  # Limit to top 10
                similar_cases = await self._find_similar_cases(state, limit=10)

                if len(similar_cases) < 3:
                    continue

                # Analyze outcome distribution
                positive = sum(1 for c in similar_cases if c.get('outcome_type') == 'positive')
                negative = sum(1 for c in similar_cases if c.get('outcome_type') == 'negative')

                if negative > positive and negative >= 3:
                    # More negative outcomes in similar cases - warn
                    alert = ClinicalAlert(
                        alert_id=self._generate_alert_id(),
                        alert_type=AlertType.SIMILAR_CASE_OUTCOME,
                        severity=AlertSeverity.WARNING,
                        patient_id_hash=state.patient_id_hash,
                        title="Similar cases show concerning outcomes",
                        description=(
                            f"Analysis of {len(similar_cases)} similar cases shows "
                            f"{negative} negative vs {positive} positive outcomes. "
                            f"Consider proactive intervention."
                        ),
                        evidence=[
                            {"type": "similar_cases", "value": len(similar_cases)},
                            {"type": "positive_outcomes", "value": positive},
                            {"type": "negative_outcomes", "value": negative}
                        ],
                        recommended_actions=[
                            "Review successful treatment approaches from similar cases",
                            "Consider early intervention",
                            "Discuss prognosis with patient"
                        ],
                        similar_cases=similar_cases[:3]
                    )

                    self._register_alert(alert)

        except Exception as e:
            logger.error(f"Error analyzing similar case outcomes: {e}", exc_info=True)

    # ==================== ALERT MANAGEMENT ====================

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        self._alert_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"ALR-{self.clinic_id[:4]}-{timestamp}-{self._alert_counter:04d}"

    def _register_alert(self, alert: ClinicalAlert):
        """Register and potentially dispatch an alert"""
        # Check for duplicate alerts
        existing = next(
            (a for a in self.alerts
             if a.patient_id_hash == alert.patient_id_hash
             and a.alert_type == alert.alert_type
             and not a.acknowledged
             and (datetime.now() - a.created_at).total_seconds() < 3600),
            None
        )

        if existing:
            logger.debug(f"Duplicate alert suppressed: {alert.alert_type.value}")
            return

        self.alerts.append(alert)
        logger.info(f"New alert: [{alert.severity.value}] {alert.title}")

        # Callback for real-time notification
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[ClinicalAlert]:
        """Get active (unacknowledged) alerts"""
        alerts = [a for a in self.alerts if not a.acknowledged]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: (
            -['info', 'warning', 'high', 'critical'].index(a.severity.value),
            a.created_at
        ), reverse=True)

    def acknowledge_alert(self, alert_id: str, doctor_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.assigned_doctor_id = doctor_id
                logger.info(f"Alert {alert_id} acknowledged by {doctor_id}")
                return True
        return False

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alerts"""
        active = self.get_active_alerts()
        return {
            "total_active": len(active),
            "critical": len([a for a in active if a.severity == AlertSeverity.CRITICAL]),
            "high": len([a for a in active if a.severity == AlertSeverity.HIGH]),
            "warning": len([a for a in active if a.severity == AlertSeverity.WARNING]),
            "info": len([a for a in active if a.severity == AlertSeverity.INFO]),
            "by_type": {
                t.value: len([a for a in active if a.alert_type == t])
                for t in AlertType
            }
        }

    def _hash_id(self, patient_id: str) -> str:
        """Hash patient ID for privacy"""
        return hashlib.sha256(patient_id.encode()).hexdigest()[:12]


# ==================== SYNCHRONOUS WRAPPER ====================

class VigilanceAgentSync:
    """Synchronous wrapper for VigilanceAgent for non-async contexts"""

    def __init__(self, clinic_id: str):
        self.agent = VigilanceAgent(clinic_id)

    def run_single_cycle(self):
        """Run a single monitoring cycle synchronously"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.agent._monitoring_cycle())
        finally:
            loop.close()

    def get_alerts(self) -> List[ClinicalAlert]:
        return self.agent.get_active_alerts()

    def get_summary(self) -> Dict[str, Any]:
        return self.agent.get_alert_summary()


# ==================== CLI ====================

def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Vigilance Agent CLI")
    parser.add_argument("--clinic-id", required=True, help="Clinic ID to monitor")
    parser.add_argument("--single-cycle", action="store_true", help="Run single cycle")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")

    args = parser.parse_args()

    def alert_handler(alert: ClinicalAlert):
        print(f"\n{'='*60}")
        print(f"[{alert.severity.value.upper()}] {alert.title}")
        print(f"Patient: {alert.patient_id_hash}")
        print(f"{alert.description}")
        print(f"Actions: {', '.join(alert.recommended_actions[:2])}")
        print(f"{'='*60}\n")

    if args.single_cycle:
        agent = VigilanceAgentSync(args.clinic_id)
        agent.run_single_cycle()
        print(f"\nAlert Summary: {agent.get_summary()}")
    else:
        agent = VigilanceAgent(
            clinic_id=args.clinic_id,
            check_interval_seconds=args.interval,
            alert_callback=alert_handler
        )
        print(f"Starting vigilance monitoring for clinic {args.clinic_id}...")
        asyncio.run(agent.start_monitoring())


if __name__ == "__main__":
    main()
