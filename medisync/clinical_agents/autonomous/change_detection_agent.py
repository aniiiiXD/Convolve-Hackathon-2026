"""
Change Detection Agent

Detects and analyzes temporal changes in patient conditions
using vector similarity and semantic analysis.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

from qdrant_client import models

from medisync.core_agents.database_agent import client
from medisync.service_agents.memory_ops_agent import COLLECTION_NAME
from medisync.service_agents.encoding_agent import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class StateChange:
    """Represents a detected state change"""
    patient_id: str
    change_type: str  # "improvement", "deterioration", "new_condition", "resolved"
    description: str
    confidence: float
    previous_embedding: List[float]
    current_embedding: List[float]
    semantic_shift: float
    key_changes: List[str]
    timestamp: datetime


class ChangeDetectionAgent:
    """
    Detects temporal changes in patient states using vector analysis.

    Uses embedding drift and semantic similarity to detect:
    - Health improvements
    - Condition deterioration
    - New conditions appearing
    - Conditions resolving
    """

    def __init__(self, clinic_id: str):
        self.clinic_id = clinic_id
        self.embedder = EmbeddingService()
        self.patient_history: Dict[str, List[Dict]] = defaultdict(list)

    def detect_changes(
        self,
        patient_id: str,
        lookback_days: int = 30,
        change_threshold: float = 0.2
    ) -> List[StateChange]:
        """
        Detect changes in a patient's state over time.

        Args:
            patient_id: Patient identifier
            lookback_days: Number of days to analyze
            change_threshold: Minimum semantic shift to report

        Returns:
            List of detected state changes
        """
        changes = []

        try:
            cutoff = datetime.now() - timedelta(days=lookback_days)
            cutoff_timestamp = cutoff.timestamp()

            # Get patient records
            records, _ = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="clinic_id",
                        match=models.MatchValue(value=self.clinic_id)
                    ),
                    models.FieldCondition(
                        key="patient_id",
                        match=models.MatchValue(value=patient_id)
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

            if len(records) < 2:
                return changes

            # Sort by timestamp
            records = sorted(records, key=lambda r: r.payload.get('timestamp', 0))

            # Analyze temporal windows
            window_size = max(1, len(records) // 3)

            # Early period
            early_records = records[:window_size]
            # Recent period
            recent_records = records[-window_size:]

            # Get embeddings
            early_embedding = self._get_period_embedding(early_records)
            recent_embedding = self._get_period_embedding(recent_records)

            if not early_embedding or not recent_embedding:
                return changes

            # Calculate semantic shift
            semantic_shift = self._calculate_semantic_shift(early_embedding, recent_embedding)

            if semantic_shift < change_threshold:
                return changes

            # Determine change type
            change_type, key_changes = self._classify_change(
                early_records, recent_records, semantic_shift
            )

            change = StateChange(
                patient_id=patient_id,
                change_type=change_type,
                description=self._generate_change_description(change_type, key_changes, semantic_shift),
                confidence=min(0.9, semantic_shift + 0.3),
                previous_embedding=early_embedding,
                current_embedding=recent_embedding,
                semantic_shift=semantic_shift,
                key_changes=key_changes,
                timestamp=datetime.now()
            )
            changes.append(change)

        except Exception as e:
            logger.error(f"Error detecting changes for patient {patient_id}: {e}", exc_info=True)

        return changes

    def detect_all_changes(
        self,
        lookback_days: int = 30,
        change_threshold: float = 0.2
    ) -> Dict[str, List[StateChange]]:
        """
        Detect changes across all patients in the clinic.

        Returns:
            Dictionary mapping patient IDs to their state changes
        """
        all_changes = {}

        try:
            cutoff = datetime.now() - timedelta(days=lookback_days)
            cutoff_timestamp = cutoff.timestamp()

            # Get all recent records
            records, _ = client.scroll(
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
                limit=1000,
                with_payload=True
            )

            # Group by patient
            patient_ids = set(r.payload.get('patient_id') for r in records if r.payload.get('patient_id'))

            for patient_id in patient_ids:
                changes = self.detect_changes(patient_id, lookback_days, change_threshold)
                if changes:
                    all_changes[patient_id] = changes

        except Exception as e:
            logger.error(f"Error detecting all changes: {e}", exc_info=True)

        return all_changes

    def _get_period_embedding(self, records) -> Optional[List[float]]:
        """Get average embedding for a period"""
        embeddings = []

        for record in records:
            if hasattr(record, 'vector') and record.vector:
                if isinstance(record.vector, dict) and 'dense_text' in record.vector:
                    embeddings.append(record.vector['dense_text'])

        if not embeddings:
            # Fallback: generate from text
            all_text = " ".join(r.payload.get('text_content', '')[:300] for r in records[:5])
            if all_text.strip():
                return self.embedder.get_dense_embedding(all_text)
            return None

        # Calculate centroid
        centroid = [sum(e[i] for e in embeddings) / len(embeddings) for i in range(len(embeddings[0]))]
        return centroid

    def _calculate_semantic_shift(self, embedding_a: List[float], embedding_b: List[float]) -> float:
        """Calculate semantic shift between two embeddings"""
        # Cosine distance
        dot_product = sum(a * b for a, b in zip(embedding_a, embedding_b))
        norm_a = sum(a ** 2 for a in embedding_a) ** 0.5
        norm_b = sum(b ** 2 for b in embedding_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0

        similarity = dot_product / (norm_a * norm_b)
        return 1 - similarity

    def _classify_change(
        self,
        early_records,
        recent_records,
        semantic_shift: float
    ) -> Tuple[str, List[str]]:
        """Classify the type of change based on record content"""
        early_text = " ".join(r.payload.get('text_content', '').lower() for r in early_records)
        recent_text = " ".join(r.payload.get('text_content', '').lower() for r in recent_records)

        improvement_indicators = ['improved', 'better', 'resolved', 'stable', 'recovered', 'healing']
        deterioration_indicators = ['worsened', 'declined', 'deteriorated', 'critical', 'unstable']
        new_condition_indicators = ['diagnosed', 'new finding', 'onset', 'presenting with', 'developed']

        key_changes = []

        # Check for improvement
        early_negative = any(ind in early_text for ind in deterioration_indicators)
        recent_positive = any(ind in recent_text for ind in improvement_indicators)

        if recent_positive and (early_negative or not any(ind in early_text for ind in improvement_indicators)):
            key_changes.append("Positive indicators in recent records")
            return "improvement", key_changes

        # Check for deterioration
        recent_negative = any(ind in recent_text for ind in deterioration_indicators)
        early_stable = not any(ind in early_text for ind in deterioration_indicators)

        if recent_negative and early_stable:
            key_changes.append("Negative indicators in recent records")
            return "deterioration", key_changes

        # Check for new condition
        new_in_recent = any(ind in recent_text for ind in new_condition_indicators)
        if new_in_recent:
            key_changes.append("New condition indicators detected")
            return "new_condition", key_changes

        # Check for condition terms that appear in recent but not early
        conditions = ['diabetes', 'hypertension', 'infection', 'fracture', 'pain', 'fever']
        for condition in conditions:
            if condition in recent_text and condition not in early_text:
                key_changes.append(f"New mention of: {condition}")
                return "new_condition", key_changes
            elif condition in early_text and condition not in recent_text:
                key_changes.append(f"No longer mentioned: {condition}")
                return "resolved", key_changes

        # Default: significant semantic shift
        key_changes.append(f"Semantic shift: {semantic_shift:.2f}")
        return "change_detected", key_changes

    def _generate_change_description(
        self,
        change_type: str,
        key_changes: List[str],
        semantic_shift: float
    ) -> str:
        """Generate human-readable change description"""
        descriptions = {
            "improvement": "Patient condition shows signs of improvement",
            "deterioration": "Patient condition appears to be deteriorating",
            "new_condition": "New condition or symptom detected",
            "resolved": "Previously documented condition may have resolved",
            "change_detected": "Significant change in patient status detected"
        }

        base = descriptions.get(change_type, "Change detected in patient records")
        changes_str = "; ".join(key_changes) if key_changes else ""

        return f"{base}. {changes_str} (semantic shift: {semantic_shift:.2%})"

    def compare_patient_trajectory(
        self,
        patient_id: str,
        reference_patient_id: str
    ) -> Dict[str, Any]:
        """
        Compare one patient's trajectory with another.

        Useful for comparing current patient with successful treatment cases.
        """
        try:
            # Get both patients' records
            patient_changes = self.detect_changes(patient_id, lookback_days=60)
            reference_changes = self.detect_changes(reference_patient_id, lookback_days=60)

            if not patient_changes or not reference_changes:
                return {"comparable": False, "reason": "Insufficient data"}

            patient_embedding = patient_changes[0].current_embedding
            reference_embedding = reference_changes[0].current_embedding

            # Calculate trajectory similarity
            trajectory_similarity = 1 - self._calculate_semantic_shift(
                patient_embedding, reference_embedding
            )

            # Compare change types
            patient_type = patient_changes[0].change_type
            reference_type = reference_changes[0].change_type

            return {
                "comparable": True,
                "trajectory_similarity": trajectory_similarity,
                "patient_trend": patient_type,
                "reference_trend": reference_type,
                "trends_match": patient_type == reference_type,
                "recommendation": self._generate_trajectory_recommendation(
                    patient_type, reference_type, trajectory_similarity
                )
            }

        except Exception as e:
            logger.error(f"Error comparing trajectories: {e}", exc_info=True)
            return {"comparable": False, "reason": str(e)}

    def _generate_trajectory_recommendation(
        self,
        patient_trend: str,
        reference_trend: str,
        similarity: float
    ) -> str:
        """Generate recommendation based on trajectory comparison"""
        if reference_trend == "improvement" and patient_trend != "improvement":
            if similarity > 0.7:
                return "Consider similar treatment approach - high trajectory similarity with successful case"
            else:
                return "Review reference case for potential treatment insights"

        if patient_trend == "improvement" and reference_trend == "improvement":
            return "Trajectory aligns with successful case - continue current approach"

        if patient_trend == "deterioration":
            return "Trajectory diverging from successful reference - consider intervention"

        return "Monitor patient progress and compare with reference outcomes"


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Change Detection Agent CLI")
    parser.add_argument("--clinic-id", required=True, help="Clinic ID")
    parser.add_argument("--patient-id", help="Specific patient ID")
    parser.add_argument("--lookback", type=int, default=30, help="Lookback days")
    parser.add_argument("--threshold", type=float, default=0.2, help="Change threshold")

    args = parser.parse_args()

    agent = ChangeDetectionAgent(args.clinic_id)

    if args.patient_id:
        changes = agent.detect_changes(
            args.patient_id,
            lookback_days=args.lookback,
            change_threshold=args.threshold
        )
        print(f"\nChanges for patient {args.patient_id}:")
        for change in changes:
            print(f"  [{change.change_type}] {change.description}")
            print(f"  Confidence: {change.confidence:.2f}")
    else:
        all_changes = agent.detect_all_changes(
            lookback_days=args.lookback,
            change_threshold=args.threshold
        )
        print(f"\nDetected changes in {len(all_changes)} patients:")
        for patient_id, changes in all_changes.items():
            for change in changes:
                print(f"  {patient_id}: [{change.change_type}] - {change.semantic_shift:.2%} shift")


if __name__ == "__main__":
    main()
