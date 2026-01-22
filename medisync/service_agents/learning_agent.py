"""
Feedback Collection Service - Qdrant Only

Tracks search behavior and user interactions using Qdrant's feedback_analytics collection.
No SQL dependencies.
"""

import hashlib
import uuid
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from qdrant_client import models
from medisync.core_agents.database_agent import client

logger = logging.getLogger(__name__)

FEEDBACK_COLLECTION = "feedback_analytics"


class FeedbackService:
    """Service for collecting and managing feedback data in Qdrant"""

    @staticmethod
    def _hash_text(text: str) -> str:
        """Hash text using SHA256 for privacy"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    @staticmethod
    def _ensure_collection():
        """Ensure feedback collection exists"""
        try:
            collections = client.get_collections().collections
            if not any(c.name == FEEDBACK_COLLECTION for c in collections):
                client.create_collection(
                    collection_name=FEEDBACK_COLLECTION,
                    vectors_config=models.VectorParams(
                        size=4,  # Minimal vector for payload-only storage
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {FEEDBACK_COLLECTION}")
        except Exception as e:
            logger.warning(f"Collection check failed: {e}")

    @staticmethod
    def record_query(
        user_id: str,
        clinic_id: str,
        query_text: str,
        query_type: str = "hybrid",
        query_intent: Optional[str] = None,
        result_count: int = 0,
        session_id: Optional[str] = None
    ) -> str:
        """
        Record a search query with hashed query text

        Returns:
            query_id: ID of the recorded query
        """
        FeedbackService._ensure_collection()

        query_id = str(uuid.uuid4())
        query_hash = FeedbackService._hash_text(query_text)

        try:
            client.upsert(
                collection_name=FEEDBACK_COLLECTION,
                points=[
                    models.PointStruct(
                        id=query_id,
                        vector=[0.0, 0.0, 0.0, 0.0],  # Dummy vector
                        payload={
                            "type": "query",
                            "user_id": user_id,
                            "clinic_id": clinic_id,
                            "query_text_hash": query_hash,
                            "query_type": query_type,
                            "query_intent": query_intent,
                            "result_count": result_count,
                            "session_id": session_id or f"session_{datetime.utcnow().timestamp()}",
                            "timestamp": datetime.utcnow().isoformat(),
                            "interactions": []
                        }
                    )
                ]
            )
            logger.info(f"Recorded query {query_id} for user {user_id}")
            return query_id

        except Exception as e:
            logger.error(f"Error recording query: {e}")
            return query_id  # Return ID anyway for demo

    @staticmethod
    def record_interaction(
        query_id: str,
        result_point_id: str,
        result_rank: int,
        result_score: float,
        interaction_type: str = "view",
        dwell_time_seconds: Optional[float] = None
    ) -> str:
        """
        Record user interaction with a search result

        Returns:
            interaction_id: ID of the recorded interaction
        """
        interaction_id = str(uuid.uuid4())

        try:
            # Get existing query record
            results = client.retrieve(
                collection_name=FEEDBACK_COLLECTION,
                ids=[query_id]
            )

            if results:
                interactions = results[0].payload.get("interactions", [])
                interactions.append({
                    "id": interaction_id,
                    "result_point_id": result_point_id,
                    "result_rank": result_rank,
                    "result_score": result_score,
                    "interaction_type": interaction_type,
                    "dwell_time_seconds": dwell_time_seconds,
                    "timestamp": datetime.utcnow().isoformat()
                })

                # Update the record
                client.set_payload(
                    collection_name=FEEDBACK_COLLECTION,
                    payload={"interactions": interactions},
                    points=[query_id]
                )

            logger.info(f"Recorded {interaction_type} interaction for query {query_id}")
            return interaction_id

        except Exception as e:
            logger.error(f"Error recording interaction: {e}")
            return interaction_id

    @staticmethod
    def record_outcome(
        query_id: str,
        patient_id: str,
        clinic_id: str,
        doctor_id: str,
        outcome_type: str,
        confidence_level: int,
        time_to_outcome_hours: Optional[float] = None
    ) -> str:
        """
        Record clinical outcome feedback

        Returns:
            outcome_id: ID of the recorded outcome
        """
        outcome_id = str(uuid.uuid4())
        patient_hash = FeedbackService._hash_text(patient_id)

        try:
            client.upsert(
                collection_name=FEEDBACK_COLLECTION,
                points=[
                    models.PointStruct(
                        id=outcome_id,
                        vector=[0.0, 0.0, 0.0, 0.0],
                        payload={
                            "type": "outcome",
                            "query_id": query_id,
                            "patient_id_hash": patient_hash,
                            "clinic_id": clinic_id,
                            "doctor_id": doctor_id,
                            "outcome_type": outcome_type,
                            "confidence_level": confidence_level,
                            "time_to_outcome_hours": time_to_outcome_hours,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                ]
            )
            logger.info(f"Recorded outcome {outcome_type} for query {query_id}")
            return outcome_id

        except Exception as e:
            logger.error(f"Error recording outcome: {e}")
            return outcome_id

    @staticmethod
    def get_query_statistics(
        clinic_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get query statistics for analytics

        Returns:
            Dictionary with statistics
        """
        try:
            # Build filter
            filter_conditions = [
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value="query")
                )
            ]

            if clinic_id:
                filter_conditions.append(
                    models.FieldCondition(
                        key="clinic_id",
                        match=models.MatchValue(value=clinic_id)
                    )
                )

            # Scroll through queries
            results, _ = client.scroll(
                collection_name=FEEDBACK_COLLECTION,
                scroll_filter=models.Filter(must=filter_conditions),
                limit=1000,
                with_payload=True
            )

            total_queries = len(results)
            queries_with_clicks = sum(
                1 for r in results
                if any(i.get("interaction_type") in ["click", "use"]
                       for i in r.payload.get("interactions", []))
            )

            total_results = sum(r.payload.get("result_count", 0) for r in results)
            avg_results = total_results / total_queries if total_queries > 0 else 0

            ctr = (queries_with_clicks / total_queries * 100) if total_queries > 0 else 0

            return {
                "total_queries": total_queries,
                "queries_with_clicks": queries_with_clicks,
                "click_through_rate": round(ctr, 2),
                "avg_results_per_query": round(avg_results, 2),
                "date_range_days": days
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                "total_queries": 0,
                "queries_with_clicks": 0,
                "click_through_rate": 0,
                "avg_results_per_query": 0,
                "date_range_days": days
            }

    @staticmethod
    def export_training_data(
        batch_name: Optional[str] = None,
        min_interactions: int = 1
    ) -> Dict[str, Any]:
        """
        Export feedback data for model training

        Returns:
            Dictionary with training data and metadata
        """
        try:
            # Get all queries with interactions
            results, _ = client.scroll(
                collection_name=FEEDBACK_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="type",
                            match=models.MatchValue(value="query")
                        )
                    ]
                ),
                limit=1000,
                with_payload=True
            )

            training_samples = []
            for r in results:
                interactions = r.payload.get("interactions", [])
                if len(interactions) >= min_interactions:
                    clicked = [i for i in interactions if i.get("interaction_type") in ["click", "use"]]
                    if clicked:
                        training_samples.append({
                            "query_id": str(r.id),
                            "query_hash": r.payload.get("query_text_hash"),
                            "positive_results": clicked,
                            "negative_results": [i for i in interactions if i not in clicked]
                        })

            batch_name = batch_name or f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            return {
                "batch_name": batch_name,
                "sample_count": len(training_samples),
                "samples": training_samples
            }

        except Exception as e:
            logger.error(f"Error exporting training data: {e}")
            return {"batch_name": batch_name, "sample_count": 0, "samples": []}
