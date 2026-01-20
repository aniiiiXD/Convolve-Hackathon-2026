"""
Feedback Collection Service for Learning Pipeline

Tracks search behavior, user interactions, and clinical outcomes
with privacy-preserving hashing of PII.
"""

import hashlib
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from medisync.models.sql_models import (
    SearchQuery,
    ResultInteraction,
    ClinicalOutcome,
    ModelTrainingBatch,
    User
)
from medisync.core.db_sql import SessionLocal

logger = logging.getLogger(__name__)


class FeedbackService:
    """Service for collecting and managing feedback data"""

    @staticmethod
    def _hash_text(text: str) -> str:
        """Hash text using SHA256 for privacy"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

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

        Args:
            user_id: User performing the search
            clinic_id: Clinic context
            query_text: Raw query text (will be hashed)
            query_type: Type of search (semantic, exact, hybrid)
            query_intent: Intent classification (diagnosis, treatment, history)
            result_count: Number of results returned
            session_id: Session identifier

        Returns:
            query_id: ID of the recorded query
        """
        db = SessionLocal()
        try:
            # Hash the query text for privacy
            query_hash = FeedbackService._hash_text(query_text)

            query = SearchQuery(
                user_id=user_id,
                clinic_id=clinic_id,
                query_text_hash=query_hash,
                query_type=query_type,
                query_intent=query_intent,
                result_count=result_count,
                session_id=session_id or f"session_{datetime.utcnow().timestamp()}"
            )

            db.add(query)
            db.commit()
            db.refresh(query)

            logger.info(f"Recorded query {query.id} for user {user_id}")
            return query.id

        except Exception as e:
            db.rollback()
            logger.error(f"Error recording query: {e}")
            raise
        finally:
            db.close()

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

        Args:
            query_id: ID of the search query
            result_point_id: Qdrant point ID of the result
            result_rank: Position in result list (1-indexed)
            result_score: Relevance score from Qdrant
            interaction_type: Type of interaction (view, click, use)
            dwell_time_seconds: Time spent viewing result

        Returns:
            interaction_id: ID of the recorded interaction
        """
        db = SessionLocal()
        try:
            interaction = ResultInteraction(
                query_id=query_id,
                result_point_id=result_point_id,
                result_rank=result_rank,
                result_score=result_score,
                interaction_type=interaction_type,
                dwell_time_seconds=dwell_time_seconds
            )

            db.add(interaction)
            db.commit()
            db.refresh(interaction)

            logger.info(f"Recorded {interaction_type} interaction for query {query_id}")
            return interaction.id

        except Exception as e:
            db.rollback()
            logger.error(f"Error recording interaction: {e}")
            raise
        finally:
            db.close()

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

        Args:
            query_id: ID of the search query
            patient_id: Patient identifier (will be hashed)
            clinic_id: Clinic context
            doctor_id: Doctor providing feedback
            outcome_type: Type of outcome (helpful, not_helpful, led_to_diagnosis)
            confidence_level: Confidence rating (1-5)
            time_to_outcome_hours: Time from query to outcome

        Returns:
            outcome_id: ID of the recorded outcome
        """
        db = SessionLocal()
        try:
            # Hash patient ID for privacy
            patient_hash = FeedbackService._hash_text(patient_id)

            outcome = ClinicalOutcome(
                query_id=query_id,
                patient_id_hash=patient_hash,
                clinic_id=clinic_id,
                doctor_id=doctor_id,
                outcome_type=outcome_type,
                confidence_level=confidence_level,
                time_to_outcome_hours=time_to_outcome_hours
            )

            db.add(outcome)
            db.commit()
            db.refresh(outcome)

            logger.info(f"Recorded outcome {outcome_type} for query {query_id}")
            return outcome.id

        except Exception as e:
            db.rollback()
            logger.error(f"Error recording outcome: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def export_training_data(
        date_range_start: datetime,
        date_range_end: datetime,
        batch_name: Optional[str] = None,
        min_interactions: int = 1
    ) -> Dict[str, Any]:
        """
        Export feedback data for model training

        Args:
            date_range_start: Start of date range
            date_range_end: End of date range
            batch_name: Name for this training batch
            min_interactions: Minimum interactions required per query

        Returns:
            Dictionary with training data and metadata
        """
        db = SessionLocal()
        try:
            # Generate batch name if not provided
            if not batch_name:
                batch_name = f"batch_{date_range_start.strftime('%Y%m%d')}_{date_range_end.strftime('%Y%m%d')}"

            # Query feedback data
            queries = db.query(SearchQuery).filter(
                and_(
                    SearchQuery.timestamp >= date_range_start,
                    SearchQuery.timestamp <= date_range_end
                )
            ).all()

            training_samples = []
            for query in queries:
                # Filter queries with sufficient interactions
                if len(query.interactions) < min_interactions:
                    continue

                # Build training sample
                clicked_results = [
                    i for i in query.interactions
                    if i.interaction_type in ['click', 'use']
                ]

                viewed_results = [
                    i for i in query.interactions
                    if i.interaction_type == 'view'
                ]

                if clicked_results:
                    sample = {
                        "query_id": query.id,
                        "query_hash": query.query_text_hash,
                        "query_type": query.query_type,
                        "query_intent": query.query_intent,
                        "positive_results": [
                            {
                                "point_id": i.result_point_id,
                                "rank": i.result_rank,
                                "score": i.result_score,
                                "dwell_time": i.dwell_time_seconds
                            }
                            for i in clicked_results
                        ],
                        "negative_results": [
                            {
                                "point_id": i.result_point_id,
                                "rank": i.result_rank,
                                "score": i.result_score
                            }
                            for i in viewed_results
                            if i.result_rank < 10 and i not in clicked_results
                        ],
                        "outcomes": [
                            {
                                "type": o.outcome_type,
                                "confidence": o.confidence_level
                            }
                            for o in query.outcomes
                        ]
                    }
                    training_samples.append(sample)

            # Create training batch record
            batch = ModelTrainingBatch(
                batch_name=batch_name,
                query_count=len(training_samples),
                date_range_start=date_range_start,
                date_range_end=date_range_end,
                training_status="exported"
            )

            db.add(batch)
            db.commit()
            db.refresh(batch)

            logger.info(
                f"Exported {len(training_samples)} training samples "
                f"in batch {batch_name}"
            )

            return {
                "batch_id": batch.id,
                "batch_name": batch_name,
                "sample_count": len(training_samples),
                "date_range": {
                    "start": date_range_start.isoformat(),
                    "end": date_range_end.isoformat()
                },
                "samples": training_samples
            }

        except Exception as e:
            db.rollback()
            logger.error(f"Error exporting training data: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def get_query_statistics(
        clinic_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get query statistics for analytics

        Args:
            clinic_id: Filter by clinic (None for all clinics)
            days: Number of days to analyze

        Returns:
            Dictionary with statistics
        """
        db = SessionLocal()
        try:
            start_date = datetime.utcnow() - timedelta(days=days)

            query_filter = [SearchQuery.timestamp >= start_date]
            if clinic_id:
                query_filter.append(SearchQuery.clinic_id == clinic_id)

            # Total queries
            total_queries = db.query(func.count(SearchQuery.id)).filter(
                and_(*query_filter)
            ).scalar()

            # Queries with interactions
            queries_with_clicks = db.query(func.count(SearchQuery.id.distinct())).join(
                ResultInteraction
            ).filter(
                and_(
                    *query_filter,
                    ResultInteraction.interaction_type.in_(['click', 'use'])
                )
            ).scalar()

            # Average results per query
            avg_results = db.query(func.avg(SearchQuery.result_count)).filter(
                and_(*query_filter)
            ).scalar()

            # Click-through rate
            ctr = (queries_with_clicks / total_queries * 100) if total_queries > 0 else 0

            return {
                "total_queries": total_queries,
                "queries_with_clicks": queries_with_clicks,
                "click_through_rate": round(ctr, 2),
                "avg_results_per_query": round(avg_results or 0, 2),
                "date_range_days": days
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def update_batch_status(
        batch_id: str,
        status: str,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Update training batch status and metrics

        Args:
            batch_id: ID of the training batch
            status: New status (training, completed, failed)
            metrics: Model performance metrics
        """
        db = SessionLocal()
        try:
            batch = db.query(ModelTrainingBatch).filter(
                ModelTrainingBatch.id == batch_id
            ).first()

            if not batch:
                raise ValueError(f"Batch {batch_id} not found")

            batch.training_status = status
            if metrics:
                batch.model_metrics = metrics

            db.commit()
            logger.info(f"Updated batch {batch_id} status to {status}")

        except Exception as e:
            db.rollback()
            logger.error(f"Error updating batch status: {e}")
            raise
        finally:
            db.close()
