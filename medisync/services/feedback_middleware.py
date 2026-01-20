"""
Feedback Middleware for Transparent Feedback Collection

Wraps agent methods to automatically log search queries, interactions,
and results for the learning pipeline.
"""

import logging
import time
import uuid
from functools import wraps
from typing import Any, Callable, Optional, List
from datetime import datetime

from medisync.services.feedback_service import FeedbackService

logger = logging.getLogger(__name__)


class FeedbackMiddleware:
    """Middleware for tracking feedback data"""

    def __init__(self, enabled: bool = True):
        """
        Initialize feedback middleware

        Args:
            enabled: Whether to collect feedback (can be disabled for testing)
        """
        self.enabled = enabled
        self.session_id = str(uuid.uuid4())
        self.current_query_id: Optional[str] = None
        self.query_start_time: Optional[float] = None

    def track_search(
        self,
        query_type: str = "hybrid",
        intent_classifier: Optional[Callable[[str], str]] = None
    ):
        """
        Decorator to track search queries

        Args:
            query_type: Type of search (semantic, exact, hybrid)
            intent_classifier: Optional function to classify query intent

        Usage:
            @feedback_middleware.track_search(query_type="hybrid")
            def search_clinic(self, query: str, limit: int = 5):
                ...
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(agent_self, query: str, *args, **kwargs):
                # Execute the search
                self.query_start_time = time.time()
                results = func(agent_self, query, *args, **kwargs)

                # Track feedback if enabled
                if self.enabled:
                    try:
                        # Classify intent if classifier provided
                        intent = None
                        if intent_classifier:
                            intent = intent_classifier(query)
                        else:
                            intent = self._infer_intent(query)

                        # Get result count
                        result_count = len(results) if hasattr(results, '__len__') else 0

                        # Record the query
                        self.current_query_id = FeedbackService.record_query(
                            user_id=agent_self.user.id,
                            clinic_id=agent_self.clinic_id,
                            query_text=query,
                            query_type=query_type,
                            query_intent=intent,
                            result_count=result_count,
                            session_id=self.session_id
                        )

                        logger.info(
                            f"Tracked search query: {self.current_query_id} "
                            f"(type={query_type}, intent={intent}, results={result_count})"
                        )

                    except Exception as e:
                        logger.error(f"Error tracking search: {e}", exc_info=True)

                return results

            return wrapper
        return decorator

    def record_result_interaction(
        self,
        result_point_id: str,
        result_rank: int,
        result_score: float,
        interaction_type: str = "view",
        dwell_time_seconds: Optional[float] = None
    ):
        """
        Record interaction with a search result

        Args:
            result_point_id: Qdrant point ID
            result_rank: Position in result list
            result_score: Relevance score
            interaction_type: Type of interaction (view, click, use)
            dwell_time_seconds: Time spent on result
        """
        if not self.enabled or not self.current_query_id:
            return

        try:
            FeedbackService.record_interaction(
                query_id=self.current_query_id,
                result_point_id=result_point_id,
                result_rank=result_rank,
                result_score=result_score,
                interaction_type=interaction_type,
                dwell_time_seconds=dwell_time_seconds
            )

            logger.debug(
                f"Recorded {interaction_type} interaction: "
                f"result_rank={result_rank}, score={result_score:.3f}"
            )

        except Exception as e:
            logger.error(f"Error recording interaction: {e}")

    def record_clinical_outcome(
        self,
        patient_id: str,
        doctor_id: str,
        outcome_type: str,
        confidence_level: int,
        time_to_outcome_hours: Optional[float] = None
    ):
        """
        Record clinical outcome feedback

        Args:
            patient_id: Patient identifier (will be hashed)
            doctor_id: Doctor providing feedback
            outcome_type: Type of outcome (helpful, not_helpful, led_to_diagnosis)
            confidence_level: Confidence rating (1-5)
            time_to_outcome_hours: Time from query to outcome
        """
        if not self.enabled or not self.current_query_id:
            return

        try:
            # Calculate time to outcome if not provided
            if time_to_outcome_hours is None and self.query_start_time:
                time_to_outcome_hours = (time.time() - self.query_start_time) / 3600

            FeedbackService.record_outcome(
                query_id=self.current_query_id,
                patient_id=patient_id,
                clinic_id="",  # Will be set from query record
                doctor_id=doctor_id,
                outcome_type=outcome_type,
                confidence_level=confidence_level,
                time_to_outcome_hours=time_to_outcome_hours
            )

            logger.info(
                f"Recorded outcome: {outcome_type} "
                f"(confidence={confidence_level}, time={time_to_outcome_hours:.2f}h)"
            )

        except Exception as e:
            logger.error(f"Error recording outcome: {e}")

    def track_result_views(self, results: List[Any], auto_log: bool = True):
        """
        Track that results were viewed

        Args:
            results: List of search results (Qdrant points)
            auto_log: Automatically log view interactions
        """
        if not self.enabled or not auto_log:
            return results

        for rank, result in enumerate(results, start=1):
            try:
                point_id = result.id if hasattr(result, 'id') else str(result)
                score = result.score if hasattr(result, 'score') else 0.0

                self.record_result_interaction(
                    result_point_id=point_id,
                    result_rank=rank,
                    result_score=score,
                    interaction_type="view"
                )
            except Exception as e:
                logger.debug(f"Error tracking view for result {rank}: {e}")

        return results

    def reset_session(self):
        """Reset session for new conversation"""
        self.session_id = str(uuid.uuid4())
        self.current_query_id = None
        self.query_start_time = None

    @staticmethod
    def _infer_intent(query: str) -> str:
        """
        Simple intent classification based on keywords

        Args:
            query: Search query text

        Returns:
            Intent category (diagnosis, treatment, history, general)
        """
        query_lower = query.lower()

        # Diagnosis keywords
        if any(word in query_lower for word in [
            'diagnos', 'symptom', 'pain', 'fever', 'cough',
            'what is', 'suffering from', 'complaining of'
        ]):
            return 'diagnosis'

        # Treatment keywords
        if any(word in query_lower for word in [
            'treat', 'medication', 'prescription', 'therapy',
            'procedure', 'surgery', 'intervention'
        ]):
            return 'treatment'

        # History keywords
        if any(word in query_lower for word in [
            'history', 'previous', 'past', 'prior',
            'before', 'earlier', 'last time'
        ]):
            return 'history'

        return 'general'


# Global middleware instance for convenience
_global_middleware = FeedbackMiddleware()


def get_middleware() -> FeedbackMiddleware:
    """Get the global feedback middleware instance"""
    return _global_middleware


def enable_feedback(enabled: bool = True):
    """Enable or disable feedback collection globally"""
    _global_middleware.enabled = enabled
