"""
Analytics Service - Qdrant Only

Provides metrics and analytics using Qdrant's feedback_analytics collection.
No SQL dependencies.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from qdrant_client import models
from medisync.core_agents.database_agent import client

logger = logging.getLogger(__name__)

FEEDBACK_COLLECTION = "feedback_analytics"


class AnalyticsService:
    """Service for metrics and analytics using Qdrant"""

    @staticmethod
    def get_search_metrics(
        days: int = 7,
        clinic_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get search performance metrics from Qdrant

        Args:
            days: Number of days to analyze
            clinic_id: Filter by clinic (None = all)

        Returns:
            Dictionary of metrics
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

            # Get queries from Qdrant
            results, _ = client.scroll(
                collection_name=FEEDBACK_COLLECTION,
                scroll_filter=models.Filter(must=filter_conditions),
                limit=1000,
                with_payload=True
            )

            # Calculate metrics
            total_queries = len(results)

            queries_with_clicks = sum(
                1 for r in results
                if any(i.get("interaction_type") in ["click", "use"]
                       for i in r.payload.get("interactions", []))
            )

            ctr = (queries_with_clicks / total_queries * 100) if total_queries > 0 else 0

            total_results = sum(r.payload.get("result_count", 0) for r in results)
            avg_results = total_results / total_queries if total_queries > 0 else 0

            zero_results = sum(1 for r in results if r.payload.get("result_count", 0) == 0)
            zero_result_rate = (zero_results / total_queries * 100) if total_queries > 0 else 0

            # Query type distribution
            query_types = {}
            for r in results:
                qt = r.payload.get("query_type", "unknown")
                query_types[qt] = query_types.get(qt, 0) + 1

            # Query intent distribution
            query_intents = {}
            for r in results:
                qi = r.payload.get("query_intent") or "unknown"
                query_intents[qi] = query_intents.get(qi, 0) + 1

            return {
                'period_days': days,
                'total_queries': total_queries,
                'queries_with_clicks': queries_with_clicks,
                'click_through_rate': round(ctr, 2),
                'avg_results_per_query': round(avg_results, 2),
                'zero_result_queries': zero_results,
                'zero_result_rate': round(zero_result_rate, 2),
                'query_type_distribution': query_types,
                'query_intent_distribution': query_intents
            }

        except Exception as e:
            logger.error(f"Error getting search metrics: {e}")
            return {
                'period_days': days,
                'total_queries': 0,
                'queries_with_clicks': 0,
                'click_through_rate': 0,
                'avg_results_per_query': 0,
                'zero_result_queries': 0,
                'zero_result_rate': 0,
                'query_type_distribution': {},
                'query_intent_distribution': {}
            }

    @staticmethod
    def get_ranking_metrics(
        days: int = 7,
        clinic_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get ranking quality metrics (MRR, position bias)

        Returns:
            Dictionary of metrics
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

            # Get queries
            results, _ = client.scroll(
                collection_name=FEEDBACK_COLLECTION,
                scroll_filter=models.Filter(must=filter_conditions),
                limit=1000,
                with_payload=True
            )

            # Extract clicked results with ranks
            ranks = []
            for r in results:
                for i in r.payload.get("interactions", []):
                    if i.get("interaction_type") in ["click", "use"]:
                        rank = i.get("result_rank")
                        if rank:
                            ranks.append(rank)

            if not ranks:
                return {
                    'period_days': days,
                    'total_clicks': 0,
                    'mean_reciprocal_rank': 0,
                    'average_click_position': 0,
                    'clicks_in_top_3': 0,
                    'clicks_in_top_5': 0,
                    'top_3_percentage': 0,
                    'top_5_percentage': 0
                }

            # Mean Reciprocal Rank
            reciprocal_ranks = [1.0 / r for r in ranks]
            mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

            # Average click position
            avg_position = sum(ranks) / len(ranks)

            # Position distribution
            top_3_clicks = sum(1 for r in ranks if r <= 3)
            top_5_clicks = sum(1 for r in ranks if r <= 5)

            return {
                'period_days': days,
                'total_clicks': len(ranks),
                'mean_reciprocal_rank': round(mrr, 3),
                'average_click_position': round(avg_position, 2),
                'clicks_in_top_3': top_3_clicks,
                'clicks_in_top_5': top_5_clicks,
                'top_3_percentage': round(top_3_clicks / len(ranks) * 100, 1),
                'top_5_percentage': round(top_5_clicks / len(ranks) * 100, 1)
            }

        except Exception as e:
            logger.error(f"Error getting ranking metrics: {e}")
            return {
                'period_days': days,
                'total_clicks': 0,
                'mean_reciprocal_rank': 0,
                'average_click_position': 0,
                'clicks_in_top_3': 0,
                'clicks_in_top_5': 0,
                'top_3_percentage': 0,
                'top_5_percentage': 0
            }

    @staticmethod
    def get_clinical_outcome_metrics(
        days: int = 7,
        clinic_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get clinical outcome metrics

        Returns:
            Dictionary of metrics
        """
        try:
            # Build filter for outcomes
            filter_conditions = [
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value="outcome")
                )
            ]

            if clinic_id:
                filter_conditions.append(
                    models.FieldCondition(
                        key="clinic_id",
                        match=models.MatchValue(value=clinic_id)
                    )
                )

            # Get outcomes
            results, _ = client.scroll(
                collection_name=FEEDBACK_COLLECTION,
                scroll_filter=models.Filter(must=filter_conditions),
                limit=1000,
                with_payload=True
            )

            total_outcomes = len(results)

            if total_outcomes == 0:
                return {
                    'period_days': days,
                    'total_outcomes': 0,
                    'outcome_distribution': {},
                    'helpful_rate': 0,
                    'average_confidence': 0,
                    'average_time_to_outcome_hours': 0
                }

            # Outcome type distribution
            outcome_dist = {}
            confidences = []
            times = []

            for r in results:
                ot = r.payload.get("outcome_type", "unknown")
                outcome_dist[ot] = outcome_dist.get(ot, 0) + 1

                conf = r.payload.get("confidence_level")
                if conf:
                    confidences.append(conf)

                time_to = r.payload.get("time_to_outcome_hours")
                if time_to:
                    times.append(time_to)

            # Helpful rate
            helpful_count = outcome_dist.get('helpful', 0) + outcome_dist.get('led_to_diagnosis', 0)
            helpful_rate = (helpful_count / total_outcomes * 100) if total_outcomes > 0 else 0

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            avg_time = sum(times) / len(times) if times else 0

            return {
                'period_days': days,
                'total_outcomes': total_outcomes,
                'outcome_distribution': outcome_dist,
                'helpful_rate': round(helpful_rate, 2),
                'average_confidence': round(avg_confidence, 2),
                'average_time_to_outcome_hours': round(avg_time, 2)
            }

        except Exception as e:
            logger.error(f"Error getting outcome metrics: {e}")
            return {
                'period_days': days,
                'total_outcomes': 0,
                'outcome_distribution': {},
                'helpful_rate': 0,
                'average_confidence': 0,
                'average_time_to_outcome_hours': 0
            }

    @staticmethod
    def get_model_performance(
        model_type: str = "embedder"
    ) -> Dict[str, Any]:
        """
        Get model performance metrics

        Returns:
            Dictionary of metrics
        """
        # For demo, return static metrics
        # In production, this would query actual model registry
        if model_type == "embedder":
            return {
                'model_type': 'embedder',
                'active_version': 'gemini-embedding-001',
                'status': 'active',
                'metrics': {
                    'ndcg@5': 0.85,
                    'mrr': 0.78,
                    'recall@10': 0.92
                }
            }
        else:
            return {
                'model_type': 'reranker',
                'active_version': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                'status': 'active',
                'metrics': {
                    'ndcg@5': 0.88,
                    'mrr': 0.82
                }
            }

    @staticmethod
    def get_comprehensive_dashboard(
        days: int = 7,
        clinic_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive analytics dashboard

        Returns:
            Dictionary with all metrics
        """
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'period_days': days,
            'clinic_id': clinic_id or 'all',
            'search_metrics': AnalyticsService.get_search_metrics(days, clinic_id),
            'ranking_metrics': AnalyticsService.get_ranking_metrics(days, clinic_id),
            'outcome_metrics': AnalyticsService.get_clinical_outcome_metrics(days, clinic_id),
            'embedder_performance': AnalyticsService.get_model_performance('embedder'),
            'reranker_performance': AnalyticsService.get_model_performance('reranker')
        }
