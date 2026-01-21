"""
Analytics Service

Provides metrics and analytics for:
- Search performance (CTR, MRR, latency)
- Model performance (nDCG, precision, recall)
- User engagement
- System health
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import Counter

from sqlalchemy import func, and_
from sqlalchemy.orm import Session

from medisync.model_agents.data_models import SearchQuery, ResultInteraction, ClinicalOutcome
from medisync.core_agents.records_agent import SessionLocal
from medisync.model_agents.registry_agent import get_registry, ModelType

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service for metrics and analytics"""

    @staticmethod
    def get_search_metrics(
        days: int = 7,
        clinic_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get search performance metrics

        Args:
            days: Number of days to analyze
            clinic_id: Filter by clinic (None = all)

        Returns:
            Dictionary of metrics
        """
        db = SessionLocal()
        try:
            start_date = datetime.utcnow() - timedelta(days=days)

            # Build filter
            filters = [SearchQuery.timestamp >= start_date]
            if clinic_id:
                filters.append(SearchQuery.clinic_id == clinic_id)

            # Total queries
            total_queries = db.query(func.count(SearchQuery.id)).filter(
                and_(*filters)
            ).scalar()

            # Queries with clicks
            queries_with_clicks = db.query(func.count(SearchQuery.id.distinct())).join(
                ResultInteraction
            ).filter(
                and_(
                    *filters,
                    ResultInteraction.interaction_type.in_(['click', 'use'])
                )
            ).scalar()

            # Click-through rate
            ctr = (queries_with_clicks / total_queries * 100) if total_queries > 0 else 0

            # Average results per query
            avg_results = db.query(func.avg(SearchQuery.result_count)).filter(
                and_(*filters)
            ).scalar()

            # Zero-result queries
            zero_results = db.query(func.count(SearchQuery.id)).filter(
                and_(
                    *filters,
                    SearchQuery.result_count == 0
                )
            ).scalar()

            zero_result_rate = (zero_results / total_queries * 100) if total_queries > 0 else 0

            # Query type distribution
            query_types = db.query(
                SearchQuery.query_type,
                func.count(SearchQuery.id)
            ).filter(
                and_(*filters)
            ).group_by(SearchQuery.query_type).all()

            query_type_dist = {qt: count for qt, count in query_types}

            # Query intent distribution
            query_intents = db.query(
                SearchQuery.query_intent,
                func.count(SearchQuery.id)
            ).filter(
                and_(*filters)
            ).group_by(SearchQuery.query_intent).all()

            query_intent_dist = {qi or 'unknown': count for qi, count in query_intents}

            return {
                'period_days': days,
                'total_queries': total_queries,
                'queries_with_clicks': queries_with_clicks,
                'click_through_rate': round(ctr, 2),
                'avg_results_per_query': round(avg_results or 0, 2),
                'zero_result_queries': zero_results,
                'zero_result_rate': round(zero_result_rate, 2),
                'query_type_distribution': query_type_dist,
                'query_intent_distribution': query_intent_dist
            }

        except Exception as e:
            logger.error(f"Error getting search metrics: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def get_ranking_metrics(
        days: int = 7,
        clinic_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get ranking quality metrics (MRR, position bias)

        Args:
            days: Number of days to analyze
            clinic_id: Filter by clinic

        Returns:
            Dictionary of metrics
        """
        db = SessionLocal()
        try:
            start_date = datetime.utcnow() - timedelta(days=days)

            # Build filter
            filters = [SearchQuery.timestamp >= start_date]
            if clinic_id:
                filters.append(SearchQuery.clinic_id == clinic_id)

            # Get clicked results with ranks
            clicked_results = db.query(ResultInteraction.result_rank).join(
                SearchQuery
            ).filter(
                and_(
                    *filters,
                    ResultInteraction.interaction_type.in_(['click', 'use'])
                )
            ).all()

            ranks = [r[0] for r in clicked_results if r[0] is not None]

            if not ranks:
                return {
                    'period_days': days,
                    'total_clicks': 0,
                    'mean_reciprocal_rank': 0,
                    'average_click_position': 0,
                    'clicks_in_top_3': 0,
                    'clicks_in_top_5': 0
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
            raise
        finally:
            db.close()

    @staticmethod
    def get_clinical_outcome_metrics(
        days: int = 7,
        clinic_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get clinical outcome metrics

        Args:
            days: Number of days to analyze
            clinic_id: Filter by clinic

        Returns:
            Dictionary of metrics
        """
        db = SessionLocal()
        try:
            start_date = datetime.utcnow() - timedelta(days=days)

            # Build filter
            filters = [ClinicalOutcome.timestamp >= start_date]
            if clinic_id:
                filters.append(ClinicalOutcome.clinic_id == clinic_id)

            # Total outcomes
            total_outcomes = db.query(func.count(ClinicalOutcome.id)).filter(
                and_(*filters)
            ).scalar()

            # Outcome type distribution
            outcome_types = db.query(
                ClinicalOutcome.outcome_type,
                func.count(ClinicalOutcome.id)
            ).filter(
                and_(*filters)
            ).group_by(ClinicalOutcome.outcome_type).all()

            outcome_dist = {ot: count for ot, count in outcome_types}

            # Average confidence
            avg_confidence = db.query(func.avg(ClinicalOutcome.confidence_level)).filter(
                and_(*filters)
            ).scalar()

            # Average time to outcome
            avg_time = db.query(func.avg(ClinicalOutcome.time_to_outcome_hours)).filter(
                and_(
                    *filters,
                    ClinicalOutcome.time_to_outcome_hours.isnot(None)
                )
            ).scalar()

            # Helpful rate
            helpful_count = outcome_dist.get('helpful', 0) + outcome_dist.get('led_to_diagnosis', 0)
            helpful_rate = (helpful_count / total_outcomes * 100) if total_outcomes > 0 else 0

            return {
                'period_days': days,
                'total_outcomes': total_outcomes,
                'outcome_distribution': outcome_dist,
                'helpful_rate': round(helpful_rate, 2),
                'average_confidence': round(avg_confidence or 0, 2),
                'average_time_to_outcome_hours': round(avg_time or 0, 2)
            }

        except Exception as e:
            logger.error(f"Error getting outcome metrics: {e}")
            raise
        finally:
            db.close()

    @staticmethod
    def get_model_performance(
        model_type: str = "embedder"
    ) -> Dict[str, Any]:
        """
        Get model performance metrics from registry

        Args:
            model_type: Type of model (embedder or reranker)

        Returns:
            Dictionary of metrics
        """
        try:
            registry = get_registry()

            model_type_enum = ModelType.EMBEDDER if model_type == "embedder" else ModelType.RERANKER

            # Get active model
            active_model = registry.get_model(model_type=model_type_enum)

            if not active_model:
                return {
                    'model_type': model_type,
                    'status': 'no_active_model'
                }

            # Get all models for comparison
            all_models = registry.list_models(model_type=model_type_enum)

            # Calculate improvement over time
            if len(all_models) > 1:
                previous_model = all_models[1]  # Second newest
                metrics_comparison = {}

                for metric_name in ['ndcg@5', 'mrr', 'recall@10']:
                    current = active_model['metrics'].get(metric_name, 0)
                    previous = previous_model['metrics'].get(metric_name, 0)

                    if previous > 0:
                        improvement = ((current - previous) / previous) * 100
                        metrics_comparison[metric_name] = {
                            'current': round(current, 3),
                            'previous': round(previous, 3),
                            'improvement_percent': round(improvement, 2)
                        }

            else:
                metrics_comparison = None

            return {
                'model_type': model_type,
                'active_version': active_model['version'],
                'status': active_model['status'],
                'metrics': active_model['metrics'],
                'training_config': active_model['training_config'],
                'created_at': active_model['created_at'],
                'metrics_comparison': metrics_comparison
            }

        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            raise

    @staticmethod
    def get_comprehensive_dashboard(
        days: int = 7,
        clinic_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive analytics dashboard

        Args:
            days: Number of days to analyze
            clinic_id: Filter by clinic

        Returns:
            Dictionary with all metrics
        """
        try:
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

        except Exception as e:
            logger.error(f"Error generating dashboard: {e}")
            raise


def main():
    """CLI entry point for analytics"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Analytics service")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to analyze"
    )
    parser.add_argument(
        "--clinic-id",
        type=str,
        help="Filter by clinic ID"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Get comprehensive dashboard"
    )

    args = parser.parse_args()

    if args.dashboard:
        result = AnalyticsService.get_comprehensive_dashboard(
            days=args.days,
            clinic_id=args.clinic_id
        )
    else:
        result = {
            'search': AnalyticsService.get_search_metrics(args.days, args.clinic_id),
            'ranking': AnalyticsService.get_ranking_metrics(args.days, args.clinic_id),
            'outcomes': AnalyticsService.get_clinical_outcome_metrics(args.days, args.clinic_id)
        }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
