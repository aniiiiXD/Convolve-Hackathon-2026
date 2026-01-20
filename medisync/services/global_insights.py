"""
Global Insights Service

Provides access to anonymized cross-clinic medical insights
with strict access control and privacy safeguards.
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import models

from medisync.core.database import client
from medisync.services.qdrant_ops import GLOBAL_INSIGHTS_COLLECTION
from medisync.services.embedding import EmbeddingService
from medisync.services.auth import User

logger = logging.getLogger(__name__)


class GlobalInsightsService:
    """Service for querying global medical insights"""

    # Authorized roles
    AUTHORIZED_ROLES = {'DOCTOR', 'SYSTEM'}

    def __init__(self):
        """Initialize global insights service"""
        self.embedder = EmbeddingService()

    @staticmethod
    def check_authorization(user: User):
        """
        Check if user is authorized to query global insights

        Args:
            user: User requesting access

        Raises:
            PermissionError: If user is not authorized
        """
        if user.role not in GlobalInsightsService.AUTHORIZED_ROLES:
            raise PermissionError(
                f"Access denied: Only {GlobalInsightsService.AUTHORIZED_ROLES} "
                "can query global insights"
            )

    def query_insights(
        self,
        query: str,
        user: User,
        limit: int = 5,
        min_sample_size: int = 20,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Query global medical insights

        Args:
            query: Search query
            user: User making the request
            limit: Maximum number of results
            min_sample_size: Minimum sample size filter
            score_threshold: Minimum similarity score

        Returns:
            List of matching insights

        Raises:
            PermissionError: If user is not authorized
        """
        # Check authorization
        self.check_authorization(user)

        try:
            # Get embeddings
            dense_vec = self.embedder.get_dense_embedding(query)
            sparse_vec = self.embedder.get_sparse_embedding(query)

            # Query Qdrant
            results = client.query_points(
                collection_name=GLOBAL_INSIGHTS_COLLECTION,
                prefetch=[
                    models.Prefetch(
                        query=dense_vec,
                        using="insight_embedding",
                        limit=limit * 2,
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="sample_size",
                                    range=models.Range(gte=min_sample_size)
                                )
                            ]
                        )
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_vec.indices,
                            values=sparse_vec.values
                        ),
                        using="sparse_keywords",
                        limit=limit * 2,
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="sample_size",
                                    range=models.Range(gte=min_sample_size)
                                )
                            ]
                        )
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                score_threshold=score_threshold
            )

            # Format results
            insights = []
            for point in results.points:
                insight = self._format_insight(point)
                insights.append(insight)

            logger.info(
                f"User {user.user_id} queried global insights: "
                f"'{query}' → {len(insights)} results"
            )

            return insights

        except Exception as e:
            logger.error(f"Error querying global insights: {e}", exc_info=True)
            raise

    def get_insight_by_id(
        self,
        insight_id: str,
        user: User
    ) -> Optional[Dict[str, Any]]:
        """
        Get specific insight by ID

        Args:
            insight_id: Insight ID
            user: User making the request

        Returns:
            Insight or None

        Raises:
            PermissionError: If user is not authorized
        """
        # Check authorization
        self.check_authorization(user)

        try:
            results = client.retrieve(
                collection_name=GLOBAL_INSIGHTS_COLLECTION,
                ids=[insight_id],
                with_payload=True
            )

            if not results:
                return None

            return self._format_insight(results[0])

        except Exception as e:
            logger.error(f"Error retrieving insight {insight_id}: {e}")
            return None

    def search_by_condition(
        self,
        condition: str,
        user: User,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search insights by medical condition

        Args:
            condition: Medical condition
            user: User making the request
            limit: Maximum number of results

        Returns:
            List of matching insights

        Raises:
            PermissionError: If user is not authorized
        """
        # Check authorization
        self.check_authorization(user)

        try:
            # Query by condition field
            results = client.scroll(
                collection_name=GLOBAL_INSIGHTS_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="condition",
                            match=models.MatchText(text=condition)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            insights = [self._format_insight(point) for point in results[0]]

            logger.info(
                f"User {user.user_id} searched by condition '{condition}': "
                f"{len(insights)} results"
            )

            return insights

        except Exception as e:
            logger.error(f"Error searching by condition: {e}")
            return []

    def search_by_treatment(
        self,
        treatment: str,
        user: User,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search insights by treatment

        Args:
            treatment: Treatment/procedure
            user: User making the request
            limit: Maximum number of results

        Returns:
            List of matching insights

        Raises:
            PermissionError: If user is not authorized
        """
        # Check authorization
        self.check_authorization(user)

        try:
            results = client.scroll(
                collection_name=GLOBAL_INSIGHTS_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="treatment",
                            match=models.MatchText(text=treatment)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            insights = [self._format_insight(point) for point in results[0]]

            logger.info(
                f"User {user.user_id} searched by treatment '{treatment}': "
                f"{len(insights)} results"
            )

            return insights

        except Exception as e:
            logger.error(f"Error searching by treatment: {e}")
            return []

    def get_statistics(self, user: User) -> Dict[str, Any]:
        """
        Get global insights statistics

        Args:
            user: User making the request

        Returns:
            Statistics dictionary

        Raises:
            PermissionError: If user is not authorized
        """
        # Check authorization
        self.check_authorization(user)

        try:
            # Get collection info
            collection_info = client.get_collection(GLOBAL_INSIGHTS_COLLECTION)

            # Get sample statistics
            results = client.scroll(
                collection_name=GLOBAL_INSIGHTS_COLLECTION,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )

            if not results[0]:
                return {
                    'total_insights': 0,
                    'total_samples': 0,
                    'avg_sample_size': 0
                }

            insights = results[0]

            total_samples = sum(
                point.payload.get('sample_size', 0) for point in insights
            )

            avg_sample_size = total_samples / len(insights) if insights else 0

            # Get unique conditions and treatments
            conditions = set(point.payload.get('condition', '') for point in insights)
            treatments = set(point.payload.get('treatment', '') for point in insights)

            stats = {
                'total_insights': collection_info.points_count,
                'total_samples': total_samples,
                'avg_sample_size': round(avg_sample_size, 1),
                'unique_conditions': len(conditions),
                'unique_treatments': len(treatments),
                'last_updated': collection_info.config
            }

            logger.info(f"User {user.user_id} retrieved global insights statistics")

            return stats

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    @staticmethod
    def _format_insight(point) -> Dict[str, Any]:
        """
        Format Qdrant point as insight

        Args:
            point: Qdrant point

        Returns:
            Formatted insight
        """
        payload = point.payload

        # Base insight
        insight = {
            'insight_id': point.id,
            'insight_type': payload.get('insight_type', 'treatment_outcome'),
            'condition': payload.get('condition', ''),
            'treatment': payload.get('treatment', ''),
            'description': payload.get('description', ''),
            'sample_size': payload.get('sample_size', 0),
            'clinic_count': payload.get('clinic_count', 0),
            'confidence': payload.get('confidence', 0),
            'created_at': payload.get('created_at', ''),
        }

        # Add statistics if present
        statistics = payload.get('statistics', {})
        if statistics:
            insight['statistics'] = statistics

        # Add score if available
        if hasattr(point, 'score'):
            insight['relevance_score'] = round(point.score, 3)

        return insight


def main():
    """CLI entry point for testing"""
    import argparse
    from medisync.services.auth import User

    parser = argparse.ArgumentParser(description="Query global insights")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query"
    )
    parser.add_argument(
        "--user-role",
        type=str,
        default="DOCTOR",
        help="User role (DOCTOR or SYSTEM)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results"
    )

    args = parser.parse_args()

    # Create test user
    test_user = User(
        id="test_user",
        username="test_doctor",
        role=args.user_role,
        clinic_id="test_clinic"
    )

    # Query insights
    service = GlobalInsightsService()
    insights = service.query_insights(
        query=args.query,
        user=test_user,
        limit=args.limit
    )

    print(f"\nFound {len(insights)} insights:\n")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight['condition']} → {insight['treatment']}")
        print(f"   Sample size: {insight['sample_size']}, Confidence: {insight['confidence']}")
        print(f"   {insight['description']}\n")


if __name__ == "__main__":
    main()
