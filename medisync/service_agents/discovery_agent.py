from qdrant_client import models
from medisync.core_agents.database_agent import client
from medisync.service_agents.memory_ops_agent import COLLECTION_NAME
from medisync.service_agents.encoding_agent import EmbeddingService
from typing import List, Optional

# Module-level embedder instance
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingService()
    return _embedder

class DiscoveryService:
    @staticmethod
    def discover_contextual(
        target_text: str,
        positive_texts: List[str],
        negative_texts: List[str],
        limit: int = 5,
        clinic_id: Optional[str] = None
    ):
        """
        Uses Qdrant's Discovery API to find points "close to" the target
        but constrained by the context (positive/negative examples).
        """
        embedder = _get_embedder()

        # Embed target
        target_vec = embedder.get_dense_embedding(target_text)

        # Filter
        filter_condition = None
        if clinic_id:
            filter_condition = models.Filter(
                must=[models.FieldCondition(key="clinic_id", match=models.MatchValue(value=clinic_id))]
            )

        # Build context pairs using ContextPair (for Discovery API)
        context = []
        if negative_texts:
            for p, n in zip(positive_texts, negative_texts):
                context.append(models.ContextPair(
                    positive=embedder.get_dense_embedding(p),
                    negative=embedder.get_dense_embedding(n)
                ))
        else:
            # Need at least one negative for context pair
            # Use a zero vector as neutral negative
            neutral_neg = [0.0] * 768  # Gemini embedding dimension
            for p in positive_texts:
                context.append(models.ContextPair(
                    positive=embedder.get_dense_embedding(p),
                    negative=neutral_neg
                ))

        # Use query_points with DiscoverQuery (modern Qdrant API)
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.DiscoverQuery(
                discover=models.DiscoverInput(
                    target=target_vec,
                    context=context
                )
            ),
            using="dense_text",
            limit=limit,
            query_filter=filter_condition,
            with_payload=True
        ).points

    @staticmethod
    def recommend_similar(
        positive_ids: List[str],
        negative_ids: List[str] = [],
        limit: int = 5,
        clinic_id: Optional[str] = None
    ):
        """
        Uses Qdrant's Recommendation API to find items similar to the positive IDs
        and dissimilar to negative IDs. Good for "Show me more cases like Patient X".
        """
        filter_condition = None
        if clinic_id:
            filter_condition = models.Filter(
                must=[models.FieldCondition(key="clinic_id", match=models.MatchValue(value=clinic_id))]
            )

        # Use query_points with RecommendQuery (modern Qdrant API)
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                positive=positive_ids,
                negative=negative_ids if negative_ids else None
            ),
            using="dense_text",
            limit=limit,
            query_filter=filter_condition,
            with_payload=True
        ).points
