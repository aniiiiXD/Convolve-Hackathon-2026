from qdrant_client import models
from medisync.core_agents.database_agent import client
from medisync.service_agents.memory_ops_agent import COLLECTION_NAME
from medisync.service_agents.encoding_agent import EmbeddingService
from typing import List, Optional

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
        # Embed everything
        target_vec = EmbeddingService.embed_dense(target_text)
        
        context_pairs = []
        for p in positive_texts:
            context_pairs.append(models.ContextExamplePair(
                positive=EmbeddingService.embed_dense(p),
                negative=EmbeddingService.embed_dense(negative_texts[0]) if negative_texts else None 
                # Note: Simplified. Ideally we pair specific positives/negatives or just list them.
                # Qdrant ContextQuery accepts list of pairs.
            ))
            
        # Simplified Context Query Approach conforming to python client
        # Target + Context
        
        # Filter
        filter_condition = None
        if clinic_id:
            filter_condition = models.Filter(
                must=[models.FieldCondition(key="clinic_id", match=models.MatchValue(value=clinic_id))]
            )

        return client.discover(
            collection_name=COLLECTION_NAME,
            target=target_vec,
            context=[
                 models.ContextExamplePair(
                     positive=EmbeddingService.embed_dense(p),
                     negative=EmbeddingService.embed_dense(n) 
                 ) for p, n in zip(positive_texts, negative_texts)
            ] if negative_texts else [models.ContextExamplePair(positive=EmbeddingService.embed_dense(p), negative=None) for p in positive_texts],
            limit=limit,
            filter=filter_condition,
            using="dense_text"
        )

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
            
        return client.recommend(
            collection_name=COLLECTION_NAME,
            positive=positive_ids,
            negative=negative_ids,
            limit=limit,
            filter=filter_condition,
            using="dense_text" # Recommendations work best on dense vectors
        )
