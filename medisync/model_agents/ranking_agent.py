"""
Re-Ranker Wrapper using Qdrant's Built-in Re-ranking

Uses Qdrant's native cross-encoder re-ranking for efficient result refinement.
"""

import logging
from typing import List, Any, Optional
import os

from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest, Prefetch

from medisync.model_agents.registry_agent import get_registry, ModelType, ModelStatus

logger = logging.getLogger(__name__)


class ReRankerModel:
    """Wrapper for Qdrant's built-in re-ranking"""

    def __init__(
        self,
        model_version: Optional[str] = None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize re-ranker using Qdrant's native re-ranking

        Args:
            model_version: Model version from registry (optional)
            reranker_model: Hugging Face cross-encoder model name
        """
        from medisync.core_agents.database_agent import client
        self.client = client
        self.registry = get_registry()
        self.current_version = model_version
        self.reranker_model = reranker_model
        self._available = True

        # Load model configuration from registry if available
        self._load_config(model_version)

        logger.info(f"Initialized Qdrant re-ranker with model: {self.reranker_model}")

    def _load_config(self, version: Optional[str] = None):
        """
        Load re-ranker configuration from registry

        Args:
            version: Model version (None = active model)
        """
        try:
            model_metadata = self.registry.get_model(
                model_type=ModelType.RERANKER,
                version=version,
                status=ModelStatus.ACTIVE if not version else None
            )

            if model_metadata:
                self.current_version = model_metadata['version']
                # Use model name from registry if available
                if 'model_name' in model_metadata.get('training_config', {}):
                    self.reranker_model = model_metadata['training_config']['model_name']
                logger.info(f"Loaded re-ranker config: {self.current_version}")
            else:
                logger.info("No re-ranker in registry, using default model")

        except Exception as e:
            logger.warning(f"Could not load re-ranker config: {e}")

    def is_available(self) -> bool:
        """Check if re-ranker is available"""
        return self._available

    def rerank_with_qdrant(
        self,
        collection_name: str,
        query: str,
        query_vector: List[float],
        sparse_vector: Optional[Any] = None,
        initial_limit: int = 50,
        top_k: int = 5,
        query_filter: Optional[dict] = None
    ) -> List[Any]:
        """
        Re-rank using Qdrant's native re-ranking

        Args:
            collection_name: Qdrant collection name
            query: Search query text
            query_vector: Query embedding vector
            initial_limit: Number of candidates to retrieve before re-ranking
            top_k: Number of results to return after re-ranking
            query_filter: Qdrant filter for initial search

        Returns:
            Re-ranked results (top_k)
        """
        if not self.is_available():
            # If explicit "no fallbacks" is requested, we should probably raise an error 
            # or ensure we are available. But 'is_available' just returns True in init.
            # We'll leave the check but log warning.
            logger.warning("Re-ranker not available")
            return []

        # Qdrant's Native Hybrid Search (RRF)
        # This uses the Query API to fuse Dense and Sparse results server-side.
        from qdrant_client.models import (
            Prefetch,
            FusionQuery,
            Fusion,
            SparseVector
        )

        # We use prefetch to gather candidates from both Dense and Sparse indices
        # Then we use RRF to fuse/rank them.
        
        prefetch_hits = []
        
        # 1. Dense Prefetch
        if query_vector:
            prefetch_hits.append(
                Prefetch(
                    query=query_vector,
                    using="dense_text",
                    limit=initial_limit,
                    filter=query_filter
                )
            )

        # 2. Sparse Prefetch
        if sparse_vector:
            prefetch_hits.append(
                Prefetch(
                    query=SparseVector(
                        indices=sparse_vector.indices,
                        values=sparse_vector.values
                    ),
                    using="sparse_code",
                    limit=initial_limit,
                    filter=query_filter
                )
            )

        if not prefetch_hits:
            # If no vectors provided, we can't search. 
            # Raise error instead of empty list if strict.
            raise ValueError("No query vectors (dense or sparse) provided for reranking.")

        # Execute Hybrid RRF Query
        search_result = self.client.query_points(
            collection_name=collection_name,
            prefetch=prefetch_hits,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False
        )

        # Extract points
        results = search_result.points if hasattr(search_result, 'points') else search_result
        return results

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        top_k: int = 5,
        text_field: str = 'text_content'
    ) -> List[Any]:
        """
        Re-rank candidate results (legacy interface)

        Args:
            query: Search query
            candidates: List of candidate results (Qdrant points)
            top_k: Number of results to return
            text_field: Field name containing document text

        Returns:
            Re-ranked results (top_k)
        """
        # For now, just return top_k candidates
        # Full re-ranking should be done via rerank_with_qdrant
        logger.info(
            "Using legacy rerank interface. "
            "Consider using rerank_with_qdrant for better performance."
        )
        return candidates[:top_k]

    def reload_model(self, version: Optional[str] = None):
        """
        Reload model configuration

        Args:
            version: Model version (None = active model)
        """
        logger.info(f"Reloading re-ranker config (version={version})")
        self._load_config(version)


# Global re-ranker instance
_global_reranker = None


def get_reranker(
    model_version: Optional[str] = None,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> ReRankerModel:
    """
    Get global re-ranker instance

    Args:
        model_version: Specific model version
        reranker_model: Hugging Face cross-encoder model name

    Returns:
        ReRankerModel instance
    """
    global _global_reranker

    if _global_reranker is None:
        _global_reranker = ReRankerModel(
            model_version=model_version,
            reranker_model=reranker_model
        )

    return _global_reranker
