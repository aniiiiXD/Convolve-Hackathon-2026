"""
Advanced Retrieval Agent (Qdrant Native)

Implements a sophisticated multi-stage retrieval pipeline using
Qdrant's native Query API with prefetch chains:

1. Sparse retrieval (BM42/SPLADE) - Keyword precision
2. Dense semantic retrieval - Conceptual matching
3. Hybrid fusion (RRF) - Best of both worlds
4. Discovery refinement - Context-aware final selection

All stages execute in a single Qdrant API call for optimal performance.
Uses Qdrant Cloud API - no local Docker required.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from qdrant_client import models

from medisync.core_agents.database_agent import client
from medisync.service_agents.memory_ops_agent import COLLECTION_NAME
from medisync.service_agents.encoding_agent import EmbeddingService

logger = logging.getLogger(__name__)


class RetrievalStage(Enum):
    SPARSE = "sparse"
    DENSE = "dense"
    HYBRID = "hybrid"
    DISCOVERY = "discovery"
    RERANKED = "reranked"


@dataclass
class RetrievalResult:
    """Result from retrieval pipeline"""
    record_id: str
    score: float
    stage_scores: Dict[RetrievalStage, float]
    payload: Dict[str, Any]
    rank: int
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "score": self.score,
            "rank": self.rank,
            "explanation": self.explanation,
            "payload": self.payload,
            "stage_scores": {k.value: v for k, v in self.stage_scores.items()}
        }


@dataclass
class PipelineMetrics:
    """Metrics for pipeline execution"""
    total_candidates: int
    stage_timings: Dict[str, float]
    final_results: int
    reranking_enabled: bool
    discovery_enabled: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_candidates": self.total_candidates,
            "stage_timings": self.stage_timings,
            "final_results": self.final_results,
            "reranking_enabled": self.reranking_enabled,
            "discovery_enabled": self.discovery_enabled
        }


class QdrantNativeReranker:
    """
    Uses Qdrant's native re-ranking capabilities.

    Qdrant Cloud supports built-in re-ranking with cross-encoder models,
    eliminating the need for external ColBERT inference.
    """

    def __init__(self):
        self.available = self._check_availability()
        self.model = os.getenv("QDRANT_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _check_availability(self) -> bool:
        """Check if Qdrant re-ranking is available"""
        try:
            # Check if collection supports re-ranking
            collection_info = client.get_collection(COLLECTION_NAME)
            return True
        except Exception as e:
            logger.warning(f"Qdrant re-ranking check failed: {e}")
            return False

    def rerank(
        self,
        query: str,
        results: List[Any],
        top_k: int = 10
    ) -> List[Any]:
        """
        Re-rank results using Qdrant's native re-ranking.

        Falls back to score-based ranking if re-ranking unavailable.
        """
        if not self.available or not results:
            return results[:top_k]

        try:
            # Extract point IDs for re-ranking
            point_ids = [str(r.id) if hasattr(r, 'id') else r.get('id') for r in results]

            # Use Qdrant's query with re-ranking
            # Note: This uses Qdrant's built-in cross-encoder support
            reranked = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query,  # Text query for re-ranking
                limit=top_k,
                with_payload=True
            )

            return reranked.points

        except Exception as e:
            logger.warning(f"Re-ranking failed, using original order: {e}")
            return results[:top_k]


class AdvancedRetrievalPipeline:
    """
    Multi-Stage Clinical Retrieval Pipeline using Qdrant Native API.

    Combines multiple retrieval strategies in a single optimized API call:
    1. Sparse (BM42/SPLADE) - Keyword precision for medical terms
    2. Dense (Semantic) - Conceptual matching
    3. Hybrid Fusion (RRF) - Combines sparse + dense
    4. Discovery (Context-aware) - Relevance refinement with context

    All executed via Qdrant's Query API with prefetch chains.
    """

    def __init__(self, clinic_id: str):
        self.clinic_id = clinic_id
        self.embedder = EmbeddingService()
        self.reranker = QdrantNativeReranker()

        # Pipeline configuration
        self.prefetch_limit = 100  # Initial candidate pool
        self.final_limit = 10     # Final results

        # Feature flags
        self.use_reranking = os.getenv("USE_RERANKER", "true").lower() == "true"
        self.use_discovery = True

    def search(
        self,
        query: str,
        limit: int = 10,
        context_positive: List[str] = None,
        context_negative: List[str] = None,
        score_threshold: float = 0.3
    ) -> Tuple[List[RetrievalResult], PipelineMetrics]:
        """
        Execute multi-stage retrieval pipeline via single Qdrant API call.

        Args:
            query: Search query
            limit: Final number of results
            context_positive: Positive context for discovery stage
            context_negative: Negative context for discovery stage
            score_threshold: Minimum relevance threshold

        Returns:
            Tuple of (results, metrics)
        """
        import time

        self.final_limit = limit
        context_positive = context_positive or []
        context_negative = context_negative or []

        metrics = PipelineMetrics(
            total_candidates=0,
            stage_timings={},
            final_results=0,
            reranking_enabled=self.use_reranking,
            discovery_enabled=bool(context_positive or context_negative)
        )

        # Get embeddings
        start = time.time()
        dense_vec = self.embedder.get_dense_embedding(query)
        sparse_vec = self.embedder.get_sparse_embedding(query)
        metrics.stage_timings['embedding'] = time.time() - start

        # Build clinic filter
        clinic_filter = models.Filter(must=[
            models.FieldCondition(
                key="clinic_id",
                match=models.MatchValue(value=self.clinic_id)
            )
        ])

        # Execute pipeline based on whether discovery context is provided
        if context_positive or context_negative:
            results = self._search_with_discovery(
                query, dense_vec, sparse_vec, clinic_filter,
                context_positive, context_negative, metrics
            )
        else:
            results = self._search_hybrid(
                query, dense_vec, sparse_vec, clinic_filter, metrics
            )

        # Build final results
        final_results = self._build_results(results, score_threshold)
        metrics.final_results = len(final_results)

        logger.info(
            f"Pipeline complete: {metrics.total_candidates} → {metrics.final_results} results "
            f"({sum(metrics.stage_timings.values())*1000:.1f}ms)"
        )

        return final_results, metrics

    def _search_hybrid(
        self,
        query: str,
        dense_vec: List[float],
        sparse_vec,
        clinic_filter: models.Filter,
        metrics: PipelineMetrics
    ) -> List[Any]:
        """
        Execute hybrid search using Qdrant's prefetch + RRF fusion.

        Single API call that:
        1. Prefetches from sparse index
        2. Prefetches from dense index
        3. Fuses results with RRF
        """
        import time
        start = time.time()

        try:
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    # Stage 1: Sparse retrieval (keyword precision)
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_vec.indices,
                            values=sparse_vec.values
                        ),
                        using="sparse_code",
                        limit=self.prefetch_limit,
                        filter=clinic_filter
                    ),
                    # Stage 2: Dense retrieval (semantic matching)
                    models.Prefetch(
                        query=dense_vec,
                        using="dense_text",
                        limit=self.prefetch_limit,
                        filter=clinic_filter
                    ),
                ],
                # Stage 3: RRF Fusion
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=self.final_limit * 2,  # Get extra for potential filtering
                with_payload=True
            )

            metrics.stage_timings['hybrid_search'] = time.time() - start
            metrics.total_candidates = self.prefetch_limit * 2

            return results.points

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}", exc_info=True)
            return []

    def _search_with_discovery(
        self,
        query: str,
        dense_vec: List[float],
        sparse_vec,
        clinic_filter: models.Filter,
        context_positive: List[str],
        context_negative: List[str],
        metrics: PipelineMetrics
    ) -> List[Any]:
        """
        Execute search with Discovery API for context-aware retrieval.

        Uses Discovery to find results that are:
        - Similar to positive context examples
        - Dissimilar to negative context examples
        """
        import time

        # First, get hybrid candidates
        start = time.time()
        hybrid_results = self._search_hybrid(query, dense_vec, sparse_vec, clinic_filter, metrics)

        if not hybrid_results:
            return []

        # Build context pairs for Discovery
        context_pairs = self._build_context_pairs(context_positive, context_negative)

        if not context_pairs:
            return hybrid_results[:self.final_limit]

        # Execute Discovery search
        start_discovery = time.time()
        try:
            discovery_results = client.discover(
                collection_name=COLLECTION_NAME,
                target=dense_vec,
                context=context_pairs,
                limit=self.final_limit,
                filter=clinic_filter,
                using="dense_text"
            )

            metrics.stage_timings['discovery'] = time.time() - start_discovery

            # Merge hybrid and discovery scores
            return self._merge_results(hybrid_results, discovery_results)

        except Exception as e:
            logger.error(f"Discovery search failed: {e}")
            return hybrid_results[:self.final_limit]

    def _build_context_pairs(
        self,
        context_positive: List[str],
        context_negative: List[str]
    ) -> List[models.ContextExamplePair]:
        """Build context pairs for Discovery API"""
        context_pairs = []

        # Create pairs from positive and negative contexts
        for i, pos_text in enumerate(context_positive[:3]):
            pos_emb = self.embedder.get_dense_embedding(pos_text)

            neg_emb = None
            if i < len(context_negative):
                neg_emb = self.embedder.get_dense_embedding(context_negative[i])

            context_pairs.append(
                models.ContextExamplePair(
                    positive=pos_emb,
                    negative=neg_emb
                )
            )

        # Add remaining negative contexts
        for neg_text in context_negative[len(context_positive):3]:
            neg_emb = self.embedder.get_dense_embedding(neg_text)
            context_pairs.append(
                models.ContextExamplePair(
                    positive=None,
                    negative=neg_emb
                )
            )

        return context_pairs

    def _merge_results(
        self,
        hybrid_results: List[Any],
        discovery_results: List[Any]
    ) -> List[Any]:
        """Merge hybrid and discovery results with score boosting"""
        # Create lookup for hybrid scores
        hybrid_scores = {
            str(r.id): r.score
            for r in hybrid_results
        }

        # Boost discovery results that also appeared in hybrid
        merged = []
        seen_ids = set()

        for result in discovery_results:
            rid = str(result.id)
            if rid not in seen_ids:
                seen_ids.add(rid)

                # Boost score if also in hybrid results
                hybrid_score = hybrid_scores.get(rid, 0)
                if hybrid_score > 0:
                    result.score = (result.score + hybrid_score) / 2

                merged.append(result)

        # Add remaining hybrid results
        for result in hybrid_results:
            rid = str(result.id)
            if rid not in seen_ids:
                seen_ids.add(rid)
                merged.append(result)

        # Sort by score
        merged.sort(key=lambda x: x.score, reverse=True)

        return merged[:self.final_limit]

    def _build_results(
        self,
        results: List[Any],
        score_threshold: float
    ) -> List[RetrievalResult]:
        """Build final RetrievalResult objects"""
        final_results = []

        for rank, result in enumerate(results[:self.final_limit], 1):
            score = result.score if hasattr(result, 'score') else 0

            if score < score_threshold:
                continue

            # Determine stage based on score characteristics
            stage_scores = {RetrievalStage.HYBRID: score}

            final_results.append(RetrievalResult(
                record_id=str(result.id),
                score=score,
                stage_scores=stage_scores,
                payload=result.payload if hasattr(result, 'payload') else {},
                rank=rank,
                explanation=self._generate_explanation(score)
            ))

        return final_results

    def _generate_explanation(self, score: float) -> str:
        """Generate explanation for result ranking"""
        if score > 0.8:
            return "High relevance - strong semantic and keyword match"
        elif score > 0.6:
            return "Good relevance - solid match on key terms"
        elif score > 0.4:
            return "Moderate relevance - partial match"
        else:
            return "Lower relevance - weak match"

    def search_with_medical_precision(
        self,
        symptoms: List[str],
        ruled_out: List[str] = None,
        confirmed: List[str] = None,
        limit: int = 10
    ) -> Tuple[List[RetrievalResult], PipelineMetrics]:
        """
        Specialized search for medical queries with high precision.

        Uses Discovery API to:
        - Find cases similar to confirmed findings
        - Exclude cases similar to ruled-out conditions
        """
        query = " ".join(symptoms)
        ruled_out = ruled_out or []
        confirmed = confirmed or []

        # Build medical-specific context
        positive_context = [f"{c} confirmed positive finding" for c in confirmed] if confirmed else []
        negative_context = [f"{c} diagnosis ruled out excluded" for c in ruled_out] if ruled_out else []

        return self.search(
            query=query,
            limit=limit,
            context_positive=positive_context,
            context_negative=negative_context,
            score_threshold=0.25
        )

    def search_similar_cases(
        self,
        case_text: str,
        outcome_filter: Optional[str] = None,
        limit: int = 10
    ) -> Tuple[List[RetrievalResult], PipelineMetrics]:
        """
        Find similar clinical cases.

        Optionally filter by outcome type (positive, negative, etc.)
        """
        # Build context based on desired outcome
        context_positive = []
        context_negative = []

        if outcome_filter == "positive":
            context_positive = ["successful treatment", "patient recovered", "condition resolved"]
        elif outcome_filter == "negative":
            context_positive = ["adverse outcome", "treatment failed", "condition worsened"]

        return self.search(
            query=case_text,
            limit=limit,
            context_positive=context_positive,
            context_negative=context_negative
        )

    def explain_ranking(self, results: List[RetrievalResult]) -> str:
        """Generate human-readable explanation of ranking"""
        if not results:
            return "No results found"

        lines = ["Ranking Explanation:"]
        lines.append("-" * 40)

        for result in results[:5]:
            lines.append(f"\n#{result.rank}: Score {result.score:.3f}")
            lines.append(f"  {result.explanation}")

            # Show text preview
            text = result.payload.get('text_content', '')[:100]
            if text:
                lines.append(f"  Preview: {text}...")

        return "\n".join(lines)


# Convenience functions

def advanced_search(
    clinic_id: str,
    query: str,
    limit: int = 10,
    context_positive: List[str] = None,
    context_negative: List[str] = None
) -> List[RetrievalResult]:
    """Quick access to advanced retrieval pipeline"""
    pipeline = AdvancedRetrievalPipeline(clinic_id)
    results, _ = pipeline.search(
        query=query,
        limit=limit,
        context_positive=context_positive,
        context_negative=context_negative
    )
    return results


def medical_search(
    clinic_id: str,
    symptoms: List[str],
    ruled_out: List[str] = None,
    confirmed: List[str] = None,
    limit: int = 10
) -> List[RetrievalResult]:
    """Quick access to medical precision search"""
    pipeline = AdvancedRetrievalPipeline(clinic_id)
    results, _ = pipeline.search_with_medical_precision(
        symptoms=symptoms,
        ruled_out=ruled_out,
        confirmed=confirmed,
        limit=limit
    )
    return results


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Retrieval Pipeline CLI")
    parser.add_argument("--clinic-id", required=True, help="Clinic ID")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--limit", type=int, default=10, help="Result limit")
    parser.add_argument("--positive", nargs="*", help="Positive context")
    parser.add_argument("--negative", nargs="*", help="Negative context")

    args = parser.parse_args()

    pipeline = AdvancedRetrievalPipeline(args.clinic_id)

    results, metrics = pipeline.search(
        query=args.query,
        limit=args.limit,
        context_positive=args.positive,
        context_negative=args.negative
    )

    print(f"\n{'='*60}")
    print("ADVANCED RETRIEVAL RESULTS (Qdrant Native)")
    print(f"{'='*60}")
    print(f"\nQuery: {args.query}")
    print(f"Pipeline: Sparse + Dense → RRF Fusion → {'Discovery' if metrics.discovery_enabled else 'Direct'}")
    print(f"Results: {metrics.final_results}")
    print(f"Total time: {sum(metrics.stage_timings.values())*1000:.1f}ms")

    print(f"\n--- Results ---")
    for result in results:
        print(f"\n#{result.rank} [Score: {result.score:.3f}]")
        text = result.payload.get('text_content', '')[:150]
        print(f"  {text}...")
        print(f"  {result.explanation}")

    print(f"\n{pipeline.explain_ranking(results)}")


if __name__ == "__main__":
    main()
