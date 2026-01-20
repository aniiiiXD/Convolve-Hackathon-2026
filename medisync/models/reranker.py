"""
Re-Ranker Inference Wrapper

Production-ready wrapper for re-ranking search results using trained cross-encoder.
"""

import logging
from typing import List, Any, Optional, Tuple
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from medisync.models.model_registry import get_registry, ModelType, ModelStatus

logger = logging.getLogger(__name__)
 

class ReRankerModel:
    """Production wrapper for re-ranking model"""

    def __init__(
        self,
        model_version: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize re-ranker

        Args:
            model_version: Specific model version (None = active model)
            device: Device to run on ('cpu', 'cuda', None=auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.registry = get_registry()
        self.model = None
        self.tokenizer = None
        self.current_version = None

        # Load model
        self._load_model(model_version)

    def _load_model(self, version: Optional[str] = None):
        """
        Load model from registry

        Args:
            version: Model version (None = active model)
        """
        try:
            # Get model metadata
            model_metadata = self.registry.get_model(
                model_type=ModelType.RERANKER,
                version=version,
                status=ModelStatus.ACTIVE if not version else None
            )

            if not model_metadata:
                logger.warning(
                    "No re-ranker model found. Re-ranking will be disabled."
                )
                return

            model_path = model_metadata['model_path']
            self.current_version = model_metadata['version']

            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()

            logger.info(
                f"Loaded re-ranker model {self.current_version} on {self.device}"
            )

        except Exception as e:
            logger.error(f"Error loading re-ranker model: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None

    def is_available(self) -> bool:
        """Check if re-ranker is available"""
        return self.model is not None and self.tokenizer is not None

    def score(
        self,
        query: str,
        document: str,
        max_length: int = 512
    ) -> float:
        """
        Score a query-document pair

        Args:
            query: Search query
            document: Document text
            max_length: Maximum sequence length

        Returns:
            Relevance score (0-1)
        """
        if not self.is_available():
            return 0.0

        try:
            # Tokenize
            inputs = self.tokenizer(
                query,
                document,
                max_length=max_length,
                truncation=True,
                return_tensors='pt'
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Score
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Probability of relevance (class 1)
                prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

            return prob

        except Exception as e:
            logger.error(f"Error scoring document: {e}")
            return 0.0

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        top_k: int = 5,
        text_field: str = 'text_content',
        score_field: str = 'rerank_score'
    ) -> List[Any]:
        """
        Re-rank candidate results

        Args:
            query: Search query
            candidates: List of candidate results (Qdrant points or dicts)
            top_k: Number of results to return
            text_field: Field name containing document text
            score_field: Field name to store re-ranking score

        Returns:
            Re-ranked results (top_k)
        """
        if not self.is_available():
            logger.warning("Re-ranker not available, returning original results")
            return candidates[:top_k]

        if not candidates:
            return []

        try:
            # Score all candidates
            scored_candidates = []

            for candidate in candidates:
                # Extract text from candidate
                if hasattr(candidate, 'payload'):
                    # Qdrant point
                    document = candidate.payload.get(text_field, '')
                elif isinstance(candidate, dict):
                    # Dictionary
                    document = candidate.get(text_field, '')
                else:
                    logger.warning(f"Unknown candidate type: {type(candidate)}")
                    scored_candidates.append((candidate, 0.0))
                    continue

                # Score
                score = self.score(query, document)

                # Store score
                if hasattr(candidate, 'payload'):
                    candidate.payload[score_field] = score
                elif isinstance(candidate, dict):
                    candidate[score_field] = score

                scored_candidates.append((candidate, score))

            # Sort by score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            # Return top-k
            reranked = [candidate for candidate, _ in scored_candidates[:top_k]]

            logger.debug(
                f"Re-ranked {len(candidates)} candidates to {len(reranked)} "
                f"(top score: {scored_candidates[0][1]:.3f})"
            )

            return reranked

        except Exception as e:
            logger.error(f"Error re-ranking results: {e}", exc_info=True)
            return candidates[:top_k]

    def batch_score(
        self,
        query: str,
        documents: List[str],
        batch_size: int = 8
    ) -> List[float]:
        """
        Score multiple documents in batches (faster)

        Args:
            query: Search query
            documents: List of document texts
            batch_size: Batch size for inference

        Returns:
            List of relevance scores
        """
        if not self.is_available():
            return [0.0] * len(documents)

        try:
            scores = []

            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]

                # Tokenize batch
                inputs = self.tokenizer(
                    [query] * len(batch_docs),
                    batch_docs,
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Score batch
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[:, 1].tolist()

                scores.extend(probs)

            return scores

        except Exception as e:
            logger.error(f"Error batch scoring: {e}", exc_info=True)
            return [0.0] * len(documents)

    def reload_model(self, version: Optional[str] = None):
        """
        Reload model (e.g., after model update)

        Args:
            version: Model version (None = active model)
        """
        logger.info(f"Reloading re-ranker model (version={version})")
        self._load_model(version)


# Global re-ranker instance for convenience
_global_reranker = None


def get_reranker(
    model_version: Optional[str] = None,
    device: Optional[str] = None
) -> ReRankerModel:
    """
    Get global re-ranker instance

    Args:
        model_version: Specific model version
        device: Device to run on

    Returns:
        ReRankerModel instance
    """
    global _global_reranker

    if _global_reranker is None:
        _global_reranker = ReRankerModel(
            model_version=model_version,
            device=device
        )

    return _global_reranker
