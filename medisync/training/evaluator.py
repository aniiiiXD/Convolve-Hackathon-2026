"""
Model Evaluation Metrics

Computes retrieval quality metrics:
- nDCG@k (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)
- Recall@k
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for retrieval models"""

    @staticmethod
    def dcg_at_k(relevances: List[float], k: int) -> float:
        """
        Compute Discounted Cumulative Gain at k

        Args:
            relevances: List of relevance scores (1 for relevant, 0 for not)
            k: Cutoff rank

        Returns:
            DCG@k score
        """
        relevances = np.array(relevances[:k])
        if relevances.size == 0:
            return 0.0

        # DCG = sum(rel_i / log2(i + 2)) for i in range(k)
        discounts = np.log2(np.arange(2, relevances.size + 2))
        return np.sum(relevances / discounts)

    @staticmethod
    def ndcg_at_k(relevances: List[float], k: int) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at k

        Args:
            relevances: List of relevance scores (1 for relevant, 0 for not)
            k: Cutoff rank

        Returns:
            nDCG@k score (0-1)
        """
        dcg = ModelEvaluator.dcg_at_k(relevances, k)

        # Ideal DCG (sorted by relevance)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = ModelEvaluator.dcg_at_k(ideal_relevances, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def reciprocal_rank(relevances: List[float]) -> float:
        """
        Compute reciprocal rank (1 / rank of first relevant result)

        Args:
            relevances: List of relevance scores

        Returns:
            Reciprocal rank
        """
        for i, rel in enumerate(relevances):
            if rel > 0:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def average_precision(relevances: List[float]) -> float:
        """
        Compute average precision

        Args:
            relevances: List of relevance scores

        Returns:
            Average precision
        """
        num_relevant = sum(1 for r in relevances if r > 0)
        if num_relevant == 0:
            return 0.0

        precision_sum = 0.0
        num_relevant_seen = 0

        for i, rel in enumerate(relevances):
            if rel > 0:
                num_relevant_seen += 1
                precision_at_i = num_relevant_seen / (i + 1)
                precision_sum += precision_at_i

        return precision_sum / num_relevant

    @staticmethod
    def recall_at_k(relevances: List[float], k: int, total_relevant: int) -> float:
        """
        Compute recall at k

        Args:
            relevances: List of relevance scores
            k: Cutoff rank
            total_relevant: Total number of relevant documents

        Returns:
            Recall@k
        """
        if total_relevant == 0:
            return 0.0

        relevant_at_k = sum(1 for r in relevances[:k] if r > 0)
        return relevant_at_k / total_relevant

    def evaluate_rankings(
        self,
        rankings: List[Dict[str, Any]],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        Evaluate a set of rankings

        Args:
            rankings: List of ranking results with relevance labels
                Format: [{"relevances": [1, 0, 1, 0, ...], "total_relevant": 5}, ...]
            k_values: List of k values to evaluate

        Returns:
            Dictionary of metrics
        """
        ndcg_scores = {k: [] for k in k_values}
        recall_scores = {k: [] for k in k_values}
        mrr_scores = []
        map_scores = []

        for ranking in rankings:
            relevances = ranking['relevances']
            total_relevant = ranking.get('total_relevant', sum(relevances))

            # Compute metrics
            mrr_scores.append(self.reciprocal_rank(relevances))
            map_scores.append(self.average_precision(relevances))

            for k in k_values:
                ndcg_scores[k].append(self.ndcg_at_k(relevances, k))
                recall_scores[k].append(
                    self.recall_at_k(relevances, k, total_relevant)
                )

        # Average metrics
        metrics = {
            'mrr': np.mean(mrr_scores),
            'map': np.mean(map_scores)
        }

        for k in k_values:
            metrics[f'ndcg@{k}'] = np.mean(ndcg_scores[k])
            metrics[f'recall@{k}'] = np.mean(recall_scores[k])

        return metrics

    def evaluate_embedder(
        self,
        model_path: str,
        test_file: str
    ) -> Dict[str, float]:
        """
        Evaluate fine-tuned embedding model

        Args:
            model_path: Path to saved embedding model
            test_file: Path to test data

        Returns:
            Dictionary of metrics
        """
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            # Load model
            model = SentenceTransformer(model_path)

            # Load test data
            test_data = []
            with open(test_file, 'r') as f:
                for line in f:
                    test_data.append(json.loads(line))

            logger.info(f"Evaluating embedder on {len(test_data)} test samples")

            # Compute rankings
            rankings = []
            for sample in test_data[:500]:  # Use subset for speed
                query = sample['query']
                positive = sample['positive']
                negatives = sample['negatives']

                # Encode
                query_emb = model.encode(query, convert_to_tensor=True)
                pos_emb = model.encode(positive, convert_to_tensor=True)
                neg_embs = model.encode(negatives, convert_to_tensor=True)

                # Compute similarities
                pos_sim = torch.nn.functional.cosine_similarity(
                    query_emb.unsqueeze(0), pos_emb.unsqueeze(0)
                ).item()

                neg_sims = torch.nn.functional.cosine_similarity(
                    query_emb.unsqueeze(0), neg_embs
                ).tolist()

                # Create ranking
                all_sims = [pos_sim] + neg_sims
                all_labels = [1] + [0] * len(neg_sims)

                # Sort by similarity
                sorted_pairs = sorted(
                    zip(all_sims, all_labels),
                    key=lambda x: x[0],
                    reverse=True
                )
                relevances = [label for _, label in sorted_pairs]

                rankings.append({
                    'relevances': relevances,
                    'total_relevant': 1
                })

            # Compute metrics
            metrics = self.evaluate_rankings(rankings)
            logger.info(f"Embedder metrics: {metrics}")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating embedder: {e}", exc_info=True)
            return {}

    def evaluate_reranker(
        self,
        model_path: str,
        test_file: str
    ) -> Dict[str, float]:
        """
        Evaluate re-ranker model

        Args:
            model_path: Path to saved re-ranker model
            test_file: Path to test data

        Returns:
            Dictionary of metrics
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            # Load model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()

            # Load test data
            test_data = []
            with open(test_file, 'r') as f:
                for line in f:
                    test_data.append(json.loads(line))

            logger.info(f"Evaluating reranker on {len(test_data)} test samples")

            # Group by query
            query_results = {}
            for sample in test_data:
                query = sample['query']
                if query not in query_results:
                    query_results[query] = []

                query_results[query].append({
                    'document': sample['document'],
                    'label': sample['label'],
                    'original_score': sample.get('score', 0)
                })

            # Compute rankings
            rankings = []
            for query, results in list(query_results.items())[:500]:  # Subset
                scores = []

                for result in results:
                    # Score query-document pair
                    inputs = tokenizer(
                        query,
                        result['document'],
                        max_length=512,
                        truncation=True,
                        return_tensors='pt'
                    )

                    with torch.no_grad():
                        outputs = model(**inputs)
                        # Probability of relevance (class 1)
                        prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

                    scores.append((prob, result['label']))

                # Sort by score
                sorted_results = sorted(scores, key=lambda x: x[0], reverse=True)
                relevances = [label for _, label in sorted_results]

                rankings.append({
                    'relevances': relevances,
                    'total_relevant': sum(r['label'] for r in results)
                })

            # Compute metrics
            metrics = self.evaluate_rankings(rankings)
            logger.info(f"Reranker metrics: {metrics}")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating reranker: {e}", exc_info=True)
            return {}


def main():
    """CLI entry point for evaluation"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=['embedder', 'reranker'],
        required=True,
        help="Type of model to evaluate"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved model"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test data"
    )

    args = parser.parse_args()

    evaluator = ModelEvaluator()

    if args.model_type == 'embedder':
        metrics = evaluator.evaluate_embedder(args.model_path, args.test_file)
    else:
        metrics = evaluator.evaluate_reranker(args.model_path, args.test_file)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
