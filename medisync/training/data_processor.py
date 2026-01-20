"""
Training Data Processor

Processes feedback data into formats suitable for:
1. Embedding fine-tuning (Multiple Negatives Ranking Loss)
2. Re-ranker training (Binary classification)
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from medisync.services.feedback_service import FeedbackService
from medisync.core.database import client
from medisync.services.qdrant_ops import COLLECTION_NAME

logger = logging.getLogger(__name__)


class TrainingDataProcessor:
    """Process feedback data for model training"""

    def __init__(self, output_dir: str = "./training_data"):
        """
        Initialize data processor

        Args:
            output_dir: Directory to save processed training data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_training_batch(
        self,
        days: int = 30,
        min_interactions: int = 1,
        batch_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export training data from feedback logs

        Args:
            days: Number of days of data to export
            min_interactions: Minimum interactions per query
            batch_name: Optional batch name

        Returns:
            Export metadata
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Export from feedback service
        export_data = FeedbackService.export_training_data(
            date_range_start=start_date,
            date_range_end=end_date,
            batch_name=batch_name,
            min_interactions=min_interactions
        )

        logger.info(
            f"Exported {export_data['sample_count']} samples "
            f"in batch {export_data['batch_name']}"
        )

        return export_data

    def prepare_embedding_training_data(
        self,
        export_data: Dict[str, Any],
        max_negatives: int = 5
    ) -> List[Dict[str, str]]:
        """
        Prepare data for embedding fine-tuning with MNR loss

        Format:
        {
            "query": "chest pain and shortness of breath",
            "positive": "Patient note text...",
            "negatives": ["Other note 1...", "Other note 2..."]
        }

        Args:
            export_data: Exported feedback data
            max_negatives: Maximum number of negative samples per query

        Returns:
            List of training samples
        """
        training_samples = []

        for sample in export_data['samples']:
            # Get positive results (clicked/used)
            positive_ids = [
                r['point_id'] for r in sample['positive_results']
            ]

            # Get negative results (viewed but not clicked, high rank)
            negative_ids = [
                r['point_id'] for r in sample['negative_results'][:max_negatives]
            ]

            if not positive_ids:
                continue

            # Fetch actual text content from Qdrant
            try:
                # Reconstruct query text (we only have hash, so use a placeholder)
                # In production, you might store actual query text separately
                query_text = f"query_{sample['query_hash'][:8]}"

                # Fetch positive document
                positive_point = client.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=[positive_ids[0]],  # Use first positive
                    with_payload=True
                )[0]

                positive_text = positive_point.payload.get('text_content', '')

                # Fetch negative documents
                negative_texts = []
                if negative_ids:
                    negative_points = client.retrieve(
                        collection_name=COLLECTION_NAME,
                        ids=negative_ids,
                        with_payload=True
                    )
                    negative_texts = [
                        p.payload.get('text_content', '')
                        for p in negative_points
                    ]

                training_samples.append({
                    "query": query_text,
                    "positive": positive_text,
                    "negatives": negative_texts
                })

            except Exception as e:
                logger.warning(f"Error fetching documents for sample: {e}")
                continue

        logger.info(f"Prepared {len(training_samples)} embedding training samples")
        return training_samples

    def prepare_reranker_training_data(
        self,
        export_data: Dict[str, Any],
        label_threshold: float = 3.0
    ) -> List[Dict[str, Any]]:
        """
        Prepare data for re-ranker training (binary classification)

        Format:
        {
            "query": "chest pain",
            "document": "Patient note text...",
            "label": 1,  # 1 for relevant (clicked), 0 for not relevant
            "score": 0.85,  # Original retrieval score
            "rank": 2  # Original rank
        }

        Args:
            export_data: Exported feedback data
            label_threshold: Confidence threshold for positive labels

        Returns:
            List of training samples
        """
        training_samples = []

        for sample in export_data['samples']:
            query_id = sample['query_id']

            # Process positive results (clicked/used)
            for result in sample['positive_results']:
                try:
                    point = client.retrieve(
                        collection_name=COLLECTION_NAME,
                        ids=[result['point_id']],
                        with_payload=True
                    )[0]

                    training_samples.append({
                        "query": f"query_{sample['query_hash'][:8]}",
                        "document": point.payload.get('text_content', ''),
                        "label": 1,
                        "score": result['score'],
                        "rank": result['rank'],
                        "dwell_time": result.get('dwell_time', 0)
                    })

                except Exception as e:
                    logger.warning(f"Error fetching positive result: {e}")

            # Process negative results (viewed but not clicked)
            for result in sample['negative_results']:
                try:
                    point = client.retrieve(
                        collection_name=COLLECTION_NAME,
                        ids=[result['point_id']],
                        with_payload=True
                    )[0]

                    training_samples.append({
                        "query": f"query_{sample['query_hash'][:8]}",
                        "document": point.payload.get('text_content', ''),
                        "label": 0,
                        "score": result['score'],
                        "rank": result['rank'],
                        "dwell_time": 0
                    })

                except Exception as e:
                    logger.warning(f"Error fetching negative result: {e}")

        logger.info(f"Prepared {len(training_samples)} re-ranker training samples")
        return training_samples

    def save_training_data(
        self,
        data: List[Dict[str, Any]],
        filename: str,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    ) -> Dict[str, str]:
        """
        Save training data and split into train/val/test

        Args:
            data: Training samples
            filename: Base filename (without extension)
            split_ratio: (train, val, test) ratio

        Returns:
            Dictionary with file paths
        """
        import random

        # Shuffle data
        random.shuffle(data)

        # Calculate split indices
        n = len(data)
        train_end = int(n * split_ratio[0])
        val_end = train_end + int(n * split_ratio[1])

        # Split data
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        # Save splits
        files = {}
        for split_name, split_data in [
            ("train", train_data),
            ("val", val_data),
            ("test", test_data)
        ]:
            filepath = self.output_dir / f"{filename}_{split_name}.jsonl"
            with open(filepath, 'w') as f:
                for sample in split_data:
                    f.write(json.dumps(sample) + '\n')

            files[split_name] = str(filepath)
            logger.info(f"Saved {len(split_data)} samples to {filepath}")

        return files

    def process_and_save(
        self,
        days: int = 30,
        batch_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete pipeline: export, process, and save training data

        Args:
            days: Number of days of data to export
            batch_name: Optional batch name

        Returns:
            Dictionary with file paths and metadata
        """
        # Export feedback data
        export_data = self.export_training_batch(
            days=days,
            batch_name=batch_name
        )

        if export_data['sample_count'] == 0:
            logger.warning("No training samples found")
            return {"status": "no_data", "sample_count": 0}

        # Prepare embedding training data
        embedding_data = self.prepare_embedding_training_data(export_data)
        embedding_files = {}
        if embedding_data:
            embedding_files = self.save_training_data(
                embedding_data,
                f"embedding_{export_data['batch_name']}"
            )

        # Prepare re-ranker training data
        reranker_data = self.prepare_reranker_training_data(export_data)
        reranker_files = {}
        if reranker_data:
            reranker_files = self.save_training_data(
                reranker_data,
                f"reranker_{export_data['batch_name']}"
            )

        result = {
            "status": "success",
            "batch_id": export_data['batch_id'],
            "batch_name": export_data['batch_name'],
            "sample_count": export_data['sample_count'],
            "embedding_samples": len(embedding_data),
            "reranker_samples": len(reranker_data),
            "embedding_files": embedding_files,
            "reranker_files": reranker_files
        }

        logger.info(f"Training data processing complete: {result}")
        return result


def main():
    """CLI entry point for data processing"""
    import argparse

    parser = argparse.ArgumentParser(description="Process feedback data for training")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of data to export"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./training_data",
        help="Output directory for training data"
    )
    parser.add_argument(
        "--batch-name",
        type=str,
        help="Optional batch name"
    )

    args = parser.parse_args()

    processor = TrainingDataProcessor(output_dir=args.output_dir)
    result = processor.process_and_save(
        days=args.days,
        batch_name=args.batch_name
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
