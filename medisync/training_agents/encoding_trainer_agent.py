"""
Embedding Fine-Tuning with Multiple Negatives Ranking Loss

Fine-tunes medical embeddings using user feedback data.
Base model: BAAI/bge-base-en-v1.5 (768-dim, medical-compatible)
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import torch
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation
)
from torch.utils.data import DataLoader

from medisync.model_agents.registry_agent import ModelRegistry, ModelType, ModelStatus
from medisync.training_agents.evaluation_agent import ModelEvaluator

logger = logging.getLogger(__name__)


class EmbeddingTrainer:
    """Trainer for fine-tuning embeddings with MNR loss"""

    def __init__(
        self,
        base_model: str = "BAAI/bge-base-en-v1.5",
        output_dir: str = "./models/embeddings"
    ):
        """
        Initialize embedding trainer

        Args:
            base_model: Base model to fine-tune
            output_dir: Directory to save fine-tuned models
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.registry = ModelRegistry()

    def load_training_data(
        self,
        train_file: str,
        val_file: Optional[str] = None
    ) -> tuple:
        """
        Load training data from JSONL files

        Args:
            train_file: Path to training data
            val_file: Path to validation data

        Returns:
            (train_examples, val_examples)
        """
        train_examples = []

        with open(train_file, 'r') as f:
            for line in f:
                sample = json.loads(line)

                # Create InputExample for each negative
                for negative in sample['negatives']:
                    train_examples.append(
                        InputExample(
                            texts=[
                                sample['query'],
                                sample['positive'],
                                negative
                            ]
                        )
                    )

        logger.info(f"Loaded {len(train_examples)} training examples")

        # Load validation data
        val_examples = []
        if val_file:
            with open(val_file, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    for negative in sample['negatives']:
                        val_examples.append(
                            InputExample(
                                texts=[
                                    sample['query'],
                                    sample['positive'],
                                    negative
                                ]
                            )
                        )

            logger.info(f"Loaded {len(val_examples)} validation examples")

        return train_examples, val_examples

    def train(
        self,
        train_file: str,
        val_file: Optional[str] = None,
        epochs: int = 3,
        batch_size: int = 16,
        warmup_steps: int = 100,
        learning_rate: float = 2e-5,
        version: Optional[str] = None
    ) -> str:
        """
        Train embedding model with MNR loss

        Args:
            train_file: Path to training data
            val_file: Path to validation data
            epochs: Number of training epochs
            batch_size: Batch size
            warmup_steps: Number of warmup steps
            learning_rate: Learning rate
            version: Model version (auto-generated if None)

        Returns:
            Model version string
        """
        logger.info(f"Starting embedding training with base model {self.base_model}")

        # Load model
        self.model = SentenceTransformer(self.base_model)

        # Load training data
        train_examples, val_examples = self.load_training_data(train_file, val_file)

        # Create data loader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size
        )

        # Define loss
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        # Calculate training steps
        num_training_steps = len(train_dataloader) * epochs

        logger.info(
            f"Training config: epochs={epochs}, batch_size={batch_size}, "
            f"steps={num_training_steps}, warmup={warmup_steps}"
        )

        # Setup evaluator
        evaluator = None
        if val_examples:
            # Create evaluation samples
            queries = []
            corpus = []
            relevant_docs = {}

            for i, example in enumerate(val_examples[:500]):  # Use subset for speed
                queries.append(example.texts[0])
                corpus.append(example.texts[1])  # Positive
                relevant_docs[i] = {len(corpus) - 1}

            evaluator = evaluation.InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name="validation"
            )

        # Train model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            evaluator=evaluator,
            evaluation_steps=500,
            output_path=None,  # Don't save intermediate checkpoints
            optimizer_params={'lr': learning_rate}
        )

        # Generate version
        if version is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            version = f"embedder-{timestamp}"

        # Save model
        model_path = self.output_dir / version
        self.model.save(str(model_path))

        logger.info(f"Saved fine-tuned model to {model_path}")

        # Evaluate on test set if available
        metrics = {}
        if val_examples:
            test_file = train_file.replace("_train.", "_test.")
            if Path(test_file).exists():
                evaluator_obj = ModelEvaluator()
                metrics = evaluator_obj.evaluate_embedder(
                    model_path=str(model_path),
                    test_file=test_file
                )

        # Register model
        training_config = {
            "base_model": self.base_model,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_samples": len(train_examples),
            "val_samples": len(val_examples)
        }

        self.registry.register_model(
            model_type=ModelType.EMBEDDER,
            model_path=str(model_path),
            version=version,
            metrics=metrics,
            training_config=training_config,
            status=ModelStatus.CANDIDATE
        )

        logger.info(f"Registered embedder version {version} with metrics: {metrics}")

        return version

    def evaluate(
        self,
        model_path: str,
        test_file: str
    ) -> Dict[str, float]:
        """
        Evaluate trained model

        Args:
            model_path: Path to saved model
            test_file: Path to test data

        Returns:
            Dictionary of metrics
        """
        evaluator = ModelEvaluator()
        return evaluator.evaluate_embedder(model_path, test_file)


def main():
    """CLI entry point for training"""
    import argparse

    parser = argparse.ArgumentParser(description="Train embedding model")
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Path to training data"
    )
    parser.add_argument(
        "--val-file",
        type=str,
        help="Path to validation data"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Model version"
    )

    args = parser.parse_args()

    trainer = EmbeddingTrainer()
    version = trainer.train(
        train_file=args.train_file,
        val_file=args.val_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        version=args.version
    )

    print(f"Training complete. Model version: {version}")


if __name__ == "__main__":
    main()
