"""
Re-Ranker Training with Cross-Encoder

Trains a cross-encoder model to re-rank search results based on user feedback.
Base model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from medisync.models.model_registry import ModelRegistry, ModelType, ModelStatus
from medisync.training.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


class RerankerDataset(Dataset):
    """Dataset for re-ranker training"""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize dataset

        Args:
            data: List of training samples
            tokenizer: Tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Tokenize query-document pair
        encoding = self.tokenizer(
            sample['query'],
            sample['document'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(sample['label'], dtype=torch.long)
        }


class RerankerTrainer:
    """Trainer for re-ranking model"""

    def __init__(
        self,
        base_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        output_dir: str = "./models/rerankers"
    ):
        """
        Initialize re-ranker trainer

        Args:
            base_model: Base model to fine-tune
            output_dir: Directory to save trained models
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = None
        self.model = None
        self.registry = ModelRegistry()

    def load_training_data(
        self,
        train_file: str,
        val_file: Optional[str] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Load training data from JSONL files

        Args:
            train_file: Path to training data
            val_file: Path to validation data

        Returns:
            (train_data, val_data)
        """
        train_data = []
        with open(train_file, 'r') as f:
            for line in f:
                train_data.append(json.loads(line))

        logger.info(f"Loaded {len(train_data)} training samples")

        val_data = []
        if val_file:
            with open(val_file, 'r') as f:
                for line in f:
                    val_data.append(json.loads(line))

            logger.info(f"Loaded {len(val_data)} validation samples")

        return train_data, val_data

    def compute_metrics(self, pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute evaluation metrics

        Args:
            pred: Model predictions

        Returns:
            Dictionary of metrics
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary'
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train(
        self,
        train_file: str,
        val_file: Optional[str] = None,
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        version: Optional[str] = None
    ) -> str:
        """
        Train re-ranker model

        Args:
            train_file: Path to training data
            val_file: Path to validation data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_ratio: Warmup ratio
            version: Model version (auto-generated if None)

        Returns:
            Model version string
        """
        logger.info(f"Starting re-ranker training with base model {self.base_model}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=2  # Binary classification
        )

        # Load training data
        train_data, val_data = self.load_training_data(train_file, val_file)

        # Create datasets
        train_dataset = RerankerDataset(train_data, self.tokenizer)
        val_dataset = RerankerDataset(val_data, self.tokenizer) if val_data else None

        # Generate version
        if version is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            version = f"reranker-{timestamp}"

        # Training arguments
        model_save_path = self.output_dir / version
        training_args = TrainingArguments(
            output_dir=str(model_save_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=warmup_ratio,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=str(model_save_path / 'logs'),
            logging_steps=100,
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=500 if val_dataset else None,
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="f1" if val_dataset else None,
            greater_is_better=True
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics if val_dataset else None
        )

        # Train
        logger.info(
            f"Training config: epochs={epochs}, batch_size={batch_size}, "
            f"lr={learning_rate}, samples={len(train_data)}"
        )

        trainer.train()

        # Save model
        trainer.save_model(str(model_save_path))
        self.tokenizer.save_pretrained(str(model_save_path))

        logger.info(f"Saved re-ranker model to {model_save_path}")

        # Evaluate on test set if available
        metrics = {}
        test_file = train_file.replace("_train.", "_test.")
        if Path(test_file).exists():
            evaluator_obj = ModelEvaluator()
            metrics = evaluator_obj.evaluate_reranker(
                model_path=str(model_save_path),
                test_file=test_file
            )

        # Register model
        training_config = {
            "base_model": self.base_model,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_samples": len(train_data),
            "val_samples": len(val_data) if val_data else 0
        }

        self.registry.register_model(
            model_type=ModelType.RERANKER,
            model_path=str(model_save_path),
            version=version,
            metrics=metrics,
            training_config=training_config,
            status=ModelStatus.CANDIDATE
        )

        logger.info(f"Registered reranker version {version} with metrics: {metrics}")

        return version


def main():
    """CLI entry point for training"""
    import argparse

    parser = argparse.ArgumentParser(description="Train re-ranker model")
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
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
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

    trainer = RerankerTrainer()
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
