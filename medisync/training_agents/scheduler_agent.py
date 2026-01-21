"""
Batch Retraining Scheduler

Automates periodic model retraining based on:
- Time-based triggers (weekly/monthly)
- Data-based triggers (minimum new feedback samples)
- Performance-based triggers (metric degradation)
"""

import os
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

from medisync.service_agents.learning_agent import FeedbackService
from medisync.training_agents.data_processor_agent import TrainingDataProcessor
from medisync.training_agents.encoding_trainer_agent import EmbeddingTrainer
from medisync.training_agents.ranking_trainer_agent import RerankerTrainer
from medisync.model_agents.registry_agent import (
    get_registry,
    ModelType,
    ModelStatus
)

logger = logging.getLogger(__name__)


class TrainingScheduler:
    """Scheduler for automated model retraining"""

    def __init__(
        self,
        output_dir: str = "./training_data",
        min_new_samples: int = 1000,
        training_days: int = 30
    ):
        """
        Initialize training scheduler

        Args:
            output_dir: Directory for training data and models
            min_new_samples: Minimum new samples before triggering training
            training_days: Days of data to use for training
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.min_new_samples = min_new_samples
        self.training_days = training_days

        self.data_processor = TrainingDataProcessor(output_dir=str(self.output_dir))
        self.embedding_trainer = EmbeddingTrainer()
        self.reranker_trainer = RerankerTrainer()
        self.registry = get_registry()

        self.state_file = self.output_dir / "scheduler_state.json"
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load scheduler state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)

        return {
            "last_training_date": None,
            "last_batch_name": None,
            "total_trainings": 0
        }

    def _save_state(self):
        """Save scheduler state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def should_trigger_training(self) -> tuple[bool, str]:
        """
        Check if training should be triggered

        Returns:
            (should_train, reason)
        """
        # Check time-based trigger
        last_training = self.state.get("last_training_date")
        if last_training:
            last_date = datetime.fromisoformat(last_training)
            days_since = (datetime.utcnow() - last_date).days

            if days_since >= 7:  # Weekly trigger
                return True, f"time_based (last training {days_since} days ago)"

        # Check data-based trigger
        stats = FeedbackService.get_query_statistics(days=self.training_days)
        if stats['total_queries'] >= self.min_new_samples:
            return True, f"data_based ({stats['total_queries']} new samples)"

        # Check performance-based trigger (simplified)
        # In production, compare current metrics with baseline
        # For now, just check if we have data and haven't trained yet
        if not last_training and stats['total_queries'] >= self.min_new_samples:
            return True, "initial_training"

        return False, "no_trigger"

    def run_training_pipeline(
        self,
        batch_name: Optional[str] = None,
        train_embedder: bool = True,
        train_reranker: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline

        Args:
            batch_name: Optional batch name
            train_embedder: Whether to train embedder
            train_reranker: Whether to train re-ranker

        Returns:
            Results dictionary
        """
        logger.info("=" * 80)
        logger.info("STARTING AUTOMATED TRAINING PIPELINE")
        logger.info("=" * 80)

        results = {
            "status": "started",
            "timestamp": datetime.utcnow().isoformat(),
            "batch_name": batch_name,
            "stages": {}
        }

        try:
            # Stage 1: Export training data
            logger.info("\n[1/5] Exporting training data...")
            export_result = self.data_processor.process_and_save(
                days=self.training_days,
                batch_name=batch_name
            )

            if export_result['status'] != 'success':
                results['status'] = 'failed'
                results['error'] = 'No training data'
                return results

            results['stages']['export'] = export_result
            batch_name = export_result['batch_name']

            # Stage 2: Train embedding model
            embedder_version = None
            if train_embedder and export_result['embedding_samples'] > 0:
                logger.info("\n[2/5] Training embedding model...")

                try:
                    train_file = export_result['embedding_files']['train']
                    val_file = export_result['embedding_files'].get('val')

                    embedder_version = self.embedding_trainer.train(
                        train_file=train_file,
                        val_file=val_file,
                        epochs=int(os.getenv("EMBEDDER_EPOCHS", "3")),
                        batch_size=int(os.getenv("EMBEDDER_BATCH_SIZE", "16"))
                    )

                    results['stages']['embedder'] = {
                        'status': 'success',
                        'version': embedder_version
                    }

                    logger.info(f"✓ Embedding model trained: {embedder_version}")

                except Exception as e:
                    logger.error(f"Error training embedder: {e}", exc_info=True)
                    results['stages']['embedder'] = {
                        'status': 'failed',
                        'error': str(e)
                    }

            # Stage 3: Train re-ranker model
            reranker_version = None
            if train_reranker and export_result['reranker_samples'] > 0:
                logger.info("\n[3/5] Training re-ranker model...")

                try:
                    train_file = export_result['reranker_files']['train']
                    val_file = export_result['reranker_files'].get('val')

                    reranker_version = self.reranker_trainer.train(
                        train_file=train_file,
                        val_file=val_file,
                        epochs=int(os.getenv("RERANKER_EPOCHS", "5")),
                        batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "8"))
                    )

                    results['stages']['reranker'] = {
                        'status': 'success',
                        'version': reranker_version
                    }

                    logger.info(f"✓ Re-ranker model trained: {reranker_version}")

                except Exception as e:
                    logger.error(f"Error training reranker: {e}", exc_info=True)
                    results['stages']['reranker'] = {
                        'status': 'failed',
                        'error': str(e)
                    }

            # Stage 4: Evaluate models
            logger.info("\n[4/5] Evaluating models...")
            # Models are already evaluated during training
            # Metrics are stored in registry

            # Stage 5: Update state
            logger.info("\n[5/5] Updating scheduler state...")
            self.state['last_training_date'] = datetime.utcnow().isoformat()
            self.state['last_batch_name'] = batch_name
            self.state['total_trainings'] += 1
            self._save_state()

            results['status'] = 'completed'
            results['embedder_version'] = embedder_version
            results['reranker_version'] = reranker_version

            logger.info("=" * 80)
            logger.info("TRAINING PIPELINE COMPLETED")
            logger.info(f"Embedder: {embedder_version}")
            logger.info(f"Re-ranker: {reranker_version}")
            logger.info("=" * 80)

            return results

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            results['status'] = 'failed'
            results['error'] = str(e)
            return results

    def auto_promote_models(
        self,
        embedder_version: Optional[str] = None,
        reranker_version: Optional[str] = None,
        min_ndcg: float = 0.70,
        min_mrr: float = 0.65
    ) -> Dict[str, bool]:
        """
        Automatically promote models if they meet quality thresholds

        Args:
            embedder_version: Embedder version to promote
            reranker_version: Re-ranker version to promote
            min_ndcg: Minimum nDCG@5 for promotion
            min_mrr: Minimum MRR for promotion

        Returns:
            Dictionary with promotion status
        """
        results = {}

        # Promote embedder
        if embedder_version:
            model_metadata = self.registry.get_model(
                model_type=ModelType.EMBEDDER,
                version=embedder_version
            )

            if model_metadata:
                ndcg = model_metadata['metrics'].get('ndcg@5', 0)

                if ndcg >= min_ndcg:
                    success = self.registry.promote_model(
                        model_type=ModelType.EMBEDDER,
                        version=embedder_version,
                        safety_check=True
                    )
                    results['embedder'] = success

                    if success:
                        logger.info(f"✓ Promoted embedder {embedder_version} to active")
                    else:
                        logger.warning(f"✗ Failed to promote embedder {embedder_version}")
                else:
                    logger.warning(
                        f"✗ Embedder {embedder_version} nDCG@5={ndcg:.3f} "
                        f"below threshold {min_ndcg}"
                    )
                    results['embedder'] = False

        # Promote re-ranker
        if reranker_version:
            model_metadata = self.registry.get_model(
                model_type=ModelType.RERANKER,
                version=reranker_version
            )

            if model_metadata:
                mrr = model_metadata['metrics'].get('mrr', 0)

                if mrr >= min_mrr:
                    success = self.registry.promote_model(
                        model_type=ModelType.RERANKER,
                        version=reranker_version,
                        safety_check=True
                    )
                    results['reranker'] = success

                    if success:
                        logger.info(f"✓ Promoted reranker {reranker_version} to active")
                    else:
                        logger.warning(f"✗ Failed to promote reranker {reranker_version}")
                else:
                    logger.warning(
                        f"✗ Re-ranker {reranker_version} MRR={mrr:.3f} "
                        f"below threshold {min_mrr}"
                    )
                    results['reranker'] = False

        return results

    def run_scheduled_training(self, auto_promote: bool = False):
        """
        Check if training should run and execute if needed

        Args:
            auto_promote: Automatically promote models if they meet thresholds
        """
        should_train, reason = self.should_trigger_training()

        if not should_train:
            logger.info(f"Training not triggered: {reason}")
            return

        logger.info(f"Training triggered: {reason}")

        # Generate batch name
        batch_name = f"auto_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Run training
        results = self.run_training_pipeline(batch_name=batch_name)

        if results['status'] == 'completed' and auto_promote:
            # Auto-promote if enabled
            logger.info("\nAttempting auto-promotion...")
            promotion_results = self.auto_promote_models(
                embedder_version=results.get('embedder_version'),
                reranker_version=results.get('reranker_version')
            )

            results['promotions'] = promotion_results

        # Save results
        results_file = self.output_dir / f"training_results_{batch_name}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {results_file}")


def main():
    """CLI entry point for scheduler"""
    import argparse

    parser = argparse.ArgumentParser(description="Training scheduler")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if training should be triggered"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run training if triggered"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force training regardless of triggers"
    )
    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Automatically promote models if they meet thresholds"
    )

    args = parser.parse_args()

    scheduler = TrainingScheduler()

    if args.check:
        should_train, reason = scheduler.should_trigger_training()
        print(f"Should train: {should_train}")
        print(f"Reason: {reason}")

    elif args.run or args.force:
        if args.force or scheduler.should_trigger_training()[0]:
            scheduler.run_scheduled_training(auto_promote=args.auto_promote)
        else:
            print("Training not triggered")

    else:
        print("Use --check, --run, or --force")


if __name__ == "__main__":
    main()
