"""
Model Registry for Version Management

Manages different versions of fine-tuned embeddings and re-ranker models
with metadata, metrics, and deployment status.
"""

import json
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model deployment status"""
    CANDIDATE = "candidate"  # Newly trained, not deployed
    SHADOW = "shadow"  # Running in shadow mode (logging only)
    AB_TEST = "ab_test"  # In A/B testing
    ACTIVE = "active"  # Currently serving traffic
    ARCHIVED = "archived"  # Deprecated/archived


class ModelType(Enum):
    """Type of model"""
    EMBEDDER = "embedder"
    RERANKER = "reranker"


class ModelRegistry:
    """Registry for managing model versions"""

    def __init__(self, registry_dir: str = "./models/registry"):
        """
        Initialize model registry

        Args:
            registry_dir: Directory to store registry metadata
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"

        # Load existing registry
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from disk"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {"embedders": {}, "rerankers": {}}

    def _save_registry(self):
        """Save registry to disk"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def register_model(
        self,
        model_type: ModelType,
        model_path: str,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        status: ModelStatus = ModelStatus.CANDIDATE
    ) -> str:
        """
        Register a new model version

        Args:
            model_type: Type of model (embedder or reranker)
            model_path: Path to saved model
            version: Version string (auto-generated if None)
            metrics: Performance metrics
            training_config: Training configuration
            status: Initial status

        Returns:
            Version string
        """
        # Generate version if not provided
        if version is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            version = f"{model_type.value}-{timestamp}"

        # Get registry for model type
        type_key = f"{model_type.value}s"
        if type_key not in self.registry:
            self.registry[type_key] = {}

        # Register model
        self.registry[type_key][version] = {
            "version": version,
            "model_path": model_path,
            "status": status.value,
            "metrics": metrics or {},
            "training_config": training_config or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        self._save_registry()
        logger.info(f"Registered {model_type.value} model: {version}")

        return version

    def get_model(
        self,
        model_type: ModelType,
        version: Optional[str] = None,
        status: Optional[ModelStatus] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get model metadata

        Args:
            model_type: Type of model
            version: Specific version (if None, returns active model)
            status: Filter by status

        Returns:
            Model metadata or None
        """
        type_key = f"{model_type.value}s"

        if type_key not in self.registry:
            return None

        models = self.registry[type_key]

        # Get specific version
        if version:
            return models.get(version)

        # Get model by status
        if status:
            for model in models.values():
                if model["status"] == status.value:
                    return model

        # Default: return active model
        for model in models.values():
            if model["status"] == ModelStatus.ACTIVE.value:
                return model

        return None

    def list_models(
        self,
        model_type: ModelType,
        status: Optional[ModelStatus] = None
    ) -> List[Dict[str, Any]]:
        """
        List all models of a given type

        Args:
            model_type: Type of model
            status: Filter by status

        Returns:
            List of model metadata
        """
        type_key = f"{model_type.value}s"

        if type_key not in self.registry:
            return []

        models = list(self.registry[type_key].values())

        # Filter by status
        if status:
            models = [m for m in models if m["status"] == status.value]

        # Sort by created_at (newest first)
        models.sort(key=lambda x: x["created_at"], reverse=True)

        return models

    def update_status(
        self,
        model_type: ModelType,
        version: str,
        status: ModelStatus
    ):
        """
        Update model status

        Args:
            model_type: Type of model
            version: Model version
            status: New status
        """
        type_key = f"{model_type.value}s"

        if type_key not in self.registry:
            raise ValueError(f"No {model_type.value} models registered")

        if version not in self.registry[type_key]:
            raise ValueError(f"Model version {version} not found")

        # If setting to active, deactivate other models
        if status == ModelStatus.ACTIVE:
            for v, model in self.registry[type_key].items():
                if model["status"] == ModelStatus.ACTIVE.value:
                    self.registry[type_key][v]["status"] = ModelStatus.ARCHIVED.value
                    self.registry[type_key][v]["updated_at"] = datetime.utcnow().isoformat()

        # Update status
        self.registry[type_key][version]["status"] = status.value
        self.registry[type_key][version]["updated_at"] = datetime.utcnow().isoformat()

        self._save_registry()
        logger.info(f"Updated {model_type.value} {version} status to {status.value}")

    def update_metrics(
        self,
        model_type: ModelType,
        version: str,
        metrics: Dict[str, float]
    ):
        """
        Update model metrics

        Args:
            model_type: Type of model
            version: Model version
            metrics: New metrics
        """
        type_key = f"{model_type.value}s"

        if type_key not in self.registry:
            raise ValueError(f"No {model_type.value} models registered")

        if version not in self.registry[type_key]:
            raise ValueError(f"Model version {version} not found")

        self.registry[type_key][version]["metrics"].update(metrics)
        self.registry[type_key][version]["updated_at"] = datetime.utcnow().isoformat()

        self._save_registry()
        logger.info(f"Updated {model_type.value} {version} metrics")

    def promote_model(
        self,
        model_type: ModelType,
        version: str,
        safety_check: bool = True
    ) -> bool:
        """
        Promote model to active status with safety checks

        Args:
            model_type: Type of model
            version: Model version to promote
            safety_check: Perform safety checks before promotion

        Returns:
            True if promoted successfully
        """
        type_key = f"{model_type.value}s"

        if version not in self.registry[type_key]:
            raise ValueError(f"Model version {version} not found")

        model = self.registry[type_key][version]

        # Safety checks
        if safety_check:
            # Check if model has metrics
            if not model.get("metrics"):
                logger.warning(f"Model {version} has no metrics")
                return False

            # Check if metrics meet minimum thresholds
            metrics = model["metrics"]
            if model_type == ModelType.EMBEDDER:
                if metrics.get("ndcg@5", 0) < 0.5:
                    logger.warning(f"Model {version} nDCG@5 below threshold")
                    return False

            elif model_type == ModelType.RERANKER:
                if metrics.get("mrr", 0) < 0.5:
                    logger.warning(f"Model {version} MRR below threshold")
                    return False

        # Promote to active
        self.update_status(model_type, version, ModelStatus.ACTIVE)
        logger.info(f"Promoted {model_type.value} {version} to active")

        return True

    def rollback(
        self,
        model_type: ModelType,
        to_version: Optional[str] = None
    ) -> str:
        """
        Rollback to previous model version

        Args:
            model_type: Type of model
            to_version: Specific version to rollback to (if None, uses last archived)

        Returns:
            Version that was activated
        """
        type_key = f"{model_type.value}s"

        if to_version:
            # Rollback to specific version
            self.update_status(model_type, to_version, ModelStatus.ACTIVE)
            return to_version

        # Find last archived model
        models = self.list_models(model_type)
        for model in models:
            if model["status"] == ModelStatus.ARCHIVED.value:
                version = model["version"]
                self.update_status(model_type, version, ModelStatus.ACTIVE)
                logger.info(f"Rolled back to {model_type.value} {version}")
                return version

        raise ValueError(f"No archived {model_type.value} models found for rollback")

    def cleanup_old_versions(
        self,
        model_type: ModelType,
        keep_count: int = 5
    ):
        """
        Archive old model versions, keeping only recent ones

        Args:
            model_type: Type of model
            keep_count: Number of recent versions to keep
        """
        type_key = f"{model_type.value}s"
        models = self.list_models(model_type)

        # Skip active and shadow models
        archived_models = [
            m for m in models
            if m["status"] not in [ModelStatus.ACTIVE.value, ModelStatus.SHADOW.value]
        ]

        # Archive old versions
        if len(archived_models) > keep_count:
            for model in archived_models[keep_count:]:
                version = model["version"]
                self.update_status(model_type, version, ModelStatus.ARCHIVED)

        logger.info(f"Cleaned up old {model_type.value} versions")


# Global registry instance
_global_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global model registry instance"""
    return _global_registry
