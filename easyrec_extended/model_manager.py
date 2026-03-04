"""
Model Manager - multi-version model lifecycle management.

Supports:
- Loading multiple model versions simultaneously
- Thread-safe active version switching for A/B testing
- Hot model reload without downtime
"""
import logging
import threading
from typing import Dict, Optional

from easyrec_extended.adapters.model_inference import EasyRecModelInference

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages multiple EasyRec model versions and provides thread-safe
    access for online inference.

    Example::

        manager = ModelManager()
        manager.load_version('v1', '/models/v1')
        manager.load_version('v2', '/models/v2')
        manager.set_active_version('v1')
        result = manager.predict({'feature_a': 1.0})
        manager.set_active_version('v2')   # live traffic switch
    """

    def __init__(self):
        self._versions: Dict[str, EasyRecModelInference] = {}
        self._active_version: Optional[str] = None
        self._lock = threading.RLock()

    def load_version(self, version: str, model_dir: str):
        """
        Load a model version from the given SavedModel directory.

        Args:
            version: A string label for this version (e.g. 'v1', 'prod').
            model_dir: Path to the EasyRec exported SavedModel directory.
        """
        logger.info(f"Loading model version '{version}' from {model_dir}")
        inference = EasyRecModelInference(model_dir)
        with self._lock:
            self._versions[version] = inference
            if self._active_version is None:
                self._active_version = version
        self._update_metrics()
        logger.info(f"Model version '{version}' loaded (active={self._active_version})")

    def set_active_version(self, version: str):
        """
        Switch the active model version (thread-safe).

        Args:
            version: The version label to activate.

        Raises:
            ValueError: If the version has not been loaded.
        """
        with self._lock:
            if version not in self._versions:
                raise ValueError(f"Model version '{version}' is not loaded")
            self._active_version = version
        self._update_metrics()
        logger.info(f"Active model version switched to '{version}'")

    def reload_version(self, version: str, new_model_dir: str):
        """
        Hot-reload an existing version from a new directory (thread-safe).

        Args:
            version: The version label to reload.
            new_model_dir: Path to the new SavedModel directory.
        """
        with self._lock:
            if version not in self._versions:
                raise ValueError(f"Model version '{version}' is not loaded; use load_version first")
            self._versions[version].reload(new_model_dir)
        logger.info(f"Model version '{version}' reloaded from {new_model_dir}")

    def predict(self, features: dict, version: Optional[str] = None) -> dict:
        """
        Run inference using the active (or specified) model version.

        Args:
            features: Feature dict for a single sample.
            version: If provided, use this version instead of the active one.

        Returns:
            Prediction result dict.  Returns empty dict if no model is loaded.
        """
        with self._lock:
            target = version or self._active_version
            if target is None or target not in self._versions:
                logger.warning("No model version available for prediction")
                return {}
            return self._versions[target].predict(features)

    def batch_predict(self, feature_batch: list, version: Optional[str] = None) -> list:
        """
        Run batch inference using the active (or specified) model version.

        Args:
            feature_batch: List of feature dicts.
            version: If provided, use this version instead of the active one.

        Returns:
            List of prediction result dicts.
        """
        with self._lock:
            target = version or self._active_version
            if target is None or target not in self._versions:
                logger.warning("No model version available for batch prediction")
                return [{} for _ in feature_batch]
            return self._versions[target].batch_predict(feature_batch)

    @property
    def active_version(self) -> Optional[str]:
        """Return the label of the currently active model version."""
        return self._active_version

    @property
    def loaded_versions(self) -> list:
        """Return a list of all loaded version labels."""
        with self._lock:
            return list(self._versions.keys())

    def is_ready(self) -> bool:
        """Return True if at least one model version is loaded and active."""
        with self._lock:
            return (
                self._active_version is not None
                and self._active_version in self._versions
                and self._versions[self._active_version].is_loaded
            )

    def _update_metrics(self):
        """Refresh Prometheus gauges for loaded / active version counts."""
        try:
            from easyrec_extended.metrics.prometheus_metrics import (
                model_loaded_versions,
                active_model_version,
            )
            with self._lock:
                n = len(self._versions)
                active = self._active_version or ""
            model_loaded_versions.set(n)
            active_model_version.info({"version": active})
        except Exception:
            pass
