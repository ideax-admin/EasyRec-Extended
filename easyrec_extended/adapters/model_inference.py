"""
EasyRec Model Inference - online serving using TF SavedModel.

Loads an EasyRec exported SavedModel and provides predict / batch_predict
methods for low-latency online inference. Supports hot model reload.
"""
import logging
import threading
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class EasyRecModelInference:
    """Online inference engine backed by an EasyRec exported SavedModel."""

    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the inference engine.

        Args:
            model_dir: Path to the EasyRec exported SavedModel directory.
                       If None, no model is loaded at construction time.
        """
        self._model_dir = model_dir
        self._model = None
        self._infer_fn = None
        self._lock = threading.RLock()
        self._loaded = False

        if model_dir:
            self._load(model_dir)

    def _load(self, model_dir: str):
        """Load a SavedModel from the given directory (internal, not thread-safe)."""
        try:
            import tensorflow as tf
            logger.info(f"Loading SavedModel from {model_dir}")
            self._model = tf.saved_model.load(model_dir)
            self._infer_fn = self._model.signatures.get('serving_default')
            if self._infer_fn is None and hasattr(self._model, '__call__'):
                self._infer_fn = self._model
            self._model_dir = model_dir
            self._loaded = True
            logger.info("SavedModel loaded successfully")
        except ImportError:
            logger.warning("TensorFlow not installed; model inference unavailable")
        except Exception as e:
            logger.error(f"Failed to load SavedModel from {model_dir}: {e}")
            raise

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference for a single feature dict.

        Args:
            features: Dict mapping feature name to value (scalar or list).

        Returns:
            Dict of output tensor names to numpy arrays.
            Returns empty dict when no model is loaded.
        """
        with self._lock:
            if not self._loaded or self._infer_fn is None:
                logger.warning("No model loaded; returning empty prediction")
                return {}
            try:
                import tensorflow as tf
                tensor_inputs = {k: tf.constant([v]) for k, v in features.items()}
                outputs = self._infer_fn(**tensor_inputs)
                return {k: v.numpy() for k, v in outputs.items()}
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                return {}

    def batch_predict(self, feature_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run inference for a list of feature dicts.

        Args:
            feature_batch: List of feature dicts.

        Returns:
            List of prediction dicts corresponding to each input.
        """
        with self._lock:
            if not self._loaded or self._infer_fn is None:
                logger.warning("No model loaded; returning empty batch prediction")
                return [{} for _ in feature_batch]
            try:
                import tensorflow as tf
                if not feature_batch:
                    return []
                keys = list(feature_batch[0].keys())
                tensor_inputs = {
                    k: tf.constant([sample[k] for sample in feature_batch])
                    for k in keys
                }
                outputs = self._infer_fn(**tensor_inputs)
                numpy_outputs = {k: v.numpy() for k, v in outputs.items()}
                results = []
                for i in range(len(feature_batch)):
                    results.append({k: v[i] for k, v in numpy_outputs.items()})
                return results
            except Exception as e:
                logger.error(f"Batch prediction failed: {e}")
                return [{} for _ in feature_batch]

    def reload(self, new_model_dir: str):
        """
        Hot-reload the model from a new directory (thread-safe).

        Args:
            new_model_dir: Path to the new SavedModel directory.
        """
        logger.info(f"Reloading model from {new_model_dir}")
        with self._lock:
            self._loaded = False
            self._load(new_model_dir)
        logger.info("Model reloaded successfully")

    @property
    def is_loaded(self) -> bool:
        """Return True if a model is currently loaded."""
        return self._loaded

    @property
    def model_dir(self) -> Optional[str]:
        """Return the directory of the currently loaded model."""
        return self._model_dir
