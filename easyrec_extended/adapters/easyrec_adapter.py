"""
EasyRec Adapter - wraps EasyRec's core training and evaluation APIs.

This module provides a bridge between EasyRec's pipeline.config-driven
workflow and the EasyRec-Extended serving framework.
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class EasyRecAdapter:
    """Adapter that wraps EasyRec's core APIs for training, export, evaluation, and prediction."""

    def __init__(self, pipeline_config_path: str):
        """
        Initialize the adapter with an EasyRec pipeline config file.

        Args:
            pipeline_config_path: Path to the protobuf pipeline.config file.
        """
        self.pipeline_config_path = pipeline_config_path
        self._pipeline_config = None
        logger.info(f"EasyRecAdapter initialized with config: {pipeline_config_path}")

    def load_config(self):
        """Load and parse the pipeline config using EasyRec's config_util."""
        try:
            from easy_rec.python.utils import config_util
            self._pipeline_config = config_util.get_configs_from_pipeline_file(
                self.pipeline_config_path
            )
            logger.info("Pipeline config loaded successfully")
            return self._pipeline_config
        except ImportError:
            logger.warning("easy_rec not installed; cannot load pipeline config")
            return None
        except Exception as e:
            logger.error(f"Failed to load pipeline config: {e}")
            raise

    def train(self, model_dir: str, **kwargs):
        """
        Train the model using EasyRec's train_and_evaluate().

        Args:
            model_dir: Directory to save checkpoints.
        """
        try:
            from easy_rec.python import main as easyrec_main
            logger.info(f"Starting EasyRec training, model_dir={model_dir}")
            easyrec_main.train_and_evaluate(
                pipeline_config_path=self.pipeline_config_path,
                model_dir=model_dir,
                **kwargs
            )
            logger.info("EasyRec training completed")
        except ImportError:
            logger.warning("easy_rec not installed; training skipped")
        except Exception as e:
            logger.error(f"EasyRec training failed: {e}")
            raise

    def export_model(self, model_dir: str, export_dir: str, **kwargs):
        """
        Export a trained model to SavedModel format.

        Args:
            model_dir: Directory with trained checkpoints.
            export_dir: Directory to export the SavedModel.
        """
        try:
            from easy_rec.python import main as easyrec_main
            logger.info(f"Exporting model from {model_dir} to {export_dir}")
            easyrec_main.export(
                pipeline_config_path=self.pipeline_config_path,
                model_dir=model_dir,
                export_dir=export_dir,
                **kwargs
            )
            logger.info("Model export completed")
        except ImportError:
            logger.warning("easy_rec not installed; export skipped")
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            raise

    def evaluate_model(self, model_dir: str, **kwargs):
        """
        Evaluate a trained model.

        Args:
            model_dir: Directory with trained checkpoints.
        """
        try:
            from easy_rec.python import main as easyrec_main
            logger.info(f"Evaluating model at {model_dir}")
            easyrec_main.evaluate(
                pipeline_config_path=self.pipeline_config_path,
                model_dir=model_dir,
                **kwargs
            )
            logger.info("Model evaluation completed")
        except ImportError:
            logger.warning("easy_rec not installed; evaluation skipped")
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise

    def predict(self, model_dir: str, output_path: str, **kwargs):
        """
        Run offline batch prediction using EasyRec.

        Args:
            model_dir: Directory with trained checkpoints.
            output_path: Path to write prediction results.
        """
        try:
            from easy_rec.python import main as easyrec_main
            logger.info(f"Running batch prediction, output={output_path}")
            easyrec_main.predict(
                pipeline_config_path=self.pipeline_config_path,
                model_dir=model_dir,
                output_path=output_path,
                **kwargs
            )
            logger.info("Batch prediction completed")
        except ImportError:
            logger.warning("easy_rec not installed; prediction skipped")
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise

    @property
    def pipeline_config(self):
        """Return the loaded pipeline config, loading it on first access."""
        if self._pipeline_config is None:
            self.load_config()
        return self._pipeline_config
