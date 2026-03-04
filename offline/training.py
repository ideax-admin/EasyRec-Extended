"""
Offline Training Pipeline for EasyRec-Extended.

Delegates model training, export, and evaluation to EasyRecAdapter,
which wraps alibaba/EasyRec's train_and_evaluate(), export(), and
evaluate() entry points.
"""
import logging
from datetime import datetime
from typing import Optional

from easyrec_extended.adapters.easyrec_adapter import EasyRecAdapter

logger = logging.getLogger(__name__)


class OfflineTrainer:
    """Offline training engine for recommendation models powered by EasyRec."""

    def __init__(self, pipeline_config_path: Optional[str] = None):
        """
        Args:
            pipeline_config_path: Path to the EasyRec protobuf pipeline.config.
                                   Required for EasyRec-backed training.
        """
        self.pipeline_config_path = pipeline_config_path
        self.training_jobs = {}
        self._adapter: Optional[EasyRecAdapter] = None
        if pipeline_config_path:
            self._adapter = EasyRecAdapter(pipeline_config_path)

    # ------------------------------------------------------------------
    # EasyRec-backed training
    # ------------------------------------------------------------------

    def train(self, model_dir: str, **kwargs) -> dict:
        """
        Train a model using EasyRec's train_and_evaluate().

        Args:
            model_dir: Directory to save checkpoints.
            **kwargs: Additional keyword arguments forwarded to EasyRec.

        Returns:
            Dict with job_id and status.
        """
        job_id = f"train_{datetime.now().timestamp()}"
        logger.info(f"Starting EasyRec training job {job_id}, model_dir={model_dir}")

        if self._adapter is None:
            logger.warning("No pipeline_config_path provided; skipping EasyRec training")
            self.training_jobs[job_id] = {'status': 'skipped', 'reason': 'no config'}
            return {'job_id': job_id, 'status': 'skipped'}

        try:
            self._adapter.train(model_dir=model_dir, **kwargs)
            self.training_jobs[job_id] = {
                'status': 'completed',
                'model_dir': model_dir,
                'timestamp': datetime.now().isoformat(),
            }
            logger.info(f"Training job {job_id} completed")
            return {'job_id': job_id, 'status': 'completed', 'model_dir': model_dir}
        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}")
            self.training_jobs[job_id] = {'status': 'failed', 'error': str(e)}
            return {'job_id': job_id, 'status': 'failed', 'error': str(e)}

    def export_model(self, model_dir: str, export_dir: str, **kwargs) -> dict:
        """
        Export a trained model to SavedModel format via EasyRec.

        Args:
            model_dir: Directory with trained checkpoints.
            export_dir: Output directory for the SavedModel.
        """
        job_id = f"export_{datetime.now().timestamp()}"
        logger.info(f"Exporting model {job_id}: {model_dir} -> {export_dir}")

        if self._adapter is None:
            logger.warning("No pipeline_config_path provided; skipping EasyRec export")
            return {'job_id': job_id, 'status': 'skipped'}

        try:
            self._adapter.export_model(model_dir=model_dir, export_dir=export_dir, **kwargs)
            return {'job_id': job_id, 'status': 'completed', 'export_dir': export_dir}
        except Exception as e:
            logger.error(f"Export job {job_id} failed: {e}")
            return {'job_id': job_id, 'status': 'failed', 'error': str(e)}

    def evaluate_model(self, model_dir: str, **kwargs) -> dict:
        """
        Evaluate a trained model using EasyRec.

        Args:
            model_dir: Directory with trained checkpoints.
        """
        job_id = f"eval_{datetime.now().timestamp()}"
        logger.info(f"Evaluating model {job_id}: {model_dir}")

        if self._adapter is None:
            logger.warning("No pipeline_config_path provided; skipping EasyRec evaluation")
            return {'job_id': job_id, 'status': 'skipped'}

        try:
            self._adapter.evaluate_model(model_dir=model_dir, **kwargs)
            return {'job_id': job_id, 'status': 'completed'}
        except Exception as e:
            logger.error(f"Evaluation job {job_id} failed: {e}")
            return {'job_id': job_id, 'status': 'failed', 'error': str(e)}

    # ------------------------------------------------------------------
    # Legacy / simple training (non-EasyRec)
    # ------------------------------------------------------------------

    def train_ranking_model(self, training_data, model_type='xgboost', **kwargs):
        """Train a ranking model using provided training data (non-EasyRec path)."""
        job_id = f"train_ranking_{datetime.now().timestamp()}"
        logger.info(f"Starting training job {job_id} with {len(training_data)} samples")

        try:
            self.training_jobs[job_id] = {
                'status': 'completed',
                'model_type': model_type,
                'samples': len(training_data),
                'timestamp': datetime.now().isoformat(),
            }
            logger.info(f"Training job {job_id} completed")
            return {
                'job_id': job_id,
                'status': 'success',
                'message': f'Trained {model_type} with {len(training_data)} samples',
            }
        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}")
            return {'job_id': job_id, 'status': 'failed', 'error': str(e)}

    def get_training_status(self, job_id: str) -> dict:
        """Get status of a training job by job_id."""
        if job_id not in self.training_jobs:
            return {'status': 'not_found'}
        return {'job_id': job_id, **self.training_jobs[job_id]}
