"""
EasyRec Config Bridge - maps EasyRec protobuf pipeline config to Python dicts.

Wraps the protobuf-based EasyRecConfig so that the rest of the serving
framework can work with plain Python structures without depending on
protobuf at every call site.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EasyRecConfigBridge:
    """
    Bridge between EasyRec's protobuf pipeline_config and Python config dicts.

    Usage::

        bridge = EasyRecConfigBridge.from_file('config/pipeline.config')
        print(bridge.feature_configs)
        print(bridge.model_config)
    """

    def __init__(self, pipeline_config=None):
        """
        Args:
            pipeline_config: A protobuf EasyRecConfig message, or None.
        """
        self._pipeline_config = pipeline_config

    @classmethod
    def from_file(cls, pipeline_config_path: str) -> 'EasyRecConfigBridge':
        """
        Load a pipeline config from file and return a new bridge instance.

        Args:
            pipeline_config_path: Path to the protobuf pipeline.config file.

        Returns:
            EasyRecConfigBridge instance.
        """
        try:
            from easy_rec.python.utils import config_util
            pipeline_config = config_util.get_configs_from_pipeline_file(
                pipeline_config_path
            )
            logger.info(f"Loaded pipeline config from {pipeline_config_path}")
            return cls(pipeline_config)
        except ImportError:
            logger.warning("easy_rec not installed; returning empty config bridge")
            return cls(None)
        except Exception as e:
            logger.error(f"Failed to load pipeline config: {e}")
            raise

    @property
    def pipeline_config(self):
        """Return the raw protobuf pipeline config object."""
        return self._pipeline_config

    @property
    def feature_configs(self) -> List[Dict[str, Any]]:
        """
        Return feature configs as a list of plain dicts.

        Each dict has at minimum ``name`` and ``feature_type`` keys.
        Returns an empty list if no config is loaded.
        """
        if self._pipeline_config is None:
            return []
        try:
            result = []
            for fc in self._pipeline_config.feature_config.features:
                result.append({
                    'name': fc.input_names[0] if fc.input_names else '',
                    'feature_type': fc.feature_type,
                })
            return result
        except Exception as e:
            logger.warning(f"Could not extract feature_configs: {e}")
            return []

    @property
    def model_config(self) -> Dict[str, Any]:
        """
        Return the model config as a plain dict.

        Returns an empty dict if no config is loaded.
        """
        if self._pipeline_config is None:
            return {}
        try:
            mc = self._pipeline_config.model_config
            return {
                'model_class': mc.WhichOneof('model') if hasattr(mc, 'WhichOneof') else str(type(mc)),
            }
        except Exception as e:
            logger.warning(f"Could not extract model_config: {e}")
            return {}

    @property
    def data_config(self) -> Dict[str, Any]:
        """
        Return the data config as a plain dict.

        Returns an empty dict if no config is loaded.
        """
        if self._pipeline_config is None:
            return {}
        try:
            dc = self._pipeline_config.data_config
            return {
                'input_fields': [f.input_name for f in dc.input_fields],
                'label_fields': list(dc.label_fields),
            }
        except Exception as e:
            logger.warning(f"Could not extract data_config: {e}")
            return {}

    @property
    def train_config(self) -> Dict[str, Any]:
        """Return train config as a plain dict."""
        if self._pipeline_config is None:
            return {}
        try:
            tc = self._pipeline_config.train_config
            return {
                'num_steps': tc.num_steps,
                'log_step_count_steps': getattr(tc, 'log_step_count_steps', 100),
            }
        except Exception as e:
            logger.warning(f"Could not extract train_config: {e}")
            return {}
