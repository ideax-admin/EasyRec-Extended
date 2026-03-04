"""
Data Pipeline Module for EasyRec-Extended.

Helps convert raw business data into the CSV format expected by EasyRec's
CSVInput data reader, and validates that training data matches the schema
described in a pipeline.config file.
"""
import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DataPipeline:
    """Converts raw business data to EasyRec-compatible training format.

    The pipeline reads raw interaction records (user–item pairs with optional
    labels and context), applies lightweight preprocessing, and writes output
    CSV files that can be consumed directly by EasyRec's ``CSVInput``.

    Args:
        pipeline_config_path: Optional path to an EasyRec protobuf
            ``pipeline.config`` file.  Used by :meth:`validate_training_data`
            to check schema conformance.
    """

    def __init__(self, pipeline_config_path: Optional[str] = None):
        """Initialise the data pipeline.

        Args:
            pipeline_config_path: Path to EasyRec pipeline.config (optional).
        """
        self.pipeline_config_path = pipeline_config_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_csv_training_data(
        self,
        raw_data_path: str,
        output_path: str,
        pipeline_config_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read raw data and write a CSV file compatible with EasyRec's CSVInput.

        The raw data file is expected to be a JSON-Lines file where each line
        contains a flat dict of features.  The output CSV uses the dict keys
        as column headers.

        Args:
            raw_data_path: Path to the input JSON-Lines file.
            output_path: Destination path for the output CSV file.
            pipeline_config_path: Optional override for the instance-level
                pipeline config path.  Reserved for future schema-ordered
                column output; currently unused.

        Returns:
            Dict with ``status``, ``rows_written``, and ``output_path``.
        """
        if not os.path.exists(raw_data_path):
            logger.error(f"Raw data file not found: {raw_data_path}")
            return {'status': 'failed', 'error': f"File not found: {raw_data_path}"}

        try:
            records = self._read_jsonlines(raw_data_path)
            if not records:
                logger.warning("No records found in raw data file")
                return {'status': 'skipped', 'rows_written': 0, 'output_path': output_path}

            fieldnames = list(records[0].keys())
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            rows_written = 0
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in records:
                    writer.writerow({k: record.get(k, '') for k in fieldnames})
                    rows_written += 1

            logger.info(f"Wrote {rows_written} rows to {output_path}")
            return {'status': 'success', 'rows_written': rows_written, 'output_path': output_path}

        except Exception as e:
            logger.error(f"generate_csv_training_data failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def validate_training_data(
        self,
        data_path: str,
        pipeline_config_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate that a training CSV matches the expected schema.

        When a pipeline.config is available the method attempts to parse the
        expected column names from the config (best-effort; full protobuf
        parsing requires EasyRec to be installed).  Without a config the
        method performs basic sanity checks (non-empty, parseable CSV).

        Args:
            data_path: Path to the CSV file to validate.
            pipeline_config_path: Optional override for the instance-level
                pipeline config path.

        Returns:
            Dict with ``status`` (``'valid'``, ``'invalid'``, or
            ``'skipped'``), ``rows``, and an optional ``errors`` list.
        """
        config_path = pipeline_config_path or self.pipeline_config_path

        if not os.path.exists(data_path):
            return {'status': 'invalid', 'errors': [f"File not found: {data_path}"]}

        errors: List[str] = []
        rows = 0

        try:
            with open(data_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                headers = reader.fieldnames or []
                if not headers:
                    errors.append("CSV file has no header row")
                for row in reader:
                    rows += 1

            if rows == 0:
                errors.append("CSV file contains no data rows")

            # Best-effort schema check when pipeline config is provided
            if config_path and os.path.exists(config_path):
                expected = self._parse_expected_columns(config_path)
                if expected:
                    missing = set(expected) - set(headers)
                    if missing:
                        errors.append(f"Missing expected columns: {sorted(missing)}")

        except Exception as e:
            errors.append(f"Validation error: {e}")

        status = 'valid' if not errors else 'invalid'
        result: Dict[str, Any] = {'status': status, 'rows': rows}
        if errors:
            result['errors'] = errors
        logger.info(f"Validation result for {data_path}: {status} ({rows} rows)")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_jsonlines(self, path: str) -> List[Dict[str, Any]]:
        """Read a JSON-Lines file and return a list of dicts."""
        records = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _parse_expected_columns(self, config_path: str) -> List[str]:
        """Attempt to extract expected column names from an EasyRec config.

        Falls back to an empty list when EasyRec / protobuf is not installed.

        Args:
            config_path: Path to the pipeline.config file.

        Returns:
            List of expected column name strings (may be empty).
        """
        try:
            from easyrec_extended.adapters.config_bridge import ConfigBridge
            bridge = ConfigBridge(config_path)
            return bridge.get_feature_names() if hasattr(bridge, 'get_feature_names') else []
        except Exception:
            pass
        return []
