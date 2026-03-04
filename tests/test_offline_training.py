"""Tests for OfflineTrainer and DataPipeline without requiring EasyRec."""
import sys
import os
import json
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline.training import OfflineTrainer
from offline.data_pipeline import DataPipeline


class TestOfflineTrainerNoConfig(unittest.TestCase):
    """OfflineTrainer tests that do not require EasyRec."""

    def setUp(self):
        # No pipeline_config_path → skipped status
        self.trainer = OfflineTrainer()

    def test_train_without_config_returns_skipped(self):
        """train() returns 'skipped' when no pipeline_config_path provided."""
        result = self.trainer.train(model_dir='/tmp/model')
        self.assertEqual(result['status'], 'skipped')

    def test_export_model_without_config_returns_skipped(self):
        """export_model() returns 'skipped' when no pipeline_config_path."""
        result = self.trainer.export_model(model_dir='/tmp/model', export_dir='/tmp/export')
        self.assertEqual(result['status'], 'skipped')

    def test_get_training_status_nonexistent_job(self):
        """get_training_status() returns not_found for unknown job_id."""
        result = self.trainer.get_training_status('nonexistent_job_id')
        self.assertEqual(result['status'], 'not_found')

    def test_train_stores_job_status(self):
        """After train(), the job is stored in training_jobs."""
        result = self.trainer.train(model_dir='/tmp/model')
        job_id = result['job_id']
        stored = self.trainer.get_training_status(job_id)
        self.assertEqual(stored['status'], 'skipped')

    def test_train_ranking_model_returns_success(self):
        """train_ranking_model() with dummy data returns success."""
        dummy_data = [{'feature': i} for i in range(10)]
        result = self.trainer.train_ranking_model(dummy_data)
        self.assertEqual(result['status'], 'success')


class TestDataPipeline(unittest.TestCase):
    """DataPipeline tests with temporary files."""

    def setUp(self):
        self.pipeline = DataPipeline()
        self.tmp_dir = tempfile.mkdtemp()

    def _write_jsonlines(self, records, filename='raw_data.jsonl'):
        path = os.path.join(self.tmp_dir, filename)
        with open(path, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        return path

    def test_generate_csv_training_data_creates_file(self):
        """generate_csv_training_data() creates the output CSV file."""
        records = [
            {'user_id': 'u1', 'item_id': 'i1', 'label': 1},
            {'user_id': 'u2', 'item_id': 'i2', 'label': 0},
        ]
        raw_path = self._write_jsonlines(records)
        out_path = os.path.join(self.tmp_dir, 'output.csv')
        result = self.pipeline.generate_csv_training_data(raw_path, out_path)
        self.assertEqual(result['status'], 'success')
        self.assertTrue(os.path.exists(out_path))

    def test_generate_csv_training_data_correct_row_count(self):
        """generate_csv_training_data() writes correct number of rows."""
        records = [{'user_id': f'u{i}', 'item_id': f'i{i}'} for i in range(5)]
        raw_path = self._write_jsonlines(records)
        out_path = os.path.join(self.tmp_dir, 'output.csv')
        result = self.pipeline.generate_csv_training_data(raw_path, out_path)
        self.assertEqual(result['rows_written'], 5)

    def test_generate_csv_training_data_missing_file_returns_failed(self):
        """generate_csv_training_data() returns 'failed' for missing input."""
        result = self.pipeline.generate_csv_training_data(
            '/nonexistent/path.jsonl', '/tmp/out.csv'
        )
        self.assertEqual(result['status'], 'failed')

    def test_validate_training_data_valid_csv(self):
        """validate_training_data() returns 'valid' for a well-formed CSV."""
        import csv
        csv_path = os.path.join(self.tmp_dir, 'train.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['user_id', 'item_id', 'label'])
            writer.writeheader()
            writer.writerow({'user_id': 'u1', 'item_id': 'i1', 'label': '1'})
        result = self.pipeline.validate_training_data(csv_path)
        self.assertEqual(result['status'], 'valid')

    def test_validate_training_data_missing_file(self):
        """validate_training_data() returns 'invalid' for missing file."""
        result = self.pipeline.validate_training_data('/nonexistent/file.csv')
        self.assertEqual(result['status'], 'invalid')

    def test_validate_training_data_empty_csv(self):
        """validate_training_data() returns 'invalid' for empty CSV."""
        csv_path = os.path.join(self.tmp_dir, 'empty.csv')
        with open(csv_path, 'w') as f:
            f.write('')
        result = self.pipeline.validate_training_data(csv_path)
        self.assertEqual(result['status'], 'invalid')


if __name__ == '__main__':
    unittest.main()
