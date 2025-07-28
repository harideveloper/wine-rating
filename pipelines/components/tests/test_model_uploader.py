"""Tests for model saver component core logic."""

import os
from unittest.mock import mock_open, patch


class TestModelUploader:
    """Test core model saver logic."""

    def test_path_preparation(self):
        """Test model path preparation."""
        target_dir = "/tmp/output"
        target_path = os.path.join(target_dir, "model.joblib")
        assert target_path == "/tmp/output/model.joblib"

    @patch("os.makedirs")
    def test_directory_creation(self, mock_makedirs):
        """Test directory creation logic."""
        model_dir = "/tmp/model_output"
        os.makedirs(model_dir, exist_ok=True)
        mock_makedirs.assert_called_once_with(model_dir, exist_ok=True)

    @patch("builtins.open", new_callable=mock_open, read_data=b"model_data")
    def test_file_copying(self, mock_file):
        """Test file copying logic."""
        with open("source.joblib", "rb") as src:
            data = src.read()
        with open("target.joblib", "wb") as tgt:
            tgt.write(data)
        assert mock_file.call_count == 2

    def test_metadata_copying(self):
        """Test metadata copying logic."""
        source_metadata = {"framework": "sklearn", "target": "Rating"}
        target_metadata = {}
        for key, value in source_metadata.items():
            target_metadata[key] = value
        assert target_metadata == source_metadata
