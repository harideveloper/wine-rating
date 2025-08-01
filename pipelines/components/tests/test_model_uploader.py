"""Tests for model uploader component."""

import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest
from kfp.dsl import Model

from pipelines.components.model_uploader import save_model

save_model_func = save_model.python_func


def test_save_model_success():
    """Test successful model saving"""
    with tempfile.TemporaryDirectory() as temp_dir:
        source_dir = os.path.join(temp_dir, "source")
        os.makedirs(source_dir)
        source_file = os.path.join(source_dir, "model.joblib")
        model_data = b"mock_model_data_12345"
        with open(source_file, "wb") as f:
            f.write(model_data)

        target_dir = os.path.join(temp_dir, "target")
        target_file = os.path.join(target_dir, "model.joblib")
        mock_evaluated_model = MagicMock(spec=Model)
        mock_evaluated_model.path = os.path.join(source_dir, "model")
        mock_evaluated_model.metadata = {
            "framework": "sklearn",
            "quality_score": "0.8500",
            "r2_score": "0.9000",
        }

        mock_saved_model = MagicMock(spec=Model)
        mock_saved_model.path = target_file
        mock_saved_model.metadata = {}

        save_model_func(
            evaluated_model=mock_evaluated_model, saved_model=mock_saved_model
        )

        # Verify model file, metadata was copied & saved model URI was set
        assert os.path.exists(target_file)
        with open(target_file, "rb") as f:
            copied_data = f.read()
        assert copied_data == model_data
        assert mock_saved_model.uri == target_dir
        assert mock_saved_model.metadata["framework"] == "sklearn"
        assert mock_saved_model.metadata["quality_score"] == "0.8500"
        assert mock_saved_model.metadata["r2_score"] == "0.9000"


def test_save_model_source_file_not_found():
    """Test error when source model file doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # simulate missing file by not creating a source file
        source_dir = os.path.join(temp_dir, "source")
        target_dir = os.path.join(temp_dir, "target")
        target_file = os.path.join(target_dir, "model.joblib")
        mock_evaluated_model = MagicMock(spec=Model)
        mock_evaluated_model.path = os.path.join(source_dir, "nonexistent_model")
        mock_evaluated_model.metadata = {}
        mock_saved_model = MagicMock(spec=Model)
        mock_saved_model.path = target_file
        mock_saved_model.metadata = {}
        with pytest.raises(FileNotFoundError):
            save_model_func(
                evaluated_model=mock_evaluated_model, saved_model=mock_saved_model
            )


def test_save_model_with_metadata_errors():
    """Test model saving with metadata errors"""

    # Storage for successful metadata
    stored_metadata = {}

    def mock_setitem(key, value):
        if key == "error_key":
            raise TypeError("Cannot set this metadata key")
        stored_metadata[key] = value

    def mock_getitem(key):
        return stored_metadata[key]

    def mock_contains(key):
        return key in stored_metadata

    with tempfile.TemporaryDirectory() as temp_dir:
        source_dir = os.path.join(temp_dir, "source")
        os.makedirs(source_dir)
        source_file = os.path.join(source_dir, "model.joblib")
        model_data = b"test_model_with_metadata_errors"
        with open(source_file, "wb") as f:
            f.write(model_data)
        target_dir = os.path.join(temp_dir, "target")
        target_file = os.path.join(target_dir, "model.joblib")

        mock_evaluated_model = MagicMock(spec=Model)
        mock_evaluated_model.path = os.path.join(source_dir, "model")
        mock_evaluated_model.metadata = {
            "valid_key": "valid_value",
            "error_key": "error_value",
        }

        mock_saved_model = MagicMock(spec=Model)
        mock_saved_model.path = target_file
        mock_metadata = MagicMock()
        mock_metadata.__setitem__.side_effect = mock_setitem
        mock_metadata.__getitem__.side_effect = mock_getitem
        mock_metadata.__contains__.side_effect = mock_contains
        mock_saved_model.metadata = mock_metadata

        save_model_func(
            evaluated_model=mock_evaluated_model, saved_model=mock_saved_model
        )

        # Verify metadata & the file was still copied despite metadata errors
        assert os.path.exists(target_file)
        assert stored_metadata["valid_key"] == "valid_value"
        assert "error_key" not in stored_metadata


def test_save_model_directory_creation_failure():
    """Test error when target directory creation fails."""
    with tempfile.TemporaryDirectory() as temp_dir:
        source_dir = os.path.join(temp_dir, "source")
        os.makedirs(source_dir)
        source_file = os.path.join(source_dir, "model.joblib")
        with open(source_file, "wb") as f:
            f.write(b"test_data")
        target_dir = os.path.join(temp_dir, "target")
        target_file = os.path.join(target_dir, "model.joblib")
        mock_evaluated_model = MagicMock(spec=Model)
        mock_evaluated_model.path = os.path.join(source_dir, "model")
        mock_evaluated_model.metadata = {}
        mock_saved_model = MagicMock(spec=Model)
        mock_saved_model.path = target_file
        mock_saved_model.metadata = {}

        with patch("os.makedirs", side_effect=OSError("Permission denied")):
            with pytest.raises(OSError, match="Permission denied"):
                save_model_func(
                    evaluated_model=mock_evaluated_model, saved_model=mock_saved_model
                )


def test_save_model_empty_metadata():
    """Test model saving when evaluated model has empty metadata."""
    with tempfile.TemporaryDirectory() as temp_dir:
        source_dir = os.path.join(temp_dir, "source")
        os.makedirs(source_dir)
        source_file = os.path.join(source_dir, "model.joblib")
        model_data = b"test_model_no_metadata"
        with open(source_file, "wb") as f:
            f.write(model_data)
        target_dir = os.path.join(temp_dir, "target")
        target_file = os.path.join(target_dir, "model.joblib")
        mock_evaluated_model = MagicMock(spec=Model)
        mock_evaluated_model.path = os.path.join(source_dir, "model")
        mock_evaluated_model.metadata = {}
        mock_saved_model = MagicMock(spec=Model)
        mock_saved_model.path = target_file
        mock_saved_model.metadata = {}
        save_model_func(
            evaluated_model=mock_evaluated_model, saved_model=mock_saved_model
        )
        assert os.path.exists(target_file)
        assert mock_saved_model.uri == target_dir
