"""Tests for model evaluator component."""

import os
import tempfile
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import pytest
from kfp.v2.dsl import Model, Dataset
from pipelines.components.model_evaluator import evaluate_model

evaluate_model_func = evaluate_model.python_func


def test_evaluate_model_success():
    """Test successful model evaluation."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([4.1, 3.8, 4.5])

    with tempfile.TemporaryDirectory() as temp_dir:
        test_data_path = os.path.join(temp_dir, "test_data.csv")

        # Create real test CSV file
        test_df = pd.DataFrame(
            {
                "price_numeric": [10.0, 20.0, 15.0],
                "Country": ["Italy", "France", "Spain"],
                "Region": ["Tuscany", "Bordeaux", "Rioja"],
                "Type": ["Red", "White", "Red"],
                "Style": ["Bold", "Light", "Bold"],
                "Grape": ["Cabernet", "Chardonnay", "Tempranillo"],
                "Rating": [4.2, 3.8, 4.1],
            }
        )
        test_df.to_csv(test_data_path, index=False)

        mock_trained_model = MagicMock(spec=Model)
        mock_trained_model.path = os.path.join(temp_dir, "model")
        mock_trained_model.uri = "gs://bucket/model"
        mock_trained_model.metadata = {
            "feature_order": "['price_numeric', 'Country', 'Region', 'Type', 'Style', 'Grape']",
            "target": "Rating",
        }

        mock_test_data = MagicMock(spec=Dataset)
        mock_test_data.path = test_data_path

        mock_evaluated_model = MagicMock(spec=Model)
        mock_evaluated_model.metadata = {}
        with patch("joblib.load", return_value=mock_model):
            model_file_path = mock_trained_model.path + ".joblib"
            with open(model_file_path, "wb") as f:
                f.write(b"fake model data")

            quality_score = evaluate_model_func(
                trained_model=mock_trained_model,
                test_data=mock_test_data,
                evaluated_model=mock_evaluated_model,
            )

        assert isinstance(quality_score, float)
        assert 0 <= quality_score <= 1
        assert mock_evaluated_model.metadata["evaluation_status"] == "completed"


def test_evaluate_model_missing_model_file():
    """Test evaluation fails when model file is missing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data_path = os.path.join(temp_dir, "test_data.csv")
        test_df = pd.DataFrame({"Rating": [4.0, 3.5]})
        test_df.to_csv(test_data_path, index=False)

        mock_trained_model = MagicMock(spec=Model)
        mock_trained_model.path = os.path.join(temp_dir, "nonexistent_model")
        mock_trained_model.metadata = {"target": "Rating"}

        mock_test_data = MagicMock(spec=Dataset)
        mock_test_data.path = test_data_path

        mock_evaluated_model = MagicMock(spec=Model)
        mock_evaluated_model.metadata = {}

        with pytest.raises(FileNotFoundError):
            evaluate_model_func(
                trained_model=mock_trained_model,
                test_data=mock_test_data,
                evaluated_model=mock_evaluated_model,
            )


def test_evaluate_model_missing_features():
    """Test evaluation fails when required features are missing."""
    mock_model = MagicMock()

    with tempfile.TemporaryDirectory() as temp_dir:
        test_data_path = os.path.join(temp_dir, "incomplete_data.csv")
        # Create data with only some features
        incomplete_df = pd.DataFrame(
            {
                "price_numeric": [10.0, 20.0],
                "Country": ["Italy", "France"],
                "Rating": [4.0, 3.5],
            }
        )
        incomplete_df.to_csv(test_data_path, index=False)

        mock_trained_model = MagicMock(spec=Model)
        mock_trained_model.path = os.path.join(temp_dir, "model")
        mock_trained_model.metadata = {
            "feature_order": "['price_numeric', 'Country', 'Region', 'Type', 'Style', 'Grape']",
            "target": "Rating",
        }

        mock_test_data = MagicMock(spec=Dataset)
        mock_test_data.path = test_data_path

        mock_evaluated_model = MagicMock(spec=Model)
        mock_evaluated_model.metadata = {}

        with patch("joblib.load", return_value=mock_model):
            model_file_path = mock_trained_model.path + ".joblib"
            with open(model_file_path, "wb") as f:
                f.write(b"fake model data")

            with pytest.raises(ValueError, match="Missing required features"):
                evaluate_model_func(
                    trained_model=mock_trained_model,
                    test_data=mock_test_data,
                    evaluated_model=mock_evaluated_model,
                )


def test_evaluate_model_empty_test_data():
    """Test evaluation fails with empty test dataset."""
    mock_model = MagicMock()

    with tempfile.TemporaryDirectory() as temp_dir:
        test_data_path = os.path.join(temp_dir, "empty_data.csv")
        # Create empty CSV with just headers
        with open(test_data_path, "w", encoding="utf-8") as f:
            f.write("price_numeric,Country,Rating\n")

        mock_trained_model = MagicMock(spec=Model)
        mock_trained_model.path = os.path.join(temp_dir, "model")
        mock_trained_model.metadata = {"target": "Rating"}

        mock_test_data = MagicMock(spec=Dataset)
        mock_test_data.path = test_data_path

        mock_evaluated_model = MagicMock(spec=Model)
        mock_evaluated_model.metadata = {}

        with patch("joblib.load", return_value=mock_model):
            model_file_path = mock_trained_model.path + ".joblib"
            with open(model_file_path, "wb") as f:
                f.write(b"fake model data")

            with pytest.raises(ValueError, match="Test dataset is empty"):
                evaluate_model_func(
                    trained_model=mock_trained_model,
                    test_data=mock_test_data,
                    evaluated_model=mock_evaluated_model,
                )


def test_evaluate_model_failure_metadata():
    """Test that failure metadata is set when evaluation fails."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data_path = os.path.join(temp_dir, "test_data.csv")
        test_df = pd.DataFrame({"Rating": [4.0, 3.5]})
        test_df.to_csv(test_data_path, index=False)

        mock_trained_model = MagicMock(spec=Model)
        mock_trained_model.path = os.path.join(temp_dir, "model")
        mock_trained_model.metadata = {"target": "Rating"}

        mock_test_data = MagicMock(spec=Dataset)
        mock_test_data.path = test_data_path

        mock_evaluated_model = MagicMock(spec=Model)
        mock_evaluated_model.metadata = {}

        with patch("joblib.load", side_effect=Exception("Model loading failed")):
            model_file_path = mock_trained_model.path + ".joblib"
            with open(model_file_path, "wb") as f:
                f.write(b"fake model data")

            with pytest.raises(Exception, match="Model loading failed"):
                evaluate_model_func(
                    trained_model=mock_trained_model,
                    test_data=mock_test_data,
                    evaluated_model=mock_evaluated_model,
                )

            metadata = mock_evaluated_model.metadata
            assert metadata["evaluation_status"] == "failed"
            assert "Model loading failed" in metadata["error_message"]
