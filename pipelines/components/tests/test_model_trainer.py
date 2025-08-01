"""Tests for model trainer component."""

import os
import tempfile
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from kfp.v2.dsl import Dataset, Model
from pipelines.components.model_trainer import train_model

train_model_func = train_model.python_func


@patch("pandas.read_csv")
def test_train_model_success(mock_read_csv):
    """Test successful model training using actual train_model function."""
    # Mock training data with wine features
    mock_df = pd.DataFrame(
        {
            "Country": ["France", "Italy", "Spain", "France", "Italy"],
            "price_numeric": [25.99, 18.50, 22.75, 35.00, 28.50],
            "Rating": [4.2, 3.8, 4.1, 4.5, 4.0],
            "Type": ["Red", "White", "Red", "Red", "White"],
            "Region": ["Bordeaux", "Tuscany", "Rioja", "Burgundy", "Piedmont"],
        }
    )
    mock_read_csv.return_value = mock_df
    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = os.path.join(temp_dir, "train.csv")
        model_path = os.path.join(temp_dir, "model")
        mock_train_data = MagicMock(spec=Dataset)
        mock_train_data.path = train_path
        mock_trained_model = MagicMock(spec=Model)
        mock_trained_model.path = model_path
        mock_trained_model.metadata = {}
        train_model_func(
            train_data=mock_train_data,
            trained_model=mock_trained_model,
            n_estimators=10,
            random_state=42,
        )

        # Verify model file was created
        model_file = model_path + ".joblib"
        assert os.path.exists(model_file)

        # Verify metadata was set
        assert mock_trained_model.metadata["framework"] == "sklearn"
        assert mock_trained_model.metadata["target"] == "Rating"
        assert "feature_order" in mock_trained_model.metadata


@patch("pandas.read_csv")
def test_train_model_feature_selection(mock_read_csv):
    """Test feature selection logic for available columns."""
    # Mock data with only some categorical features available
    mock_df = pd.DataFrame(
        {
            "Country": ["France", "Italy"],
            "price_numeric": [25.99, 18.50],
            "Rating": [4.2, 3.8],
            "Type": ["Red", "White"],
            # Missing: Region, Style, Grape
        }
    )
    mock_read_csv.return_value = mock_df

    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = os.path.join(temp_dir, "train.csv")
        model_path = os.path.join(temp_dir, "model")

        mock_train_data = MagicMock(spec=Dataset)
        mock_train_data.path = train_path
        mock_trained_model = MagicMock(spec=Model)
        mock_trained_model.path = model_path
        mock_trained_model.metadata = {}
        train_model_func(
            train_data=mock_train_data,
            trained_model=mock_trained_model,
            n_estimators=5,
            random_state=42,
        )

        # Verify model was trained successfully
        model_file = model_path + ".joblib"
        assert os.path.exists(model_file)

        # Check that feature_order only includes available columns
        feature_order_str = mock_trained_model.metadata["feature_order"]
        assert "price_numeric" in feature_order_str
        assert "Country" in feature_order_str
        assert "Type" in feature_order_str


@patch("pandas.read_csv")
def test_train_model_numeric_features_only(mock_read_csv):
    """Test training with only numeric features available."""
    # Mock data with only numeric features
    mock_df = pd.DataFrame(
        {"price_numeric": [25.99, 18.50, 22.75, 35.00], "Rating": [4.2, 3.8, 4.1, 4.5]}
    )
    mock_read_csv.return_value = mock_df

    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = os.path.join(temp_dir, "train.csv")
        model_path = os.path.join(temp_dir, "model")

        mock_train_data = MagicMock(spec=Dataset)
        mock_train_data.path = train_path
        mock_trained_model = MagicMock(spec=Model)
        mock_trained_model.path = model_path
        mock_trained_model.metadata = {}

        train_model_func(
            train_data=mock_train_data,
            trained_model=mock_trained_model,
            n_estimators=5,
            random_state=42,
        )

        # Verify training succeeded
        model_file = model_path + ".joblib"
        assert os.path.exists(model_file)

        # Verify metadata
        feature_order_str = mock_trained_model.metadata["feature_order"]
        assert "price_numeric" in feature_order_str


@patch("pandas.read_csv")
def test_train_model_missing_target_column(mock_read_csv):
    """Test training fails gracefully when target column is missing."""
    # Mock data without Rating column
    mock_df = pd.DataFrame(
        {
            "Country": ["France", "Italy"],
            "price_numeric": [25.99, 18.50],
            # Missing: Rating
        }
    )
    mock_read_csv.return_value = mock_df

    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = os.path.join(temp_dir, "train.csv")
        model_path = os.path.join(temp_dir, "model")

        mock_train_data = MagicMock(spec=Dataset)
        mock_train_data.path = train_path
        mock_trained_model = MagicMock(spec=Model)
        mock_trained_model.path = model_path
        mock_trained_model.metadata = {}

        # Test that training fails when target is missing
        with pytest.raises(KeyError):
            train_model_func(
                train_data=mock_train_data,
                trained_model=mock_trained_model,
                n_estimators=5,
                random_state=42,
            )


@patch("pandas.read_csv")
def test_train_model_hyperparameters(mock_read_csv):
    """Test that hyperparameters are properly used."""
    # Mock training data
    mock_df = pd.DataFrame(
        {
            "Country": ["France", "Italy", "Spain"],
            "price_numeric": [25.99, 18.50, 22.75],
            "Rating": [4.2, 3.8, 4.1],
        }
    )
    mock_read_csv.return_value = mock_df

    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = os.path.join(temp_dir, "train.csv")
        model_path = os.path.join(temp_dir, "model")

        mock_train_data = MagicMock(spec=Dataset)
        mock_train_data.path = train_path
        mock_trained_model = MagicMock(spec=Model)
        mock_trained_model.path = model_path
        mock_trained_model.metadata = {}

        # Test with specific parameters
        train_model_func(
            train_data=mock_train_data,
            trained_model=mock_trained_model,
            n_estimators=50,
            random_state=123,
        )

        # Verify model was created
        model_file = model_path + ".joblib"
        assert os.path.exists(model_file)

        # Verify metadata includes framework and target
        assert mock_trained_model.metadata["framework"] == "sklearn"
        assert mock_trained_model.metadata["target"] == "Rating"
