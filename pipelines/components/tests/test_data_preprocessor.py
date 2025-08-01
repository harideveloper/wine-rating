"""Tests for data preprocessor component."""

import os
import tempfile
from unittest.mock import MagicMock, patch
import pandas as pd
from kfp.v2.dsl import Dataset

from pipelines.components.data_preprocessor import preprocess_data

preprocess_data_func = preprocess_data.python_func


@patch("pandas.read_csv")
def test_preprocess_data_success(mock_read_csv):
    """Test successful data preprocessing using actual preprocess_data function."""
    # Mock wine data
    mock_df = pd.DataFrame(
        {
            "Country": ["France", "Italy", "Spain"],
            "Price": ["$25.99", "€18.50", "$22.75"],
            "Rating": [4.2, 3.8, 4.1],
            "Type": ["Red", "White", "Red"],
        }
    )
    mock_read_csv.return_value = mock_df
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "output.csv")
        train_path = os.path.join(temp_dir, "train.csv")
        test_path = os.path.join(temp_dir, "test.csv")
        mock_input = MagicMock(spec=Dataset)
        mock_input.path = input_path
        mock_output = MagicMock(spec=Dataset)
        mock_output.path = output_path
        mock_train = MagicMock(spec=Dataset)
        mock_train.path = train_path
        mock_test = MagicMock(spec=Dataset)
        mock_test.path = test_path
        preprocess_data_func(
            input_data=mock_input,
            output_data=mock_output,
            train_data=mock_train,
            test_data=mock_test,
            test_size=0.2,
            random_state=42,
        )

        assert os.path.exists(output_path)
        assert os.path.exists(train_path)
        assert os.path.exists(test_path)
        processed_df = pd.read_csv(output_path)
        assert "price_numeric" in processed_df.columns
        assert processed_df["price_numeric"].iloc[0] == 25.99


@patch("pandas.read_csv")
def test_preprocess_data_handles_missing_price_column(mock_read_csv):
    """Test preprocessing when Price column is missing."""
    # Mock DataFrame without Price column
    mock_df = pd.DataFrame({"Country": ["France", "Italy"], "Rating": [4.2, 3.8]})
    mock_read_csv.return_value = mock_df
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "output.csv")
        train_path = os.path.join(temp_dir, "train.csv")
        test_path = os.path.join(temp_dir, "test.csv")
        mock_input = MagicMock(spec=Dataset)
        mock_input.path = input_path
        mock_output = MagicMock(spec=Dataset)
        mock_output.path = output_path
        mock_train = MagicMock(spec=Dataset)
        mock_train.path = train_path
        mock_test = MagicMock(spec=Dataset)
        mock_test.path = test_path
        preprocess_data_func(
            input_data=mock_input,
            output_data=mock_output,
            train_data=mock_train,
            test_data=mock_test,
        )

        # Verify price_numeric defaults to 0
        processed_df = pd.read_csv(output_path)
        assert "price_numeric" in processed_df.columns
        assert all(processed_df["price_numeric"] == 0)


@patch("pandas.read_csv")
def test_preprocess_data_creates_synthetic_rating(mock_read_csv):
    """Test synthetic Rating creation when Rating column is missing."""
    # Mock DataFrame without Rating column
    mock_df = pd.DataFrame(
        {"Country": ["France", "Italy"], "Price": ["$30.00", "$40.00"]}
    )
    mock_read_csv.return_value = mock_df
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "output.csv")
        train_path = os.path.join(temp_dir, "train.csv")
        test_path = os.path.join(temp_dir, "test.csv")
        mock_input = MagicMock(spec=Dataset)
        mock_input.path = input_path
        mock_output = MagicMock(spec=Dataset)
        mock_output.path = output_path
        mock_train = MagicMock(spec=Dataset)
        mock_train.path = train_path
        mock_test = MagicMock(spec=Dataset)
        mock_test.path = test_path
        preprocess_data_func(
            input_data=mock_input,
            output_data=mock_output,
            train_data=mock_train,
            test_data=mock_test,
        )

        # Verify rating was created
        processed_df = pd.read_csv(output_path)
        assert "Rating" in processed_df.columns
        # Capped at 5.0
        # First row: $30.00 -> 30.0 * 0.2 + 3.0 = 9.0, capped to 5.0
        # Second row: $40.00 -> 40.0 * 0.2 + 3.0 = 11.0, capped to 5.0
        assert all(processed_df["Rating"] <= 5.0)
        assert processed_df["Rating"].iloc[0] == 5.0
        assert processed_df["Rating"].iloc[1] == 5.0


@patch("pandas.read_csv")
def test_preprocess_data_handles_null_categorical_features(mock_read_csv):
    """Test handling of null categorical features."""
    # Mock DataFrame with null categorical values
    mock_df = pd.DataFrame(
        {
            "Country": ["France", None, "Spain"],
            "Region": ["Bordeaux", "Tuscany", None],
            "Type": [None, "White", "Red"],
            "Price": ["$25.99", "$18.50", "$22.75"],
            "Rating": [4.2, 3.8, 4.1],
        }
    )
    mock_read_csv.return_value = mock_df
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "output.csv")
        train_path = os.path.join(temp_dir, "train.csv")
        test_path = os.path.join(temp_dir, "test.csv")
        mock_input = MagicMock(spec=Dataset)
        mock_input.path = input_path
        mock_output = MagicMock(spec=Dataset)
        mock_output.path = output_path
        mock_train = MagicMock(spec=Dataset)
        mock_train.path = train_path
        mock_test = MagicMock(spec=Dataset)
        mock_test.path = test_path
        preprocess_data_func(
            input_data=mock_input,
            output_data=mock_output,
            train_data=mock_train,
            test_data=mock_test,
        )

        # Verify null values were filled with "Unknown"
        processed_df = pd.read_csv(output_path)
        assert processed_df["Country"].iloc[1] == "Unknown"
        assert processed_df["Region"].iloc[2] == "Unknown"
        assert processed_df["Type"].iloc[0] == "Unknown"


@patch("pandas.read_csv")
def test_preprocess_data_price_extraction_regex(mock_read_csv):
    """Test price extraction regex logic with various price formats."""
    # Mock DataFrame with different price formats
    mock_df = pd.DataFrame(
        {
            "Country": ["France", "Italy", "Spain", "Germany", "Portugal"],
            "Price": ["$25.99", "€18.50", "£22.75", "invalid_price", None],
            "Rating": [4.2, 3.8, 4.1, 4.0, 3.9],
        }
    )
    mock_read_csv.return_value = mock_df
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "output.csv")
        train_path = os.path.join(temp_dir, "train.csv")
        test_path = os.path.join(temp_dir, "test.csv")
        mock_input = MagicMock(spec=Dataset)
        mock_input.path = input_path
        mock_output = MagicMock(spec=Dataset)
        mock_output.path = output_path
        mock_train = MagicMock(spec=Dataset)
        mock_train.path = train_path
        mock_test = MagicMock(spec=Dataset)
        mock_test.path = test_path
        preprocess_data_func(
            input_data=mock_input,
            output_data=mock_output,
            train_data=mock_train,
            test_data=mock_test,
        )

        # Verify price extraction results
        processed_df = pd.read_csv(output_path)
        assert processed_df["price_numeric"].iloc[0] == 25.99  # $25.99
        assert processed_df["price_numeric"].iloc[1] == 18.50  # €18.50
        assert processed_df["price_numeric"].iloc[2] == 22.75  # £22.75
        assert processed_df["price_numeric"].iloc[3] == 0  # invalid_price -> 0
        assert processed_df["price_numeric"].iloc[4] == 0  # None -> 0
