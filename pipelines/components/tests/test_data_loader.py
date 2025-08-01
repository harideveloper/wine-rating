"""Tests for data loader component."""

import os
import tempfile
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from kfp.v2.dsl import Dataset


from pipelines.components.data_loader import load_data

load_data_func = load_data.python_func


def test_load_data_invalid_path():
    """Test load_data with invalid GCS path."""
    mock_output = MagicMock(spec=Dataset)
    mock_output.path = "/tmp/test_output.csv"

    with pytest.raises(ValueError, match="data_path must be a valid GCS URL"):
        load_data_func("", mock_output)

    with pytest.raises(ValueError, match="data_path must be a valid GCS URL"):
        load_data_func("s3://bucket/file.csv", mock_output)

    with pytest.raises(ValueError, match="data_path must be a valid GCS URL"):
        load_data_func(None, mock_output)


@patch("google.cloud.storage.Client")
@patch("pandas.read_csv")
def test_load_data_success(mock_read_csv, mock_storage_client):
    """Test successful data loading using actual load_data function."""
    # Mock the CSV data
    mock_df = pd.DataFrame(
        {
            "Country": ["France", "Italy"],
            "price_numeric": [25.99, 18.50],
            "Rating": [4.2, 3.8],
        }
    )
    mock_read_csv.return_value = mock_df
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_storage_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    mock_blob.download_as_text.return_value = (
        "Country,price_numeric,Rating\nFrance,25.99,4.2"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "output.csv")
        mock_output = MagicMock(spec=Dataset)
        mock_output.path = output_path
        load_data_func("gs://test-bucket/wine_data.csv", mock_output)
        assert os.path.exists(output_path)


@patch("google.cloud.storage.Client")
@patch("pandas.read_csv")
def test_load_data_adds_id_column(mock_read_csv, mock_storage_client):
    """Test that load_data adds ID column when missing."""
    # Mock DataFrame without ID column
    mock_df = pd.DataFrame({"Country": ["France", "Italy"], "Rating": [4.2, 3.8]})
    mock_read_csv.return_value = mock_df
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_storage_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    mock_blob.download_as_text.return_value = "Country,Rating\nFrance,4.2\nItaly,3.8"
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "output.csv")
        mock_output = MagicMock(spec=Dataset)
        mock_output.path = output_path
        load_data_func("gs://test-bucket/wine_data.csv", mock_output)
        # Verify ID column was added
        saved_df = pd.read_csv(output_path)
        assert "id" in saved_df.columns
        assert list(saved_df["id"]) == [1, 2]


@patch("google.cloud.storage.Client")
@patch("pandas.read_csv")
def test_load_data_empty_dataset(mock_read_csv, mock_storage_client):
    """Test load_data with empty dataset."""
    # Mock empty DataFrame
    mock_df = pd.DataFrame()
    mock_read_csv.return_value = mock_df
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_storage_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    mock_blob.download_as_text.return_value = ""

    mock_output = MagicMock(spec=Dataset)
    mock_output.path = "/tmp/test_output.csv"

    with pytest.raises(ValueError, match="Loaded dataset is empty"):
        load_data_func("gs://test-bucket/empty.csv", mock_output)


@patch("google.cloud.storage.Client")
def test_load_data_gcs_error_handling(mock_storage_client):
    """Test load_data handles GCS errors properly."""
    # Mock GCS client to raise exception
    mock_storage_client.side_effect = Exception("GCS connection failed")

    mock_output = MagicMock(spec=Dataset)
    mock_output.path = "/tmp/test_output.csv"

    with pytest.raises(Exception, match="GCS connection failed"):
        load_data_func("gs://test-bucket/wine_data.csv", mock_output)
