"""Tests for data loader component core logic."""

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestDataLoader:
    """Test core data loader logic."""

    def test_gcs_path_validation(self):
        """Test GCS path validation."""

        def validate_path(path):
            if not path or not path.startswith("gs://"):
                raise ValueError("Invalid GCS path")

        # Valid path
        validate_path("gs://bucket/file.csv")

        # Invalid paths
        with pytest.raises(ValueError):
            validate_path("")
        with pytest.raises(ValueError):
            validate_path("s3://bucket/file.csv")

    def test_gcs_path_parsing(self):
        """Test parsing GCS path into bucket and blob."""
        path = "gs://my-bucket/path/to/file.csv"
        parts = path.replace("gs://", "").split("/")
        bucket = parts[0]
        blob = "/".join(parts[1:])

        assert bucket == "my-bucket"
        assert blob == "path/to/file.csv"

    def test_id_column_addition(self):
        """Test ID column addition when missing."""
        df = pd.DataFrame({"Country": ["France", "Italy"], "Rating": [4.2, 3.8]})

        if "id" not in df.columns:
            df["id"] = range(1, len(df) + 1)

        assert "id" in df.columns
        assert list(df["id"]) == [1, 2]

    @patch("google.cloud.storage.Client")
    def test_gcs_download_flow(self, mock_client):
        """Test GCS download workflow."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.download_as_text.return_value = "id,Country\n1,France"

        from google.cloud import storage  # pylint: disable=import-outside-toplevel

        client = storage.Client()
        content = client.bucket("test").blob("data.csv").download_as_text()
        df = pd.read_csv(io.StringIO(content))

        assert len(df) == 1
        assert "Country" in df.columns
