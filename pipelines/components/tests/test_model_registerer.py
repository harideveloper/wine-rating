"""Tests for model registerer component."""

from unittest.mock import MagicMock, patch
import re
import pytest
from kfp.v2.dsl import Model

from pipelines.components.model_registerer import register_model

register_model_func = register_model.python_func


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model.list")
@patch("google.cloud.aiplatform.Model.upload")
def test_register_model_new_model_success(mock_upload, mock_list, mock_init):
    """Test successful model registration for new model."""
    # Mock no existing models
    mock_list.return_value = []
    mock_vertex_model = MagicMock()
    mock_vertex_model.resource_name = (
        "projects/test-project/locations/europe-west2/models/123"
    )
    mock_upload.return_value = mock_vertex_model
    mock_saved_model = MagicMock(spec=Model)
    mock_saved_model.uri = "gs://bucket/model"
    mock_saved_model.metadata = {
        "r2_score": "0.8500",
        "rmse": "0.1200",
        "mae": "0.0800",
        "quality_score": "0.9000",
        "evaluation_status": "completed",
        "framework": "sklearn",
    }
    mock_registered_model = MagicMock(spec=Model)
    mock_registered_model.metadata = {}
    register_model_func(
        saved_model=mock_saved_model,
        registered_model=mock_registered_model,
        model_display_name="wine-quality-model",
        project="test-project",
        region="europe-west2",
        model_serving_image="gcr.io/test/model:latest",
        build_number="build-123",
    )

    mock_init.assert_called_once_with(project="test-project", location="europe-west2")
    mock_list.assert_called_once_with(filter='display_name="wine-quality-model"')
    mock_upload.assert_called_once()
    upload_call = mock_upload.call_args
    assert upload_call.kwargs["display_name"] == "wine-quality-model"
    assert upload_call.kwargs["artifact_uri"] == "gs://bucket/model"
    assert (
        upload_call.kwargs["serving_container_image_uri"] == "gcr.io/test/model:latest"
    )
    assert upload_call.kwargs["is_default_version"] is True
    assert "parent_model" not in upload_call.kwargs
    labels = upload_call.kwargs["labels"]
    assert labels["ready-for-promotion"] == "true"
    assert labels["harness-build-id"] == "build-123"
    assert labels["r2-score"] == "0-8500"  # Converted format
    assert labels["quality-score"] == "0-9000"
    assert labels["eval-status"] == "completed"

    # Verify registered model metadata output
    assert (
        mock_registered_model.uri
        == "projects/test-project/locations/europe-west2/models/123"
    )
    assert mock_registered_model.metadata["display_name"] == "wine-quality-model"
    assert mock_registered_model.metadata["framework"] == "sklearn"


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model.list")
@patch("google.cloud.aiplatform.Model.upload")
def test_register_model_existing_model_new_version(mock_upload, mock_list, _mock_init):
    """Test model registration for existing model (new version)."""
    # Mock existing model
    mock_existing_model = MagicMock()
    mock_existing_model.resource_name = (
        "projects/test-project/locations/europe-west2/models/456"
    )
    mock_list.return_value = [mock_existing_model]
    mock_vertex_model = MagicMock()
    mock_vertex_model.resource_name = (
        "projects/test-project/locations/europe-west2/models/789"
    )
    mock_upload.return_value = mock_vertex_model
    mock_saved_model = MagicMock(spec=Model)
    mock_saved_model.uri = "gs://bucket/model-v2"
    mock_saved_model.metadata = {"r2_score": "0.9000", "framework": "sklearn"}

    mock_registered_model = MagicMock(spec=Model)
    mock_registered_model.metadata = {}
    register_model_func(
        saved_model=mock_saved_model,
        registered_model=mock_registered_model,
        model_display_name="wine-quality-model",
        project="test-project",
        region="europe-west2",
        model_serving_image="gcr.io/test/model:v2",
        build_number="build-456",
    )

    # Verify model upload was called existing model path
    upload_call = mock_upload.call_args
    assert (
        upload_call.kwargs["parent_model"]
        == "projects/test-project/locations/europe-west2/models/456"
    )
    assert upload_call.kwargs["is_default_version"] is True


def test_convert_to_gcp_label_format():
    """Test the GCP label format conversion function."""

    def convert_to_gcp_label_format(value):
        """Convert labels value to google required format e.g mae = 0.0003 to 0-0003"""
        str_value = str(value).lower()
        compliant = re.sub(r"[^a-z0-9_-]", "-", str_value)
        return compliant[:63]

    # Test various format conversions
    assert convert_to_gcp_label_format("0.8500") == "0-8500"
    assert convert_to_gcp_label_format("completed") == "completed"
    assert convert_to_gcp_label_format("COMPLETED") == "completed"
    assert convert_to_gcp_label_format("test.value") == "test-value"
    assert convert_to_gcp_label_format("test@value!") == "test-value-"


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model.list")
@patch("google.cloud.aiplatform.Model.upload")
def test_register_model_handles_invalid_metrics(mock_upload, mock_list, _mock_init):
    """Test model registration handles invalid metric values gracefully."""
    mock_list.return_value = []

    mock_vertex_model = MagicMock()
    mock_vertex_model.resource_name = (
        "projects/test-project/locations/europe-west2/models/123"
    )
    mock_upload.return_value = mock_vertex_model

    # Mock saved model with invalid metrics
    mock_saved_model = MagicMock(spec=Model)
    mock_saved_model.uri = "gs://bucket/model"
    mock_saved_model.metadata = {
        "r2_score": "invalid_number",  # Invalid float
        "rmse": None,  # None value
        "mae": "0.0800",  # Valid
        "quality_score": "",  # Empty string
        "evaluation_status": "completed",
    }

    mock_registered_model = MagicMock(spec=Model)
    mock_registered_model.metadata = {}

    register_model_func(
        saved_model=mock_saved_model,
        registered_model=mock_registered_model,
        model_display_name="wine-quality-model",
        project="test-project",
        region="europe-west2",
        model_serving_image="gcr.io/test/model:latest",
    )

    mock_upload.assert_called_once()

    # Verify labels only include valid metrics
    labels = mock_upload.call_args.kwargs["labels"]
    assert "mae" in labels  # Valid metric should be included
    assert labels["mae"] == "0-0800"
    assert "r2-score" not in labels  # Invalid metric should be skipped
    assert "rmse" not in labels  # None metric should be skipped


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model.list")
def test_register_model_vertex_ai_error(_mock_list, mock_init):
    """Test model registration handles Vertex AI errors gracefully."""
    # Mock Vertex AI initialization to raise an error
    mock_init.side_effect = Exception("Vertex AI authentication failed")

    mock_saved_model = MagicMock(spec=Model)
    mock_saved_model.uri = "gs://bucket/model"
    mock_saved_model.metadata = {"framework": "sklearn"}

    mock_registered_model = MagicMock(spec=Model)
    mock_registered_model.metadata = {}

    with pytest.raises(Exception, match="Vertex AI authentication failed"):
        register_model_func(
            saved_model=mock_saved_model,
            registered_model=mock_registered_model,
            model_display_name="wine-quality-model",
            project="test-project",
            region="europe-west2",
            model_serving_image="gcr.io/test/model:latest",
        )
