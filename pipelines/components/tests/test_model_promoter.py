"""Tests for model promoter component."""

from unittest.mock import MagicMock, patch
from kfp.dsl import Model
import pytest

from pipelines.components.model_promoter import promote_model

promote_model_func = promote_model.python_func


@patch("datetime.datetime")
@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model")
def test_promote_model_creates_new_model_when_none_exists(
    mock_model_class, _mock_init, mock_datetime
):
    """Test promotion creates new model when none exists in target project."""
    # Mock current date
    mock_datetime.utcnow.return_value.date.return_value.isoformat.return_value = (
        "2024-01-15"
    )

    # Mock KFP Model input
    mock_fetched_model = MagicMock(spec=Model)
    mock_fetched_model.uri = "projects/source-project/locations/europe-west2/models/123"
    mock_fetched_model.metadata = {
        "display_name": "wine-quality-model",
        "resource_name": "projects/source-project/locations/europe-west2/models/123",
        "quality_score": "0.96",
        "eval_status": "completed",
        "ready_for_promotion": "true",
        "harness_build_id": "build-123",
    }

    # Mock source model from Vertex AI
    mock_source_model = MagicMock()
    mock_source_model.display_name = "wine-quality-model"
    mock_source_model.uri = "gs://source-bucket/model"
    mock_source_model.labels = {"ready-for-promotion": "true", "version": "v1"}
    mock_source_model.container_spec.image_uri = "gcr.io/source/model:latest"

    # Mock uploaded model in target project
    mock_target_model = MagicMock()
    mock_target_model.resource_name = (
        "projects/target-project/locations/europe-west2/models/456"
    )
    mock_target_model.display_name = "wine-quality-model"
    mock_target_model.version_id = "1"

    # Mock no existing models in target project
    mock_model_class.list.return_value = []  # No existing models
    mock_model_class.side_effect = [mock_source_model]
    mock_model_class.upload.return_value = mock_target_model

    # Mock promoted model output
    mock_promoted_model = MagicMock(spec=Model)
    mock_promoted_model.metadata = {}

    # Test the promotion function
    promote_model_func(
        fetched_model=mock_fetched_model,
        source_project="source-project",
        target_project="target-project",
        location="europe-west2",
        promoted_model=mock_promoted_model,
    )

    # Verify model list was called to check for existing models
    mock_model_class.list.assert_called_once_with(
        filter='display_name="wine-quality-model"'
    )

    # Verify model upload called without parent_model (new model)
    upload_call = mock_model_class.upload.call_args
    assert upload_call.kwargs["display_name"] == "wine-quality-model"
    assert "parent_model" not in upload_call.kwargs  # No parent for new model
    assert upload_call.kwargs["is_default_version"] is True

    # Verify promoted model metadata includes versioning info
    assert mock_promoted_model.metadata["model_version"] == "1"
    assert mock_promoted_model.metadata["is_new_model"] is True


@patch("datetime.datetime")
@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model")
def test_promote_model_creates_new_version_when_model_exists(
    mock_model_class, _mock_init, mock_datetime
):
    """Test promotion creates new version when model already exists in target project."""
    # Mock current date
    mock_datetime.utcnow.return_value.date.return_value.isoformat.return_value = (
        "2024-01-20"
    )

    # Mock KFP Model input
    mock_fetched_model = MagicMock(spec=Model)
    mock_fetched_model.uri = "projects/source-project/locations/europe-west2/models/123"
    mock_fetched_model.metadata = {
        "display_name": "wine-quality-model",
        "quality_score": "0.97",
        "eval_status": "completed",
    }

    # Mock source model
    mock_source_model = MagicMock()
    mock_source_model.display_name = "wine-quality-model"
    mock_source_model.uri = "gs://source-bucket/model-v2"
    mock_source_model.labels = {"version": "v2"}
    mock_source_model.container_spec.image_uri = "gcr.io/source/model:v2"

    # Mock existing model in target project
    mock_existing_model = MagicMock()
    mock_existing_model.resource_name = (
        "projects/target-project/locations/europe-west2/models/existing"
    )
    mock_existing_model.display_name = "wine-quality-model"

    # Mock new version model
    mock_target_model = MagicMock()
    mock_target_model.resource_name = (
        "projects/target-project/locations/europe-west2/models/456"
    )
    mock_target_model.display_name = "wine-quality-model"
    mock_target_model.version_id = "2"

    # Mock existing models found
    mock_model_class.list.return_value = [mock_existing_model]  # Existing model found
    mock_model_class.side_effect = [mock_source_model]
    mock_model_class.upload.return_value = mock_target_model

    # Mock promoted model output
    mock_promoted_model = MagicMock(spec=Model)
    mock_promoted_model.metadata = {}

    # Test promotion
    promote_model_func(
        fetched_model=mock_fetched_model,
        source_project="source-project",
        target_project="target-project",
        location="europe-west2",
        promoted_model=mock_promoted_model,
    )

    # Verify model upload called with parent_model (new version)
    upload_call = mock_model_class.upload.call_args
    assert upload_call.kwargs["display_name"] == "wine-quality-model"
    assert (
        upload_call.kwargs["parent_model"]
        == "projects/target-project/locations/europe-west2/models/existing"
    )
    assert upload_call.kwargs["is_default_version"] is True

    # Verify promoted model metadata includes versioning info
    assert mock_promoted_model.metadata["model_version"] == "2"
    assert mock_promoted_model.metadata["is_new_model"] is False


@patch("datetime.datetime")
@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model")
def test_promote_model_handles_model_without_version_id(
    mock_model_class, _mock_init, mock_datetime
):
    """Test promotion handles models that don't have version_id attribute."""
    # Mock current date
    mock_datetime.utcnow.return_value.date.return_value.isoformat.return_value = (
        "2024-01-25"
    )

    # Mock KFP Model input
    mock_fetched_model = MagicMock(spec=Model)
    mock_fetched_model.uri = "projects/source-project/locations/europe-west2/models/123"
    mock_fetched_model.metadata = {"display_name": "wine-quality-model"}

    # Mock source model
    mock_source_model = MagicMock()
    mock_source_model.display_name = "wine-quality-model"
    mock_source_model.uri = "gs://source-bucket/model"
    mock_source_model.labels = {}
    mock_source_model.container_spec.image_uri = "gcr.io/source/model:latest"

    # Mock target model without version_id attribute
    mock_target_model = MagicMock()
    mock_target_model.resource_name = (
        "projects/target-project/locations/europe-west2/models/456"
    )
    mock_target_model.display_name = "wine-quality-model"
    # Remove version_id attribute entirely
    del mock_target_model.version_id

    # Mock no existing models
    mock_model_class.list.return_value = []
    mock_model_class.side_effect = [mock_source_model]
    mock_model_class.upload.return_value = mock_target_model

    # Mock promoted model output
    mock_promoted_model = MagicMock(spec=Model)
    mock_promoted_model.metadata = {}

    # Test promotion
    promote_model_func(
        fetched_model=mock_fetched_model,
        source_project="source-project",
        target_project="target-project",
        location="europe-west2",
        promoted_model=mock_promoted_model,
    )

    # Verify default version is used when version_id not available
    assert mock_promoted_model.metadata["model_version"] == "1"


@patch("datetime.datetime")
@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model")
def test_promote_model_with_list_operation(mock_model_class, _mock_init, mock_datetime):
    """Test promotion with successful model listing (covers the list operation)."""
    # Mock current date
    mock_datetime.utcnow.return_value.date.return_value.isoformat.return_value = (
        "2024-01-26"
    )

    # Mock KFP Model input
    mock_fetched_model = MagicMock(spec=Model)
    mock_fetched_model.uri = "projects/source-project/locations/europe-west2/models/123"
    mock_fetched_model.metadata = {
        "display_name": "wine-quality-model",
        "quality_score": "0.95",
    }

    # Mock source model
    mock_source_model = MagicMock()
    mock_source_model.display_name = "wine-quality-model"
    mock_source_model.uri = "gs://source-bucket/model"
    mock_source_model.labels = {}
    mock_source_model.container_spec.image_uri = "gcr.io/source/model:latest"

    # Mock target model
    mock_target_model = MagicMock()
    mock_target_model.resource_name = (
        "projects/target-project/locations/europe-west2/models/456"
    )
    mock_target_model.display_name = "wine-quality-model"
    mock_target_model.version_id = "1"

    # Mock successful list operation (no existing models)
    mock_model_class.list.return_value = []
    mock_model_class.side_effect = [mock_source_model]
    mock_model_class.upload.return_value = mock_target_model

    # Mock promoted model output
    mock_promoted_model = MagicMock(spec=Model)
    mock_promoted_model.metadata = {}

    # Test should complete successfully
    promote_model_func(
        fetched_model=mock_fetched_model,
        source_project="source-project",
        target_project="target-project",
        location="europe-west2",
        promoted_model=mock_promoted_model,
    )

    # Verify list was called
    mock_model_class.list.assert_called_once_with(
        filter='display_name="wine-quality-model"'
    )

    # Verify upload called without parent_model (new model)
    upload_call = mock_model_class.upload.call_args
    assert "parent_model" not in upload_call.kwargs

    # Verify metadata
    assert mock_promoted_model.metadata["is_new_model"] is True
    assert mock_promoted_model.metadata["model_version"] == "1"


@patch("datetime.datetime")
@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model")
def test_promote_model_preserves_all_metadata_from_fetched_model(
    mock_model_class, _mock_init, mock_datetime
):
    """Test that all metadata from fetched model is preserved."""
    # Mock current date
    mock_datetime.utcnow.return_value.date.return_value.isoformat.return_value = (
        "2024-01-30"
    )

    # Mock KFP Model input with comprehensive metadata
    mock_fetched_model = MagicMock(spec=Model)
    mock_fetched_model.uri = "projects/source-project/locations/europe-west2/models/123"
    mock_fetched_model.metadata = {
        "display_name": "wine-quality-model",
        "resource_name": "projects/source-project/locations/europe-west2/models/123",
        "quality_score": "0.98",
        "r2_score": "0.97",
        "mae": "0.002",
        "rmse": "0.015",
        "eval_status": "completed",
        "ready_for_promotion": "true",
        "harness_build_id": "build-456",
    }

    # Mock source and target models
    mock_source_model = MagicMock()
    mock_source_model.display_name = "wine-quality-model"
    mock_source_model.uri = "gs://source-bucket/model"
    mock_source_model.labels = {}
    mock_source_model.container_spec.image_uri = "gcr.io/source/model:latest"

    mock_target_model = MagicMock()
    mock_target_model.resource_name = (
        "projects/target-project/locations/europe-west2/models/999"
    )
    mock_target_model.display_name = "wine-quality-model"
    mock_target_model.version_id = "3"

    # Mock no existing models
    mock_model_class.list.return_value = []
    mock_model_class.side_effect = [mock_source_model]
    mock_model_class.upload.return_value = mock_target_model

    mock_promoted_model = MagicMock(spec=Model)
    mock_promoted_model.metadata = {}

    # Test promotion
    promote_model_func(
        fetched_model=mock_fetched_model,
        source_project="source-project",
        target_project="target-project",
        location="europe-west2",
        promoted_model=mock_promoted_model,
    )

    # Verify all original metadata keys are preserved
    original_keys = [
        "quality_score",
        "r2_score",
        "mae",
        "rmse",
        "eval_status",
        "ready_for_promotion",
        "harness_build_id",
    ]

    for key in original_keys:
        assert key in mock_promoted_model.metadata
        assert mock_promoted_model.metadata[key] == mock_fetched_model.metadata[key]

    # Verify promotion-specific metadata is added
    assert mock_promoted_model.metadata["source_project"] == "source-project"
    assert mock_promoted_model.metadata["target_project"] == "target-project"
    assert mock_promoted_model.metadata["promotion_date"] == "2024-01-30"
    assert mock_promoted_model.metadata["model_version"] == "3"
    assert mock_promoted_model.metadata["is_new_model"] is True


@patch("datetime.datetime")
@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model")
def test_promote_model_error_preserves_original_metadata(
    mock_model_class, _mock_init, mock_datetime
):
    """Test that original metadata is preserved even when promotion fails."""
    # Mock current date
    mock_datetime.utcnow.return_value.date.return_value.isoformat.return_value = (
        "2024-02-01"
    )

    # Mock KFP Model input
    mock_fetched_model = MagicMock(spec=Model)
    mock_fetched_model.uri = "projects/source-project/locations/europe-west2/models/123"
    mock_fetched_model.metadata = {
        "display_name": "wine-quality-model",
        "quality_score": "0.95",
        "eval_status": "completed",
        "harness_build_id": "build-789",
    }

    # Mock source model (successful)
    mock_source_model = MagicMock()
    mock_source_model.display_name = "wine-quality-model"
    mock_source_model.uri = "gs://source-bucket/model"
    mock_source_model.labels = {"version": "v1"}
    mock_source_model.container_spec.image_uri = "gcr.io/source/model:latest"

    # Mock upload failure
    mock_model_class.list.return_value = []
    mock_model_class.side_effect = [mock_source_model]
    mock_model_class.upload.side_effect = Exception("Upload failed")

    mock_promoted_model = MagicMock(spec=Model)
    mock_promoted_model.metadata = {}

    # Test should raise upload exception
    with pytest.raises(Exception, match="Upload failed"):
        promote_model_func(
            fetched_model=mock_fetched_model,
            source_project="source-project",
            target_project="target-project",
            location="europe-west2",
            promoted_model=mock_promoted_model,
        )

    # Verify original metadata is preserved in error case
    assert mock_promoted_model.metadata["quality_score"] == "0.95"
    assert mock_promoted_model.metadata["eval_status"] == "completed"
    assert mock_promoted_model.metadata["harness_build_id"] == "build-789"

    # Verify error metadata is added
    assert mock_promoted_model.metadata["promotion_status"] == "failed"
    assert mock_promoted_model.metadata["error_message"] == "Upload failed"
