"""Tests for model fetcher component"""

from unittest.mock import MagicMock, patch
from datetime import datetime
import pytest

from pipelines.components.model_fetcher import fetch_model

fetch_model_func = fetch_model.python_func


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model.list")
def test_fetch_model_success_with_metadata_extraction(mock_model_list, mock_init):
    """Test successful model fetching with metadata extraction."""
    # Mock model with promotion label and metrics
    mock_model = MagicMock()
    mock_model.resource_name = "projects/test-project/locations/europe-west2/models/123"
    mock_model.display_name = "wine-quality-model"
    mock_model.create_time = datetime(2024, 1, 15, 10, 0, 0)
    mock_model.labels = {
        "ready-for-promotion": "true",
        "eval-status": "completed",
        "quality-score": "0-9992",  # Dashed format
        "mae": "0-0003",
        "mse": "0-0000",
        "r2-score": "0-9998",
        "rmse": "0-0020",
        "harness-build-id": "build-123",
        "env": "prd-test",
    }

    mock_model_list.return_value = [mock_model]
    mock_fetched_model = MagicMock()
    mock_fetched_model.metadata = {}
    fetch_model_func(
        fetched_model=mock_fetched_model,
        model_display_name="wine-quality-model",
        project="test-project",
        location="europe-west2",
    )
    mock_init.assert_called_once_with(project="test-project", location="europe-west2")
    mock_model_list.assert_called_once_with(filter='display_name="wine-quality-model"')
    assert (
        mock_fetched_model.uri
        == "projects/test-project/locations/europe-west2/models/123"
    )

    # Verify metadata extraction
    expected_metadata = {
        "display_name": "wine-quality-model",
        "resource_name": "projects/test-project/locations/europe-west2/models/123",
        "harness_build_id": "build-123",
        "eval_status": "completed",
        "ready_for_promotion": "true",
        "quality_score": "0.9992",  # Converted from dashed format
        "mae": "0.0003",
        "mse": "0.0",
        "r2_score": "0.9998",
        "rmse": "0.002",
    }

    assert mock_fetched_model.metadata == expected_metadata


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model.list")
def test_fetch_model_dashed_format_conversion(mock_model_list, _mock_init):
    """Test proper conversion of dashed format metrics to float."""
    mock_model = MagicMock()
    mock_model.resource_name = "projects/test-project/locations/europe-west2/models/123"
    mock_model.display_name = "wine-quality-model"
    mock_model.create_time = datetime(2024, 1, 15, 10, 0, 0)
    mock_model.labels = {
        "ready-for-promotion": "true",
        "eval-status": "completed",
        "quality-score": "0-8750",  # Should become 0.8750
        "mae": "0-1234",  # Should become 0.1234
        "mse": "0-0000",  # Should become 0.0
        "r2-score": "0-9999",  # Should become 0.9999
        "rmse": "0-0567",  # Should become 0.0567
        "harness-build-id": "local",
    }

    mock_model_list.return_value = [mock_model]
    mock_fetched_model = MagicMock()
    mock_fetched_model.metadata = {}
    fetch_model_func(
        fetched_model=mock_fetched_model,
        model_display_name="wine-quality-model",
        project="test-project",
        location="europe-west2",
    )

    # Verify metric conversions (float conversion removes trailing zeros)
    assert (
        mock_fetched_model.metadata["quality_score"] == "0.875"
    )  # 0.8750 becomes 0.875
    assert mock_fetched_model.metadata["mae"] == "0.1234"
    assert mock_fetched_model.metadata["mse"] == "0.0"
    assert mock_fetched_model.metadata["r2_score"] == "0.9999"
    assert mock_fetched_model.metadata["rmse"] == "0.0567"


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model.list")
def test_fetch_model_multiple_models_returns_newest_promoted(
    mock_model_list, _mock_init
):
    """Test fetching returns newest promoted model when multiple exist."""
    # Mock older promoted model
    mock_model_old = MagicMock()
    mock_model_old.resource_name = (
        "projects/test-project/locations/europe-west2/models/111"
    )
    mock_model_old.display_name = "wine-quality-model"
    mock_model_old.create_time = datetime(2024, 1, 10, 10, 0, 0)  # Older
    mock_model_old.labels = {
        "ready-for-promotion": "true",
        "eval-status": "completed",
        "quality-score": "0-8500",
        "mae": "0-0050",
        "mse": "0-0001",
        "r2-score": "0-9500",
        "rmse": "0-0100",
        "harness-build-id": "build-old",
    }

    # Mock newer promoted model
    mock_model_new = MagicMock()
    mock_model_new.resource_name = (
        "projects/test-project/locations/europe-west2/models/222"
    )
    mock_model_new.display_name = "wine-quality-model"
    mock_model_new.create_time = datetime(2024, 1, 20, 10, 0, 0)  # Newer
    mock_model_new.labels = {
        "ready-for-promotion": "true",
        "eval-status": "completed",
        "quality-score": "0-9200",
        "mae": "0-0030",
        "mse": "0-0000",
        "r2-score": "0-9800",
        "rmse": "0-0050",
        "harness-build-id": "build-new",
    }

    # Mock non-promoted model (newest but not promoted)
    mock_model_latest = MagicMock()
    mock_model_latest.resource_name = (
        "projects/test-project/locations/europe-west2/models/333"
    )
    mock_model_latest.display_name = "wine-quality-model"
    mock_model_latest.create_time = datetime(2024, 1, 25, 10, 0, 0)  # Newest
    mock_model_latest.labels = {"ready-for-promotion": "false"}

    # Return models in random order (to test sorting)
    mock_model_list.return_value = [mock_model_old, mock_model_latest, mock_model_new]

    mock_fetched_model = MagicMock()
    mock_fetched_model.metadata = {}
    fetch_model_func(
        fetched_model=mock_fetched_model,
        model_display_name="wine-quality-model",
        project="test-project",
        location="europe-west2",
    )

    # Should return the newer promoted model (not the newest non-promoted one)
    assert (
        mock_fetched_model.uri
        == "projects/test-project/locations/europe-west2/models/222"
    )
    assert mock_fetched_model.metadata["harness_build_id"] == "build-new"
    assert mock_fetched_model.metadata["quality_score"] == "0.92"  # 0.9200 becomes 0.92


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model.list")
def test_fetch_model_no_models_found(mock_model_list, _mock_init):
    """Test error when no models found with display name."""
    mock_model_list.return_value = []
    mock_fetched_model = MagicMock()
    with pytest.raises(
        ValueError, match="No models found with display name: wine-quality-model"
    ):
        fetch_model_func(
            fetched_model=mock_fetched_model,
            model_display_name="wine-quality-model",
            project="test-project",
            location="europe-west2",
        )

    # Verify model list was called
    mock_model_list.assert_called_once_with(filter='display_name="wine-quality-model"')


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model.list")
def test_fetch_model_no_promoted_models(mock_model_list, _mock_init):
    """Test error when no models have promotion label."""
    # Mock models without promotion labels
    mock_model1 = MagicMock()
    mock_model1.resource_name = (
        "projects/test-project/locations/europe-west2/models/111"
    )
    mock_model1.create_time = datetime(2024, 1, 15, 10, 0, 0)
    mock_model1.labels = {"env": "dev"}  # No promotion label

    mock_model2 = MagicMock()
    mock_model2.resource_name = (
        "projects/test-project/locations/europe-west2/models/222"
    )
    mock_model2.create_time = datetime(2024, 1, 10, 10, 0, 0)
    mock_model2.labels = {"ready-for-promotion": "false"}

    mock_model_list.return_value = [mock_model1, mock_model2]
    mock_fetched_model = MagicMock()

    with pytest.raises(
        ValueError,
        match="No models with 'ready-for-promotion=true' found among 2 models",
    ):
        fetch_model_func(
            fetched_model=mock_fetched_model,
            model_display_name="wine-quality-model",
            project="test-project",
            location="europe-west2",
        )


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model.list")
def test_fetch_model_handles_case_insensitive_promotion_label(
    mock_model_list, _mock_init
):
    """Test that promotion label check is case insensitive."""
    mock_model1 = MagicMock()
    mock_model1.resource_name = (
        "projects/test-project/locations/europe-west2/models/111"
    )
    mock_model1.display_name = "wine-quality-model"
    mock_model1.create_time = datetime(2024, 1, 15, 10, 0, 0)
    mock_model1.labels = {
        "ready-for-promotion": "TRUE",  # Uppercase
        "eval-status": "completed",
        "quality-score": "0-9000",
        "mae": "0-0040",
        "mse": "0-0000",
        "r2-score": "0-9500",
        "rmse": "0-0080",
        "harness-build-id": "build-upper",
    }

    mock_model2 = MagicMock()
    mock_model2.resource_name = (
        "projects/test-project/locations/europe-west2/models/222"
    )
    mock_model2.display_name = "wine-quality-model"
    mock_model2.create_time = datetime(2024, 1, 10, 10, 0, 0)
    mock_model2.labels = {
        "ready-for-promotion": "True",  # Mixed case
        "eval-status": "completed",
        "quality-score": "0-8500",
        "harness-build-id": "build-mixed",
    }

    mock_model_list.return_value = [mock_model1, mock_model2]
    mock_fetched_model = MagicMock()
    mock_fetched_model.metadata = {}

    fetch_model_func(
        fetched_model=mock_fetched_model,
        model_display_name="wine-quality-model",
        project="test-project",
        location="europe-west2",
    )

    # Should return the newer model with uppercase "TRUE"
    assert (
        mock_fetched_model.uri
        == "projects/test-project/locations/europe-west2/models/111"
    )
    assert mock_fetched_model.metadata["harness_build_id"] == "build-upper"


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model.list")
def test_fetch_model_handles_missing_metrics_gracefully(mock_model_list, _mock_init):
    """Test handling of models with missing or invalid metrics."""
    mock_model = MagicMock()
    mock_model.resource_name = "projects/test-project/locations/europe-west2/models/123"
    mock_model.display_name = "wine-quality-model"
    mock_model.create_time = datetime(2024, 1, 15, 10, 0, 0)
    mock_model.labels = {
        "ready-for-promotion": "true",
        "eval-status": "completed",
        "quality-score": "invalid-score",  # Invalid format
        "mae": "0-0030",  # Valid
        # Missing mse, r2-score, rmse
        "harness-build-id": "build-123",
    }

    mock_model_list.return_value = [mock_model]
    mock_fetched_model = MagicMock()
    mock_fetched_model.metadata = {}

    # Test the function
    fetch_model_func(
        fetched_model=mock_fetched_model,
        model_display_name="wine-quality-model",
        project="test-project",
        location="europe-west2",
    )

    # Verify it handles missing/invalid metrics gracefully
    assert (
        mock_fetched_model.metadata["quality_score"] == "0.0"
    )  # Invalid converted to 0.0
    assert (
        mock_fetched_model.metadata["mae"] == "0.003"
    )  # Valid conversion (0.0030 becomes 0.003)
    assert mock_fetched_model.metadata["mse"] == "0.0"  # Missing defaults to 0.0
    assert mock_fetched_model.metadata["r2_score"] == "0.0"  # Missing defaults to 0.0
    assert mock_fetched_model.metadata["rmse"] == "0.0"  # Missing defaults to 0.0
    assert mock_fetched_model.metadata["harness_build_id"] == "build-123"


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model.list")
def test_fetch_model_error_handling_sets_error_metadata(mock_model_list, _mock_init):
    """Test that errors set appropriate error metadata."""
    # Simulate an exception during model listing
    mock_model_list.side_effect = Exception("API Error")

    mock_fetched_model = MagicMock()
    mock_fetched_model.metadata = {}

    with pytest.raises(Exception, match="API Error"):
        fetch_model_func(
            fetched_model=mock_fetched_model,
            model_display_name="wine-quality-model",
            project="test-project",
            location="europe-west2",
        )
    assert mock_fetched_model.uri == ""
    assert mock_fetched_model.metadata["display_name"] == "wine-quality-model"
    assert mock_fetched_model.metadata["resource_name"] == ""
    assert mock_fetched_model.metadata["harness_build_id"] == "unknown"
    assert mock_fetched_model.metadata["eval_status"] == "failed"
    assert mock_fetched_model.metadata["ready_for_promotion"] == "false"
    assert mock_fetched_model.metadata["quality_score"] == "0.0"
    assert mock_fetched_model.metadata["mae"] == "0.0"
    assert mock_fetched_model.metadata["mse"] == "0.0"
    assert mock_fetched_model.metadata["r2_score"] == "0.0"
    assert mock_fetched_model.metadata["rmse"] == "0.0"
    assert mock_fetched_model.metadata["error_message"] == "API Error"
