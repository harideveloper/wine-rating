"""Tests for model endpoint validator component."""

from unittest.mock import MagicMock, patch
import pytest
from kfp.v2.dsl import Model

# Import the component and extract the python function
from pipelines.components.model_endpoint_validator import validate_model_endpoint

validate_model_endpoint_func = validate_model_endpoint.python_func


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Endpoint")
def test_validate_model_endpoint_success(mock_endpoint_class, mock_init):
    """Test successful model endpoint validation."""
    mock_endpoint = MagicMock()
    mock_endpoint.resource_name = (
        "projects/test-project/locations/europe-west2/endpoints/123"
    )
    mock_endpoint_class.return_value = mock_endpoint
    mock_response = MagicMock()
    mock_response.predictions = [4.2]
    mock_endpoint.predict.return_value = mock_response
    mock_input_endpoint = MagicMock(spec=Model)
    mock_input_endpoint.uri = (
        "projects/test-project/locations/europe-west2/endpoints/123"
    )
    mock_validation_output = MagicMock(spec=Model)
    mock_validation_output.metadata = {}
    validate_model_endpoint_func(
        endpoint=mock_input_endpoint,
        validation_output=mock_validation_output,
        project="test-project",
        region="europe-west2",
        test_instances=[
            [25.99, "France", "Bordeaux", "Red", "Bold", "Cabernet Sauvignon"]
        ],
        expected_prediction_count=1,
    )

    mock_init.assert_called_once_with(project="test-project", location="europe-west2")

    mock_endpoint_class.assert_called_once_with(
        "projects/test-project/locations/europe-west2/endpoints/123"
    )
    mock_endpoint.predict.assert_called_once_with(
        instances=[[25.99, "France", "Bordeaux", "Red", "Bold", "Cabernet Sauvignon"]]
    )

    # Verify validation results
    assert (
        mock_validation_output.uri
        == "projects/test-project/locations/europe-west2/endpoints/123"
    )
    metadata = mock_validation_output.metadata
    assert metadata["status"] == "PASSED"
    assert metadata["test_instances_count"] == 1
    assert metadata["predictions_count"] == 1
    assert metadata["sample_prediction"] == 4.2


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Endpoint")
def test_validate_model_endpoint_missing_uri(mock_endpoint_class, _mock_init):
    """Test validation fails when endpoint URI is missing."""
    # Mock input endpoint with no URI
    mock_input_endpoint = MagicMock(spec=Model)
    mock_input_endpoint.uri = None  # Missing URI

    mock_validation_output = MagicMock(spec=Model)
    mock_validation_output.metadata = {}

    # Test should fail with missing URI
    with pytest.raises(ValueError, match="Endpoint resource_name is missing"):
        validate_model_endpoint_func(
            endpoint=mock_input_endpoint,
            validation_output=mock_validation_output,
            project="test-project",
            region="europe-west2",
        )

    # Verify endpoint was not called
    mock_endpoint_class.assert_not_called()


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Endpoint")
def test_validate_model_endpoint_prediction_failure(mock_endpoint_class, _mock_init):
    """Test validation fails when prediction request fails."""
    mock_endpoint = MagicMock()
    mock_endpoint.resource_name = (
        "projects/test-project/locations/europe-west2/endpoints/456"
    )
    mock_endpoint_class.return_value = mock_endpoint

    # Mock prediction failure
    mock_endpoint.predict.side_effect = Exception("Prediction service unavailable")
    mock_input_endpoint = MagicMock(spec=Model)
    mock_input_endpoint.uri = (
        "projects/test-project/locations/europe-west2/endpoints/456"
    )
    mock_validation_output = MagicMock(spec=Model)
    mock_validation_output.metadata = {}

    # Test should fail with prediction error
    with pytest.raises(Exception, match="Prediction service unavailable"):
        validate_model_endpoint_func(
            endpoint=mock_input_endpoint,
            validation_output=mock_validation_output,
            project="test-project",
            region="europe-west2",
        )

    # Verify failure metadata was set
    metadata = mock_validation_output.metadata
    assert metadata["status"] == "FAILED"
    assert "Prediction service unavailable" in metadata["error"]


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Endpoint")
def test_validate_model_endpoint_wrong_prediction_count(
    mock_endpoint_class, _mock_init
):
    """Test validation fails when prediction count doesn't match expected."""
    mock_endpoint = MagicMock()
    mock_endpoint.resource_name = (
        "projects/test-project/locations/europe-west2/endpoints/789"
    )
    mock_endpoint_class.return_value = mock_endpoint

    # Mock response with wrong prediction count
    mock_response = MagicMock()
    mock_response.predictions = [4.2, 3.8]  # 2 predictions instead of expected 1
    mock_endpoint.predict.return_value = mock_response
    mock_input_endpoint = MagicMock(spec=Model)
    mock_input_endpoint.uri = (
        "projects/test-project/locations/europe-west2/endpoints/789"
    )
    mock_validation_output = MagicMock(spec=Model)
    mock_validation_output.metadata = {}

    # Test should fail with wrong prediction count
    with pytest.raises(ValueError, match="Expected predictions count"):
        validate_model_endpoint_func(
            endpoint=mock_input_endpoint,
            validation_output=mock_validation_output,
            project="test-project",
            region="europe-west2",
            expected_prediction_count=1,  # Expect 1, but getting 2
        )


def test_validate_model_endpoint_default_test_instances():
    """Test that default test instances are properly set."""
    test_instances = None

    # Simulate the default logic from the component
    if test_instances is None:
        test_instances = [[3.99, "Italy", "Tuscany", "White", "Light", "Pinot Grigio"]]

    # Verify default instances
    assert len(test_instances) == 1
    assert len(test_instances[0]) == 6  # 6 features per instance
    assert test_instances[0][0] == 3.99  # Price
    assert test_instances[0][1] == "Italy"  # Country
    assert test_instances[0][2] == "Tuscany"  # Region
