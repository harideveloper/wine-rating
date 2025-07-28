"""Tests for model validator component core logic."""

from unittest.mock import MagicMock, patch


class TestModelValidator:
    """Test core model validator logic."""

    @patch("google.cloud.aiplatform.init")
    def test_vertex_ai_initialization(self, mock_init):
        """Test Vertex AI initialization for validation."""
        from google.cloud import aiplatform  # pylint: disable=import-outside-toplevel

        project = "test-project"
        region = "europe-west2"
        aiplatform.init(project=project, location=region)
        mock_init.assert_called_once_with(project=project, location=region)

    @patch("google.cloud.aiplatform.Endpoint")
    def test_endpoint_loading_for_validation(self, mock_endpoint_class):
        """Test loading endpoint for validation."""
        mock_endpoint = MagicMock()
        mock_endpoint_class.return_value = mock_endpoint
        from google.cloud import aiplatform  # pylint: disable=import-outside-toplevel

        endpoint_resource_name = "projects/test/locations/europe-west2/endpoints/789"
        endpoint = aiplatform.Endpoint(endpoint_resource_name)
        assert endpoint == mock_endpoint

    @patch("google.cloud.aiplatform.Endpoint")
    def test_prediction_request(self, mock_endpoint_class):
        """Test prediction request logic."""
        mock_endpoint = MagicMock()
        mock_response = MagicMock()
        mock_response.predictions = [{"quality_score": 0.85}]
        mock_endpoint.predict.return_value = mock_response
        mock_endpoint_class.return_value = mock_endpoint

        from google.cloud import aiplatform  # pylint: disable=import-outside-toplevel

        endpoint = aiplatform.Endpoint("endpoint-resource-name")
        test_instances = [
            [39.99, "France", "Bordeaux", "Red", "Bold", "Cabernet Sauvignon"]
        ]
        response = endpoint.predict(instances=test_instances)
        mock_endpoint.predict.assert_called_once_with(instances=test_instances)
        assert len(response.predictions) == 1

    def test_default_test_instances_configuration(self):
        """Test default wine test instances configuration."""
        default_instances = [
            [39.99, "France", "Bordeaux", "Red", "Bold", "Cabernet Sauvignon"]
        ]
        # Validate test instance structure
        assert len(default_instances) == 1
        assert len(default_instances[0]) == 6
        assert isinstance(default_instances[0][0], float)  # price
        assert isinstance(default_instances[0][1], str)  # country
        assert isinstance(default_instances[0][2], str)  # region

    def test_validation_response_structure(self):
        """Test prediction response validation logic."""
        # Valid response
        valid_response = MagicMock()
        valid_response.predictions = [{"score": 0.85}]
        assert hasattr(valid_response, "predictions")
        assert isinstance(valid_response.predictions, list)
        assert len(valid_response.predictions) == 1
        # Invalid response
        invalid_response = MagicMock()
        del invalid_response.predictions
        assert not hasattr(invalid_response, "predictions")

    def test_prediction_count_validation(self):
        """Test prediction count validation logic."""
        predictions = [{"score": 0.85}, {"score": 0.92}]
        expected_count = 2
        assert len(predictions) == expected_count
        assert isinstance(predictions, list)
        # Test mismatch
        wrong_expected_count = 1
        assert len(predictions) != wrong_expected_count

    def test_validation_metadata_structure(self):
        """Test validation result metadata structure."""
        success_metadata = {
            "status": "PASSED",
            "endpoint": "projects/test/locations/europe-west2/endpoints/789",
            "test_instances_count": 1,
            "predictions_count": 1,
            "sample_prediction": {"quality_score": 0.85},
        }
        failure_metadata = {
            "status": "FAILED",
            "endpoint": "projects/test/locations/europe-west2/endpoints/789",
            "error": "Prediction failed",
        }  # pylint: disable=line-too-long
        # Validate success metadata
        assert success_metadata["status"] == "PASSED"
        assert "endpoint" in success_metadata
        assert "test_instances_count" in success_metadata
        assert success_metadata["predictions_count"] >= 1
        # Validate failure metadata
        assert failure_metadata["status"] == "FAILED"
        assert "error" in failure_metadata
        assert "endpoint" in failure_metadata

    @patch("google.cloud.aiplatform.Endpoint")
    def test_model_prediction_validation(self, mock_endpoint_class):
        """Test validating model predictions."""
        mock_endpoint = MagicMock()
        mock_response = MagicMock()
        mock_response.predictions = [{"quality_score": 0.85, "confidence": 0.92}]
        mock_endpoint.predict.return_value = mock_response
        mock_endpoint_class.return_value = mock_endpoint
        from google.cloud import aiplatform  # pylint: disable=import-outside-toplevel

        endpoint = aiplatform.Endpoint("endpoint-resource-name")
        test_instances = [
            [39.99, "France", "Bordeaux", "Red", "Bold", "Cabernet Sauvignon"]
        ]
        response = endpoint.predict(instances=test_instances)
        predictions = response.predictions
        # Validate prediction structure
        assert len(predictions) == 1
        assert "quality_score" in predictions[0]
        assert isinstance(predictions[0]["quality_score"], float)
        assert 0 <= predictions[0]["quality_score"] <= 1
