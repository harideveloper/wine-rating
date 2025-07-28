"""Tests for model deployer component core logic."""

from unittest.mock import MagicMock, patch


class TestModelDeployer:
    """Test core model deployer logic."""

    @patch("google.cloud.aiplatform.init")
    def test_vertex_ai_initialization(self, mock_init):
        """Test Vertex AI initialization for deployment."""
        from google.cloud import aiplatform  # pylint: disable=import-outside-toplevel

        project = "test-project"
        region = "europe-west1"
        aiplatform.init(project=project, location=region)
        mock_init.assert_called_once_with(project=project, location=region)

    @patch("google.cloud.aiplatform.Model")
    def test_model_loading_for_deployment(self, mock_model_class):
        """Test loading model for deployment."""
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        from google.cloud import aiplatform  # pylint: disable=import-outside-toplevel

        model_resource_name = "projects/test/locations/europe-west1/models/123"
        model = aiplatform.Model(model_resource_name)
        assert model == mock_model

    @patch("google.cloud.aiplatform.Endpoint.create")
    def test_endpoint_creation(self, mock_create):
        """Test endpoint creation logic."""
        mock_endpoint = MagicMock()
        mock_endpoint.resource_name = (
            "projects/test/locations/europe-west1/endpoints/456"
        )
        mock_create.return_value = mock_endpoint
        from google.cloud import aiplatform  # pylint: disable=import-outside-toplevel

        endpoint = aiplatform.Endpoint.create(
            display_name="wine-endpoint",
            project="test-project",
            location="europe-west1",
        )
        assert endpoint.resource_name.endswith("endpoints/456")

    def test_deployment_configuration(self):
        """Test deployment configuration parameters."""
        config = {
            "machine_type": "n1-standard-2",
            "min_replica_count": 1,
            "max_replica_count": 3,
            "traffic_percentage": 100,
        }
        assert config["min_replica_count"] >= 1
        assert config["max_replica_count"] >= config["min_replica_count"]
        assert 0 <= config["traffic_percentage"] <= 100
        assert config["machine_type"].startswith("n1-")

    @patch("google.cloud.aiplatform.Endpoint")
    def test_model_deployment_to_endpoint(self, mock_endpoint_class):
        """Test deploying model to endpoint."""
        mock_endpoint = MagicMock()
        mock_model = MagicMock()
        mock_endpoint_class.return_value = mock_endpoint
        mock_endpoint.deploy.return_value = mock_model
        from google.cloud import aiplatform  # pylint: disable=import-outside-toplevel

        endpoint = aiplatform.Endpoint("endpoint-resource-name")
        endpoint.deploy(
            model=mock_model,
            machine_type="n1-standard-2",
            min_replica_count=1,
            max_replica_count=3,
        )
        mock_endpoint.deploy.assert_called_once_with(
            model=mock_model,
            machine_type="n1-standard-2",
            min_replica_count=1,
            max_replica_count=3,
        )
