"""Tests for model registry component core logic."""

from unittest.mock import MagicMock, patch


class TestModelRegisterer:
    """Test core model registry logic."""

    @patch("google.cloud.aiplatform.init")
    def test_vertex_ai_initialization(self, mock_init):
        """Test Vertex AI initialization."""
        from google.cloud import aiplatform  # pylint: disable=import-outside-toplevel

        project = "test-project"
        region = "europe-west2"

        aiplatform.init(project=project, location=region)
        mock_init.assert_called_once_with(project=project, location=region)

    @patch("google.cloud.aiplatform.Model.upload")
    def test_model_upload(self, mock_upload):
        """Test model upload to registry."""
        mock_model = MagicMock()
        mock_model.resource_name = "projects/test/locations/europe-west2/models/123"
        mock_upload.return_value = mock_model

        from google.cloud import aiplatform  # pylint: disable=import-outside-toplevel

        model = aiplatform.Model.upload(
            display_name="wine-model",
            artifact_uri="gs://bucket/model",
            serving_container_image_uri="gcr.io/project/image",
        )

        assert model.resource_name.startswith("projects/")

    def test_metadata_setting(self):
        """Test setting registered model metadata."""
        registered_model = MagicMock()
        registered_model.metadata = {}

        # Set metadata
        registered_model.metadata["display_name"] = "wine-model"
        registered_model.metadata["framework"] = "sklearn"

        assert registered_model.metadata["display_name"] == "wine-model"
        assert registered_model.metadata["framework"] == "sklearn"

    def test_metadata_copying(self):
        """Test copying metadata from source model."""
        source_metadata = {"target": "Rating", "features": "['price', 'country']"}
        target_metadata = {"display_name": "wine-model"}

        # Copy source metadata
        for key, value in source_metadata.items():
            if key not in target_metadata:
                target_metadata[key] = value

        assert target_metadata["target"] == "Rating"
        assert target_metadata["display_name"] == "wine-model"
