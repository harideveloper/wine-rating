"""Tests for model deployer component."""

from unittest.mock import MagicMock, patch
import pytest
from kfp.v2.dsl import Model


from pipelines.components.model_deployer import deploy_model

deploy_model_func = deploy_model.python_func


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model")
@patch("google.cloud.aiplatform.Endpoint.list")
@patch("google.cloud.aiplatform.Endpoint.create")
def test_deploy_model_to_new_endpoint_success(
    mock_endpoint_create, mock_endpoint_list, mock_model_class, mock_init
):
    """Test successful model deployment to new endpoint."""
    # Mock no existing endpoints
    mock_endpoint_list.return_value = []
    mock_model = MagicMock()
    mock_model.resource_name = "projects/test-project/locations/europe-west2/models/123"
    mock_model_class.return_value = mock_model
    mock_endpoint = MagicMock()
    mock_endpoint.resource_name = (
        "projects/test-project/locations/europe-west2/endpoints/456"
    )
    mock_endpoint_create.return_value = mock_endpoint
    mock_registered_model = MagicMock(spec=Model)
    mock_registered_model.metadata = {
        "resource_name": "projects/test-project/locations/europe-west2/models/123",
        "display_name": "wine-quality-model",
    }

    mock_deployed_model = MagicMock(spec=Model)
    deploy_model_func(
        registered_model=mock_registered_model,
        deployed_model=mock_deployed_model,
        endpoint_display_name="wine-quality-endpoint",
        project="test-project",
        region="europe-west2",
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3,
    )

    mock_init.assert_called_once_with(project="test-project", location="europe-west2")
    mock_model_class.assert_called_once_with(
        "projects/test-project/locations/europe-west2/models/123"
    )

    mock_endpoint_list.assert_called_once_with(
        filter='display_name="wine-quality-endpoint"',
        order_by="create_time desc",
        project="test-project",
        location="europe-west2",
    )
    mock_endpoint_create.assert_called_once_with(
        display_name="wine-quality-endpoint",
        project="test-project",
        location="europe-west2",
    )
    mock_endpoint.deploy.assert_called_once_with(
        model=mock_model,
        deployed_model_display_name="wine-quality-model-deployed",
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3,
        traffic_split={"0": 100},
    )

    # Verify deployed model output
    assert (
        mock_deployed_model.uri
        == "projects/test-project/locations/europe-west2/endpoints/456"
    )


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model")
@patch("google.cloud.aiplatform.Endpoint.list")
def test_deploy_model_to_existing_endpoint_success(
    mock_endpoint_list, mock_model_class, _mock_init
):
    """Test successful model deployment to existing endpoint."""
    # Mock existing endpoint
    mock_existing_endpoint = MagicMock()
    mock_existing_endpoint.resource_name = (
        "projects/test-project/locations/europe-west2/endpoints/789"
    )
    mock_endpoint_list.return_value = [mock_existing_endpoint]
    mock_model = MagicMock()
    mock_model.resource_name = "projects/test-project/locations/europe-west2/models/123"
    mock_model_class.return_value = mock_model
    mock_registered_model = MagicMock(spec=Model)
    mock_registered_model.metadata = {
        "resource_name": "projects/test-project/locations/europe-west2/models/123",
        "display_name": "wine-quality-model-v2",
    }
    mock_deployed_model = MagicMock(spec=Model)
    deploy_model_func(
        registered_model=mock_registered_model,
        deployed_model=mock_deployed_model,
        endpoint_display_name="wine-quality-endpoint",
        project="test-project",
        region="europe-west2",
        machine_type="n1-standard-2",
        min_replica_count=2,
        max_replica_count=5,
    )
    mock_endpoint_list.assert_called_once_with(
        filter='display_name="wine-quality-endpoint"',
        order_by="create_time desc",
        project="test-project",
        location="europe-west2",
    )
    mock_existing_endpoint.deploy.assert_called_once_with(
        model=mock_model,
        deployed_model_display_name="wine-quality-model-v2-deployed",
        machine_type="n1-standard-2",
        min_replica_count=2,
        max_replica_count=5,
        traffic_split={"0": 100},
    )

    # Verify deployed model URI is set to existing endpoint
    assert (
        mock_deployed_model.uri
        == "projects/test-project/locations/europe-west2/endpoints/789"
    )


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model")
def test_deploy_model_missing_resource_name(mock_model_class, _mock_init):
    """Test deployment fails when model resource_name is missing from metadata."""
    # Mock registered model without resource_name
    mock_registered_model = MagicMock(spec=Model)
    mock_registered_model.metadata = {
        "display_name": "wine-quality-model"
        # Missing: resource_name
    }

    mock_deployed_model = MagicMock(spec=Model)

    # Test should fail with missing resource_name
    with pytest.raises(
        ValueError, match="Model resource_name is missing from metadata"
    ):
        deploy_model_func(
            registered_model=mock_registered_model,
            deployed_model=mock_deployed_model,
            endpoint_display_name="wine-quality-endpoint",
            project="test-project",
            region="europe-west2",
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=3,
        )

    # Verify model retrieval was not attempted
    mock_model_class.assert_not_called()


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model")
def test_deploy_model_invalid_model_retrieval(mock_model_class, _mock_init):
    """Test deployment fails when model retrieval fails."""
    # Mock model retrieval to raise exception
    mock_model_class.side_effect = Exception("Model not found in Vertex AI")

    # Mock registered model input
    mock_registered_model = MagicMock(spec=Model)
    mock_registered_model.metadata = {
        "resource_name": "projects/test-project/locations/europe-west2/models/invalid",
        "display_name": "wine-quality-model",
    }

    mock_deployed_model = MagicMock(spec=Model)

    # Test should fail with model retrieval error
    with pytest.raises(Exception, match="Model not found in Vertex AI"):
        deploy_model_func(
            registered_model=mock_registered_model,
            deployed_model=mock_deployed_model,
            endpoint_display_name="wine-quality-endpoint",
            project="test-project",
            region="europe-west2",
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=3,
        )


@patch("google.cloud.aiplatform.init")
@patch("google.cloud.aiplatform.Model")
@patch("google.cloud.aiplatform.Endpoint.list")
@patch("google.cloud.aiplatform.Endpoint.create")
def test_deploy_model_deployment_failure_with_endpoint_uri_set(
    mock_endpoint_create, mock_endpoint_list, mock_model_class, _mock_init
):
    """Test deployment failure still sets endpoint URI when endpoint exists."""
    # Mock no existing endpoints
    mock_endpoint_list.return_value = []
    mock_model = MagicMock()
    mock_model.resource_name = "projects/test-project/locations/europe-west2/models/123"
    mock_model_class.return_value = mock_model
    mock_endpoint = MagicMock()
    mock_endpoint.resource_name = (
        "projects/test-project/locations/europe-west2/endpoints/456"
    )
    mock_endpoint_create.return_value = mock_endpoint
    mock_endpoint.deploy.side_effect = Exception(
        "Deployment failed due to resource limits"
    )
    mock_registered_model = MagicMock(spec=Model)
    mock_registered_model.metadata = {
        "resource_name": "projects/test-project/locations/europe-west2/models/123",
        "display_name": "wine-quality-model",
    }
    mock_deployed_model = MagicMock(spec=Model)

    # Test deployment failure
    with pytest.raises(Exception, match="Deployment failed due to resource limits"):
        deploy_model_func(
            registered_model=mock_registered_model,
            deployed_model=mock_deployed_model,
            endpoint_display_name="wine-quality-endpoint",
            project="test-project",
            region="europe-west2",
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=3,
        )

    # Verify endpoint URI was still set despite deployment failure
    assert (
        mock_deployed_model.uri
        == "projects/test-project/locations/europe-west2/endpoints/456"
    )
