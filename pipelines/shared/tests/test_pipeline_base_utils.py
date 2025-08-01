"""Tests for pipeline base utilities."""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from google.oauth2.credentials import Credentials

from shared.pipeline_base_utils import (
    PipelineConfig,
    initialize_aiplatform,
    run_pipeline_with_error_handling,
    validate_pipeline_inputs,
    setup_credentials,
)


@dataclass
class MockPipelineConfig(PipelineConfig):
    """Mock pipeline config for testing."""

    project_id: str = "test-project"
    region: str = "europe-west2"
    pipeline_bucket: str = "test-bucket"
    pipeline_sa: str = "test@service-account.com"
    pipeline_file: str = "test.json"
    auth_token: str = "test-token"
    build_number: str = "123"
    is_local: bool = True


def test_pipeline_config_creation():
    """Test PipelineConfig can be created with required fields."""
    config = PipelineConfig(
        project_id="test-project",
        region="europe-west2",
        pipeline_bucket="test-bucket",
        pipeline_sa="test@service-account.com",
        pipeline_file="test.json",
        auth_token="test-token",
        build_number="123",
        is_local=True,
    )

    assert config.project_id == "test-project"
    assert config.region == "europe-west2"
    assert config.is_local is True
    assert config.enable_caching is True  # default value
    assert config.sync is True  # default value


@patch("shared.pipeline_base_utils.aiplatform")
@patch("shared.pipeline_base_utils.logger")
def test_initialize_aiplatform_local(mock_logger, mock_aiplatform):
    """Test AI Platform initialization for local environment."""
    initialize_aiplatform(
        project_id="test-project", region="europe-west2", is_local=True
    )

    mock_aiplatform.init.assert_called_once_with(
        project="test-project", location="europe-west2"
    )
    mock_logger.info.assert_called()


@patch("shared.pipeline_base_utils.aiplatform")
@patch("shared.pipeline_base_utils.logger")
def test_initialize_aiplatform_non_local(mock_logger, mock_aiplatform):
    """Test AI Platform initialization for CI/production environment."""
    mock_credentials = MagicMock(spec=Credentials)

    initialize_aiplatform(
        project_id="test-project",
        region="europe-west2",
        is_local=False,
        credentials=mock_credentials,
        service_account="test@service-account.com",
    )

    mock_aiplatform.init.assert_called_once_with(
        project="test-project",
        location="europe-west2",
        credentials=mock_credentials,
        service_account="test@service-account.com",
    )


def test_initialize_aiplatform_missing_credentials():
    """Test that ValueError is raised when credentials are missing."""
    with pytest.raises(ValueError, match="Credentials are required"):
        initialize_aiplatform(
            project_id="test-project", region="europe-west2", is_local=False
        )


def test_validate_pipeline_inputs_success():
    """Test successful validation."""
    config = MockPipelineConfig()
    mandatory_fields = ["project_id", "region"]

    # Should not raise any exception
    validate_pipeline_inputs(config, mandatory_fields)


def test_validate_pipeline_inputs_missing_field():
    """Test validation with missing field."""
    config = MockPipelineConfig(project_id="")  # Empty project_id
    mandatory_fields = ["project_id", "region"]

    with pytest.raises(ValueError, match="Missing Config Parameter for: project_id"):
        validate_pipeline_inputs(config, mandatory_fields)


def test_validate_pipeline_inputs_missing_auth_token():
    """Test validation with missing auth token for non-local."""
    config = MockPipelineConfig(is_local=False, auth_token="")
    mandatory_fields = ["project_id"]

    with pytest.raises(ValueError, match="AUTH_TOKEN is required"):
        validate_pipeline_inputs(config, mandatory_fields)


def test_setup_credentials_local():
    """Test credential setup for local environment."""
    config = MockPipelineConfig(is_local=True)
    credentials = setup_credentials(config)

    assert credentials is None


@patch("shared.pipeline_base_utils.Credentials")
def test_setup_credentials_non_local(mock_credentials_class):
    """Test credential setup for non-local environment."""
    mock_credentials = MagicMock()
    mock_credentials_class.return_value = mock_credentials

    config = MockPipelineConfig(is_local=False, auth_token="test-token")
    credentials = setup_credentials(config)

    mock_credentials_class.assert_called_once_with("test-token")
    assert credentials == mock_credentials


@patch("shared.pipeline_base_utils.run_pipeline")
@patch("shared.pipeline_base_utils.logger")
def test_run_pipeline_with_error_handling_success(mock_logger, mock_run_pipeline):
    """Test successful pipeline execution."""
    mock_job = MagicMock()
    mock_job.state = "PIPELINE_STATE_SUCCEEDED"
    mock_run_pipeline.return_value = mock_job

    def mock_compile_function(**kwargs):
        return "gs://test-bucket/test-pipeline.json"

    # Should not raise any exception
    run_pipeline_with_error_handling(
        config=MockPipelineConfig(),
        compile_function=mock_compile_function,
        job_display_name="test-job",
        storage_bucket="test-bucket",
        parameter_values={"param1": "value1"},
        mandatory_fields=["project_id", "region"],
        success_message="Test successful",
    )

    # Verify success message was logged
    mock_logger.info.assert_called_with("Test successful")


@patch("shared.pipeline_base_utils.sys.exit")
@patch("shared.pipeline_base_utils.logger")
def test_run_pipeline_with_error_handling_failure(mock_logger, mock_exit):
    """Test pipeline execution with failure."""

    def mock_compile_function(**kwargs):
        raise ValueError("Test error")

    run_pipeline_with_error_handling(
        config=MockPipelineConfig(),
        compile_function=mock_compile_function,
        job_display_name="test-job",
        storage_bucket="test-bucket",
        parameter_values={"param1": "value1"},
        mandatory_fields=["project_id", "region"],
        success_message="Test successful",
    )

    # Verify error was logged and sys.exit was called
    mock_logger.error.assert_called()
    mock_exit.assert_called_with(1)
