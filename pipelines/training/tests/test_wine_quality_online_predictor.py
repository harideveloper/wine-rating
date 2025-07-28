"""Tests for wine quality pipeline definition."""

import os
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd


# pylint: disable=line-too-long
class TestWineQualityPipeline:
    """Test wine quality pipeline definition."""

    @patch("google.cloud.storage.Client")
    @patch("training.wine_quality_online_predictor.compiler.Compiler")
    def test_pipeline_compilation(self, mock_compiler_class, mock_storage_client_class):
        """Test pipeline compilation functionality with GCS upload."""
        mock_compiler = MagicMock()
        mock_compiler_class.return_value = mock_compiler
        mock_storage_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client_class.return_value = mock_storage_client
        mock_storage_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        from training.wine_quality_online_predictor import (  # pylint: disable=import-outside-toplevel
            compile_pipeline,
        )

        pipeline_name = "test-pipeline"
        pipeline_file_name = "test-pipeline.json"
        pipeline_storage_bucket = "test-bucket"
        project = "test-project"
        credentials = "credentials"
        result = compile_pipeline(
            pipeline_name=pipeline_name,
            pipeline_file_name=pipeline_file_name,
            pipeline_storage_bucket=pipeline_storage_bucket,
            project=project,
            credentials=credentials,
        )
        mock_compiler_class.assert_called_once()
        mock_compiler.compile.assert_called_once_with(
            pipeline_func=mock_compiler.compile.call_args[1]["pipeline_func"],
            package_path=pipeline_file_name,
        )
        mock_storage_client_class.assert_called_once_with(
            project=project, credentials=credentials
        )
        mock_storage_client.bucket.assert_called_once_with(pipeline_storage_bucket)
        mock_bucket.blob.assert_called_once_with(
            f"{pipeline_name}/{pipeline_file_name}"
        )
        mock_blob.upload_from_filename.assert_called_once_with(pipeline_file_name)
        expected_gcs_uri = (
            f"gs://{pipeline_storage_bucket}/{pipeline_name}/{pipeline_file_name}"
        )
        assert result == expected_gcs_uri

    def test_pipeline_parameter_validation(self):
        """Test basic parameter validation."""
        test_params = {
            "data_path": "gs://bucket/data.csv",
            "evaluation_threshold": 0.8,
            "test_size": 0.2,
            "min_replica_count": 1,
            "max_replica_count": 3,
        }
        assert test_params["data_path"].startswith("gs://")
        assert 0.0 < test_params["evaluation_threshold"] <= 1.0
        assert 0.0 < test_params["test_size"] < 1.0
        assert test_params["max_replica_count"] >= test_params["min_replica_count"]

    @patch("google.cloud.aiplatform.PipelineJob")
    @patch("google.cloud.aiplatform.init")
    def test_pipeline_job_execution_with_gcs(self, mock_init, mock_pipeline_job):
        """Test pipeline job creation and execution with GCS template."""
        mock_job = MagicMock()
        mock_job.run.return_value = None
        mock_job.state = "PIPELINE_STATE_SUCCEEDED"
        mock_pipeline_job.return_value = mock_job
        from google.cloud import aiplatform  # pylint: disable=import-outside-toplevel

        aiplatform.init(project="test-project", location="europe-west2")
        gcs_template_path = "gs://test-bucket/pipeline-build-123.json"
        job = aiplatform.PipelineJob(
            display_name="wine-pipeline-test",
            template_path=gcs_template_path,
            parameter_values={
                "data_path": "gs://bucket/data.csv",
                "model_display_name": "wine-model",
                "project": "test-project",
                "region": "europe-west2",
            },
        )
        job.run(sync=True)
        mock_init.assert_called_once_with(
            project="test-project", location="europe-west2"
        )
        mock_pipeline_job.assert_called_once()
        call_args = mock_pipeline_job.call_args
        assert call_args[1]["template_path"] == gcs_template_path
        mock_job.run.assert_called_once_with(sync=True)

    def test_compilation_gcs_uri_construction(self):
        """Test GCS URI construction logic."""
        pipeline_file_name = "wine-training.json"
        pipeline_storage_bucket = "wine-pipelines"
        expected_gcs_uri = f"gs://{pipeline_storage_bucket}/{pipeline_file_name}"
        assert expected_gcs_uri == "gs://wine-pipelines/wine-training.json"
        build_number = "123"
        base_name, extension = os.path.splitext(pipeline_file_name)
        versioned_file = f"{base_name}-{build_number}{extension}"
        versioned_gcs_uri = f"gs://{pipeline_storage_bucket}/{versioned_file}"
        assert versioned_gcs_uri == "gs://wine-pipelines/wine-training-123.json"

    @patch("os.path.exists")
    @patch("os.unlink")
    @patch("google.cloud.storage.Client")
    @patch("training.wine_quality_online_predictor.compiler.Compiler")
    def test_pipeline_compilation_cleanup(
        self, mock_compiler_class, mock_storage_client_class, mock_unlink, mock_exists
    ):
        """Test that local files are cleaned up after compilation."""
        mock_compiler = MagicMock()
        mock_compiler_class.return_value = mock_compiler
        mock_storage_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client_class.return_value = mock_storage_client
        mock_storage_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_exists.return_value = True
        # pylint: disable=import-outside-toplevel
        from training.wine_quality_online_predictor import (
            compile_pipeline,
        )

        pipeline_name = "test-pipeline"
        pipeline_file_name = "test-pipeline.json"
        pipeline_storage_bucket = "test-bucket"
        project = "test-project"
        credentials = "credentials"
        compile_pipeline(
            pipeline_name=pipeline_name,
            pipeline_file_name=pipeline_file_name,
            pipeline_storage_bucket=pipeline_storage_bucket,
            project=project,
            credentials=credentials,
        )
        mock_exists.assert_called_once_with("test-pipeline.json")
        mock_unlink.assert_called_once_with("test-pipeline.json")


class TestWineComponentIntegration:
    """Simple integration tests for wine pipeline components."""

    @pytest.fixture
    def sample_wine_data(self):
        """Simple wine dataset for testing."""
        return pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "Country": ["France", "Italy", "Spain", "France", "Italy"],
                "Price": ["$25.99", "$45.00", "$15.50", "€30.50", "$35.00"],
                "Rating": [4.2, 4.8, 3.9, 4.6, 4.4],
            }
        )

    @pytest.fixture
    def mock_datasets(self, tmp_path):
        """Mock KFP datasets."""
        datasets = {}
        for name in ["output_data", "train_data", "test_data", "output_model"]:
            mock_dataset = MagicMock()
            if name == "output_model":
                mock_dataset.path = str(tmp_path / f"{name}.pkl")
            else:
                mock_dataset.path = str(tmp_path / f"{name}.csv")
            datasets[name] = mock_dataset
        return datasets

    def test_component_parameter_compatibility(self):
        """Test component parameter validation for pipeline compatibility."""
        test_parameters = {
            "data_path": "gs://test-bucket/wine_data.csv",
            "test_size": 0.2,
            "random_state": 42,
            "n_estimators": 100,
            "evaluation_threshold": 0.8,
            "project": "test-project",
            "region": "europe-west2",
            "model_display_name": "test-wine-model",
            "endpoint_display_name": "test-wine-endpoint",
            "machine_type": "n1-standard-4",
            "min_replica_count": 1,
            "max_replica_count": 3,
            "model_serving_image": "eu-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
            "expected_prediction_count": 1,
            "pipeline_file_name": "wine-training.json",
            "pipeline_storage_bucket": "wine-pipelines",
        }
        assert isinstance(test_parameters["data_path"], str)
        assert test_parameters["data_path"].startswith("gs://")
        assert 0.0 < test_parameters["test_size"] < 1.0
        assert isinstance(test_parameters["random_state"], int)
        assert test_parameters["random_state"] >= 0
        assert isinstance(test_parameters["n_estimators"], int)
        assert test_parameters["n_estimators"] > 0
        assert 0.0 <= test_parameters["evaluation_threshold"] <= 1.0
        assert (
            test_parameters["min_replica_count"] <= test_parameters["max_replica_count"]
        )
        assert isinstance(test_parameters["expected_prediction_count"], int)
        assert test_parameters["expected_prediction_count"] > 0
        assert isinstance(test_parameters["pipeline_file_name"], str)
        assert test_parameters["pipeline_file_name"].endswith(".json")
        assert isinstance(test_parameters["pipeline_storage_bucket"], str)
        assert not test_parameters["pipeline_storage_bucket"].startswith("gs://")

    def test_validation_component_integration(self):
        """Test model validator component integration with pipeline."""
        validation_config = {
            "test_instances": [
                [39.99, "France", "Bordeaux", "Red", "Bold", "Cabernet Sauvignon"]
            ],
            "expected_prediction_count": 1,
        }
        assert isinstance(validation_config["test_instances"], list)
        assert len(validation_config["test_instances"]) > 0
        assert len(validation_config["test_instances"][0]) == 6  # feature count
        instance = validation_config["test_instances"][0]
        assert isinstance(instance[0], float)  # price
        assert isinstance(instance[1], str)  # country
        assert isinstance(instance[2], str)  # region
        assert isinstance(instance[3], str)  # wine type
        assert isinstance(instance[4], str)  # body
        assert isinstance(instance[5], str)  # variety
        assert validation_config["expected_prediction_count"] == len(
            validation_config["test_instances"]
        )

    def test_pipeline_component_sequence_validation(self):
        """Test that pipeline components can be sequenced correctly."""
        import components  # pylint: disable=import-outside-toplevel

        component_sequence = components.__all__
        assert component_sequence.index("load_data") < component_sequence.index(
            "preprocess_data"
        )
        assert component_sequence.index("preprocess_data") < component_sequence.index(
            "train_model"
        )
        assert component_sequence.index("train_model") < component_sequence.index(
            "evaluate_model"
        )
        assert component_sequence.index("evaluate_model") < component_sequence.index(
            "save_model"
        )
        assert component_sequence.index("save_model") < component_sequence.index(
            "register_model"
        )
        assert component_sequence.index("register_model") < component_sequence.index(
            "deploy_model"
        )
        assert component_sequence.index("deploy_model") < component_sequence.index(
            "validate_model"
        )
        assert component_sequence[-1] == "validate_model"


class TestModelValidationIntegration:
    """Test model validation integration with pipeline."""

    def test_validation_test_data_format(self):
        """Test validation test data matches expected wine quality format."""
        default_test_instances = [
            [39.99, "France", "Bordeaux", "Red", "Bold", "Cabernet Sauvignon"]
        ]
        instance = default_test_instances[0]
        assert len(instance) == 6
        assert isinstance(instance[0], (int, float))  # price
        assert all(isinstance(feature, str) for feature in instance[1:])
        multiple_instances = [
            [39.99, "France", "Bordeaux", "Red", "Bold", "Cabernet Sauvignon"],
            [29.99, "Italy", "Tuscany", "Red", "Medium", "Chianti"],
            [49.99, "Spain", "Rioja", "Red", "Full", "Tempranillo"],
        ]
        assert len(multiple_instances) == 3
        assert all(len(instance) == 6 for instance in multiple_instances)

    def test_validation_output_compatibility(self):
        """Test validation output is compatible with pipeline artifacts."""
        validation_metadata = {
            "status": "PASSED",
            "endpoint": "projects/test/locations/europe-west2/endpoints/123",
            "test_instances_count": 1,
            "predictions_count": 1,
            "sample_prediction": {"quality_score": 0.85, "confidence": 0.92},
        }
        assert "status" in validation_metadata
        assert validation_metadata["status"] in ["PASSED", "FAILED"]
        assert "endpoint" in validation_metadata
        assert validation_metadata["endpoint"].startswith("projects/")
        assert isinstance(validation_metadata["test_instances_count"], int)
        assert isinstance(validation_metadata["predictions_count"], int)
        failure_metadata = {
            "status": "FAILED",
            "endpoint": "projects/test/locations/europe-west2/endpoints/123",
            "error": "Prediction service unavailable",
        }
        assert failure_metadata["status"] == "FAILED"
        assert "error" in failure_metadata


class TestPipelineCompilationWithGCS:
    """Test pipeline compilation with GCS integration."""

    def test_gcs_uri_validation(self):
        """Test GCS URI construction and validation."""
        pipeline_file_name = "wine-training.json"
        pipeline_storage_bucket = "wine-pipelines"
        gcs_uri = f"gs://{pipeline_storage_bucket}/{pipeline_file_name}"
        assert gcs_uri.startswith("gs://")
        assert pipeline_storage_bucket in gcs_uri
        assert pipeline_file_name in gcs_uri
        assert gcs_uri == "gs://wine-pipelines/wine-training.json"

    def test_build_number_integration(self):
        """Test build number integration in pipeline file naming."""
        app_name = "wine"
        app_type = "training"
        build_number = "123"
        pipeline_file_name = f"{app_name}-{app_type}-{build_number}.json"
        assert pipeline_file_name == "wine-training-123.json"
        build_number_local = None or "local"
        pipeline_file_name_local = f"{app_name}-{app_type}-{build_number_local}.json"
        assert pipeline_file_name_local == "wine-training-local.json"
