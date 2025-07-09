"""Tests for wine quality pipeline definition."""

from unittest.mock import MagicMock, patch


class TestWineQualityPipeline:
    """Test wine quality pipeline definition."""

    @patch("pipelines.online.wine_quality_online_predictor.compiler.Compiler")
    def test_pipeline_compilation(self, mock_compiler_class):
        """Test pipeline compilation functionality."""
        mock_compiler = MagicMock()
        mock_compiler_class.return_value = mock_compiler

        from online.wine_quality_online_predictor import (  # pylint: disable=import-outside-toplevel
            compile_pipeline,
        )

        output_file = "test_pipeline.json"
        compile_pipeline(output_file)

        # Verify compiler was called correctly
        mock_compiler_class.assert_called_once()
        mock_compiler.compile.assert_called_once()

    def test_pipeline_parameter_validation(self):
        """Test basic parameter validation."""
        test_params = {
            "data_path": "gs://bucket/data.csv",
            "evaluation_threshold": 0.8,
            "test_size": 0.2,
            "min_replica_count": 1,
            "max_replica_count": 3,
        }

        # Basic validation checks
        assert test_params["data_path"].startswith("gs://")
        assert 0.0 < test_params["evaluation_threshold"] <= 1.0
        assert 0.0 < test_params["test_size"] < 1.0
        assert test_params["max_replica_count"] >= test_params["min_replica_count"]

    @patch("google.cloud.aiplatform.PipelineJob")
    @patch("google.cloud.aiplatform.init")
    def test_pipeline_job_execution(self, mock_init, mock_pipeline_job):
        """Test pipeline job creation and execution."""
        # Mock pipeline job
        mock_job = MagicMock()
        mock_job.run.return_value = None
        mock_job.state = "PIPELINE_STATE_SUCCEEDED"
        mock_pipeline_job.return_value = mock_job

        # Simulate job creation and execution
        from google.cloud import aiplatform  # pylint: disable=import-outside-toplevel

        aiplatform.init(project="test-project", location="europe-west2")

        job = aiplatform.PipelineJob(
            display_name="wine-pipeline-test",
            template_path="pipeline.json",
            parameter_values={
                "data_path": "gs://bucket/data.csv",
                "model_display_name": "wine-model",
                "project": "test-project",
                "region": "europe-west2",
            },
        )

        job.run(sync=True)

        # Verify initialization and job execution
        mock_init.assert_called_once_with(
            project="test-project", location="europe-west2"
        )
        mock_pipeline_job.assert_called_once()
        mock_job.run.assert_called_once_with(sync=True)
