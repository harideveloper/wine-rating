"""Run the wine quality prediction online pipeline on Vertex AI."""

import logging
import sys
from dataclasses import dataclass
from typing import Optional
from google.cloud import aiplatform
from google.oauth2.credentials import Credentials
from training import compile_pipeline
from constants import (
    AUTH_TOKEN,
    PROJECT_ID,
    DATA_PATH,
    REGION,
    PIPELINE_FILE,
    PIPELINE_JOB_DISPLAY_NAME,
    MODEL_DISPLAY_NAME,
    MODEL_ENDPOINT_DISPLAY_NAME,
    EVALUATION_THRESHOLD,
    TEST_SIZE,
    RANDOM_STATE,
    N_ESTIMATORS,
    MACHINE_TYPE,
    MIN_REPLICA_COUNT,
    MAX_REPLICA_COUNT,
    MODEL_SERVING_IMAGE,
    DATA_BUCKET,
    PIPELINE_SA,
    BUILD_NUMBER,
    PIPELINE_BUCKET,
)


@dataclass
class PipelineConfig:
    """Configuration for wine quality pipeline."""

    # pylint: disable=too-many-instance-attributes
    project_id: str = PROJECT_ID
    data_bucket: str = DATA_BUCKET
    data_path: str = DATA_PATH
    pipeline_bucket: str = PIPELINE_BUCKET
    pipeline_sa: str = PIPELINE_SA
    model_display_name: str = MODEL_DISPLAY_NAME
    endpoint_display_name: str = MODEL_ENDPOINT_DISPLAY_NAME
    region: str = REGION
    pipeline_root: Optional[str] = None
    pipeline_file: str = PIPELINE_FILE
    evaluation_threshold: float = EVALUATION_THRESHOLD
    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE
    n_estimators: int = N_ESTIMATORS
    machine_type: str = MACHINE_TYPE
    min_replica_count: int = MIN_REPLICA_COUNT
    max_replica_count: int = MAX_REPLICA_COUNT
    model_serving_image: str = MODEL_SERVING_IMAGE
    auth_token: str = AUTH_TOKEN
    build_number: str = BUILD_NUMBER
    enable_caching: bool = True
    sync: bool = True


def validate_inputs(config: PipelineConfig) -> None:
    """Validate input parameters."""
    print("mandatory config")
    mandatory_configs = [
        "project_id",
        "data_bucket",
        "data_path",
        "pipeline_bucket",
        "pipeline_sa",
        "model_display_name",
        "endpoint_display_name",
        "region",
        "pipeline_file",
        "evaluation_threshold",
        "test_size",
        "n_estimators",
        "machine_type",
        "min_replica_count",
        "min_replica_count",
        "model_serving_image",
    ]
    for mandatory_config in mandatory_configs:
        value = getattr(config, mandatory_config)
        if not value or (isinstance(value, str) and value.strip() == ""):
            raise ValueError(f"Missing Config Parameter for: {mandatory_config}")


def run_wine_quality_online_predictor(config: PipelineConfig):
    """
    Run the wine quality prediction pipeline on Vertex AI.
    Args:
        config: Pipeline configuration object
    Returns:
        aiplatform.PipelineJob: The pipeline job object
    """
    try:
        logging.info("Starting Wine Quality Online Predictor Pipeline")
        # Validate inputs
        validate_inputs(config)
        # Initialize Vertex AI
        logging.info("Initializing Vertex AI")
        if not AUTH_TOKEN:
            raise ValueError(
                "CLOUDSDK_AUTH_ACCESS_TOKEN environment variable is not set"
            )
        credentials = Credentials(AUTH_TOKEN)
        aiplatform.init(
            project=config.project_id,
            location=config.region,
            credentials=credentials,
            service_account=PIPELINE_SA,
        )  # pylint: disable=line-too-long
        # Compile pipeline
        logging.info("Compiling pipeline")
        pipeline_file_gcs_uri = compile_pipeline(
            pipeline_name=PIPELINE_JOB_DISPLAY_NAME,
            pipeline_file_name=PIPELINE_FILE,
            pipeline_storage_bucket=DATA_BUCKET,
            project=config.project_id,
            credentials=credentials,  # pylint: disable=line-too-long
        )
        # Create pipeline job
        logging.info("Creating pipeline job")
        job = aiplatform.PipelineJob(
            display_name=PIPELINE_JOB_DISPLAY_NAME,
            template_path=pipeline_file_gcs_uri,
            pipeline_root=config.pipeline_root,
            parameter_values={
                "data_path": config.data_path,
                "model_display_name": config.model_display_name,
                "endpoint_display_name": config.endpoint_display_name,
                "project": config.project_id,
                "region": config.region,
                "evaluation_threshold": config.evaluation_threshold,
                "test_size": config.test_size,
                "random_state": config.random_state,
                "n_estimators": config.n_estimators,
                "machine_type": config.machine_type,
                "min_replica_count": config.min_replica_count,
                "max_replica_count": config.max_replica_count,
                "model_serving_image": config.model_serving_image,
            },
            enable_caching=config.enable_caching,
        )
        # Run pipeline job
        logging.info("Starting pipeline execution")
        job.run(sync=config.sync)
        if config.sync:
            if job.state == "PIPELINE_STATE_SUCCEEDED":
                logging.info("Pipeline completed successfully")
            else:
                logging.error("Pipeline failed with state: %s", job.state)
        else:
            logging.info("Pipeline started asynchronously")
        return job
    except ValueError as ve:
        logging.error("Input validation failed: %s", ve)
        raise
    except Exception as e:
        logging.error("Pipeline execution failed: %s", e)
        raise


if __name__ == "__main__":
    try:
        pipeline_config = PipelineConfig()
        result = run_wine_quality_online_predictor(pipeline_config)
        if result.state == "PIPELINE_STATE_SUCCEEDED":
            logging.info("Success! Wine quality online predictions are ready")
        else:
            logging.warning("Pipeline completed with state: %s", result.state)
    except (ValueError, RuntimeError) as exc:
        logging.error("Pipeline execution failed: %s", exc)
        sys.exit(1)
