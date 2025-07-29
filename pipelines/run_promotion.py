import logging
import sys
from dataclasses import dataclass
from typing import Optional
from google.cloud import aiplatform
from google.oauth2.credentials import Credentials

from promotion import compile_pipeline
from promotion_constants import (
    AUTH_TOKEN,
    SOURCE_PROJECT_ID,
    TARGET_PROJECT_ID,
    LOCATION,
    PIPELINE_FILE,
    PROMOTION_JOB_DISPLAY_NAME,
    PIPELINE_BUCKET,
    PIPELINE_SA,
    BUILD_NUMBER,
    PIPELINE_ROOT_SUFFIX,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PromotionConfig:
    """Configuration for model promotion pipeline."""
    
    source_project_id: str = SOURCE_PROJECT_ID
    target_project_id: str = TARGET_PROJECT_ID
    location: str = LOCATION

    pipeline_bucket: str = PIPELINE_BUCKET
    pipeline_sa: str = PIPELINE_SA
    pipeline_file: str = PIPELINE_FILE
    pipeline_root: Optional[str] = None

    model_display_name: str = "wine-quality-online-prediction-model"

    auth_token: str = AUTH_TOKEN
    build_number: str = BUILD_NUMBER
    enable_caching: bool = True
    sync: bool = True


def validate_inputs(config: PromotionConfig) -> None:
    """Ensure all required config values are present."""
    logger.info("Validating configuration parameters")
    
    required_fields = [
        "source_project_id",
        "target_project_id",
        "location",
        "pipeline_bucket",
        "pipeline_sa",
        "pipeline_file"
    ]
    
    for field in required_fields:
        value = getattr(config, field)
        if not value:
            raise ValueError(f"Missing config parameter: {field}")
    
    logger.info("Configuration validation passed")


def run_model_promotion_pipeline(config: PromotionConfig):
    """Compile and launch the promotion pipeline."""
    try:
        logger.info("Starting model promotion pipeline")
        validate_inputs(config)

        if not config.auth_token:
            raise ValueError("AUTH_TOKEN is not set")

        credentials = Credentials(config.auth_token)

        aiplatform.init(project=config.target_project_id, location=config.location)

        if not config.pipeline_root:
            config.pipeline_root = f"gs://{config.pipeline_bucket}/{PIPELINE_ROOT_SUFFIX}"

        logger.info("Compiling pipeline")
        pipeline_file_gcs_uri = compile_pipeline(
            pipeline_name=PROMOTION_JOB_DISPLAY_NAME,
            pipeline_file_name=config.pipeline_file,
            pipeline_storage_bucket=config.pipeline_bucket,
            project=config.target_project_id,
            credentials=credentials,
        )

        parameter_values = {
            "source_project": config.source_project_id,
            "target_project": config.target_project_id,
            "location": config.location,
            "model_display_name": config.model_display_name,
        }

        logger.info("Pipeline parameters:")
        for key, value in parameter_values.items():
            logger.info(f"  {key}: {value}")

        logger.info("Creating pipeline job")
        job = aiplatform.PipelineJob(
            display_name=PROMOTION_JOB_DISPLAY_NAME,
            template_path=pipeline_file_gcs_uri,
            pipeline_root=config.pipeline_root,
            parameter_values=parameter_values,
            enable_caching=config.enable_caching,
        )

        logger.info("Submitting pipeline job")
        job.submit(service_account=config.pipeline_sa)

        if config.sync:
            logger.info("Waiting for pipeline completion...")
            job.wait()
            logger.info(f"Pipeline completed with state: {job.state}")
        else:
            logger.info(f"Pipeline submitted. Job name: {job.name}")

        return job

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


def main():
    try:
        config = PromotionConfig()

        logger.info("=== Clean Model Promotion Pipeline ===")
        logger.info(f"Source Project: {config.source_project_id}")
        logger.info(f"Target Project: {config.target_project_id}")
        logger.info(f"Location: {config.location}")
        logger.info(f"Model Display Name: {config.model_display_name}")

        job = run_model_promotion_pipeline(config)

        logger.info("Pipeline execution completed successfully")
        logger.info(f"Job Name: {job.name}")
        logger.info(f"Job State: {job.state}")

    except Exception as e:
        logger.error(f"Failed to run promotion pipeline: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
