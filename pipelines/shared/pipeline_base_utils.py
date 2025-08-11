"""Base utilities for pipeline execution and management"""

# pylint: disable=too-many-positional-arguments
import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable
from kfp.v2 import compiler
from google.cloud import aiplatform, storage
from google.oauth2.credentials import Credentials
from pipelines.shared.log_utils import get_logger

logger = get_logger(__name__)


def initialize_aiplatform(
    project_id: str,
    region: str,
    is_local: bool,
    credentials: Optional[Credentials] = None,
    service_account: Optional[str] = None,
) -> None:
    """
    Initialize AI Platform with appropriate configuration based on environment flag.

    Args:
        project_id: GCP project ID
        region: GCP region
        is_local: Flag indicating if running in local environment
        credentials: OAuth2 credentials (used when not local)
        service_account: Service account email (used when not local)
    """
    if is_local:
        logger.info("Initializing AI Platform for local environment")
        aiplatform.init(project=project_id, location=region)
    else:
        logger.info("Initializing AI Platform for CI/production environment")
        if not credentials:
            raise ValueError("Credentials are required for non-local environment")
        if not service_account:
            raise ValueError("Service account is required for non-local environment")

        aiplatform.init(
            project=project_id,
            location=region,
            credentials=credentials,
            service_account=service_account,
        )

    logger.info(
        "AI Platform initialized for project: %s, region: %s", project_id, region
    )


# pylint: disable=too-many-instance-attributes
@dataclass
class PipelineConfig:
    """Base configuration dataclass for all pipelines."""

    project_id: str
    region: str
    pipeline_bucket: str
    pipeline_sa: str
    pipeline_file: str
    auth_token: str
    build_number: str
    is_local: bool
    pipeline_root: Optional[str] = None
    enable_caching: bool = True
    sync: bool = True


def validate_pipeline_inputs(
    config: PipelineConfig, mandatory_fields: List[str]
) -> None:
    """
    Validate input parameters for pipeline configuration.

    Args:
        config: Pipeline configuration object
        mandatory_fields: List of mandatory field names to validate
    """
    logger.info("Validating configuration parameters")

    for field in mandatory_fields:
        value = getattr(config, field)
        if not value or (isinstance(value, str) and value.strip() == ""):
            raise ValueError( # pylint: disable=raising-format-tuple
                "Missing Config Parameter for: %s", field
            )  

    # Validate non-local environment requirements
    if not config.is_local and not config.auth_token:
        raise ValueError("AUTH_TOKEN is required for non-local environment")

    logger.info("Configuration validation passed")


def setup_credentials(config: PipelineConfig) -> Optional[Credentials]:
    """
    Setup credentials for non-local environments.

    Args:
        config: Pipeline configuration

    Returns:
        Optional[Credentials]: Credentials object or None for local environments
    """
    if not config.is_local:
        if not config.auth_token:
            raise ValueError(
                "CLOUDSDK_AUTH_ACCESS_TOKEN environment variable is not set"
            )
        return Credentials(config.auth_token)
    return None


def setup_pipeline_root(config: PipelineConfig) -> str:
    """
    Set pipeline root if not already provided.

    Args:
        config: Pipeline configuration

    Returns:
        str: Pipeline root path
    """
    if not config.pipeline_root:
        config.pipeline_root = f"gs://{config.pipeline_bucket}/pipeline_root"
    return config.pipeline_root


def compile_and_get_pipeline_uri(
    compile_function: Callable,
    config: PipelineConfig,
    credentials: Optional[Credentials],
    job_display_name: str,
    storage_bucket: str,
) -> str:
    """
    Compile the pipeline and return GCS URI.

    Args:
        compile_function: Function to compile the pipeline
        config: Pipeline configuration
        credentials: OAuth2 credentials
        job_display_name: Display name for the pipeline job
        storage_bucket: GCS bucket for pipeline storage

    Returns:
        str: GCS URI of the compiled pipeline
    """
    logger.info("Compiling pipeline")
    return compile_function(
        pipeline_name=job_display_name,
        pipeline_file_name=config.pipeline_file,
        pipeline_storage_bucket=storage_bucket,
        project=config.project_id,
        credentials=credentials,
    )


def create_pipeline_job(
    config: PipelineConfig,
    pipeline_file_gcs_uri: str,
    job_display_name: str,
    parameter_values: Dict[str, Any],
) -> aiplatform.PipelineJob:
    """
    Create and return pipeline job.

    Args:
        config: Pipeline configuration
        pipeline_file_gcs_uri: GCS URI of the compiled pipeline
        job_display_name: Display name for the pipeline job
        parameter_values: Parameters for the pipeline

    Returns:
        aiplatform.PipelineJob: Created pipeline job
    """
    logger.info("Creating pipeline job")
    return aiplatform.PipelineJob(
        display_name=job_display_name,
        template_path=pipeline_file_gcs_uri,
        pipeline_root=config.pipeline_root,
        parameter_values=parameter_values,
        enable_caching=config.enable_caching,
    )


def execute_pipeline_job(
    config: PipelineConfig, job: aiplatform.PipelineJob
) -> aiplatform.PipelineJob:
    """
    Execute the pipeline job and handle sync/async execution.

    Args:
        config: Pipeline configuration
        job: Pipeline job to execute

    Returns:
        aiplatform.PipelineJob: The executed pipeline job
    """
    logger.info("Starting pipeline execution")
    job.run(sync=config.sync)

    if config.sync:
        if job.state == 4:
            logger.info("Pipeline completed successfully")
        else:
            logger.error("Pipeline failed with state: %s", job.state)
    else:
        logger.info("Pipeline started asynchronously")

    return job


# pylint: disable=too-many-arguments
def run_pipeline(
    config: PipelineConfig,
    compile_function: Callable,
    job_display_name: str,
    storage_bucket: str,
    parameter_values: Dict[str, Any],
    mandatory_fields: List[str],
) -> aiplatform.PipelineJob:
    """
    Execute the complete pipeline workflow.

    Args:
        config: Pipeline configuration
        compile_function: Function to compile the pipeline
        job_display_name: Display name for the pipeline job
        storage_bucket: GCS bucket for pipeline storage
        parameter_values: Parameters for the pipeline
        mandatory_fields: List of mandatory configuration fields

    Returns:
        aiplatform.PipelineJob: The executed pipeline job
    """
    try:
        logger.info("Starting Job : %s", job_display_name)

        # Validate configuration
        validate_pipeline_inputs(config, mandatory_fields)

        # Setup credentials
        credentials = setup_credentials(config)

        # Initialize Vertex AI
        logger.info("Initializing Vertex AI")
        initialize_aiplatform(
            project_id=config.project_id,
            region=config.region,
            is_local=config.is_local,
            credentials=credentials,
            service_account=config.pipeline_sa,
        )

        # Set pipeline root
        setup_pipeline_root(config)

        # Compile pipeline
        pipeline_file_gcs_uri = compile_and_get_pipeline_uri(
            compile_function, config, credentials, job_display_name, storage_bucket
        )

        # Create pipeline job
        job = create_pipeline_job(
            config, pipeline_file_gcs_uri, job_display_name, parameter_values
        )

        # Execute pipeline
        return execute_pipeline_job(config, job)

    except ValueError as ve:
        logger.error("Input validation failed: %s", ve)
        raise
    except Exception as e:
        logger.error("Pipeline execution failed: %s", e)
        raise


def compile_and_upload_pipeline(
    pipeline_function: Callable,
    pipeline_name: str,
    pipeline_file_name: str,
    pipeline_storage_bucket: str,
    project: str,
    credentials: Optional[Credentials] = None,
) -> str:
    """
    Compile a Kubeflow pipeline and upload it to Cloud Storage.

    Args:
        pipeline_function: The pipeline function decorated with @dsl.pipeline
        pipeline_name: Pipeline name for organizing in GCS (e.g., "wine_quality_pipeline")
        pipeline_file_name: Pipeline file name (e.g., "training.json")
        pipeline_storage_bucket: GCS bucket name (without gs:// prefix)
        project: Google Cloud project ID
        credentials: Google Cloud credentials (optional for local environments)

    Returns:
        str: GCS URI of the compiled pipeline JSON

    Raises:
        Exception: If compilation or upload fails
    """
    pipeline_gcs_path = f"{pipeline_name}/{pipeline_file_name}"
    pipeline_file_gcs_uri = f"gs://{pipeline_storage_bucket}/{pipeline_gcs_path}"

    try:
        # Compile pipeline to local file
        logger.info("Compiling pipeline to %s", pipeline_file_name)
        compiler.Compiler().compile(
            pipeline_func=pipeline_function, package_path=pipeline_file_name
        )

        # Upload to GCS
        logger.info("Uploading %s to %s", pipeline_file_name, pipeline_file_gcs_uri)
        storage_client = storage.Client(project=project, credentials=credentials)
        bucket = storage_client.bucket(pipeline_storage_bucket)
        blob = bucket.blob(pipeline_gcs_path)
        blob.upload_from_filename(pipeline_file_name)

        logger.info(
            "Pipeline compilation and upload completed successfully to %s",
            pipeline_file_gcs_uri,
        )
        return pipeline_file_gcs_uri

    except Exception as e:
        logger.error(
            "Pipeline compilation failed for project %s with error %s", project, str(e)
        )
        raise e
    finally:
        #  cleanup local file
        if os.path.exists(pipeline_file_name):
            os.unlink(pipeline_file_name)
            logger.info("Cleaned up local file: %s", pipeline_file_name)


def run_pipeline_with_error_handling(
    config: PipelineConfig,
    compile_function: Callable,
    job_display_name: str,
    storage_bucket: str,
    parameter_values: Dict[str, Any],
    mandatory_fields: List[str],
    success_message: str,
) -> None:
    """
    Run pipeline with standardized error handling and logging.

    Args:
        config: Pipeline configuration
        compile_function: Function to compile the pipeline
        job_display_name: Display name for the pipeline job
        storage_bucket: GCS bucket for pipeline storage
        parameter_values: Parameters for the pipeline
        mandatory_fields: List of mandatory configuration fields
        success_message: Message to log on successful completion
    """
    try:
        result = run_pipeline(
            config,
            compile_function,
            job_display_name,
            storage_bucket,
            parameter_values,
            mandatory_fields,
        )

        if result.state == 4:
            logger.info(success_message)
            return True
        
        logger.warning("Pipeline completed with state: %s", result.state)
        return False

    except (ValueError, RuntimeError) as exc:
        logger.error("Pipeline execution failed: %s", exc)
        return True
