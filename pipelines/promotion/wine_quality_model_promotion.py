"""Wine quality model registration and deployment pipeline."""

from typing import Optional
from kfp.v2 import dsl
from google.oauth2.credentials import Credentials
from pipelines.components import (
    register_model_gcs,
    deploy_model,
    validate_model_endpoint,
)
from pipelines.shared.pipeline_base_utils import compile_and_upload_pipeline
from pipelines.promotion.constants import (
    PROMOTION_JOB_DISPLAY_NAME,
    PROMOTION_JOB_DISPLAY_DESC,
    MODEL_SERVING_IMAGE,
)


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, no-value-for-parameter
@dsl.pipeline(name=PROMOTION_JOB_DISPLAY_NAME, description=PROMOTION_JOB_DISPLAY_DESC)
def model_promotion_pipeline(
    model_display_name: str,
    endpoint_display_name: str,
    source_project: str,          # Added back for compatibility (unused)
    target_project: str,
    region: str,
    machine_type: str,
    min_replica_count: int,
    max_replica_count: int,
    promotion_threshold: float,   # Added back for compatibility (unused)
    model_gcs_uri: str,
    model_serving_image: str = MODEL_SERVING_IMAGE,
    build_number: str = "unknown",
):
    """
    Create simplified promotion pipeline: Register → Deploy → Validate.

    Pipeline Flow:
    1. Register model from pre-promoted GCS bucket to Vertex AI Model Registry
    2. Deploy registered model to endpoint
    3. Validate deployment endpoint with sample requests

    Args:
        model_display_name: model display name for registry
        endpoint_display_name: model endpoint display name
        source_project: source gcp project (unused, for compatibility)
        target_project: gcp project for prod workload
        region: gcp region
        machine_type: machine type for deployment
        min_replica_count: Minimum replicas for serving
        max_replica_count: Maximum replicas for serving
        promotion_threshold: Quality threshold (unused, for compatibility)
        model_gcs_uri: GCS URI where promoted model artifacts are stored
        model_serving_image: Container image for serving
        build_number: Build identifier for tracking
    """

    # Step 1: Register model from GCS bucket to Vertex AI Model Registry
    register_task = register_model_gcs(
        model_gcs_uri=model_gcs_uri,
        model_display_name=model_display_name,
        project=target_project,
        region=region,
        model_serving_image=model_serving_image,
        build_number=build_number,
    )

    # Step 2: Deploy registered model to endpoint
    deploy_task = deploy_model(
        registered_model=register_task.outputs["registered_model"],
        endpoint_display_name=endpoint_display_name,
        project=target_project,
        region=region,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
    )

    # Step 3: Validate deployment endpoint
    validate_task = validate_model_endpoint(
        endpoint=deploy_task.outputs["deployed_model"],
        project=target_project,
        region=region,
    )

    # Ensure proper execution order
    deploy_task.after(register_task)
    validate_task.after(deploy_task)


def compile_pipeline(
    pipeline_name: str,
    pipeline_file_name: str,
    pipeline_storage_bucket: str,
    project: str,
    credentials: Optional[Credentials] = None,
) -> str:
    """
    Compile the model promotion pipeline and upload to Cloud Storage.

    Args:
        pipeline_name: Pipeline name (e.g., "wine_quality_promotion")
        pipeline_file_name: Pipeline file name (e.g., "promotion.json")
        pipeline_storage_bucket: GCS bucket name (without gs:// prefix)
        project: Google Cloud project ID
        credentials: Google Cloud credentials (optional for local environments)

    Returns:
        str: GCS URI of the compiled pipeline JSON
    """
    return compile_and_upload_pipeline(
        pipeline_function=model_promotion_pipeline,
        pipeline_name=pipeline_name,
        pipeline_file_name=pipeline_file_name,
        pipeline_storage_bucket=pipeline_storage_bucket,
        project=project,
        credentials=credentials,
    )