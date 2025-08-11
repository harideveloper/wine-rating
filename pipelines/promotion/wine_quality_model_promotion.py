"""Wine quality online prediction model promotion pipeline."""

from typing import Optional
from kfp.v2 import dsl
from google.oauth2.credentials import Credentials
from pipelines.components import (
    fetch_model,
    promotion_gate,
    promote_model,
    deploy_model,
    validate_model_endpoint,
)
from pipelines.shared.pipeline_base_utils import compile_and_upload_pipeline
from pipelines.promotion.constants import (
    PROMOTION_JOB_DISPLAY_NAME,
    PROMOTION_JOB_DISPLAY_DESC,
)


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, no-value-for-parameter
@dsl.pipeline(name=PROMOTION_JOB_DISPLAY_NAME, description=PROMOTION_JOB_DISPLAY_DESC)
def model_promotion_pipeline(
    model_display_name: str,
    endpoint_display_name: str,
    source_project: str,
    target_project: str,
    region: str,
    machine_type: str,
    min_replica_count: int,
    max_replica_count: int,
    promotion_threshold: float = 0.95,
):
    """
    Create wine quality online promotion pipeline with gate.

    Pipeline Flow:
    1. Fetch model and extract metadata from registry
    2. Check promotion gate (validate promotion criteria)
    3. Promote model (only if gate passes)
    4. Deploy to target environment
    5. Validate deployment endpoint sending sample requests

    Args:
        model_display_name: model display name from source model registry
        endpoint_display_name: model endpoint display name
        source_project: gcp project for prod test spoke
        target_project: gcp project for prod workload spoke
        region: gcp region
        machine_type: machine type for deployment
        min_replica_count: Minimum replicas for serving
        max_replica_count: Maximum replicas for serving
        promotion_threshold: Quality score threshold for promotion gate (default: 0.95)
    """

    # Step 1: Fetch model and extract metadata from registry labels
    fetch_model_task = fetch_model(
        model_display_name=model_display_name,
        project=source_project,
        location=region,
    )

    # Step 2: Check promotion gate (validate criteria)
    promotion_gate_task = promotion_gate(
        fetched_model=fetch_model_task.outputs["fetched_model"],
        promotion_threshold=promotion_threshold,
    )

    # Step 3: Conditional promotion based on gate result
    with dsl.Condition(
        promotion_gate_task.output == True,  # pylint: disable=no-member, singleton-comparison
        name="promotion_gate_passed",
    ):
        # Step 4: Promote model to target project
        promote_model_task = promote_model(
            fetched_model=fetch_model_task.outputs["fetched_model"],
            source_project=source_project,
            target_project=target_project,
            location=region,
        )

        # Step 5: Deploy promoted model
        deploy_task = deploy_model(
            registered_model=promote_model_task.outputs["promoted_model"],
            endpoint_display_name=endpoint_display_name,
            project=target_project,
            region=region,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
        )

        # Step 6: Validate deployment endpoint
        validate_task = validate_model_endpoint(
            endpoint=deploy_task.outputs["deployed_model"],
            project=target_project,
            region=region,
        )

        promote_model_task.after(promotion_gate_task)
        deploy_task.after(promote_model_task)
        validate_task.after(deploy_task)


def compile_pipeline(
    pipeline_name: str,
    pipeline_file_name: str,
    pipeline_storage_bucket: str,
    project: str,
    credentials: Optional[Credentials] = None,
) -> str:
    """
    Compile the wine quality prediction pipeline and upload to Cloud Storage.

    Args:
        pipeline_name: Pipeline name (e.g., "wine_quality_pipeline")
        pipeline_file_name: Pipeline file name (e.g., "training.json")
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
