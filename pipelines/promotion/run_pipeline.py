"""Run the wine quality prediction promotion pipeline."""

import sys
from dataclasses import dataclass
from typing import Dict, Any, List
from pipelines.promotion.wine_quality_model_promotion import (
    compile_pipeline,
)
from pipelines.shared.pipeline_base_utils import (
    PipelineConfig,
    run_pipeline_with_error_handling,
)
from pipelines.promotion.constants import (
    AUTH_TOKEN,
    SOURCE_PROJECT_ID,
    TARGET_PROJECT_ID,
    REGION,
    PIPELINE_FILE,
    PROMOTION_JOB_DISPLAY_NAME,
    MODEL_DISPLAY_NAME,
    MODEL_ENDPOINT_DISPLAY_NAME,
    PIPELINE_BUCKET,
    PIPELINE_SA,
    BUILD_NUMBER,
    MACHINE_TYPE,
    MIN_REPLICA_COUNT,
    MAX_REPLICA_COUNT,
    IS_LOCAL,
    PROMOTION_THRESHOLD,
    MODEL_GCS_URI,
    MODEL_SERVING_IMAGE,
)


# pylint: disable=too-many-instance-attributes, duplicate-code
@dataclass
class PromotionPipelineConfig(PipelineConfig):
    """Configuration for model promotion pipeline."""

    # Base config
    project_id: str = TARGET_PROJECT_ID
    region: str = REGION
    pipeline_bucket: str = PIPELINE_BUCKET
    pipeline_sa: str = PIPELINE_SA
    pipeline_file: str = PIPELINE_FILE
    auth_token: str = AUTH_TOKEN
    build_number: str = BUILD_NUMBER
    is_local: bool = IS_LOCAL

    # Promotion config
    source_project_id: str = SOURCE_PROJECT_ID
    target_project_id: str = TARGET_PROJECT_ID
    model_display_name: str = MODEL_DISPLAY_NAME
    endpoint_display_name: str = MODEL_ENDPOINT_DISPLAY_NAME
    machine_type: str = MACHINE_TYPE
    min_replica_count: int = MIN_REPLICA_COUNT
    max_replica_count: int = MAX_REPLICA_COUNT
    promotion_threshold: float = PROMOTION_THRESHOLD
    model_gcs_uri: str = MODEL_GCS_URI
    model_serving_image: str = MODEL_SERVING_IMAGE


def get_mandatory_fields() -> List[str]:
    """Return list of mandatory configuration fields."""
    return [
        "source_project_id",
        "target_project_id",
        "region",
        "pipeline_bucket",
        "pipeline_sa",
        "pipeline_file",
        "model_display_name",
        "machine_type",
        "min_replica_count",
        "max_replica_count",
        "promotion_threshold",
        "model_gcs_uri",
        "model_serving_image",
    ]


def get_parameter_values(config: PromotionPipelineConfig) -> Dict[str, Any]:
    """Return parameter values for the promotion pipeline."""
    return {
        "source_project": config.source_project_id,
        "target_project": config.target_project_id,
        "region": config.region,
        "model_display_name": config.model_display_name,
        "endpoint_display_name": config.endpoint_display_name,
        "machine_type": config.machine_type,
        "min_replica_count": config.min_replica_count,
        "max_replica_count": config.max_replica_count,
        "promotion_threshold": config.promotion_threshold,
        "model_gcs_uri": config.model_gcs_uri,
        "model_serving_image": config.model_serving_image,
        "build_number": config.build_number,
    }


def main():
    """Main function for promotion pipeline execution."""
    config = PromotionPipelineConfig()

    return run_pipeline_with_error_handling(
        config=config,
        compile_function=compile_pipeline,
        job_display_name=PROMOTION_JOB_DISPLAY_NAME,
        storage_bucket=PIPELINE_BUCKET,
        parameter_values=get_parameter_values(config),
        mandatory_fields=get_mandatory_fields(),
        success_message="Success! Wine quality model promotion pipeline completed",
    )


if __name__ == "__main__":
    sys.exit(0 if main() else 1)