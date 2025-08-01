"""Run the wine quality prediction training pipeline."""
from dataclasses import dataclass
from typing import Dict, Any, List
from training import compile_pipeline
from shared.pipeline_base_utils import (
    PipelineConfig,
    run_pipeline_with_error_handling
)
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
    IS_LOCAL,
)


@dataclass
class TrainingPipelineConfig(PipelineConfig):
    """Configuration for wine quality training pipeline."""
    # Base fields
    project_id: str = PROJECT_ID
    region: str = REGION
    pipeline_bucket: str = PIPELINE_BUCKET
    pipeline_sa: str = PIPELINE_SA
    pipeline_file: str = PIPELINE_FILE
    auth_token: str = AUTH_TOKEN
    build_number: str = BUILD_NUMBER
    is_local: bool = IS_LOCAL
    
    # Training-specific fields
    data_bucket: str = DATA_BUCKET
    data_path: str = DATA_PATH
    model_display_name: str = MODEL_DISPLAY_NAME
    endpoint_display_name: str = MODEL_ENDPOINT_DISPLAY_NAME
    evaluation_threshold: float = EVALUATION_THRESHOLD
    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE
    n_estimators: int = N_ESTIMATORS
    machine_type: str = MACHINE_TYPE
    min_replica_count: int = MIN_REPLICA_COUNT
    max_replica_count: int = MAX_REPLICA_COUNT
    model_serving_image: str = MODEL_SERVING_IMAGE


def get_mandatory_fields() -> List[str]:
    """Return list of mandatory configuration fields."""
    return [
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
        "max_replica_count",
        "model_serving_image",
        "build_number",
    ]


def get_parameter_values(config: TrainingPipelineConfig) -> Dict[str, Any]:
    """Return parameter values for the training pipeline."""
    return {
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
        "build_number": config.build_number,
    }


if __name__ == "__main__":
    config = TrainingPipelineConfig()
    
    run_pipeline_with_error_handling(
        config=config,
        compile_function=compile_pipeline,
        job_display_name=PIPELINE_JOB_DISPLAY_NAME,
        storage_bucket=DATA_BUCKET,
        parameter_values=get_parameter_values(config),
        mandatory_fields=get_mandatory_fields(),
        success_message="Success! Wine quality online predictions are ready"
    )