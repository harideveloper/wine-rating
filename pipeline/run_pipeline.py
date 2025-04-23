"""
Run the wine rating prediction pipeline on Vertex AI.
"""
from google.cloud import aiplatform
from wine_rating_pipeline import compile_pipeline
from typing import Optional
from constants import (
    PROJECT_ID,
    GCS_BUCKET,
    DATA_PATH,
    REGION,
    PIPELINE_FILE,
    PIPELINE_JOB_DISPLAY_NAME,
    PIPELINE_ROOT_SUFFIX,
    MODEL_DISPLAY_NAME,
    ENDPOINT_DISPLAY_NAME,
    EVALUATION_THRESHOLD,
    TEST_SIZE,
    RANDOM_STATE,
    N_ESTIMATORS,
    SERVING_CONTAINER_IMAGE_URI,
    MACHINE_TYPE,
    MIN_REPLICA_COUNT,
    MAX_REPLICA_COUNT
)


def run_pipeline(
    project_id: str,
    gcs_bucket: str,
    data_path: str,
    model_display_name: str = MODEL_DISPLAY_NAME,
    endpoint_display_name: str = ENDPOINT_DISPLAY_NAME,
    region: str = REGION,
    pipeline_root: Optional[str] = None,
    evaluation_threshold: float = EVALUATION_THRESHOLD,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    n_estimators: int = N_ESTIMATORS,
    serving_container_image_uri: str = SERVING_CONTAINER_IMAGE_URI,
    machine_type: str = MACHINE_TYPE,
    min_replica_count: int = MIN_REPLICA_COUNT,
    max_replica_count: int = MAX_REPLICA_COUNT
):
    """Run the wine rating prediction pipeline on Vertex AI."""
    aiplatform.init(project=project_id, location=region)

    if pipeline_root is None:
        pipeline_root = f"gs://{gcs_bucket}/{PIPELINE_ROOT_SUFFIX}"

    compile_pipeline(PIPELINE_FILE)

    job = aiplatform.PipelineJob(
        display_name=PIPELINE_JOB_DISPLAY_NAME,
        template_path=PIPELINE_FILE,
        pipeline_root=pipeline_root,
        parameter_values={
            "data_path": data_path,
            "model_display_name": model_display_name,
            "endpoint_display_name": endpoint_display_name,
            "project": project_id,
            "region": region,
            "evaluation_threshold": evaluation_threshold,
            "test_size": test_size,
            "random_state": random_state,
            "n_estimators": n_estimators,
            "serving_container_image_uri": serving_container_image_uri,
            "machine_type": machine_type,
            "min_replica_count": min_replica_count,
            "max_replica_count": max_replica_count
        },
        enable_caching=True
    )

    job.run(sync=True)

    print(f"Pipeline job launched: {job.display_name}")
    print(f"Pipeline job ID: {job.resource_name}")

    return job


if __name__ == "__main__":
    job = run_pipeline(
        project_id=PROJECT_ID,
        gcs_bucket=GCS_BUCKET,
        data_path=DATA_PATH,
        model_display_name=MODEL_DISPLAY_NAME,
        endpoint_display_name=ENDPOINT_DISPLAY_NAME,
        region=REGION,
        evaluation_threshold=EVALUATION_THRESHOLD,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        n_estimators=N_ESTIMATORS,
        serving_container_image_uri=SERVING_CONTAINER_IMAGE_URI,
        machine_type=MACHINE_TYPE,
        min_replica_count=MIN_REPLICA_COUNT,
        max_replica_count=MAX_REPLICA_COUNT
    )

    print(f"Pipeline job completed with state: {job.state}")
