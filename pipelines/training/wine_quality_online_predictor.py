"""Wine quality online prediction pipeline component."""

from typing import Optional
from kfp.v2 import dsl
from google.oauth2.credentials import Credentials
from pipelines.components import (
    load_data,
    preprocess_data,
    train_model,
    evaluate_model,
    save_model,
    register_model,
    deploy_model,
    validate_model_endpoint,
)
from pipelines.training.constants import (
    PIPELINE_JOB_DISPLAY_NAME,
    PIPELINE_JOB_DISPLAY_DESC,
)
from pipelines.shared.pipeline_base_utils import compile_and_upload_pipeline


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, no-value-for-parameter
@dsl.pipeline(name=PIPELINE_JOB_DISPLAY_NAME, description=PIPELINE_JOB_DISPLAY_DESC)
def wine_quality_online_predictor_pipeline(
    data_path: str,
    model_display_name: str,
    endpoint_display_name: str,
    project: str,
    region: str,
    evaluation_threshold: float,
    test_size: float,
    random_state: int,
    n_estimators: int,
    machine_type: str,
    min_replica_count: int,
    max_replica_count: int,
    model_serving_image: str,
    build_number: str,
):
    """
    Create wine quality online prediction pipeline.

    Args:
        data_path: GCS path to training data
        model_display_name: Display name for the model
        endpoint_display_name: Display name for the endpoint
        project: GCP project ID
        region: GCP region
        evaluation_threshold: Threshold for model evaluation
        test_size: Test split size
        random_state: Random seed
        n_estimators: Number of estimators for random forest
        machine_type: Machine type for training
        min_replica_count: Minimum replicas for serving
        max_replica_count: Maximum replicas for serving
        model_serving_image: Container image for serving
        build_number: Build number for the training pipeline
    """
    # Load and preprocess data
    load_task = load_data(data_path=data_path)

    preprocess_task = preprocess_data(
        input_data=load_task.outputs["output_data"],
        test_size=test_size,
        random_state=random_state,
    )

    # Train model
    train_task = train_model(
        train_data=preprocess_task.outputs["train_data"],
        n_estimators=n_estimators,
        random_state=random_state,
    )

    # Evaluate model
    evaluate_task = evaluate_model(
        trained_model=train_task.outputs["trained_model"],
        test_data=preprocess_task.outputs["test_data"],
    )

    # Conditional deployment based on evaluation threshold
    with dsl.Condition(
        evaluate_task.outputs["Output"] >= evaluation_threshold,
        name="model deployment",
    ):
        # Save model
        save_task = save_model(evaluated_model=evaluate_task.outputs["evaluated_model"])

        # Register model
        register_task = register_model(
            saved_model=save_task.outputs["saved_model"],
            model_display_name=model_display_name,
            project=project,
            region=region,
            model_serving_image=model_serving_image,
            build_number=build_number,
        )

        # # Deploy model
        # deploy_task = deploy_model(
        #     registered_model=register_task.outputs["registered_model"],
        #     endpoint_display_name=endpoint_display_name,
        #     project=project,
        #     region=region,
        #     machine_type=machine_type,
        #     min_replica_count=min_replica_count,
        #     max_replica_count=max_replica_count,
        # )

        # # Validate endpoint
        # validate_task = validate_model_endpoint(
        #     endpoint=deploy_task.outputs["deployed_model"],
        #     project=project,
        #     region=region,
        # )
        # validate_task.after(deploy_task)


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
        pipeline_function=wine_quality_online_predictor_pipeline,
        pipeline_name=pipeline_name,
        pipeline_file_name=pipeline_file_name,
        pipeline_storage_bucket=pipeline_storage_bucket,
        project=project,
        credentials=credentials,
    )
