"""Wine quality online prediction pipeline definition."""

from kfp.v2 import dsl, compiler
import logging
from components import (
    load_data,
    preprocess_data,
    train_model,
    evaluate_model,
    save_model,
    register_model,
    deploy_model,
    validate_model
)

from constants import (
    PIPELINE_JOB_DISPLAY_NAME,
    PIPELINE_JOB_DISPLAY_DESC,
)


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
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments,no-value-for-parameter
    load_task = load_data(data_path=data_path)
    preprocess_task = preprocess_data(
        input_data=load_task.outputs["output_data"],
        test_size=test_size,
        random_state=random_state,
    )
    train_task = train_model(
        train_data=preprocess_task.outputs["train_data"],
        n_estimators=n_estimators,
        random_state=random_state,
    )
    evaluate_task = evaluate_model(
        model_artifact=train_task.outputs["output_model"],
        test_data=preprocess_task.outputs["test_data"],
    )
    with dsl.Condition(
        evaluate_task.output >= evaluation_threshold, name="model deployment"
    ):
        save_task = save_model(model_artifact=train_task.outputs["output_model"])
        register_task = register_model(
            model_artifact=save_task.outputs["uploaded_model_artifact"],
            model_display_name=model_display_name,
            project=project,
            region=region,
            model_serving_image=model_serving_image,
        )
        deploy_task = deploy_model(
            model_registry_name=register_task.outputs["registered_model"],
            endpoint_display_name=endpoint_display_name,
            project=project,
            region=region,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
        )
        validate_task = validate_model(
            endpoint=deploy_task.outputs["endpoint"],
            project=project,
            region=region,
        )
        deploy_task.after(register_task)
        validate_task.after(deploy_task)

def compile_pipeline(pipeline_file_name: str, pipeline_storage_bucket: str, project: str) -> str:
    """
    Compile the wine quality prediction pipeline and upload to Cloud Storage.
    
    Args:
        pipeline_file_name: Name of the pipeline file (e.g., "training.json" or "pipeline-build-123.json")
        pipeline_storage_bucket: GCS bucket name to store the pipeline (without gs:// prefix)
        project: Google Cloud project ID
        
    Returns:
        str: GCS URI of the compiled pipeline JSON (gs://bucket/pipeline_file_name)
        
    Raises:
        Exception: If compilation or upload fails
    """
    import os
    from google.cloud import storage
    
    # Construct GCS URI
    pipeline_file_gcs_uri = f"gs://{pipeline_storage_bucket}/{pipeline_file_name}"
    
    try:
        logging.info("Starting pipeline compilation for project %s", project)
        
        # Compile pipeline to local file
        compiler.Compiler().compile(
            pipeline_func=wine_quality_online_predictor_pipeline,
            package_path=pipeline_file_name
        )
        
        # Upload to GCS
        storage_client = storage.Client(project=project)
        bucket = storage_client.bucket(pipeline_storage_bucket)
        blob = bucket.blob(pipeline_file_name)
        blob.upload_from_filename(pipeline_file_name)
        
        logging.info("Pipeline compilation completed successfully to %s", pipeline_file_gcs_uri)
        return pipeline_file_gcs_uri
        
    except Exception as e:
        logging.error("Pipeline compilation failed for project %s: %s", project, str(e))
        raise e
        
    finally:
        # Clean up local file
        if os.path.exists(pipeline_file_name):
            os.unlink(pipeline_file_name)