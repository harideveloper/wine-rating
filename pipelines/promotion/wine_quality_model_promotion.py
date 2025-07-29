import logging
from kfp.v2 import dsl,compiler
from kfp.v2.dsl import component, Model, Output, Condition


@component(packages_to_install=["google-cloud-aiplatform"])
def fetch_model(
    model_display_name: str,
    project: str,
    location: str,
) -> str:
    """Fetch latest model with given name that has label `ready-for-promotion: true`."""
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=location)
    models = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')

    if not models:
        raise ValueError(f"No models found with display name: {model_display_name}")

    # Sort by creation time (latest first)
    sorted_models = sorted(models, key=lambda m: m.create_time, reverse=True)

    for model in sorted_models:
        labels = model.labels or {}
        if labels.get("ready-for-promotion", "").lower() == "true":
            return model.resource_name

    raise ValueError(f"No models with required labels: ready-for-promotion=true")


@component(packages_to_install=["google-cloud-aiplatform"])
def promote_model(
    model_resource_name: str,
    source_project: str,
    target_project: str,
    location: str,
    promoted_model: Output[Model]
):
    """Copy model to target project with production tag."""
    from google.cloud import aiplatform
    import datetime

    aiplatform.init(project=source_project, location=location)
    source_model = aiplatform.Model(model_resource_name)

    aiplatform.init(project=target_project, location=location)
    prod_labels = (source_model.labels or {}).copy()
    prod_labels.update({
        "environment": "production",
        "promoted-from": source_project,
        "promotion-date": datetime.datetime.utcnow().date().isoformat()
    })

    target_display_name = f"{source_model.display_name}_prod"

    new_model = aiplatform.Model.upload(
        display_name=target_display_name,
        artifact_uri=source_model.uri,
        serving_container_image_uri=source_model.container_spec.image_uri,
        labels=prod_labels,
        description=f"Promoted from {source_project}",
        sync=True
    )

    promoted_model.uri = new_model.resource_name
    promoted_model.metadata["display_name"] = new_model.display_name


@dsl.pipeline(
    name="simplified-model-promotion",
    description="Promote model with minimal validation"
)
def model_promotion_pipeline(
    model_display_name: str,
    source_project: str,
    target_project: str,
    location: str,
):
    # Step 1: Get latest promotable model
    fetch_model_task = fetch_model(
        model_display_name=model_display_name,
        project=source_project,
        location=location,
    )
    # evaluate_metrics()

    # Step 2: Promote to target project
    promote_model(
        model_resource_name=fetch_model_task.output,
        source_project=source_project,
        target_project=target_project,
        location=location
    )


def compile_pipeline(
    pipeline_name: str,
    pipeline_file_name: str,
    pipeline_storage_bucket: str,
    project: str,
    credentials
) -> str:
    """Compile the model promotion pipeline and upload to Cloud Storage."""
    try:
        logging.info("Compiling model promotion pipeline")
        
        # Compile pipeline
        compiler.Compiler().compile(
            pipeline_func=model_promotion_pipeline,
            package_path=pipeline_file_name
        )
        
        # Upload to GCS
        from google.cloud import storage
        
        client = storage.Client(project=project)
        bucket = client.bucket(pipeline_storage_bucket)
        blob = bucket.blob(pipeline_file_name)
        
        with open(pipeline_file_name, 'rb') as f:
            blob.upload_from_file(f)
        
        gcs_uri = f"gs://{pipeline_storage_bucket}/{pipeline_file_name}"
        logging.info(f"Pipeline compiled and uploaded to: {gcs_uri}")
        
        return gcs_uri
        
    except Exception as e:
        logging.error(f"Pipeline compilation failed: {str(e)}")
        raise