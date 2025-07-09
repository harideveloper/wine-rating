"""Model registry component for wine quality pipeline."""

from kfp.v2.dsl import Model, Input, Output, component
from constants import BASE_CONTAINER_IMAGE


# pylint: disable=too-many-arguments, too-many-positional-arguments
@component(
    packages_to_install=["google-cloud-aiplatform"], base_image=BASE_CONTAINER_IMAGE
)
def register_model(
    model_artifact: Input[Model],
    registered_model: Output[Model],
    model_display_name: str,
    project: str,
    region: str,
    model_serving_image: str,
):
    """Registers the wine rating model to Vertex AI Model Registry."""
    # pylint: disable=import-outside-toplevel
    from google.cloud import aiplatform
    import logging

    try:
        logging.info("Starting model registration")
        logging.info("Registering model %s to Vertex AI", model_display_name)

        # Initialize Vertex AI
        aiplatform.init(project=project, location=region)

        # Upload model to registry
        logging.info("Uploading model to Vertex AI Model Registry")
        model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=model_artifact.uri,
            serving_container_image_uri=model_serving_image,
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
        )

        logging.info("Model registered successfully")

        # Set registered model metadata
        registered_model.uri = model.resource_name
        registered_model.metadata["display_name"] = model_display_name
        registered_model.metadata["resource_name"] = model.resource_name

        # Copy original metadata
        metadata_count = 0
        for key, value in model_artifact.metadata.items():
            if key not in registered_model.metadata:
                registered_model.metadata[key] = value
                metadata_count += 1

        logging.info("Copied %s metadata items", metadata_count)
        logging.info("Model registration completed successfully")

    except Exception as e:
        logging.error("Data loading failed: %s", e)
        raise
