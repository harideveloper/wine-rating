"""Model registry component for direct GCS registration."""

from kfp.v2.dsl import Model, Output, component
from pipelines.components.constants import BASE_CONTAINER_IMAGE


# pylint: disable=import-outside-toplevel, too-many-arguments, broad-exception-caught, too-many-positional-arguments, too-many-locals
@component(
    packages_to_install=["google-cloud-aiplatform"], base_image=BASE_CONTAINER_IMAGE
)
def register_model_gcs(
    registered_model: Output[Model],
    model_gcs_uri: str,
    model_display_name: str,
    project: str,
    region: str,
    model_serving_image: str,
    build_number: str = "unknown",
):
    """
    Register model from GCS URI to Vertex AI Model Registry.
    
    Args:
        registered_model: Output model artifact
        model_gcs_uri: GCS URI where model artifacts are stored
        model_display_name: Display name for the model
        project: GCP project ID
        region: GCP region
        model_serving_image: Container image for serving
        build_number: Build number for tracking
    """
    from google.cloud import aiplatform
    import logging
    import re

    def convert_to_gcp_label_format(value):
        """Convert value to GCP label format."""
        str_value = str(value).lower()
        compliant = re.sub(r"[^a-z0-9_-]", "-", str_value)
        return compliant[:63]

    try:
        logging.info("Starting model registration for %s", model_display_name)
        logging.info("Model GCS URI: %s", model_gcs_uri)
        
        aiplatform.init(project=project, location=region)
        
        # Check for existing models
        existing_models = aiplatform.Model.list(
            filter=f'display_name="{model_display_name}"'
        )

        # Create labels for tracking
        labels = {
            "env": "production",
            "build-id": convert_to_gcp_label_format(build_number),
            "promoted": "true",
        }

        logging.info("Using labels: %s", labels)

        if existing_models:
            logging.info("Creating new version for existing model")
            model = aiplatform.Model.upload(
                display_name=model_display_name,
                artifact_uri=model_gcs_uri,
                serving_container_image_uri=model_serving_image,
                serving_container_predict_route="/predict",
                serving_container_health_route="/health",
                parent_model=existing_models[0].resource_name,
                is_default_version=True,
                labels=labels,
                sync=True,
            )
        else:
            logging.info("Creating new model")
            model = aiplatform.Model.upload(
                display_name=model_display_name,
                artifact_uri=model_gcs_uri,
                serving_container_image_uri=model_serving_image,
                serving_container_predict_route="/predict",
                serving_container_health_route="/health",
                is_default_version=True,
                labels=labels,
                sync=True,
            )

        logging.info("Model registered successfully: %s", model.resource_name)
        
        # Set output model metadata
        registered_model.uri = model.resource_name
        registered_model.metadata["display_name"] = model_display_name
        registered_model.metadata["resource_name"] = model.resource_name
        registered_model.metadata["model_gcs_uri"] = model_gcs_uri
        registered_model.metadata["build_number"] = build_number
        registered_model.metadata["registration_status"] = "completed"

        logging.info("Model registration completed successfully")
        
    except Exception as e:
        logging.error("Model registration failed: %s", e)
        # Set error metadata
        registered_model.metadata["registration_status"] = "failed"
        registered_model.metadata["error_message"] = str(e)
        raise