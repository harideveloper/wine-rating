"""Model registry component for wine quality pipeline."""

from kfp.v2.dsl import Model, Input, Output, component
from constants import BASE_CONTAINER_IMAGE


# pylint: disable=too-many-arguments
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
    build_number: str = "1"
):
    """Registers the wine rating model to Vertex AI Model Registry."""
    # pylint: disable=import-outside-toplevel
    from google.cloud import aiplatform
    import logging
    import re

    def convert_to_gcp_label_format(value):
        """Make label value compliant with GCP requirements."""
        # Convert to string, lowercase, replace invalid chars
        str_value = str(value).lower()
        compliant = re.sub(r'[^a-z0-9_-]', '-', str_value)
        # Limit length to 63 characters
        return compliant[:63]

    try:
        logging.info("Starting model registration for %s", model_display_name)
        aiplatform.init(project=project, location=region)

        # list existing model
        existing_models = aiplatform.Model.list(
            filter=f'display_name="{model_display_name}"'
        )

        # Start with essential promotion labels (GCP compliant)
        promotion_labels = {
            "ready-for-promotion": "true",
            "harness-build-id": build_number,
            "env": "prd-test"
        }

        # Add key evaluation metrics as labels (limit to most important ones)
        try:
            metadata = model_artifact.metadata or {}
            
            # Only add the most important metrics to stay under 64 label limit
            key_metrics = {
                "r2-score": metadata.get("r2_score"),
                "rmse": metadata.get("rmse"),
                "mae": metadata.get("mae"),
                "mse": metadata.get("mse"),
                "quality-score": metadata.get("quality_score"),
                "eval-status": metadata.get("evaluation_status")
            }
            
            # Add metrics with safe conversion
            for metric_key, metric_value in key_metrics.items():
                if metric_value is not None:
                    try:
                        if metric_key == "eval-status":
                            # Handle string values
                            promotion_labels[metric_key] = convert_to_gcp_label_format(metric_value)
                        else:
                            # Handle numeric values
                            float_value = float(metric_value)
                            # Format as string without decimal point for label compliance
                            formatted_value = f"{float_value:.3f}".replace(".", "-")
                            promotion_labels[metric_key] = formatted_value
                        
                        logging.info("Added metric label %s: %s", metric_key, promotion_labels[metric_key])
                    except (ValueError, TypeError) as e:
                        logging.info("Skipping invalid metric %s: %s (%s)", metric_key, metric_value, e)
                        
        except Exception as metrics_error:
            logging.info("Could not extract metrics for labels: %s", metrics_error)

        # Ensure we don't exceed label limits
        if len(promotion_labels) > 10:  # Keep well under 64 limit
            logging.info("Too many labels (%d), keeping only essential ones", len(promotion_labels))
            # Keep only essential labels
            promotion_labels = {
                "ready-for-promotion": "true",
                "build-id": build_number,
                "environment": "prd-test"
            }

        logging.info("Final promotion labels (%d): %s", len(promotion_labels), promotion_labels)

        # Upload model (creates new version if parent exists) with labels
        if existing_models:
            logging.info("Creating new version for existing model")
            model = aiplatform.Model.upload(
                display_name=model_display_name,
                artifact_uri=model_artifact.uri,
                serving_container_image_uri=model_serving_image,
                serving_container_predict_route="/predict",
                serving_container_health_route="/health",
                parent_model=existing_models[0].resource_name,
                is_default_version=True,
                labels=promotion_labels  # Add labels during upload
            )
        else:
            logging.info("Creating new model")
            model = aiplatform.Model.upload(
                display_name=model_display_name,
                artifact_uri=model_artifact.uri,
                serving_container_image_uri=model_serving_image,
                serving_container_predict_route="/predict",
                serving_container_health_route="/health",
                is_default_version=True,
                labels=promotion_labels  # Add labels during upload
            )
        
        logging.info("Model registered with promotion labels successfully")

        registered_model.uri = model.resource_name
        registered_model.metadata["display_name"] = model_display_name
        registered_model.metadata["resource_name"] = model.resource_name
        for key, value in model_artifact.metadata.items():
            if key not in registered_model.metadata:
                registered_model.metadata[key] = value
        logging.info("Model registration completed successfully")
        
    except Exception as e:
        logging.error("Model registration failed: %s", e)
        raise