"""Model registry component for wine quality pipeline."""

from kfp.v2.dsl import Model, Input, Output, component
from pipelines.components.constants import BASE_CONTAINER_IMAGE


# pylint: disable=import-outside-toplevel, too-many-arguments, broad-exception-caught, too-many-positional-arguments, too-many-locals
@component(
    packages_to_install=["google-cloud-aiplatform"], base_image=BASE_CONTAINER_IMAGE
)
def register_model(
    saved_model: Input[Model],
    registered_model: Output[Model],
    model_display_name: str,
    project: str,
    region: str,
    model_serving_image: str,
    build_number: str = "unknown",
):
    """Registers the wine rating model to Vertex AI Model Registry."""
    from google.cloud import aiplatform
    import logging
    import re

    def convert_to_gcp_label_format(value):
        """Convert labels value to google required format e.g mae = 0.0003 to 0-0003"""
        str_value = str(value).lower()
        compliant = re.sub(r"[^a-z0-9_-]", "-", str_value)
        return compliant[:63]

    try:
        logging.info("Starting model registration for %s", model_display_name)
        aiplatform.init(project=project, location=region)
        existing_models = aiplatform.Model.list(
            filter=f'display_name="{model_display_name}"'
        )

        promotion_labels = {
            "ready-for-promotion": "true",
            "harness-build-id": build_number,
            "env": "prd-test",
        }
        metadata = saved_model.metadata or {}

        # Vertex apis will throw limit error more thna 64 labels
        model_metrics = {
            "r2-score": metadata.get("r2_score"),
            "rmse": metadata.get("rmse"),
            "mae": metadata.get("mae"),
            "mse": metadata.get("mse"),
            "quality-score": metadata.get("quality_score"),
            "eval-status": metadata.get("evaluation_status"),
        }

        for metric_key, metric_value in model_metrics.items():
            if metric_value is not None:
                try:
                    if metric_key == "eval-status":
                        promotion_labels[metric_key] = convert_to_gcp_label_format(
                            metric_value
                        )
                    else:
                        converted_metric_value = float(metric_value)
                        formatted_metric_value = (
                            f"{converted_metric_value:.4f}".replace(".", "-")
                        )
                        promotion_labels[metric_key] = formatted_metric_value
                    logging.info(
                        "Added metric label %s: %s",
                        metric_key,
                        promotion_labels[metric_key],
                    )
                except (ValueError, TypeError) as e:
                    logging.info(
                        "Skipping invalid metric %s: %s (%s)",
                        metric_key,
                        metric_value,
                        e,
                    )

        # Additional check to ensure that we don't more than 64 labels
        if len(promotion_labels) > 10:
            logging.info(
                "Too many labels (%d), keeping only essential ones",
                len(promotion_labels),
            )
            promotion_labels = {
                "ready-for-promotion": "true",
                "build-id": build_number,
                "environment": "prd-test",
            }

        logging.info(
            "Final promotion labels (%d): %s", len(promotion_labels), promotion_labels
        )

        if existing_models:
            logging.info("Creating new version for existing model")
            model = aiplatform.Model.upload(
                display_name=model_display_name,
                artifact_uri=saved_model.uri,
                serving_container_image_uri=model_serving_image,
                serving_container_predict_route="/predict",
                serving_container_health_route="/health",
                parent_model=existing_models[0].resource_name,
                is_default_version=True,
                labels=promotion_labels,
            )
        else:
            logging.info("Creating new model")
            model = aiplatform.Model.upload(
                display_name=model_display_name,
                artifact_uri=saved_model.uri,
                serving_container_image_uri=model_serving_image,
                serving_container_predict_route="/predict",
                serving_container_health_route="/health",
                is_default_version=True,
                labels=promotion_labels,
            )

        logging.info("Model registered with promotion labels successfully")
        registered_model.uri = model.resource_name
        registered_model.metadata["display_name"] = model_display_name
        registered_model.metadata["resource_name"] = model.resource_name

        for key, value in saved_model.metadata.items():
            if key not in registered_model.metadata:
                registered_model.metadata[key] = value

        logging.info("Model registration completed successfully")
    except Exception as e:
        logging.error("Model registration failed: %s", e)
        raise
