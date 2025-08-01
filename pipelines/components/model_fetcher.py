"""Simplified model fetch component with clean metadata extraction."""

from kfp.v2.dsl import Model, Output, component
from pipelines.components.constants import BASE_CONTAINER_IMAGE


# pylint: disable=import-outside-toplevel, too-many-arguments, broad-exception-caught, too-many-positional-arguments, too-many-locals
@component(
    packages_to_install=["google-cloud-aiplatform"],
    base_image=BASE_CONTAINER_IMAGE,
)
def fetch_model(
    fetched_model: Output[Model],
    model_display_name: str,
    project: str,
    location: str,
):
    """
    Fetch latest model with `ready-for-promotion: true` and extract essential metrics.

    Sets only essential metadata for evaluation:
    - Core model info: display_name, resource_name
    - Evaluation metrics: eval_status, quality_score, ready_for_promotion
    - Performance metrics: mae, mse, r2_score, rmse
    """
    from google.cloud import aiplatform
    import logging

    logging.info(
        "Fetching model: %s from project: %s, location: %s",
        model_display_name,
        project,
        location,
    )

    try:
        aiplatform.init(project=project, location=location)
        models = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')

        if not models:
            raise ValueError(f"No models found with display name: {model_display_name}")

        # Get latest model with ready-for-promotion = true
        sorted_models = sorted(models, key=lambda m: m.create_time, reverse=True)

        selected_model = None
        for model in sorted_models:
            labels = model.labels or {}
            if labels.get("ready-for-promotion", "").lower() == "true":
                selected_model = model
                break

        if not selected_model:
            raise ValueError(
                f"No models with 'ready-for-promotion=true' found among {len(sorted_models)} models"
            )

        # Extract labels and convert dashed format to float
        labels = selected_model.labels or {}

        def parse_metric(value_str: str) -> float:
            """Convert dashed format to float: '0-9992' -> 0.9992"""
            try:
                return (
                    float(value_str.replace("-", "."))
                    if "-" in value_str
                    else float(value_str)
                )
            except (ValueError, TypeError):
                return 0.0

        fetched_model.uri = selected_model.resource_name

        # Essential metadata only
        fetched_model.metadata = {
            "display_name": selected_model.display_name,
            "resource_name": selected_model.resource_name,
            "harness_build_id": labels.get("harness-build-id", "unknown"),
            "eval_status": labels.get("eval-status", "unknown"),
            "ready_for_promotion": labels.get("ready-for-promotion", "false"),
            "quality_score": str(parse_metric(labels.get("quality-score", "0"))),
            "mae": str(parse_metric(labels.get("mae", "0"))),
            "mse": str(parse_metric(labels.get("mse", "0"))),
            "r2_score": str(parse_metric(labels.get("r2-score", "0"))),
            "rmse": str(parse_metric(labels.get("rmse", "0"))),
        }

        logging.info("Model fetched successfully: %s", selected_model.display_name)
        logging.info(
            "Key metrics - Quality: %s, Eval status: %s, Ready: %s",
            fetched_model.metadata["quality_score"],
            fetched_model.metadata["eval_status"],
            fetched_model.metadata["ready_for_promotion"],
        )
    except Exception as e:
        logging.error("Model fetch failed: %s", e)

        # Minimal error metadata
        fetched_model.uri = ""
        fetched_model.metadata = {
            "display_name": model_display_name,
            "resource_name": "",
            "harness_build_id": "unknown",
            "eval_status": "failed",
            "ready_for_promotion": "false",
            "quality_score": "0.0",
            "mae": "0.0",
            "mse": "0.0",
            "r2_score": "0.0",
            "rmse": "0.0",
            "error_message": str(e),
        }
        raise
